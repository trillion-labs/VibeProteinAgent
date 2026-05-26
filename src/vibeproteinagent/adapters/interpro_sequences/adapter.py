from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.fasta import write_fasta
from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.interpro import interpro_accessions
from vibeproteinagent.adapters._remote.temporal import before_cutoff
from vibeproteinagent.adapters._remote.uniprot import (
    UniProtRecord,
    fetch_accession,
    to_fasta_record,
)
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import stable_candidate_id


def _keep_record(
    record: UniProtRecord,
    params: InterProSequencesInput,
) -> tuple[bool, str | None]:
    # ``taxon_id`` and ``reviewed`` are now applied server-side via the
    # InterPro URL path (see :func:`_remote.interpro._build_accession_url`)
    # so the InterPro page only contains accessions that already match.
    # We keep the same checks here as a defence-in-depth safety net — if
    # the API ever changes its filter semantics, we still don't claim a
    # human protein when it's actually from a different taxon.
    if params.taxon_id is not None and not _record_matches_taxon(record, params.taxon_id):
        return False, "taxon_filter"
    if (
        params.reviewed is not None
        and record.reviewed is not None
        and record.reviewed != params.reviewed
    ):
        return False, "reviewed_filter"
    if params.min_length is not None and (record.length or 0) < params.min_length:
        return False, "length_filter"
    if params.max_length is not None and (record.length or 0) > params.max_length:
        return False, "length_filter"
    if params.temporal_cutoff and not record.first_public_date and not params.include_undated:
        return False, "undated"
    if not before_cutoff(
        record.first_public_date,
        params.temporal_cutoff,
        include_undated=params.include_undated,
    ):
        return False, "temporal_filter"
    if not params.include_fragments and _is_fragment_record(record.metadata):
        return False, "fragment_filter"
    return True, None


def _is_fragment_record(metadata: Any) -> bool:
    """Return True iff the UniProt record is marked as a fragment.

    The canonical place for this flag in UniProt JSON is
    ``proteinDescription.flag``, which carries one of:

    * ``"Fragment"``           — incomplete sequence (one end missing)
    * ``"Fragments"``          — multiple non-contiguous pieces
    * ``"Precursor"``          — proproteolytic precursor (NOT a fragment;
                                  do NOT drop)
    * ``"Fragment,Precursor"`` — both flags simultaneously

    Only entries whose ``flag`` contains ``"fragment"`` (case-insensitive)
    are dropped. Previous implementation did a substring search on the
    full serialised metadata, which incorrectly rejected intact proteins
    whose references, keywords, GO terms, or feature descriptions
    happened to contain the word "fragment".
    """
    if not isinstance(metadata, dict):
        return False
    pd = metadata.get("proteinDescription")
    if not isinstance(pd, dict):
        return False
    flag = pd.get("flag")
    if isinstance(flag, str) and "fragment" in flag.lower():
        return True
    return False


def _record_matches_taxon(record: UniProtRecord, taxon_id: int) -> bool:
    organism = record.metadata.get("organism") if isinstance(record.metadata, dict) else None
    if isinstance(organism, dict):
        observed = organism.get("taxonId") or organism.get("taxon_id")
        if observed is not None:
            return str(observed) == str(taxon_id)
    lineage = json.dumps(record.metadata).lower()
    return f"ncbitaxon:{taxon_id}" in lineage or f"taxonid={taxon_id}" in lineage


def _fetch_records(
    client: HttpClient,
    accessions: list[str],
    *,
    max_concurrency: int,
    fetcher: Callable[[HttpClient, str], UniProtRecord | None],
) -> list[tuple[str, UniProtRecord | None, Exception | None]]:
    if max_concurrency <= 1:
        out = []
        for accession in accessions:
            try:
                out.append((accession, fetcher(client, accession), None))
            except Exception as exc:
                out.append((accession, None, exc))
        return out

    def _one(accession: str) -> tuple[str, UniProtRecord | None, Exception | None]:
        try:
            return accession, fetcher(client, accession), None
        except Exception as exc:
            return accession, None, exc

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        return list(pool.map(_one, accessions))


class InterProSequencesInput(BaseModel):
    interpro_id: str = Field(pattern=r"^IPR[0-9]{6}$")
    max_sequences: int = Field(1000, ge=1, le=10000)
    reviewed: bool | None = None
    taxon_id: int | None = None
    min_length: int | None = Field(None, ge=1)
    max_length: int | None = Field(None, ge=1)
    temporal_cutoff: str | None = None
    include_fragments: bool = False
    include_undated: bool = False
    deduplicate_sequences: bool = True
    page_size: int = Field(200, ge=1, le=500)
    max_concurrency: int = Field(5, ge=1, le=20)
    timeout_s: float = Field(20.0, gt=0, le=120)
    max_scan_accessions: int | None = Field(None, ge=1, le=50000)


class InterProSequencesOutput(BaseModel):
    interpro_id: str
    retrieved_accessions: int
    kept_sequences: int
    dropped_by_temporal_filter: int
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    fasta_path: str
    metadata_jsonl: str
    summary: dict[str, Any]
    artifacts: dict[str, str]
    runtime_ms: int


@register_adapter("descriptor.yaml")
class InterProSequencesAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: InterProSequencesInput,
        ctx: AdapterRunContext,
    ) -> InterProSequencesOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=out_dir / "http_cache")
        max_scan = params.max_scan_accessions or min(
            50000,
            max(params.max_sequences * 20, params.max_sequences + 200),
        )
        accessions, raw_pages = interpro_accessions(
            client,
            params.interpro_id,
            max_records=max_scan,
            page_size=params.page_size,
            taxon_id=params.taxon_id,
            reviewed=params.reviewed,
        )
        ctx.emit_progress(0.25)
        kept: list[UniProtRecord] = []
        dropped: list[dict[str, Any]] = []
        seen_sequences: set[str] = set()
        fetched: list[tuple[str, UniProtRecord | None, Exception | None]] = []
        fetch_batch_size = max(1, min(params.page_size, params.max_sequences, max_scan))
        for start in range(0, len(accessions), fetch_batch_size):
            if len(kept) >= params.max_sequences:
                break
            batch = accessions[start : start + fetch_batch_size]
            batch_results = _fetch_records(
                client,
                batch,
                max_concurrency=params.max_concurrency,
                fetcher=fetch_accession,
            )
            fetched.extend(batch_results)
            for accession, record, error in batch_results:
                if len(kept) >= params.max_sequences:
                    break
                if error is not None:
                    dropped.append(
                        {
                            "accession": accession,
                            "reason": "unresolved",
                            "error": str(error),
                        }
                    )
                    continue
                if record is None:
                    dropped.append({"accession": accession, "reason": "unresolved"})
                    continue
                keep, reason = _keep_record(record, params)
                if keep and params.deduplicate_sequences and record.sequence in seen_sequences:
                    keep, reason = False, "duplicate_sequence"
                if keep:
                    seen_sequences.add(record.sequence)
                    kept.append(record)
                else:
                    dropped.append(
                        {
                            "accession": accession,
                            "reason": reason,
                            "first_public_date": record.first_public_date,
                        }
                    )
            ctx.emit_progress(
                min(0.9, 0.25 + 0.65 * min(len(fetched), len(accessions)) / max(1, len(accessions)))
            )
        fasta_path = out_dir / "interpro_sequences.fasta"
        metadata_path = out_dir / "interpro_sequences_metadata.jsonl"
        summary_path = out_dir / "interpro_sequences_summary.json"
        dropped_path = out_dir / "interpro_dropped_records.jsonl"
        raw_path = out_dir / "interpro_raw_pages.json"
        raw_uniprot_path = out_dir / "uniprot_raw_records.jsonl"
        write_fasta([to_fasta_record(r) for r in kept], fasta_path)
        with metadata_path.open("w") as fh:
            for r in kept:
                fh.write(
                    json.dumps(
                        {
                            "accession": r.accession,
                            "first_public_date": r.first_public_date,
                            "reviewed": r.reviewed,
                            "length": r.length,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        with dropped_path.open("w") as fh:
            for row in dropped:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        with raw_uniprot_path.open("w") as fh:
            for _accession, record, _error in fetched:
                if record is not None:
                    fh.write(json.dumps(record.metadata, sort_keys=True) + "\n")
        lengths = [r.length or 0 for r in kept]
        summary = {
            "candidate_accessions": len(accessions),
            "scanned_accessions": len(fetched),
            "max_scan_accessions": max_scan,
            "scan_exhausted_before_fill": len(kept) < params.max_sequences
            and len(fetched) >= len(accessions),
            "length_min": min(lengths) if lengths else None,
            "length_median": statistics.median(lengths) if lengths else None,
            "length_max": max(lengths) if lengths else None,
            "reviewed_count": sum(1 for r in kept if r.reviewed),
            "dropped": {
                reason: sum(1 for d in dropped if d.get("reason") == reason)
                for reason in sorted({str(d.get("reason")) for d in dropped})
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        raw_path.write_text(json.dumps(raw_pages, indent=2, sort_keys=True) + "\n")
        ctx.emit_progress(1.0)
        return InterProSequencesOutput(
            interpro_id=params.interpro_id,
            retrieved_accessions=len(accessions),
            kept_sequences=len(kept),
            dropped_by_temporal_filter=sum(
                1 for d in dropped if d.get("reason") == "temporal_filter"
            ),
            candidates=[
                _canonical_candidate_from_record(record, index)
                for index, record in enumerate(kept)
            ],
            fasta_path=str(fasta_path),
            metadata_jsonl=str(metadata_path),
            summary=summary,
            artifacts={
                "summary_json": str(summary_path),
                "dropped_jsonl": str(dropped_path),
                "raw_pages_json": str(raw_path),
                "uniprot_raw_records_jsonl": str(raw_uniprot_path),
            },
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: InterProSequencesInput) -> int:
        return 0


def _canonical_candidate_from_record(record: UniProtRecord, index: int) -> dict[str, Any]:
    local_id = f"seq_{index:04d}"
    return {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            "interpro_sequences",
            local_id,
            native_id=record.accession,
            sequence=record.sequence,
            index=index,
        ),
        "source": {
            "tool": "interpro_sequences",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": record.accession,
            "index": index,
        },
        "sequence": record.sequence,
        "scores": {
            "interpro_sequences": {
                "length": record.length,
                "reviewed": record.reviewed,
            }
        },
        "artifacts": {},
    }


__all__ = [
    "InterProSequencesAdapter",
    "InterProSequencesInput",
    "InterProSequencesOutput",
]
