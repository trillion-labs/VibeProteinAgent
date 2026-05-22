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
from vibeproteinagent.adapters._remote.geneontology import (
    extract_uniprot_accessions,
    go_associations,
)
from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.temporal import before_cutoff
from vibeproteinagent.adapters._remote.uniprot import (
    UniProtRecord,
    fetch_accession,
    search_accessions,
    to_fasta_record,
)
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class GOSequencesInput(BaseModel):
    go_id: str = Field(pattern=r"^GO:[0-9]{7}$")
    aspect: str | None = Field(
        None,
        description=(
            "GO aspect endpoint: function/molecular_function, "
            "process/biological_process, or component/cellular_component."
        ),
    )
    max_sequences: int = Field(1000, ge=1, le=10000)
    reviewed: bool | None = None
    taxon_id: int | None = None
    evidence_codes: list[str] = Field(default_factory=list)
    exclude_iea: bool = False
    min_length: int | None = Field(None, ge=1)
    max_length: int | None = Field(None, ge=1)
    temporal_cutoff: str | None = None
    include_undated: bool = False
    deduplicate_sequences: bool = True
    page_size: int = Field(200, ge=1, le=500)
    max_concurrency: int = Field(5, ge=1, le=20)
    timeout_s: float = Field(20.0, gt=0, le=120)
    max_scan_accessions: int | None = Field(None, ge=1, le=50000)


class GOSequencesOutput(BaseModel):
    go_id: str
    go_label: str | None
    association_records: int
    resolved_uniprot_accessions: int
    kept_sequences: int
    fasta_path: str
    metadata_jsonl: str
    dropped: dict[str, int]
    summary: dict[str, Any]
    artifacts: dict[str, str]
    runtime_ms: int


def _evidence_ok(row: dict[str, Any], params: GOSequencesInput) -> bool:
    raw_evidence = row.get("evidence")
    evidence = str(
        raw_evidence.get("type", "")
        if isinstance(raw_evidence, dict)
        else row.get("evidence", "")
    )
    if params.exclude_iea and evidence.upper().endswith("IEA"):
        return False
    if params.evidence_codes and not any(
        code.upper() in evidence.upper() for code in params.evidence_codes
    ):
        return False
    return True


def _keep_record(
    record: UniProtRecord,
    params: GOSequencesInput,
) -> tuple[bool, str | None]:
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
    return True, None


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


@register_adapter("descriptor.yaml")
class GOSequencesAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: GOSequencesInput, ctx: AdapterRunContext) -> GOSequencesOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=out_dir / "http_cache")
        associations, raw_go = go_associations(
            client,
            params.go_id,
            rows=params.page_size,
            aspect=params.aspect,
            taxon_id=params.taxon_id,
        )
        filtered_assoc = [row for row in associations if _evidence_ok(row, params)]
        accessions, unresolved = extract_uniprot_accessions(filtered_assoc)
        fallback_used = False
        fallback_suppressed = False
        evidence_constrained = bool(params.evidence_codes or params.exclude_iea)
        max_scan = params.max_scan_accessions or min(
            50000,
            max(params.max_sequences * 20, params.max_sequences + 200),
        )
        if len(accessions) < min(params.max_sequences, 10) and not evidence_constrained:
            fallback_used = True
            query = f"go:{params.go_id.split(':', 1)[1].lower()}"
            if params.reviewed is True:
                query += " AND reviewed:true"
            if params.taxon_id:
                query += f" AND organism_id:{params.taxon_id}"
            accessions.extend(search_accessions(client, query, size=max_scan))
        elif len(accessions) < min(params.max_sequences, 10) and evidence_constrained:
            fallback_suppressed = True
        accessions = list(dict.fromkeys(accessions))[:max_scan]
        ctx.emit_progress(0.30)
        kept: list[UniProtRecord] = []
        dropped_rows: list[dict[str, Any]] = []
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
                    dropped_rows.append(
                        {
                            "accession": accession,
                            "reason": "unresolved_identifier",
                            "error": str(error),
                        }
                    )
                    continue
                if record is None:
                    dropped_rows.append(
                        {"accession": accession, "reason": "unresolved_identifier"}
                    )
                    continue
                keep, reason = _keep_record(record, params)
                if keep and params.deduplicate_sequences and record.sequence in seen_sequences:
                    keep, reason = False, "duplicate_sequence"
                if keep:
                    seen_sequences.add(record.sequence)
                    kept.append(record)
                else:
                    dropped_rows.append(
                        {
                            "accession": accession,
                            "reason": reason,
                            "first_public_date": record.first_public_date,
                        }
                    )
            ctx.emit_progress(
                min(0.9, 0.30 + 0.60 * min(len(fetched), len(accessions)) / max(1, len(accessions)))
            )
        fasta_path = out_dir / "go_sequences.fasta"
        metadata_path = out_dir / "go_sequences_metadata.jsonl"
        association_path = out_dir / "go_association_records.jsonl"
        unresolved_path = out_dir / "go_unresolved_records.jsonl"
        summary_path = out_dir / "go_sequences_summary.json"
        raw_path = out_dir / "go_raw_response.json"
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
        with association_path.open("w") as fh:
            for row in filtered_assoc:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        with unresolved_path.open("w") as fh:
            for row in unresolved + dropped_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        with raw_uniprot_path.open("w") as fh:
            for _accession, record, _error in fetched:
                if record is not None:
                    fh.write(json.dumps(record.metadata, sort_keys=True) + "\n")
        raw_path.write_text(json.dumps(raw_go, indent=2, sort_keys=True) + "\n")
        dropped = {
            reason: sum(1 for d in dropped_rows if d.get("reason") == reason)
            for reason in sorted({str(d.get("reason")) for d in dropped_rows})
        }
        dropped["unresolved_identifier"] = dropped.get("unresolved_identifier", 0) + len(unresolved)
        lengths = [r.length or 0 for r in kept]
        summary = {
            "fallback_used": fallback_used,
            "fallback_suppressed_by_evidence_filter": fallback_suppressed,
            "candidate_accessions": len(accessions),
            "scanned_accessions": len(fetched),
            "max_scan_accessions": max_scan,
            "scan_exhausted_before_fill": len(kept) < params.max_sequences
            and len(fetched) >= len(accessions),
            "length_min": min(lengths) if lengths else None,
            "length_median": statistics.median(lengths) if lengths else None,
            "length_max": max(lengths) if lengths else None,
            "reviewed_count": sum(1 for r in kept if r.reviewed),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        ctx.emit_progress(1.0)
        return GOSequencesOutput(
            go_id=params.go_id,
            go_label=raw_go.get("label") or raw_go.get("name"),
            association_records=len(filtered_assoc),
            resolved_uniprot_accessions=sum(1 for _a, record, _e in fetched if record is not None),
            kept_sequences=len(kept),
            fasta_path=str(fasta_path),
            metadata_jsonl=str(metadata_path),
            dropped=dropped,
            summary=summary,
            artifacts={
                "association_records_jsonl": str(association_path),
                "unresolved_records_jsonl": str(unresolved_path),
                "summary_json": str(summary_path),
                "raw_response_json": str(raw_path),
                "uniprot_raw_records_jsonl": str(raw_uniprot_path),
            },
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: GOSequencesInput) -> int:
        return 0


__all__ = ["GOSequencesAdapter", "GOSequencesInput", "GOSequencesOutput"]
