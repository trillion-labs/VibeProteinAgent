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
from vibeproteinagent.adapters._remote.sequence_relevance import (
    intent_quota_top_k,
    sequence_relevance_score,
)
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
    interpro_id: str | None = Field(default=None, pattern=r"^IPR[0-9]{6}$")
    interpro_terms: list[dict[str, Any]] = Field(default_factory=list)
    intent_id: str | None = None
    intent_label: str | None = None
    term_relevance_score: float | None = None
    term_score_breakdown: dict[str, Any] = Field(default_factory=dict)
    max_sequences: int = Field(1000, ge=1, le=10000)
    max_downstream_candidates: int = Field(
        25,
        ge=1,
        le=1000,
        description=(
            "Maximum scored sequence candidates emitted downstream. Retrieval "
            "artifacts may contain more kept sequences; candidates are bounded "
            "to the top-K by semantic relevance."
        ),
    )
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
    empty_facets: list[dict[str, Any]] = Field(default_factory=list)
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
        term_specs = _interpro_term_specs(params)
        if not params.interpro_id and not term_specs:
            return _empty_no_term_output(params, ctx)
        if len(term_specs) > 1:
            return self._run_multi_term(params, ctx, term_specs)
        if not params.interpro_id and term_specs:
            params = _params_for_interpro_term(params, term_specs[0])
        assert params.interpro_id is not None
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
        raw_path.write_text(json.dumps(raw_pages, indent=2, sort_keys=True) + "\n")
        ctx.emit_progress(1.0)
        candidate_rows = []
        for record in kept:
            semantic_relevance, relevance_breakdown = sequence_relevance_score(
                metadata=record.metadata,
                reviewed=record.reviewed,
                length=record.length,
                term_match_score=float(params.term_relevance_score or 0.0),
                intent_query=" ".join(
                    item
                    for item in [params.interpro_id, params.intent_label]
                    if item
                ),
            )
            candidate_rows.append(
                {
                    "record": record,
                    "sequence_relevance_score": semantic_relevance,
                    "term_relevance_score": params.term_relevance_score,
                    "score_breakdown": relevance_breakdown,
                    "intent_id": params.intent_id or params.interpro_id,
                }
            )
        bounded_rows = intent_quota_top_k(
            candidate_rows,
            max_candidates=min(params.max_downstream_candidates, len(candidate_rows) or 1),
        )
        candidates = [
            _canonical_candidate_from_record(
                row["record"],
                index,
                semantic_relevance=row.get("sequence_relevance_score"),
                semantic_relevance_breakdown=row.get("score_breakdown"),
                intent={
                    "intent_id": params.intent_id or params.interpro_id,
                    "source": "interpro_sequences",
                    "term_id": params.interpro_id,
                    "label": params.intent_label,
                    "term_relevance_score": params.term_relevance_score,
                    "score_breakdown": params.term_score_breakdown,
                    "selected_reason": _selected_reason(params),
                },
            )
            for index, row in enumerate(bounded_rows)
        ]
        summary.update(
            {
                "downstream_candidate_count": len(candidates),
                "max_downstream_candidates": params.max_downstream_candidates,
                "sequence_relevance_bounded": True,
            }
        )
        empty_facets = _retrieval_empty_facets(params, candidates)
        if empty_facets:
            summary["empty_facets"] = empty_facets
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return InterProSequencesOutput(
            interpro_id=params.interpro_id,
            retrieved_accessions=len(accessions),
            kept_sequences=len(kept),
            dropped_by_temporal_filter=sum(
                1 for d in dropped if d.get("reason") == "temporal_filter"
            ),
            candidates=candidates,
            empty_facets=empty_facets,
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

    def _run_multi_term(
        self,
        params: InterProSequencesInput,
        ctx: AdapterRunContext,
        term_specs: list[dict[str, Any]],
    ) -> InterProSequencesOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        child_outputs: list[InterProSequencesOutput] = []
        total = max(1, len(term_specs))
        for index, term in enumerate(term_specs[:5]):
            child_params = _params_for_interpro_term(params, term)
            child_ctx = AdapterRunContext(
                job_id=ctx.job_id,
                workspace_root=ctx.workspace_root,
                job_dir=out_dir / "terms" / f"interpro_{index:02d}_{child_params.interpro_id}",
                is_cancelled=ctx.is_cancelled,
                log=ctx.log,
                emit_progress=lambda pct, index=index, total=total: ctx.emit_progress(
                    min(0.99, (index + pct) / total)
                ),
                fusion_inputs=ctx.fusion_inputs,
            )
            child_outputs.append(self.run(child_params, child_ctx))
        candidate_rows = []
        for child in child_outputs:
            for candidate in child.candidates:
                score = (((candidate.get("scores") or {}).get("interpro_sequences") or {}).get("sequence_relevance_score") or 0.0)
                intent = ((candidate.get("scores") or {}).get("interpro_sequences") or {}).get("intent") or {}
                candidate_rows.append(
                    {
                        "candidate": candidate,
                        "sequence_relevance_score": score,
                        "intent_id": intent.get("intent_id"),
                    }
                )
        bounded = intent_quota_top_k(
            candidate_rows,
            max_candidates=min(params.max_downstream_candidates, len(candidate_rows) or 1),
        )
        candidates = [dict(row["candidate"]) for row in bounded]
        for index, candidate in enumerate(candidates):
            source = candidate.get("source")
            if isinstance(source, dict):
                source["index"] = index
        summary = {
            "multi_intent": True,
            "term_count": len(child_outputs),
            "terms": [child.interpro_id for child in child_outputs],
            "downstream_candidate_count": len(candidates),
            "max_downstream_candidates": params.max_downstream_candidates,
            "sequence_relevance_bounded": True,
        }
        empty_facets = [
            facet
            for child in child_outputs
            for facet in getattr(child, "empty_facets", [])
            if isinstance(facet, dict)
        ]
        if empty_facets:
            summary["empty_facets"] = empty_facets
        summary_path = out_dir / "interpro_sequences_summary.json"
        metadata_path = out_dir / "interpro_sequences_metadata.jsonl"
        fasta_path = out_dir / "interpro_sequences.fasta"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        _merge_text_artifacts(
            [Path(child.metadata_jsonl) for child in child_outputs],
            metadata_path,
        )
        _merge_text_artifacts(
            [Path(child.fasta_path) for child in child_outputs],
            fasta_path,
        )
        ctx.emit_progress(1.0)
        first = child_outputs[0] if child_outputs else None
        return InterProSequencesOutput(
            interpro_id=first.interpro_id if first else str(term_specs[0].get("term_id") or term_specs[0].get("id")),
            retrieved_accessions=sum(child.retrieved_accessions for child in child_outputs),
            kept_sequences=sum(child.kept_sequences for child in child_outputs),
            dropped_by_temporal_filter=sum(child.dropped_by_temporal_filter for child in child_outputs),
            candidates=candidates,
            empty_facets=empty_facets,
            fasta_path=str(fasta_path),
            metadata_jsonl=str(metadata_path),
            summary=summary,
            artifacts={
                "summary_json": str(summary_path),
                "child_metadata_jsonl": str(metadata_path),
                "child_fasta": str(fasta_path),
            },
            runtime_ms=int((time.time() - t0) * 1000),
        )


def _canonical_candidate_from_record(
    record: UniProtRecord,
    index: int,
    *,
    semantic_relevance: Any = None,
    semantic_relevance_breakdown: Any = None,
    intent: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_id = f"seq_{index:04d}"
    semantic_relevance_payload = {
        "sequence_relevance_score": semantic_relevance,
        "term_relevance_score": (intent or {}).get("term_relevance_score")
        if isinstance(intent, dict)
        else None,
        "score_breakdown": dict(semantic_relevance_breakdown or {}),
        "intent_id": (intent or {}).get("intent_id") if isinstance(intent, dict) else None,
        "selected_reason": _selected_reason_from_intent(intent),
        "intent": dict(intent or {}),
    }
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
        "capabilities": {
            "has_sequence": True,
            "has_candidate_scores": True,
        },
        "scores": {
            "semantic_relevance": semantic_relevance_payload,
            "interpro_sequences": {
                "length": record.length,
                "reviewed": record.reviewed,
                "sequence_relevance_score": semantic_relevance,
                "term_relevance_score": (intent or {}).get("term_relevance_score")
                if isinstance(intent, dict)
                else None,
                "semantic_relevance": {
                    "score": semantic_relevance,
                    "score_breakdown": dict(semantic_relevance_breakdown or {}),
                    "selected_reason": _selected_reason_from_intent(intent),
                    "facets": [dict(intent or {})],
                },
                "semantic_relevance_breakdown": semantic_relevance_breakdown,
                "intent": intent,
            }
        },
        "artifacts": {},
    }


def _interpro_term_specs(params: InterProSequencesInput) -> list[dict[str, Any]]:
    if params.interpro_terms:
        return [
            {
                **dict(term),
                "term_id": term.get("term_id") or term.get("id"),
            }
            for term in params.interpro_terms[:5]
            if isinstance(term, dict) and (term.get("term_id") or term.get("id"))
        ]
    if params.interpro_id:
        return [
            {
                "term_id": params.interpro_id,
                "intent_id": params.intent_id or params.interpro_id,
                "intent_label": params.intent_label,
                "term_relevance_score": params.term_relevance_score,
                "score_breakdown": params.term_score_breakdown,
            }
        ]
    return []


def _params_for_interpro_term(
    params: InterProSequencesInput,
    term: dict[str, Any],
) -> InterProSequencesInput:
    return params.model_copy(
        update={
            "interpro_id": str(term.get("term_id") or term.get("id")),
            "interpro_terms": [],
            "intent_id": term.get("intent_id") or term.get("term_id") or term.get("id"),
            "intent_label": term.get("intent_label") or term.get("label"),
            "term_relevance_score": term.get("term_relevance_score"),
            "term_score_breakdown": term.get("score_breakdown") or {},
        }
    )


def _empty_no_term_output(
    params: InterProSequencesInput,
    ctx: AdapterRunContext,
) -> InterProSequencesOutput:
    out_dir = ctx.job_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "interpro_sequences.fasta"
    metadata_path = out_dir / "interpro_sequences_metadata.jsonl"
    summary_path = out_dir / "interpro_sequences_summary.json"
    dropped_path = out_dir / "interpro_dropped_records.jsonl"
    raw_path = out_dir / "interpro_raw_pages.json"
    raw_uniprot_path = out_dir / "uniprot_raw_records.jsonl"
    for path in (fasta_path, metadata_path, dropped_path, raw_uniprot_path):
        path.write_text("")
    raw_path.write_text("[]\n")
    empty_facets = [_no_term_empty_event(params)]
    summary = {
        "retrieval_status": "no_terms",
        "reason": "no InterPro terms were provided by resolver or user input",
        "candidate_accessions": 0,
        "scanned_accessions": 0,
        "downstream_candidate_count": 0,
        "max_downstream_candidates": params.max_downstream_candidates,
        "sequence_relevance_bounded": True,
        "empty_facets": empty_facets,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    ctx.emit_progress(1.0)
    return InterProSequencesOutput(
        interpro_id=params.interpro_id or "",
        retrieved_accessions=0,
        kept_sequences=0,
        dropped_by_temporal_filter=0,
        candidates=[],
        empty_facets=empty_facets,
        fasta_path=str(fasta_path),
        metadata_jsonl=str(metadata_path),
        summary=summary,
        artifacts={
            "summary_json": str(summary_path),
            "dropped_jsonl": str(dropped_path),
            "raw_pages_json": str(raw_path),
            "uniprot_raw_records_jsonl": str(raw_uniprot_path),
        },
        runtime_ms=0,
    )


def _no_term_empty_event(params: InterProSequencesInput) -> dict[str, Any]:
    return {
        "event_type": "empty_candidate_event",
        "intent_id": params.intent_id,
        "label": params.intent_label or "No InterPro term selected",
        "query": params.intent_label,
        "selected_term": None,
        "term_id": None,
        "term_relevance_score": params.term_relevance_score,
        "score_breakdown": dict(params.term_score_breakdown or {}),
        "selected_reason": "resolver did not select any InterPro term for this retrieval stream",
        "resolver_evidence": {
            "intent_id": params.intent_id,
            "intent_label": params.intent_label,
            "term_score_breakdown": dict(params.term_score_breakdown or {}),
        },
        "retrieval_status": "no_terms",
        "candidate_count": 0,
        "stage": "interpro_sequences",
    }


def _retrieval_empty_facets(
    params: InterProSequencesInput,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidates:
        return []
    if not params.intent_id and not params.interpro_id:
        return []
    return [
        {
            "event_type": "empty_candidate_event",
            "intent_id": params.intent_id or params.interpro_id,
            "label": params.intent_label,
            "selected_term": params.interpro_id,
            "term_id": params.interpro_id,
            "term_relevance_score": params.term_relevance_score,
            "score_breakdown": dict(params.term_score_breakdown or {}),
            "selected_reason": _selected_reason(params),
            "resolver_evidence": {
                "intent_id": params.intent_id or params.interpro_id,
                "intent_label": params.intent_label,
                "term_score_breakdown": dict(params.term_score_breakdown or {}),
            },
            "retrieval_status": "empty",
            "candidate_count": 0,
            "stage": "interpro_sequences",
        }
    ]


def _selected_reason(params: InterProSequencesInput) -> str:
    if params.intent_label:
        return f"selected for query facet {params.intent_label!r}"
    if params.intent_id and params.interpro_id and params.intent_id != params.interpro_id:
        return f"selected term {params.interpro_id} for resolver facet {params.intent_id}"
    if params.interpro_id:
        return f"selected term {params.interpro_id}"
    return "selected by resolver"


def _selected_reason_from_intent(intent: Any) -> str:
    if isinstance(intent, dict):
        reason = intent.get("selected_reason")
        if isinstance(reason, str) and reason.strip():
            return reason
        label = intent.get("label")
        if isinstance(label, str) and label.strip():
            return f"selected for query facet {label!r}"
        term_id = intent.get("term_id")
        if isinstance(term_id, str) and term_id.strip():
            return f"selected term {term_id}"
    return "selected by resolver"


def _merge_text_artifacts(inputs: list[Path], output: Path) -> None:
    with output.open("w") as out:
        for path in inputs:
            if not path.is_file():
                continue
            text = path.read_text()
            if not text:
                continue
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")


__all__ = [
    "InterProSequencesAdapter",
    "InterProSequencesInput",
    "InterProSequencesOutput",
]
