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

from vibeproteinagent.adapters._remote.fasta import FastaRecord, write_fasta
from vibeproteinagent.adapters._remote.geneontology import (
    evidence_letter_code,
    extract_uniprot_accessions,
    go_associations,
)
from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.sequence_relevance import (
    intent_quota_top_k,
    sequence_relevance_score,
)
from vibeproteinagent.adapters._remote.temporal import before_cutoff
from vibeproteinagent.adapters._remote.uniprot import (
    UniProtRecord,
    fetch_accession,
    search_accessions,
    to_fasta_record,
)
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import stable_candidate_id


class GOSequencesInput(BaseModel):
    go_id: str | None = Field(default=None, pattern=r"^GO:[0-9]{7}$")
    go_terms: list[dict[str, Any]] = Field(default_factory=list)
    intent_id: str | None = None
    intent_label: str | None = None
    term_relevance_score: float | None = None
    term_score_breakdown: dict[str, Any] = Field(default_factory=dict)
    aspect: str | None = Field(
        None,
        description=(
            "GO aspect endpoint: function/molecular_function, "
            "process/biological_process, or component/cellular_component."
        ),
    )
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
    raw_result_count: int
    kept_sequences: int
    kept_sequence_count: int
    dropped_empty_sequence_count: int
    sequences: list[dict[str, Any]]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    empty_facets: list[dict[str, Any]] = Field(default_factory=list)
    fasta_path: str
    metadata_jsonl: str
    dropped: dict[str, int]
    summary: dict[str, Any]
    artifacts: dict[str, str]
    runtime_ms: int


def _evidence_ok(row: dict[str, Any], params: GOSequencesInput) -> bool:
    """Apply the user's evidence-code filters against one association row.

    The GO REST API expresses evidence as ECO identifiers
    (e.g. ``"ECO:0000501"``) on the ``evidence`` field and an
    ``evidence_types: [{id: ...}]`` list. The user, by convention,
    expresses filters with three-letter GOC codes
    (``"IEA"``, ``"IDA"``, ...). :func:`evidence_letter_code`
    normalises whatever shape arrives into the matching letter code so
    filters work regardless of which API form the server uses.

    Unknown / unrecognised codes return ``""`` from the helper. We
    treat that as "can't classify": ``exclude_iea`` lets it through
    (we don't know it's IEA), but a ``evidence_codes`` allowlist
    drops it (the user asked for specific codes and this isn't one).
    """
    letter = evidence_letter_code(row.get("evidence"))
    if not letter:
        # Try the alternative ``evidence_types`` payload before giving up.
        evidence_types = row.get("evidence_types")
        if isinstance(evidence_types, list) and evidence_types:
            letter = evidence_letter_code(evidence_types[0])
    if params.exclude_iea and letter == "IEA":
        return False
    if params.evidence_codes:
        wanted = {str(code).strip().upper() for code in params.evidence_codes if code}
        if letter not in wanted:
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
        term_specs = _go_term_specs(params)
        if not params.go_id and not term_specs:
            return _empty_no_term_output(params, ctx)
        if len(term_specs) > 1:
            return self._run_multi_term(params, ctx, term_specs)
        if not params.go_id and term_specs:
            params = _params_for_go_term(params, term_specs[0])
        assert params.go_id is not None
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
                if not str(record.sequence or "").strip():
                    dropped_rows.append(
                        {"accession": accession, "reason": "empty_sequence"}
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
        dropped_empty_sequence_count = dropped.get("empty_sequence", 0)
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
        intent_query = " ".join(
            str(item)
            for item in (
                params.go_id,
                raw_go.get("label") or raw_go.get("name") or "",
                raw_go.get("definition") or "",
            )
            if item
        )
        sequences = []
        scored_candidate_rows = []
        for r in kept:
            semantic_relevance, relevance_breakdown = sequence_relevance_score(
                metadata=r.metadata,
                reviewed=r.reviewed,
                length=r.length,
                term_match_score=float(params.term_relevance_score or 0.0),
                evidence_quality_score=_evidence_quality_score(filtered_assoc),
                intent_query=intent_query,
            )
            sequences.append(
                {
                    "candidate_id": f"go:{params.go_id}:{r.accession}",
                    "accession": r.accession,
                    "sequence": r.sequence,
                    "length": r.length,
                    "reviewed": r.reviewed,
                }
            )
            scored_candidate_rows.append(
                {
                    "candidate_id": f"go:{params.go_id}:{r.accession}",
                    "accession": r.accession,
                    "sequence": r.sequence,
                    "length": r.length,
                    "reviewed": r.reviewed,
                    "sequence_relevance_score": semantic_relevance,
                    "term_relevance_score": params.term_relevance_score,
                    "score_breakdown": relevance_breakdown,
                    "intent_id": params.intent_id or params.go_id,
                    "intent": {
                        "intent_id": params.intent_id or params.go_id,
                        "source": "go_sequences",
                        "term_id": params.go_id,
                        "label": params.intent_label or raw_go.get("label") or raw_go.get("name"),
                        "term_relevance_score": params.term_relevance_score,
                        "score_breakdown": params.term_score_breakdown,
                        "selected_reason": _selected_reason(params),
                    },
                }
            )
        candidate_rows = intent_quota_top_k(
            scored_candidate_rows,
            max_candidates=min(params.max_downstream_candidates, len(sequences) or 1),
        )
        candidates = [
            _canonical_candidate_from_sequence(row, index)
            for index, row in enumerate(candidate_rows)
        ]
        summary.update(
            {
                "downstream_candidate_count": len(candidates),
                "max_downstream_candidates": params.max_downstream_candidates,
                "sequence_relevance_bounded": True,
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return GOSequencesOutput(
            go_id=params.go_id,
            go_label=raw_go.get("label") or raw_go.get("name"),
            association_records=len(filtered_assoc),
            resolved_uniprot_accessions=sum(1 for _a, record, _e in fetched if record is not None),
            raw_result_count=len(fetched),
            kept_sequences=len(kept),
            kept_sequence_count=len(kept),
            dropped_empty_sequence_count=dropped_empty_sequence_count,
            sequences=sequences,
            candidates=candidates,
            empty_facets=_retrieval_empty_facets(params, candidates),
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

    def _run_multi_term(
        self,
        params: GOSequencesInput,
        ctx: AdapterRunContext,
        term_specs: list[dict[str, Any]],
    ) -> GOSequencesOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        child_outputs: list[GOSequencesOutput] = []
        total = max(1, len(term_specs))
        for index, term in enumerate(term_specs[:5]):
            child_params = _params_for_go_term(params, term)
            child_ctx = AdapterRunContext(
                job_id=ctx.job_id,
                workspace_root=ctx.workspace_root,
                job_dir=out_dir / "terms" / f"go_{index:02d}_{str(child_params.go_id).replace(':', '_')}",
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
                score = (((candidate.get("scores") or {}).get("go_sequences") or {}).get("sequence_relevance_score") or 0.0)
                intent = ((candidate.get("scores") or {}).get("go_sequences") or {}).get("intent") or {}
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
        sequences = [
            item
            for child in child_outputs
            for item in child.sequences
            if isinstance(item, dict)
        ]
        summary = {
            "multi_intent": True,
            "term_count": len(child_outputs),
            "terms": [child.go_id for child in child_outputs],
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
        summary_path = out_dir / "go_sequences_summary.json"
        fasta_path = out_dir / "go_sequences.fasta"
        metadata_path = out_dir / "go_sequences_metadata.jsonl"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        write_fasta(
            [
                FastaRecord(
                    str(item.get("accession") or item.get("candidate_id")),
                    str(item.get("sequence") or ""),
                )
                for item in sequences
                if item.get("sequence")
            ],
            fasta_path,
        )
        with metadata_path.open("w") as fh:
            for item in sequences:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
        ctx.emit_progress(1.0)
        first = child_outputs[0] if child_outputs else None
        return GOSequencesOutput(
            go_id=first.go_id if first else str(term_specs[0].get("term_id") or term_specs[0].get("id")),
            go_label=first.go_label if first else None,
            association_records=sum(child.association_records for child in child_outputs),
            resolved_uniprot_accessions=sum(child.resolved_uniprot_accessions for child in child_outputs),
            raw_result_count=sum(child.raw_result_count for child in child_outputs),
            kept_sequences=sum(child.kept_sequences for child in child_outputs),
            kept_sequence_count=sum(child.kept_sequence_count for child in child_outputs),
            dropped_empty_sequence_count=sum(child.dropped_empty_sequence_count for child in child_outputs),
            sequences=sequences,
            candidates=candidates,
            empty_facets=empty_facets,
            fasta_path=str(fasta_path),
            metadata_jsonl=str(metadata_path),
            dropped={},
            summary=summary,
            artifacts={"summary_json": str(summary_path)},
            runtime_ms=int((time.time() - t0) * 1000),
        )


def _canonical_candidate_from_sequence(row: dict[str, Any], index: int) -> dict[str, Any]:
    local_id = f"seq_{index:04d}"
    native_id = row.get("accession")
    sequence = str(row.get("sequence") or "")
    semantic_relevance = {
        "sequence_relevance_score": row.get("sequence_relevance_score"),
        "term_relevance_score": row.get("term_relevance_score"),
        "score_breakdown": dict(row.get("score_breakdown") or {}),
        "intent_id": (row.get("intent") or {}).get("intent_id")
        if isinstance(row.get("intent"), dict)
        else None,
        "selected_reason": _selected_reason_from_intent(row.get("intent")),
        "intent": dict(row.get("intent") or {}),
    }
    return {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            "go_sequences",
            local_id,
            native_id=str(native_id) if native_id else None,
            sequence=sequence,
            index=index,
        ),
        "source": {
            "tool": "go_sequences",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": str(native_id) if native_id else None,
            "index": index,
        },
        "sequence": sequence,
        "capabilities": {
            "has_sequence": True,
            "has_candidate_scores": True,
        },
        "scores": {
            "semantic_relevance": semantic_relevance,
            "go_sequences": {
                "length": row.get("length"),
                "reviewed": row.get("reviewed"),
                "sequence_relevance_score": row.get("sequence_relevance_score"),
                "term_relevance_score": row.get("term_relevance_score"),
                "semantic_relevance": {
                    "score": row.get("sequence_relevance_score"),
                    "score_breakdown": dict(row.get("score_breakdown") or {}),
                    "selected_reason": _selected_reason_from_intent(row.get("intent")),
                    "facets": [dict(row.get("intent") or {})],
                },
                "semantic_relevance_breakdown": row.get("score_breakdown"),
                "intent": row.get("intent"),
            }
        },
        "artifacts": {},
    }


def _go_term_specs(params: GOSequencesInput) -> list[dict[str, Any]]:
    if params.go_terms:
        return [
            {
                **dict(term),
                "term_id": term.get("term_id") or term.get("id"),
            }
            for term in params.go_terms[:5]
            if isinstance(term, dict) and (term.get("term_id") or term.get("id"))
        ]
    if params.go_id:
        return [
            {
                "term_id": params.go_id,
                "intent_id": params.intent_id or params.go_id,
                "intent_label": params.intent_label,
                "term_relevance_score": params.term_relevance_score,
                "score_breakdown": params.term_score_breakdown,
            }
        ]
    return []


def _params_for_go_term(params: GOSequencesInput, term: dict[str, Any]) -> GOSequencesInput:
    return params.model_copy(
        update={
            "go_id": str(term.get("term_id") or term.get("id")),
            "go_terms": [],
            "intent_id": term.get("intent_id") or term.get("term_id") or term.get("id"),
            "intent_label": term.get("intent_label") or term.get("label"),
            "term_relevance_score": term.get("term_relevance_score"),
            "term_score_breakdown": term.get("score_breakdown") or {},
        }
    )


def _empty_no_term_output(
    params: GOSequencesInput,
    ctx: AdapterRunContext,
) -> GOSequencesOutput:
    out_dir = ctx.job_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = out_dir / "go_sequences.fasta"
    metadata_path = out_dir / "go_sequences_metadata.jsonl"
    association_path = out_dir / "go_association_records.jsonl"
    unresolved_path = out_dir / "go_unresolved_records.jsonl"
    summary_path = out_dir / "go_sequences_summary.json"
    raw_path = out_dir / "go_raw_response.json"
    raw_uniprot_path = out_dir / "uniprot_raw_records.jsonl"
    for path in (
        fasta_path,
        metadata_path,
        association_path,
        unresolved_path,
        raw_uniprot_path,
    ):
        path.write_text("")
    raw_path.write_text("{}\n")
    empty_facets = [_no_term_empty_event(params)]
    summary = {
        "retrieval_status": "no_terms",
        "reason": "no GO terms were provided by resolver or user input",
        "candidate_accessions": 0,
        "scanned_accessions": 0,
        "downstream_candidate_count": 0,
        "max_downstream_candidates": params.max_downstream_candidates,
        "sequence_relevance_bounded": True,
        "empty_facets": empty_facets,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    ctx.emit_progress(1.0)
    return GOSequencesOutput(
        go_id=params.go_id or "",
        go_label=None,
        association_records=0,
        resolved_uniprot_accessions=0,
        raw_result_count=0,
        kept_sequences=0,
        kept_sequence_count=0,
        dropped_empty_sequence_count=0,
        sequences=[],
        candidates=[],
        empty_facets=empty_facets,
        fasta_path=str(fasta_path),
        metadata_jsonl=str(metadata_path),
        dropped={},
        summary=summary,
        artifacts={
            "association_records_jsonl": str(association_path),
            "unresolved_records_jsonl": str(unresolved_path),
            "summary_json": str(summary_path),
            "raw_response_json": str(raw_path),
            "uniprot_raw_records_jsonl": str(raw_uniprot_path),
        },
        runtime_ms=0,
    )


def _no_term_empty_event(params: GOSequencesInput) -> dict[str, Any]:
    return {
        "event_type": "empty_candidate_event",
        "intent_id": params.intent_id,
        "label": params.intent_label or "No GO term selected",
        "query": params.intent_label,
        "selected_term": None,
        "term_id": None,
        "term_relevance_score": params.term_relevance_score,
        "score_breakdown": dict(params.term_score_breakdown or {}),
        "selected_reason": "resolver did not select any GO term for this retrieval stream",
        "resolver_evidence": {
            "intent_id": params.intent_id,
            "intent_label": params.intent_label,
            "term_score_breakdown": dict(params.term_score_breakdown or {}),
        },
        "retrieval_status": "no_terms",
        "candidate_count": 0,
        "stage": "go_sequences",
    }


def _retrieval_empty_facets(
    params: GOSequencesInput,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if candidates:
        return []
    if not params.intent_id and not params.go_id:
        return []
    return [
        {
            "event_type": "empty_candidate_event",
            "intent_id": params.intent_id or params.go_id,
            "label": params.intent_label,
            "selected_term": params.go_id,
            "term_id": params.go_id,
            "term_relevance_score": params.term_relevance_score,
            "score_breakdown": dict(params.term_score_breakdown or {}),
            "selected_reason": _selected_reason(params),
            "resolver_evidence": {
                "intent_id": params.intent_id or params.go_id,
                "intent_label": params.intent_label,
                "term_score_breakdown": dict(params.term_score_breakdown or {}),
            },
            "retrieval_status": "empty",
            "candidate_count": 0,
            "stage": "go_sequences",
        }
    ]


def _selected_reason(params: GOSequencesInput) -> str:
    if params.intent_label:
        return f"selected for query facet {params.intent_label!r}"
    if params.intent_id and params.go_id and params.intent_id != params.go_id:
        return f"selected term {params.go_id} for resolver facet {params.intent_id}"
    if params.go_id:
        return f"selected term {params.go_id}"
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


def _evidence_quality_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    experimental = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP"}
    letters = [evidence_letter_code(row.get("evidence")) for row in rows]
    if not letters:
        return 0.0
    return min(1.0, sum(1 for letter in letters if letter in experimental) / max(1, len(letters)))


__all__ = ["GOSequencesAdapter", "GOSequencesInput", "GOSequencesOutput"]
