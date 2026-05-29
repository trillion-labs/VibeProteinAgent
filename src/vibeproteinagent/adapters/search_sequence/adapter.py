from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.sequence_relevance import intent_quota_top_k
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.function_term_resolver.adapter import (
    FunctionTermResolverAdapter,
    FunctionTermResolverInput,
    FunctionTermResolverOutput,
)
from vibeproteinagent.adapters.go_sequences.adapter import (
    GOSequencesAdapter,
    GOSequencesInput,
    GOSequencesOutput,
)
from vibeproteinagent.adapters.interpro_sequences.adapter import (
    InterProSequencesAdapter,
    InterProSequencesInput,
    InterProSequencesOutput,
)


class SearchSequenceInput(BaseModel):
    context: str | None = Field(default=None, description="Natural-language sequence retrieval context used for internal GO/InterPro term resolution when exact terms are absent or semantic facets are present.")
    go_term: str | None = Field(default=None, pattern=r"^GO:[0-9]{7}$", description="Optional explicit Gene Ontology identifier in GO:NNNNNNN format. When provided, GO sequence retrieval is always run.", examples=["GO:0052855", None])
    interpro_term: str | None = Field(default=None, pattern=r"^IPR[0-9]{6}$", description="Optional explicit InterPro accession in IPRNNNNNN format. When provided, InterPro sequence retrieval is always run.", examples=["IPR000001", None])
    go_terms: list[dict[str, Any]] = Field(default_factory=list, description="Bounded GO term records, usually from a resolver. Each record may include term_id, intent_id, label, term_relevance_score, and score_breakdown.")
    interpro_terms: list[dict[str, Any]] = Field(default_factory=list, description="Bounded InterPro term records, usually from a resolver. Each record may include term_id, intent_id, label, term_relevance_score, and score_breakdown.")
    semantic_intents: list[dict[str, Any]] = Field(default_factory=list, description="Optional semantic facets to resolve independently. Each item may include intent_id, label, query, and required.")
    reviewed: bool | None = Field(default=True, description="If true, prefer reviewed Swiss-Prot records where source APIs support the filter; false restricts to unreviewed records; null returns both.")
    taxon_id: int | None = Field(default=None, description="Optional NCBI taxonomy id to restrict returned UniProt sequences. Null disables taxonomy filtering.")
    temporal_cutoff: str | None = Field(default=None, description="Optional ISO date (YYYY-MM-DD); only sequences with UniProt entry dates on or before this date are retained.")
    include_fragments: bool = Field(default=False, description="If true, include InterPro sequences flagged by UniProt as fragments. GO retrieval currently ignores this field.")
    max_sequences: int = Field(default=300, ge=1, le=10000, description="Maximum source sequences scanned or kept per retrieval source before downstream candidate bounding.")
    max_downstream_candidates: int = Field(default=40, ge=1, le=1000, description="Maximum final canonical Candidate[] records emitted after source merge, deduplication, scoring, and intent-quota top-K.")
    timeout_s: float = Field(default=20.0, gt=0, le=120, description="HTTP timeout in seconds for resolver and source retrieval calls.")
    include_undated: bool = Field(default=False, description="When temporal_cutoff is set, controls whether records lacking a parseable UniProt entry date are included.")
    min_length: int | None = Field(default=None, ge=1, description="Optional minimum sequence length in amino acids.")
    max_length: int | None = Field(default=None, ge=1, description="Optional maximum sequence length in amino acids.")
    page_size: int = Field(default=200, ge=1, le=500, description="Number of records fetched per remote API page where supported.")
    max_concurrency: int = Field(default=5, ge=1, le=20, description="Maximum parallel HTTP fetches used by source retrieval adapters.")


class SearchSequenceOutput(BaseModel):
    query: str
    retrieval_status: Literal["completed", "partial", "empty", "no_terms", "no_hits"]
    retrieval_sources: list[Literal["go", "interpro"]]
    resolved_terms: dict[str, list[dict[str, Any]]]
    source_summaries: dict[str, dict[str, Any]]
    sequences: list[dict[str, Any]]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    empty_facets: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any]
    artifacts: dict[str, Any]
    runtime_ms: int


@dataclass
class SequenceSourceResult:
    source: Literal["go", "interpro"]
    status: Literal["completed", "no_terms", "no_hits", "failed"]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    sequences: list[dict[str, Any]] = field(default_factory=list)
    empty_facets: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    term_count: int = 0
    error: dict[str, Any] | None = None


@register_adapter("descriptor.yaml")
class SearchSequenceAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.config = config
        self.weights_dir = weights_dir
        self.device = device

    def run(
        self,
        params: SearchSequenceInput,
        ctx: AdapterRunContext,
    ) -> SearchSequenceOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        query = _query_text(params)
        explicit_go_terms = _explicit_go_terms(params)
        explicit_interpro_terms = _explicit_interpro_terms(params)
        should_resolve = bool(params.context or params.semantic_intents or not (explicit_go_terms or explicit_interpro_terms))

        resolver_output: FunctionTermResolverOutput | None = None
        resolver_empty_facets: list[dict[str, Any]] = []
        resolver_go_terms: list[dict[str, Any]] = []
        resolver_interpro_terms: list[dict[str, Any]] = []
        if should_resolve and query:
            resolver_output = self._run_resolver(params, ctx, query)
            resolver_empty_facets = [
                _with_stage(dict(facet), "search_sequence")
                for facet in resolver_output.empty_facets
                if isinstance(facet, dict)
            ]
            resolver_go_terms = _terms_from_resolver(resolver_output, "go")
            resolver_interpro_terms = _terms_from_resolver(resolver_output, "interpro")
        elif should_resolve:
            resolver_empty_facets = [_no_query_empty_event()]

        go_terms = _dedupe_terms(explicit_go_terms + resolver_go_terms)
        interpro_terms = _dedupe_terms(explicit_interpro_terms + resolver_interpro_terms)
        source_results: list[SequenceSourceResult] = []

        if go_terms:
            source_results.append(self._run_go(params, ctx, go_terms))
        elif should_resolve:
            source_results.append(_no_terms_source_result("go", params, resolver_output))

        if interpro_terms:
            source_results.append(self._run_interpro(params, ctx, interpro_terms))
        elif should_resolve:
            source_results.append(_no_terms_source_result("interpro", params, resolver_output))

        if not source_results:
            source_results = [
                _no_terms_source_result("go", params, resolver_output),
                _no_terms_source_result("interpro", params, resolver_output),
            ]

        failed = [result for result in source_results if result.status == "failed"]
        successful_candidates = [
            candidate
            for result in source_results
            for candidate in result.candidates
            if isinstance(candidate, dict)
        ]
        if failed and not successful_candidates and all(result.status == "failed" for result in source_results):
            first = failed[0]
            raise RuntimeError(
                f"search_sequence source {first.source} failed: "
                f"{(first.error or {}).get('message') or 'unknown error'}"
            )

        merged_candidates = _merge_dedupe_top_k(
            successful_candidates,
            max_candidates=params.max_downstream_candidates,
            explicit_go_terms=explicit_go_terms,
            explicit_interpro_terms=explicit_interpro_terms,
        )
        sequences = _sequences_from_candidates(merged_candidates)
        empty_facets = resolver_empty_facets + [
            _normalize_source_empty_facet(facet, result.status)
            for result in source_results
            for facet in result.empty_facets
            if isinstance(facet, dict)
        ]
        source_summaries = {
            result.source: {
                "status": result.status,
                "candidate_count": len(result.candidates),
                "term_count": result.term_count,
                **({"error": result.error} if result.error else {}),
            }
            for result in source_results
        }
        retrieval_status = _overall_status(source_results, merged_candidates)
        retrieval_sources = [
            result.source
            for result in source_results
            if result.term_count > 0 and result.status != "failed"
        ]
        summary = {
            "retrieval_sources": retrieval_sources,
            "source_summaries": source_summaries,
            "resolved_terms": {
                "go_terms": go_terms,
                "interpro_terms": interpro_terms,
            },
            "source_candidate_count": len(successful_candidates),
            "downstream_candidate_count": len(merged_candidates),
            "max_downstream_candidates": params.max_downstream_candidates,
            "sequence_relevance_bounded": True,
            "empty_facets": empty_facets,
        }
        summary_path = out_dir / "search_sequence_summary.json"
        candidates_path = out_dir / "search_sequence_candidates.json"
        sequences_path = out_dir / "search_sequence_sequences.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        candidates_path.write_text(json.dumps(merged_candidates, indent=2, sort_keys=True) + "\n")
        sequences_path.write_text(json.dumps(sequences, indent=2, sort_keys=True) + "\n")
        artifacts: dict[str, Any] = {
            "summary_json": str(summary_path),
            "candidates_json": str(candidates_path),
            "sequences_json": str(sequences_path),
            "sources": {
                result.source: result.artifacts for result in source_results
            },
        }
        ctx.emit_progress(1.0)
        return SearchSequenceOutput(
            query=query,
            retrieval_status=retrieval_status,
            retrieval_sources=retrieval_sources,
            resolved_terms={"go_terms": go_terms, "interpro_terms": interpro_terms},
            source_summaries=source_summaries,
            sequences=sequences,
            candidates=merged_candidates,
            empty_facets=empty_facets,
            summary=summary,
            artifacts=artifacts,
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: SearchSequenceInput) -> int:
        del params
        return 0

    def _run_resolver(
        self,
        params: SearchSequenceInput,
        ctx: AdapterRunContext,
        query: str,
    ) -> FunctionTermResolverOutput:
        child_ctx = _child_context(ctx, "resolver", 0.0, 0.15)
        adapter = FunctionTermResolverAdapter(self.config, self.weights_dir, self.device)
        resolver_params = FunctionTermResolverInput(
            query=query,
            semantic_intents=params.semantic_intents,
            preferred_namespaces=["go", "interpro"],
            timeout_s=min(float(params.timeout_s), 60.0),
        )
        return adapter.run(resolver_params, child_ctx)

    def _run_go(
        self,
        params: SearchSequenceInput,
        ctx: AdapterRunContext,
        terms: list[dict[str, Any]],
    ) -> SequenceSourceResult:
        try:
            child_ctx = _child_context(ctx, "go", 0.15, 0.55)
            adapter = GOSequencesAdapter(self.config, self.weights_dir, self.device)
            out = adapter.run(_go_params(params, terms), child_ctx)
            return _source_result_from_go(out, len(terms))
        except Exception as exc:
            return _failed_source_result("go", exc, len(terms))

    def _run_interpro(
        self,
        params: SearchSequenceInput,
        ctx: AdapterRunContext,
        terms: list[dict[str, Any]],
    ) -> SequenceSourceResult:
        try:
            child_ctx = _child_context(ctx, "interpro", 0.55, 0.95)
            adapter = InterProSequencesAdapter(self.config, self.weights_dir, self.device)
            out = adapter.run(_interpro_params(params, terms), child_ctx)
            return _source_result_from_interpro(out, len(terms))
        except Exception as exc:
            return _failed_source_result("interpro", exc, len(terms))


def _query_text(params: SearchSequenceInput) -> str:
    if params.context and params.context.strip():
        return params.context.strip()
    labels = [
        str(item.get("query") or item.get("label") or "").strip()
        for item in params.semantic_intents
        if isinstance(item, dict)
    ]
    labels = [item for item in labels if item]
    if labels:
        return "; ".join(labels[:5])
    terms = [params.go_term, params.interpro_term]
    terms.extend(str(t.get("term_id") or t.get("id")) for t in params.go_terms if isinstance(t, dict))
    terms.extend(str(t.get("term_id") or t.get("id")) for t in params.interpro_terms if isinstance(t, dict))
    return " ".join(term for term in terms if term)


def _explicit_go_terms(params: SearchSequenceInput) -> list[dict[str, Any]]:
    terms = [_normalize_term(term, namespace="go", explicit=True) for term in params.go_terms if isinstance(term, dict)]
    if params.go_term:
        terms.insert(0, {"term_id": params.go_term, "id": params.go_term, "intent_id": params.go_term, "explicit": True})
    return [term for term in terms if term.get("term_id")]


def _explicit_interpro_terms(params: SearchSequenceInput) -> list[dict[str, Any]]:
    terms = [
        _normalize_term(term, namespace="interpro", explicit=True)
        for term in params.interpro_terms
        if isinstance(term, dict)
    ]
    if params.interpro_term:
        terms.insert(0, {"term_id": params.interpro_term, "id": params.interpro_term, "intent_id": params.interpro_term, "explicit": True})
    return [term for term in terms if term.get("term_id")]


def _terms_from_resolver(
    resolver_output: FunctionTermResolverOutput,
    namespace: Literal["go", "interpro"],
) -> list[dict[str, Any]]:
    raw_terms = (
        resolver_output.selected_go_terms
        if namespace == "go"
        else resolver_output.selected_interpro_terms
    )
    return [
        _normalize_term(term, namespace=namespace, explicit=False)
        for term in raw_terms
        if isinstance(term, dict)
    ]


def _normalize_term(
    term: dict[str, Any],
    *,
    namespace: Literal["go", "interpro"],
    explicit: bool,
) -> dict[str, Any]:
    term_id = term.get("term_id") or term.get("id")
    if namespace == "interpro":
        term_id = term_id or term.get("accession")
    if not term_id:
        return {}
    out = dict(term)
    out["term_id"] = str(term_id)
    out["id"] = str(term_id)
    out["explicit"] = bool(term.get("explicit", explicit))
    if out.get("intent_label") is None and out.get("label") is not None:
        out["intent_label"] = out.get("label")
    if out.get("intent_id") is None:
        out["intent_id"] = out["term_id"]
    return out


def _dedupe_terms(terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for term in terms:
        term_id = str(term.get("term_id") or term.get("id") or "")
        if not term_id:
            continue
        intent_id = str(term.get("intent_id") or term_id)
        key = (intent_id, term_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
    return out[:5]


def _go_params(params: SearchSequenceInput, terms: list[dict[str, Any]]) -> GOSequencesInput:
    return GOSequencesInput(
        go_id=None,
        go_terms=terms,
        max_sequences=params.max_sequences,
        max_downstream_candidates=params.max_downstream_candidates,
        reviewed=params.reviewed,
        taxon_id=params.taxon_id,
        temporal_cutoff=params.temporal_cutoff,
        include_undated=params.include_undated,
        min_length=params.min_length,
        max_length=params.max_length,
        page_size=params.page_size,
        max_concurrency=params.max_concurrency,
        timeout_s=params.timeout_s,
    )


def _interpro_params(
    params: SearchSequenceInput,
    terms: list[dict[str, Any]],
) -> InterProSequencesInput:
    return InterProSequencesInput(
        interpro_id=None,
        interpro_terms=terms,
        max_sequences=params.max_sequences,
        max_downstream_candidates=params.max_downstream_candidates,
        reviewed=params.reviewed,
        taxon_id=params.taxon_id,
        temporal_cutoff=params.temporal_cutoff,
        include_fragments=params.include_fragments,
        include_undated=params.include_undated,
        min_length=params.min_length,
        max_length=params.max_length,
        page_size=params.page_size,
        max_concurrency=params.max_concurrency,
        timeout_s=params.timeout_s,
    )


def _source_result_from_go(out: GOSequencesOutput, term_count: int) -> SequenceSourceResult:
    candidates = list(out.candidates or [])
    return SequenceSourceResult(
        source="go",
        status="completed" if candidates else "no_hits",
        candidates=candidates,
        sequences=list(out.sequences or []),
        empty_facets=list(out.empty_facets or []),
        summary=dict(out.summary or {}),
        artifacts=dict(out.artifacts or {}),
        term_count=term_count,
    )


def _source_result_from_interpro(
    out: InterProSequencesOutput,
    term_count: int,
) -> SequenceSourceResult:
    candidates = list(out.candidates or [])
    return SequenceSourceResult(
        source="interpro",
        status="completed" if candidates else "no_hits",
        candidates=candidates,
        sequences=_sequences_from_candidates(candidates),
        empty_facets=list(out.empty_facets or []),
        summary=dict(out.summary or {}),
        artifacts=dict(out.artifacts or {}),
        term_count=term_count,
    )


def _failed_source_result(
    source: Literal["go", "interpro"],
    exc: Exception,
    term_count: int,
) -> SequenceSourceResult:
    return SequenceSourceResult(
        source=source,
        status="failed",
        term_count=term_count,
        error={"type": type(exc).__name__, "message": str(exc)},
    )


def _no_terms_source_result(
    source: Literal["go", "interpro"],
    params: SearchSequenceInput,
    resolver_output: FunctionTermResolverOutput | None,
) -> SequenceSourceResult:
    return SequenceSourceResult(
        source=source,
        status="no_terms",
        empty_facets=_source_no_term_events(source, params, resolver_output),
        summary={"retrieval_status": "no_terms", "candidate_count": 0},
        term_count=0,
    )


def _source_no_term_events(
    source: Literal["go", "interpro"],
    params: SearchSequenceInput,
    resolver_output: FunctionTermResolverOutput | None,
) -> list[dict[str, Any]]:
    label = "No GO term selected" if source == "go" else "No InterPro term selected"
    stage = "go_sequences" if source == "go" else "interpro_sequences"
    targets = list(getattr(resolver_output, "targets", []) or [])
    if not targets:
        return [
            {
                "event_type": "empty_candidate_event",
                "intent_id": None,
                "label": label,
                "query": params.context,
                "selected_term": None,
                "term_relevance_score": None,
                "score_breakdown": {},
                "selected_reason": f"resolver did not select any {source.upper()} term for this retrieval stream",
                "resolver_evidence": {},
                "retrieval_status": "no_terms",
                "candidate_count": 0,
                "stage": stage,
            }
        ]
    selected_key = "selected_go_id" if source == "go" else "selected_interpro_id"
    out = []
    for target in targets:
        if target.get(selected_key):
            continue
        out.append(
            {
                "event_type": "empty_candidate_event",
                "intent_id": target.get("intent_id"),
                "label": target.get("label") or label,
                "query": target.get("query"),
                "selected_term": None,
                "term_relevance_score": None,
                "score_breakdown": {},
                "selected_reason": f"resolver did not select any {source.upper()} term for this retrieval stream",
                "resolver_evidence": {
                    "intent_id": target.get("intent_id"),
                    "label": target.get("label"),
                    "query": target.get("query"),
                    "errors": target.get("errors") or [],
                },
                "retrieval_status": "no_terms",
                "candidate_count": 0,
                "stage": stage,
            }
        )
    return out


def _no_query_empty_event() -> dict[str, Any]:
    return {
        "event_type": "empty_candidate_event",
        "intent_id": None,
        "label": "No semantic context",
        "query": None,
        "selected_term": None,
        "term_relevance_score": None,
        "score_breakdown": {},
        "selected_reason": "no explicit term or semantic context was provided",
        "resolver_evidence": {},
        "retrieval_status": "no_terms",
        "candidate_count": 0,
        "stage": "search_sequence",
    }


def _with_stage(facet: dict[str, Any], stage: str) -> dict[str, Any]:
    facet.setdefault("stage", stage)
    if facet.get("retrieval_status") == "empty":
        facet["retrieval_status"] = "no_terms"
    return facet


def _merge_dedupe_top_k(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
    explicit_go_terms: list[dict[str, Any]],
    explicit_interpro_terms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    explicit_term_ids = {
        "go_sequences": {str(term.get("term_id")) for term in explicit_go_terms if term.get("term_id")},
        "interpro_sequences": {str(term.get("term_id")) for term in explicit_interpro_terms if term.get("term_id")},
    }
    best_by_sequence: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        sequence = str(candidate.get("sequence") or "")
        if not sequence:
            continue
        incoming = deepcopy(candidate)
        current = best_by_sequence.get(sequence)
        if current is None:
            best_by_sequence[sequence] = incoming
            continue
        if _candidate_rank_key(incoming, explicit_term_ids) < _candidate_rank_key(current, explicit_term_ids):
            best_by_sequence[sequence] = _merge_candidate_score_details(incoming, current)
        else:
            best_by_sequence[sequence] = _merge_candidate_score_details(current, incoming)
    rows = [
        {
            "candidate": candidate,
            "sequence_relevance_score": _semantic_score(candidate),
            "intent_id": _intent_id(candidate),
        }
        for candidate in best_by_sequence.values()
    ]
    bounded = intent_quota_top_k(rows, max_candidates=max_candidates)
    out = [dict(row["candidate"]) for row in bounded]
    for index, candidate in enumerate(out):
        source = candidate.get("source")
        if isinstance(source, dict):
            source["index"] = index
    return out


def _candidate_rank_key(
    candidate: dict[str, Any],
    explicit_term_ids: dict[str, set[str]],
) -> tuple[float, int, int, int, str]:
    return (
        -_semantic_score(candidate),
        -int(_reviewed(candidate)),
        -int(_explicit_term(candidate, explicit_term_ids)),
        _source_priority(candidate),
        str(candidate.get("candidate_id") or ""),
    )


def _merge_candidate_score_details(
    winner: dict[str, Any],
    other: dict[str, Any],
) -> dict[str, Any]:
    out = deepcopy(winner)
    out_scores = out.setdefault("scores", {})
    other_scores = other.get("scores") if isinstance(other.get("scores"), dict) else {}
    if isinstance(out_scores, dict):
        for key, value in other_scores.items():
            if key == "semantic_relevance":
                continue
            if key not in out_scores and isinstance(value, dict):
                out_scores[key] = deepcopy(value)
    return out


def _semantic_score(candidate: dict[str, Any]) -> float:
    scores = candidate.get("scores")
    if not isinstance(scores, dict):
        return 0.0
    semantic = scores.get("semantic_relevance")
    if isinstance(semantic, dict):
        score = semantic.get("sequence_relevance_score")
        if isinstance(score, (int, float)):
            return float(score)
    for value in scores.values():
        if isinstance(value, dict):
            score = value.get("sequence_relevance_score")
            if isinstance(score, (int, float)):
                return float(score)
    return 0.0


def _intent_id(candidate: dict[str, Any]) -> str:
    semantic = ((candidate.get("scores") or {}).get("semantic_relevance") or {})
    if isinstance(semantic, dict) and semantic.get("intent_id"):
        return str(semantic["intent_id"])
    for value in (candidate.get("scores") or {}).values():
        if isinstance(value, dict):
            intent = value.get("intent")
            if isinstance(intent, dict) and intent.get("intent_id"):
                return str(intent["intent_id"])
    return "default"


def _reviewed(candidate: dict[str, Any]) -> bool:
    for value in (candidate.get("scores") or {}).values():
        if isinstance(value, dict) and value.get("reviewed") is True:
            return True
    return False


def _explicit_term(
    candidate: dict[str, Any],
    explicit_term_ids: dict[str, set[str]],
) -> bool:
    source_tool = str(((candidate.get("source") or {}).get("tool")) or "")
    terms = explicit_term_ids.get(source_tool, set())
    if not terms:
        return False
    for value in (candidate.get("scores") or {}).values():
        if not isinstance(value, dict):
            continue
        intent = value.get("intent")
        if isinstance(intent, dict) and str(intent.get("term_id") or "") in terms:
            return True
    return False


def _source_priority(candidate: dict[str, Any]) -> int:
    source_tool = str(((candidate.get("source") or {}).get("tool")) or "")
    return {"go_sequences": 0, "interpro_sequences": 1}.get(source_tool, 9)


def _sequences_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sequences = []
    for candidate in candidates:
        sequence = candidate.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            continue
        source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
        sequences.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "accession": source.get("native_id") if isinstance(source, dict) else None,
                "sequence": sequence,
                "source_tool": source.get("tool") if isinstance(source, dict) else None,
            }
        )
    return sequences


def _overall_status(
    source_results: list[SequenceSourceResult],
    candidates: list[dict[str, Any]],
) -> Literal["completed", "partial", "empty", "no_terms", "no_hits"]:
    if candidates:
        if any(result.status != "completed" for result in source_results):
            return "partial"
        return "completed"
    if all(result.status == "no_terms" for result in source_results):
        return "no_terms"
    if any(result.term_count > 0 for result in source_results):
        return "no_hits"
    return "empty"


def _normalize_source_empty_facet(
    facet: dict[str, Any],
    source_status: str,
) -> dict[str, Any]:
    out = dict(facet)
    if source_status == "no_hits" and out.get("retrieval_status") in {None, "empty"}:
        out["retrieval_status"] = "no_hits"
    return out


def _child_context(
    ctx: AdapterRunContext,
    name: str,
    start: float,
    span: float,
) -> AdapterRunContext:
    return AdapterRunContext(
        job_id=ctx.job_id,
        workspace_root=ctx.workspace_root,
        job_dir=ctx.job_dir / "outputs" / "sources" / name,
        is_cancelled=ctx.is_cancelled,
        log=ctx.log,
        emit_progress=lambda pct: ctx.emit_progress(min(0.99, start + span * pct)),
        fusion_inputs=ctx.fusion_inputs,
    )


__all__ = ["SearchSequenceAdapter", "SearchSequenceInput", "SearchSequenceOutput"]
