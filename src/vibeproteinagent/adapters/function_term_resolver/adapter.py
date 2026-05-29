from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.http import HttpClient, HttpError
from vibeproteinagent.adapters._remote.resolvers import (
    search_go_terms,
    search_interpro_entries,
    select_best_identifier,
    term_relevance_payload,
)
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class FunctionTermResolverInput(BaseModel):
    query: str = Field(min_length=1)
    target_names: list[str] = Field(default_factory=list)
    semantic_intents: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Optional bounded semantic facets to resolve independently. Each "
            "item may include intent_id, label, query, and required."
        ),
    )
    preferred_namespaces: list[Literal["go", "interpro"]] = Field(default_factory=lambda: ["go", "interpro"])
    aspect: str | None = None
    limit_per_target: int = Field(default=5, ge=1, le=25)
    timeout_s: float = Field(default=15.0, gt=0, le=60)


class FunctionTermResolverOutput(BaseModel):
    query: str
    status: str = "resolved"
    matches: list[dict[str, Any]] = Field(default_factory=list)
    semantic_intents: list[dict[str, Any]] = Field(default_factory=list)
    empty_facets: list[dict[str, Any]] = Field(default_factory=list)
    selected_go_terms: list[dict[str, Any]] = Field(default_factory=list)
    selected_interpro_terms: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    targets: list[dict[str, Any]]
    ready_for_execution: bool
    ambiguous: bool
    selected_go_id: str | None = None
    selected_interpro_id: str | None = None
    selection_reason: str | None = None
    runtime_ms: int


@register_adapter("descriptor.yaml")
class FunctionTermResolverAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: FunctionTermResolverInput,
        ctx: AdapterRunContext,
    ) -> FunctionTermResolverOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=out_dir / "http_cache")
        semantic_intents = _semantic_intents_from_params(params)
        targets: list[dict[str, Any]] = []
        for intent in semantic_intents:
            label = str(intent["label"])
            target_query = str(intent["query"])
            go_candidates: list[dict[str, Any]] = []
            interpro_candidates: list[dict[str, Any]] = []
            errors: list[str] = []
            if "go" in params.preferred_namespaces:
                try:
                    go_candidates, _raw = search_go_terms(
                        client,
                        target_query,
                        aspect=params.aspect,
                        limit=params.limit_per_target,
                    )
                except HttpError as exc:
                    errors.append(f"go: {exc}")
                except Exception as exc:
                    errors.append(f"go: {type(exc).__name__}: {exc}")
            if "interpro" in params.preferred_namespaces:
                try:
                    interpro_candidates, _raw = search_interpro_entries(
                        client,
                        target_query,
                        limit=params.limit_per_target,
                    )
                except HttpError as exc:
                    errors.append(f"interpro: {exc}")
                except Exception as exc:
                    errors.append(f"interpro: {type(exc).__name__}: {exc}")
            go_selection = select_best_identifier(
                query=target_query,
                candidates=go_candidates,
                id_key="id",
                label_key="name",
                namespace="GO term",
            )
            interpro_selection = select_best_identifier(
                query=target_query,
                candidates=interpro_candidates,
                id_key="accession",
                label_key="name",
                namespace="InterPro entry",
            )
            errors.extend(go_selection.warnings)
            errors.extend(interpro_selection.warnings)
            ambiguous = go_selection.ambiguous or interpro_selection.ambiguous
            go_candidates = [
                _with_intent_term_score(row, intent=intent, namespace="go")
                for row in go_candidates
            ]
            interpro_candidates = [
                _with_intent_term_score(row, intent=intent, namespace="interpro")
                for row in interpro_candidates
            ]
            target = {
                "intent_id": intent["intent_id"],
                "label": label,
                "query": target_query,
                "required": bool(intent.get("required", True)),
                "go_candidates": go_candidates,
                "interpro_candidates": interpro_candidates,
                "selected_go_id": go_selection.selected_id,
                "selected_interpro_id": interpro_selection.selected_id,
                "go_selection_reason": go_selection.selection_reason,
                "interpro_selection_reason": interpro_selection.selection_reason,
                "ambiguous": ambiguous,
                "errors": errors,
            }
            target["ready_for_execution"] = (
                go_selection.ready_for_execution
                or interpro_selection.ready_for_execution
            )
            targets.append(target)
        ambiguous = any(bool(t["ambiguous"]) for t in targets)
        ready = bool(targets) and any(bool(t["ready_for_execution"]) for t in targets)
        matches = _resolver_matches(targets)
        empty_facets = _empty_facets(targets)
        selected_go_terms = _selected_terms(targets, namespace="go")
        selected_interpro_terms = _selected_terms(targets, namespace="interpro")
        warnings = [
            str(error)
            for target in targets
            for error in target.get("errors", [])
            if error
        ]
        status = "resolved" if ready else "no_match"
        if warnings and not matches:
            status = "backend_error"
        selected_go_id = _selected_identifier(targets, namespace="go")
        selected_interpro_id = _selected_identifier(targets, namespace="interpro")
        return FunctionTermResolverOutput(
            query=params.query,
            status=status,
            matches=matches,
            semantic_intents=semantic_intents,
            empty_facets=empty_facets,
            selected_go_terms=selected_go_terms,
            selected_interpro_terms=selected_interpro_terms,
            warnings=warnings,
            targets=targets,
            ready_for_execution=ready,
            ambiguous=ambiguous,
            selected_go_id=selected_go_id,
            selected_interpro_id=selected_interpro_id,
            selection_reason=_combined_selection_reason(
                targets,
                selected_go_id=selected_go_id,
                selected_interpro_id=selected_interpro_id,
            ),
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: FunctionTermResolverInput) -> int:
        del params
        return 0


def _selected_identifier(targets: list[dict[str, Any]], *, namespace: str) -> str | None:
    if namespace == "go":
        key = "selected_go_id"
    else:
        key = "selected_interpro_id"
    for target in targets:
        value = target.get(key)
        if value:
            return str(value)
    return None


def _combined_selection_reason(
    targets: list[dict[str, Any]],
    *,
    selected_go_id: str | None,
    selected_interpro_id: str | None,
) -> str | None:
    for target in targets:
        if selected_go_id and target.get("selected_go_id") == selected_go_id:
            reason = target.get("go_selection_reason")
            return str(reason) if reason else None
        if selected_interpro_id and target.get("selected_interpro_id") == selected_interpro_id:
            reason = target.get("interpro_selection_reason")
            return str(reason) if reason else None
    return None


def _resolver_matches(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for target in targets:
        for row in target.get("go_candidates", []) or []:
            if isinstance(row, dict) and row.get("id"):
                matches.append(
                    {
                        "source": row.get("source") or "go_term_search",
                        "intent_id": target.get("intent_id"),
                        "id": row.get("id"),
                        "label": row.get("name"),
                        "confidence": row.get("score"),
                        "term_relevance_score": row.get("term_relevance_score"),
                        "score_breakdown": row.get("score_breakdown"),
                        "selected_reason": target.get("go_selection_reason"),
                    }
                )
        for row in target.get("interpro_candidates", []) or []:
            if isinstance(row, dict) and row.get("accession"):
                matches.append(
                    {
                        "source": row.get("source") or "interpro_entry_search",
                        "intent_id": target.get("intent_id"),
                        "id": row.get("accession"),
                        "label": row.get("name"),
                        "confidence": row.get("score"),
                        "term_relevance_score": row.get("term_relevance_score"),
                        "score_breakdown": row.get("score_breakdown"),
                        "selected_reason": target.get("interpro_selection_reason"),
                    }
                )
    return sorted(
        matches,
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )


def _semantic_intents_from_params(params: FunctionTermResolverInput) -> list[dict[str, Any]]:
    if params.semantic_intents:
        intents = [
            _normalize_intent(item, fallback_query=params.query, index=index)
            for index, item in enumerate(params.semantic_intents[:5])
            if isinstance(item, dict)
        ]
        if intents:
            return intents
    if params.target_names:
        return [
            _normalize_intent(
                {"label": name, "query": f"{params.query} {name}", "required": True},
                fallback_query=params.query,
                index=index,
            )
            for index, name in enumerate(params.target_names[:5])
        ]
    facets = _split_query_facets(params.query)
    return [
        _normalize_intent(
            {"label": facet, "query": facet, "required": True},
            fallback_query=params.query,
            index=index,
        )
        for index, facet in enumerate(facets[:5])
    ]


def _normalize_intent(
    item: dict[str, Any],
    *,
    fallback_query: str,
    index: int,
) -> dict[str, Any]:
    label = str(item.get("label") or item.get("query") or fallback_query).strip()
    query = str(item.get("query") or label or fallback_query).strip()
    intent_id = str(item.get("intent_id") or _intent_id(label, index)).strip()
    return {
        "intent_id": intent_id,
        "label": label or fallback_query,
        "query": query or fallback_query,
        "required": bool(item.get("required", True)),
    }


def _split_query_facets(query: str) -> list[str]:
    cleaned = query.strip()
    if not cleaned:
        return [query]
    cleaned = _strip_workflow_prefix(cleaned)
    raw_parts = [
        part.strip(" .;:")
        for part in cleaned.replace(" and ", ";").replace(",", ";").split(";")
        if part.strip(" .;:")
    ]
    parts = [part for part in raw_parts if _looks_like_semantic_facet(part)]
    if len(parts) <= 1:
        return [cleaned]
    return parts[:5]


def _strip_workflow_prefix(query: str) -> str:
    lowered = query.lower()
    prefixes = (
        "find and rank ",
        "find ",
        "rank ",
        "retrieve ",
        "search for ",
        "identify ",
        "resolve ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return query[len(prefix):].strip()
    return query


def _looks_like_semantic_facet(text: str) -> bool:
    tokens = {
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if token
    }
    if len(tokens) >= 3:
        return True
    semantic_markers = {
        "activity",
        "function",
        "domain",
        "family",
        "enzyme",
        "complex",
        "ligand",
        "assay",
        "organism",
        "disease",
        "binding",
        "transporter",
        "kinase",
        "protease",
        "receptor",
    }
    return bool(tokens & semantic_markers)


def _intent_id(label: str, index: int) -> str:
    tokens = [
        token
        for token in "".join(ch.lower() if ch.isalnum() else " " for ch in label).split()
        if token
    ]
    base = "_".join(tokens[:5]) or f"intent_{index + 1}"
    return base[:64]


def _with_intent_term_score(
    row: dict[str, Any],
    *,
    intent: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    label_key = "name"
    definition_key = "definition" if namespace == "go" else "description"
    if row.get("term_relevance_score") is not None and row.get("score_breakdown"):
        scored = dict(row)
    else:
        scored = {
            **row,
            **term_relevance_payload(
                query=str(intent.get("query") or ""),
                label=row.get(label_key),
                definition=row.get(definition_key),
                source=str(row.get("source") or ""),
            ),
        }
    scored["intent_id"] = intent.get("intent_id")
    scored["intent_label"] = intent.get("label")
    return scored


def _empty_facets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for target in targets:
        if target.get("ready_for_execution"):
            continue
        selected_term = target.get("selected_go_id") or target.get("selected_interpro_id")
        score_breakdown = {}
        term_relevance_score = None
        selected_reason = target.get("selection_reason")
        if selected_term:
            for key in ("go_candidates", "interpro_candidates"):
                for candidate in target.get(key) or []:
                    candidate_id = candidate.get("id") or candidate.get("accession")
                    if candidate_id == selected_term:
                        score_breakdown = dict(candidate.get("score_breakdown") or {})
                        term_relevance_score = candidate.get("term_relevance_score")
                        selected_reason = (
                            target.get("go_selection_reason")
                            or target.get("interpro_selection_reason")
                            or candidate.get("selected_reason")
                            or selected_reason
                        )
                        break
                if term_relevance_score is not None:
                    break
        out.append(
            {
                "event_type": "empty_candidate_event",
                "intent_id": target.get("intent_id"),
                "label": target.get("label"),
                "query": target.get("query"),
                "selected_term": selected_term,
                "selected_go_id": target.get("selected_go_id"),
                "selected_interpro_id": target.get("selected_interpro_id"),
                "term_relevance_score": term_relevance_score,
                "score_breakdown": score_breakdown,
                "selected_reason": selected_reason
                or f"resolver facet {target.get('intent_id') or target.get('label') or target.get('query')} did not produce an executable term",
                "resolver_evidence": {
                    "intent_id": target.get("intent_id"),
                    "label": target.get("label"),
                    "query": target.get("query"),
                    "errors": target.get("errors") or [],
                    "selected_go_id": target.get("selected_go_id"),
                    "selected_interpro_id": target.get("selected_interpro_id"),
                },
                "retrieval_status": "empty",
                "candidate_count": 0,
                "errors": target.get("errors") or [],
            }
        )
    return out


def _selected_terms(targets: list[dict[str, Any]], *, namespace: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    if namespace == "go":
        candidates_key = "go_candidates"
        selected_key = "selected_go_id"
        term_id_key = "id"
        reason_key = "go_selection_reason"
    else:
        candidates_key = "interpro_candidates"
        selected_key = "selected_interpro_id"
        term_id_key = "accession"
        reason_key = "interpro_selection_reason"
    for target in targets:
        selected = target.get(selected_key)
        if not selected:
            continue
        rows = target.get(candidates_key) or []
        row = next(
            (
                item
                for item in rows
                if isinstance(item, dict) and item.get(term_id_key) == selected
            ),
            {},
        )
        key = (str(target.get("intent_id") or ""), str(selected))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "intent_id": target.get("intent_id"),
                "intent_label": target.get("label"),
                "query": target.get("query"),
                "required": bool(target.get("required", True)),
                "term_id": selected,
                "id": selected,
                "label": row.get("name") if isinstance(row, dict) else None,
                "term_relevance_score": row.get("term_relevance_score")
                if isinstance(row, dict)
                else None,
                "score_breakdown": row.get("score_breakdown")
                if isinstance(row, dict)
                else None,
                "selected_reason": target.get(reason_key),
            }
        )
    return out
