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
)
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class FunctionTermResolverInput(BaseModel):
    query: str = Field(min_length=1)
    target_names: list[str] = Field(default_factory=list)
    preferred_namespaces: list[Literal["go", "interpro"]] = Field(default_factory=lambda: ["go", "interpro"])
    aspect: str | None = None
    limit_per_target: int = Field(default=5, ge=1, le=25)
    timeout_s: float = Field(default=15.0, gt=0, le=60)


class FunctionTermResolverOutput(BaseModel):
    query: str
    status: str = "resolved"
    matches: list[dict[str, Any]] = Field(default_factory=list)
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
        labels = params.target_names or [params.query]
        targets: list[dict[str, Any]] = []
        for label in labels:
            target_query = params.query if label == params.query else f"{params.query} {label}"
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
            target = {
                "label": label,
                "query": target_query,
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
                        "id": row.get("id"),
                        "label": row.get("name"),
                        "confidence": row.get("score"),
                    }
                )
        for row in target.get("interpro_candidates", []) or []:
            if isinstance(row, dict) and row.get("accession"):
                matches.append(
                    {
                        "source": row.get("source") or "interpro_entry_search",
                        "id": row.get("accession"),
                        "label": row.get("name"),
                        "confidence": row.get("score"),
                    }
                )
    return sorted(
        matches,
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )
