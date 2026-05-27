from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.resolvers import (
    search_go_terms,
    select_best_identifier,
)
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class GOTermSearchInput(BaseModel):
    query: str = Field(min_length=1)
    aspect: str | None = None
    limit: int = Field(default=10, ge=1, le=50)
    timeout_s: float = Field(default=15.0, gt=0, le=60)


class GOTermSearchOutput(BaseModel):
    query: str
    status: str = "resolved"
    matches: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_terms: list[dict[str, Any]]
    ambiguous: bool
    ready_for_execution: bool
    selected_go_id: str | None = None
    selection_reason: str | None = None
    raw_result_count: int
    runtime_ms: int


@register_adapter("descriptor.yaml")
class GOTermSearchAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: GOTermSearchInput, ctx: AdapterRunContext) -> GOTermSearchOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=out_dir / "http_cache")
        ctx.emit_progress(0.1)
        warnings: list[str] = []
        try:
            terms, _raw = search_go_terms(
                client,
                params.query,
                aspect=params.aspect,
                limit=params.limit,
            )
        except Exception as exc:
            terms = []
            warnings.append(f"go_term_search backend_error: {type(exc).__name__}: {exc}")
        ctx.emit_progress(0.9)
        selection = select_best_identifier(
            query=params.query,
            candidates=terms,
            id_key="id",
            label_key="name",
            namespace="GO term",
        )
        warnings.extend(selection.warnings)
        status = "resolved" if selection.selected_id else "no_match"
        if warnings and not selection.selected_id:
            status = "backend_error"
        ctx.emit_progress(1.0)
        return GOTermSearchOutput(
            query=params.query,
            status=status,
            matches=_go_matches(terms),
            warnings=warnings,
            resolved_terms=terms,
            ambiguous=selection.ambiguous,
            ready_for_execution=selection.ready_for_execution,
            selected_go_id=selection.selected_id,
            selection_reason=selection.selection_reason,
            raw_result_count=len(terms),
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: GOTermSearchInput) -> int:
        del params
        return 0

def _go_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": row.get("source") or "go_term_search",
            "id": row.get("id"),
            "label": row.get("name"),
            "confidence": row.get("score"),
        }
        for row in rows
        if row.get("id")
    ]
