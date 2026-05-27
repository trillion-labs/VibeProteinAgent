from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.resolvers import (
    search_interpro_entries,
    select_best_identifier,
)
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class InterProEntrySearchInput(BaseModel):
    query: str = Field(min_length=1)
    entry_type: str | None = None
    limit: int = Field(default=10, ge=1, le=50)
    timeout_s: float = Field(default=15.0, gt=0, le=60)


class InterProEntrySearchOutput(BaseModel):
    query: str
    status: str = "resolved"
    matches: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    resolved_entries: list[dict[str, Any]]
    ambiguous: bool
    ready_for_execution: bool
    selected_interpro_id: str | None = None
    selection_reason: str | None = None
    raw_result_count: int
    runtime_ms: int


@register_adapter("descriptor.yaml")
class InterProEntrySearchAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: InterProEntrySearchInput,
        ctx: AdapterRunContext,
    ) -> InterProEntrySearchOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=out_dir / "http_cache")
        warnings: list[str] = []
        try:
            entries, _raw = search_interpro_entries(
                client,
                params.query,
                entry_type=params.entry_type,
                limit=params.limit,
            )
        except Exception as exc:
            entries = []
            warnings.append(
                f"interpro_entry_search backend_error: {type(exc).__name__}: {exc}"
            )
        selection = select_best_identifier(
            query=params.query,
            candidates=entries,
            id_key="accession",
            label_key="name",
            namespace="InterPro entry",
        )
        warnings.extend(selection.warnings)
        status = "resolved" if selection.selected_id else "no_match"
        if warnings and not selection.selected_id:
            status = "backend_error"
        return InterProEntrySearchOutput(
            query=params.query,
            status=status,
            matches=_interpro_matches(entries),
            warnings=warnings,
            resolved_entries=entries,
            ambiguous=selection.ambiguous,
            ready_for_execution=selection.ready_for_execution,
            selected_interpro_id=selection.selected_id,
            selection_reason=selection.selection_reason,
            raw_result_count=len(entries),
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: InterProEntrySearchInput) -> int:
        del params
        return 0

def _interpro_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source": row.get("source") or "interpro_entry_search",
            "id": row.get("accession"),
            "label": row.get("name"),
            "confidence": row.get("score"),
        }
        for row in rows
        if row.get("accession")
    ]
