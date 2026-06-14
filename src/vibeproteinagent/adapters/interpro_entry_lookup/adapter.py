from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.resolvers import lookup_interpro_entry
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class InterProEntryLookupInput(BaseModel):
    accession: str = Field(pattern=r"^(IPR[0-9]{6}|PF[0-9]{5}|PTHR[0-9]+(?::SF[0-9]+)?)$")
    database: str = "interpro"
    timeout_s: float = Field(default=15.0, gt=0, le=60)


class InterProEntryLookupOutput(BaseModel):
    accession: str
    status: str = "resolved"
    matches: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    valid: bool
    database: str
    name: str | None = None
    entry_type: str | None = None
    description: str | None = None
    go_terms: list[dict[str, Any]] = Field(default_factory=list)
    resolved_entries: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


@register_adapter("descriptor.yaml")
class InterProEntryLookupAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: InterProEntryLookupInput,
        ctx: AdapterRunContext,
    ) -> InterProEntryLookupOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=out_dir / "http_cache")
        warnings: list[str] = []
        try:
            row, _raw = lookup_interpro_entry(
                client,
                params.accession,
                database=params.database,
            )
        except Exception as exc:
            row = {"accession": params.accession, "valid": False}
            warnings.append(
                f"interpro_entry_lookup backend_error: {type(exc).__name__}: {exc}"
            )
        entry = {
            "accession": row.get("accession") or params.accession,
            "name": row.get("name"),
            "type": row.get("type"),
            "description": row.get("description"),
            "go_terms": row.get("go_terms") or [],
            "score": 1.0 if row.get("valid") else 0.0,
            "source": "interpro_api",
        }
        return InterProEntryLookupOutput(
            accession=str(entry["accession"]),
            status=(
                "resolved"
                if row.get("valid")
                else ("backend_error" if warnings else "no_match")
            ),
            matches=[
                {
                    "source": entry.get("source") or "interpro_entry_lookup",
                    "id": entry["accession"],
                    "label": entry.get("name"),
                    "confidence": entry.get("score"),
                }
            ] if row.get("valid") else [],
            warnings=warnings,
            valid=bool(row.get("valid")),
            database=params.database,
            name=entry.get("name"),
            entry_type=entry.get("type"),
            description=entry.get("description"),
            go_terms=list(entry.get("go_terms") or []),
            resolved_entries=[entry] if row.get("valid") else [],
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: InterProEntryLookupInput) -> int:
        del params
        return 0
