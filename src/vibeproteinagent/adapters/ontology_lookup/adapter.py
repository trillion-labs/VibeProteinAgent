from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.resolvers import lookup_go_term
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class OntologyLookupInput(BaseModel):
    ontology: Literal["go"] = "go"
    term_id: str = Field(pattern=r"^GO:[0-9]{7}$")
    timeout_s: float = Field(default=15.0, gt=0, le=60)


class OntologyLookupOutput(BaseModel):
    id: str
    status: str = "resolved"
    matches: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    valid: bool
    ontology: str
    obsolete: bool = False
    name: str | None = None
    definition: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    resolved_terms: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


@register_adapter("descriptor.yaml")
class OntologyLookupAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: OntologyLookupInput, ctx: AdapterRunContext) -> OntologyLookupOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=out_dir / "http_cache")
        warnings: list[str] = []
        try:
            row, _raw = lookup_go_term(client, params.term_id)
        except Exception as exc:
            row = {"id": params.term_id, "valid": False}
            warnings.append(f"ontology_lookup backend_error: {type(exc).__name__}: {exc}")
        term = {
            "id": row.get("id") or params.term_id,
            "name": row.get("name"),
            "definition": row.get("definition"),
            "synonyms": row.get("synonyms") or [],
            "aspect": row.get("aspect"),
            "obsolete": bool(row.get("obsolete", False)),
            "score": 1.0 if row.get("valid") else 0.0,
            "source": row.get("source"),
        }
        return OntologyLookupOutput(
            id=str(term["id"]),
            status=(
                "resolved"
                if row.get("valid")
                else ("backend_error" if warnings else "no_match")
            ),
            matches=[
                {
                    "source": term.get("source") or "ontology_lookup",
                    "id": term["id"],
                    "label": term.get("name"),
                    "confidence": term.get("score"),
                }
            ] if row.get("valid") else [],
            warnings=warnings,
            valid=bool(row.get("valid")),
            ontology=params.ontology,
            obsolete=bool(term["obsolete"]),
            name=term.get("name"),
            definition=term.get("definition"),
            synonyms=list(term.get("synonyms") or []),
            resolved_terms=[term] if row.get("valid") else [],
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: OntologyLookupInput) -> int:
        del params
        return 0
