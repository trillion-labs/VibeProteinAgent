from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, Field, field_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.schema import validate_candidate_v1
from vibeproteinagent.candidates.merge import merge_candidate_records


class MergeCandidatesInput(BaseModel):
    candidate_sets: list[list[dict[str, Any]]] = Field(min_length=1)
    dedup_policy: Literal["keep_all"] = "keep_all"

    @field_validator("candidate_sets")
    @classmethod
    def _validate_candidate_sets(
        cls,
        candidate_sets: list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        for candidate_set in candidate_sets:
            for candidate in candidate_set:
                try:
                    validate_candidate_v1(candidate)
                except JsonSchemaValidationError as exc:
                    raise ValueError(f"invalid candidate_v1: {exc.message}") from exc
        return candidate_sets


class MergeCandidatesOutput(BaseModel):
    candidates: list[dict[str, Any]]
    merge_audit: dict[str, Any]


@register_adapter("descriptor.yaml")
class MergeCandidatesAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: MergeCandidatesInput,
        ctx: AdapterRunContext,
    ) -> MergeCandidatesOutput:
        ctx.emit_progress(0.0)
        result = merge_candidate_records(params.candidate_sets)
        ctx.emit_progress(1.0)
        return MergeCandidatesOutput(
            candidates=result.candidates,
            merge_audit={
                "input_counts": [len(candidate_set) for candidate_set in params.candidate_sets],
                "output_count": len(result.candidates),
                "dedup_policy": params.dedup_policy,
                "duplicates_removed": result.audit["duplicates_removed"],
                "duplicate_groups": result.audit["duplicate_groups"],
                "conflicts": result.audit["conflicts"],
            },
        )

    def estimate_vram(self, params: MergeCandidatesInput) -> int:
        return 0

__all__ = ["MergeCandidatesAdapter", "MergeCandidatesInput", "MergeCandidatesOutput"]
