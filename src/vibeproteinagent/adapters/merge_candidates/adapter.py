from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


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
                validate_candidate_v1(candidate)
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
        merged, duplicates_removed = _merge(params.candidate_sets)
        ctx.emit_progress(1.0)
        return MergeCandidatesOutput(
            candidates=merged,
            merge_audit={
                "input_counts": [len(candidate_set) for candidate_set in params.candidate_sets],
                "output_count": len(merged),
                "dedup_policy": params.dedup_policy,
                "duplicates_removed": duplicates_removed,
            },
        )

    def estimate_vram(self, params: MergeCandidatesInput) -> int:
        return 0


def _merge(
    candidate_sets: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], int]:
    return [candidate for candidate_set in candidate_sets for candidate in candidate_set], 0


__all__ = ["MergeCandidatesAdapter", "MergeCandidatesInput", "MergeCandidatesOutput"]
