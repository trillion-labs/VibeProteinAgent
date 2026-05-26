from __future__ import annotations

from typing import Any

from vibeproteinagent.adapters.base import register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.pxdesign.adapter import (
    PXDesignAdapter,
    PXDesignDesign,
    PXDesignInput,
    PXDesignOutput,
    _pxdesign_candidates,
)


@register_adapter("descriptor.yaml")
class PXDesignPipelineMultiGpuAdapter(PXDesignAdapter):
    """PXDesign pipeline alias that requires multi-worker distributed dispatch."""

    def run(self, params: PXDesignInput, ctx: AdapterRunContext) -> PXDesignOutput:
        params.mode = "pipeline"
        out = super().run(params, ctx)
        out.candidates = _pxdesign_pipeline_multi_gpu_candidates(params, out.designs)
        return out


def _pxdesign_pipeline_multi_gpu_candidates(
    params: PXDesignInput,
    designs: list[PXDesignDesign],
) -> list[dict[str, Any]]:
    candidates = _pxdesign_candidates(params, designs)
    for candidate in candidates:
        candidate["source"] = dict(candidate["source"])
        candidate["source"]["tool"] = "pxdesign_pipeline_multi_gpu"
        scores = dict(candidate.get("scores") or {})
        px_scores = scores.pop("pxdesign", {})
        scores["pxdesign_pipeline_multi_gpu"] = px_scores
        candidate["scores"] = scores
    return candidates


__all__ = ["PXDesignPipelineMultiGpuAdapter", "_pxdesign_pipeline_multi_gpu_candidates"]
