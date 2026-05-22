from __future__ import annotations

from vibeproteinagent.adapters.base import register_adapter
from vibeproteinagent.adapters.pxdesign.adapter import PXDesignAdapter


@register_adapter("descriptor.yaml")
class PXDesignPipelineMultiGpuAdapter(PXDesignAdapter):
    """PXDesign pipeline alias that requires multi-worker distributed dispatch."""


__all__ = ["PXDesignPipelineMultiGpuAdapter"]
