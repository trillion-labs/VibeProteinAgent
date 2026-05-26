from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext


class NoopInput(BaseModel):
    payload: str


class NoopOutput(BaseModel):
    payload: str
    echoed_at: float


@register_adapter("descriptor.yaml")
class NoopAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: NoopInput, ctx: AdapterRunContext) -> NoopOutput:
        ctx.emit_progress(0.0)
        time.sleep(0.05)
        ctx.emit_progress(0.5)
        time.sleep(0.05)
        ctx.emit_progress(1.0)
        return NoopOutput(payload=params.payload, echoed_at=time.time())

    def estimate_vram(self, params: NoopInput) -> int:
        return 0
