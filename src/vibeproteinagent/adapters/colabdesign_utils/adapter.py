from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext


class ColabDesignUtilsInput(BaseModel):
    action: Literal["capabilities"] = "capabilities"
    check_import: bool = Field(
        False,
        description=(
            "When true, also check whether colabdesign is importable in the "
            "current Python interpreter. Sidecar runtimes are always probed."
        ),
    )


class ColabDesignUtility(BaseModel):
    name: str
    status: Literal["sidecar_available", "package_available", "unavailable"]
    reason: str


class ColabDesignUtilsOutput(BaseModel):
    action: Literal["capabilities"]
    package_available: bool | None
    import_error: str | None = None
    utilities: list[ColabDesignUtility]
    deferred: bool
    runtime_ms: int


def _sidecar_runtime_reasons() -> dict[str, str | None]:
    """Return ColabDesign sidecar readiness by runtime name."""
    reasons: dict[str, str | None] = {}

    try:
        from vibeproteinagent.adapters.pxdesign import (
            availability_skip_reason as pxdesign_skip_reason,
        )

        reasons["pxdesign_pipeline"] = pxdesign_skip_reason()
    except Exception as exc:  # pragma: no cover - defensive dependency probe
        reasons["pxdesign_pipeline"] = f"{type(exc).__name__}: {exc}"

    try:
        from vibeproteinagent.adapters.bindcraft import (
            availability_skip_reason as bindcraft_skip_reason,
        )

        reasons["bindcraft"] = bindcraft_skip_reason()
    except Exception as exc:  # pragma: no cover - defensive dependency probe
        reasons["bindcraft"] = f"{type(exc).__name__}: {exc}"

    return reasons


def colabdesign_runtime_skip_reason() -> str | None:
    """Return a concrete reason if no ColabDesign execution runtime is ready."""
    if importlib.util.find_spec("colabdesign") is not None:
        return None

    reasons = _sidecar_runtime_reasons()
    if any(reason is None for reason in reasons.values()):
        return None

    details = "; ".join(f"{name}: {reason}" for name, reason in sorted(reasons.items()))
    return (
        "colabdesign is not importable in the shared vpa environment and no "
        f"ColabDesign sidecar runtime is ready ({details})."
    )


@register_adapter("descriptor.yaml")
class ColabDesignUtilsAdapter(ToolAdapter):
    """Machine-readable accounting surface for ColabDesign utility support.

    This adapter is intentionally lightweight. It registers ColabDesign utility
    capability metadata without importing JAX, AF2 params, or ColabDesign during
    discovery and normal fast tests.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.config = dict(config) if config else {}
        self.weights_dir = Path(weights_dir)
        self.device = device

    def run(
        self, params: ColabDesignUtilsInput, ctx: AdapterRunContext
    ) -> ColabDesignUtilsOutput:
        t0 = time.time()
        ctx.emit_progress(0.25)
        package_available: bool | None = None
        import_error: str | None = None
        sidecar_reasons = _sidecar_runtime_reasons()
        sidecar_ready = [name for name, skip in sidecar_reasons.items() if skip is None]
        status: Literal["sidecar_available", "package_available", "unavailable"]
        if sidecar_ready:
            status = "sidecar_available"
            reason = (
                "ColabDesign execution is available through verified persistent "
                f"sidecar runtime(s): {', '.join(sorted(sidecar_ready))}."
            )
        else:
            status = "unavailable"
            reason = (
                "No ColabDesign execution runtime is available. "
                + "; ".join(
                    f"{name}: {sidecar_reasons[name]}" for name in sorted(sidecar_reasons)
                )
            )

        if params.check_import:
            try:
                package_available = importlib.util.find_spec("colabdesign") is not None
            except Exception as exc:  # pragma: no cover - defensive importlib guard
                package_available = False
                import_error = f"{type(exc).__name__}: {exc}"
            if package_available:
                status = "package_available"
                reason = "colabdesign is importable in this Python environment."

        utilities = [
            ColabDesignUtility(name="binder_design", status=status, reason=reason),
            ColabDesignUtility(name="partial_sequence_design", status=status, reason=reason),
            ColabDesignUtility(name="hallucination", status=status, reason=reason),
            ColabDesignUtility(name="af2_validation", status=status, reason=reason),
        ]
        ctx.emit_progress(1.0)
        return ColabDesignUtilsOutput(
            action=params.action,
            package_available=package_available,
            import_error=import_error,
            utilities=utilities,
            deferred=status == "unavailable",
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: ColabDesignUtilsInput) -> int:
        return 0


__all__ = [
    "ColabDesignUtility",
    "ColabDesignUtilsAdapter",
    "ColabDesignUtilsInput",
    "ColabDesignUtilsOutput",
    "colabdesign_runtime_skip_reason",
]
