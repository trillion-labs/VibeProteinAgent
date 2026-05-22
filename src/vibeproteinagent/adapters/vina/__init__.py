from __future__ import annotations

from .adapter import VinaAdapter, VinaInput, VinaMode, VinaOutput, _find_binary


def availability_skip_reason() -> str | None:
    try:
        _find_binary()
        return None
    except FileNotFoundError as exc:
        return f"AutoDock Vina binary unavailable: {exc}. Install `vina` or set $VPA_VINA_BINARY."


__all__ = [
    "VinaAdapter",
    "VinaInput",
    "VinaMode",
    "VinaOutput",
    "availability_skip_reason",
]
