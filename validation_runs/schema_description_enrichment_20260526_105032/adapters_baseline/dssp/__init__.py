from __future__ import annotations

from .adapter import DSSPAdapter, DSSPInput, DSSPOutput, _find_binary


def availability_skip_reason() -> str | None:
    try:
        _find_binary()
        return None
    except FileNotFoundError as exc:
        return f"DSSP binary unavailable: {exc}. Install `mkdssp`/`dssp` or set $VPA_DSSP_BINARY."


__all__ = ["DSSPAdapter", "DSSPInput", "DSSPOutput", "availability_skip_reason"]
