from __future__ import annotations

from .adapter import FoldseekAdapter, FoldseekInput, FoldseekOutput, _find_binary


def availability_skip_reason() -> str | None:
    try:
        _find_binary()
        return None
    except FileNotFoundError as exc:
        return (
            f"Foldseek binary unavailable: {exc}. Install `foldseek` or set "
            "$VPA_FOLDSEEK_BINARY."
        )


__all__ = ["FoldseekAdapter", "FoldseekInput", "FoldseekOutput", "availability_skip_reason"]
