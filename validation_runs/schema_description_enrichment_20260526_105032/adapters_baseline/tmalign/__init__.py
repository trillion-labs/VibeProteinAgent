from __future__ import annotations

from .adapter import TMalignAdapter, TMalignInput, TMalignOutput, _find_binary


def availability_skip_reason() -> str | None:
    try:
        _find_binary()
        return None
    except FileNotFoundError as exc:
        return (
            f"TM-align binary unavailable: {exc}. Install `TMalign`/`tmalign` or set "
            "$VPA_TMALIGN_BINARY."
        )


__all__ = ["TMalignAdapter", "TMalignInput", "TMalignOutput", "availability_skip_reason"]
