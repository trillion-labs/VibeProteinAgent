from __future__ import annotations

import time
from collections.abc import Callable

from vibeproteinagent.workspace import JobStore


def poll_cancel(
    job_store: JobStore,
    job_id: str,
    interval_s: float = 1.0,
) -> Callable[[], bool]:
    """Return a closure suitable for :class:`AdapterRunContext.is_cancelled`.

    The closure rate-limits DB reads to once per ``interval_s`` seconds; in between,
    it returns the last observed value. Once it sees cancellation, it stays True.
    """
    state = {"last_check": 0.0, "cancelled": False}

    def _check() -> bool:
        if state["cancelled"]:
            return True
        now = time.monotonic()
        if now - state["last_check"] < interval_s:
            return state["cancelled"]
        state["last_check"] = now
        try:
            state["cancelled"] = job_store.is_cancel_requested(job_id)
        except Exception:
            return state["cancelled"]
        return state["cancelled"]

    return _check


__all__ = ["poll_cancel"]
