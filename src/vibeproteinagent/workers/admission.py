"""Shared VRAM admission helper.

Both the supervisor's scheduler (when picking a candidate worker for a
queued job) and the worker (just before loading an adapter) must agree on
the amount of VRAM a job needs. If they disagree, the scheduler can dispatch
a job that the worker then rejects with ``E_VRAM_EXHAUSTED`` — wasted
admission round-trip plus a confused operator.

This module is the single owner of the admission formula:

  required = descriptor.vram_min_gib * 1024     # MiB
  if an adapter is loaded for these (descriptor, params):
      required = max(required, adapter.estimate_vram(params))

The scheduler-side caller does NOT have an instantiated adapter for
not-yet-loaded tools, so it falls back to the descriptor's declared
minimum. The worker-side caller, after load, has the live adapter and
can use ``estimate_vram`` for a tighter bound.
"""

from __future__ import annotations

from typing import Any

_DEFAULT_VRAM_GIB = 4
"""Fallback when a descriptor does not declare ``vram_min_gib``."""


def required_vram_mib(
    descriptor: Any,
    adapter: Any | None = None,
    params: Any | None = None,
) -> int:
    """Return the MiB of VRAM a (tool, params) pair needs.

    ``descriptor`` is a :class:`ToolDescriptor` (or any object exposing
    ``vram_min_gib``). ``adapter`` is an optional live :class:`ToolAdapter`
    instance — supplied by the worker once load completes; ``None`` from
    the scheduler before placement.

    ``params`` is the validated pydantic params model (or ``None`` when
    placement is happening before params are bound). When both adapter and
    params are present, ``adapter.estimate_vram(params)`` raises the
    minimum; any exception falls back to the descriptor's declaration so
    a buggy ``estimate_vram`` cannot brick the scheduler.
    """
    gib = getattr(descriptor, "vram_min_gib", None)
    if gib is None:
        gib = _DEFAULT_VRAM_GIB
    base = int(gib) * 1024
    if adapter is not None and params is not None:
        try:
            estimated = int(adapter.estimate_vram(params))
        except Exception:
            return base
        return max(base, estimated)
    return base


__all__ = ["required_vram_mib"]
