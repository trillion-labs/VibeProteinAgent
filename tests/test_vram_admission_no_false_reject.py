"""Codex round-4 RC2: VRAM admission must not double-count resident weights.

The pre-fix bug: :class:`ToolAdapter.estimate_vram(params)` was
contractually ambiguous — could be read as "incremental need" or "total
including resident model". The worker's gate-(b) check assumed
"incremental" but the in-tree adapters had been written interpreting
"total" (descriptor.vram_min_gib is a total). Result: once an adapter
was loaded, the check effectively required
``free >= total + already_used + headroom`` — the resident weights were
counted twice, false-rejecting jobs that would actually fit.

The fix:

1. ``ToolAdapter.estimate_vram`` now documents that it returns the
   TOTAL VRAM the job needs (including resident weights).
2. ``Worker._check_params_admission`` subtracts the adapter's cached
   ``vram_used_mib`` from that total to get the incremental need.

This test reproduces the false-reject scenario from the task brief
(large resident model, modest free headroom) and asserts the new
admission path admits the job.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from vibeproteinagent.core.errors import VramExhaustedError
from vibeproteinagent.workers import worker as worker_mod
from vibeproteinagent.workers.worker import AdapterMeta, Worker


@dataclass
class _Descriptor:
    name: str = "synthetic"
    version: str = "1.0.0"
    vram_min_gib: int = 4


class _TotalEstimateAdapter:
    """Adapter whose ``estimate_vram`` returns the TOTAL — the new contract."""

    def __init__(self, total_mib: int) -> None:
        self._total = total_mib

    def estimate_vram(self, params: Any) -> int:
        return self._total


def _bare_worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_mib: int,
    total_mib: int = 200_000,
    min_free_mib: int = 2_048,
) -> Worker:
    """Worker skeleton without subprocess setup, with mocked VRAM probe."""
    monkeypatch.setattr(worker_mod, "_setup_torch", lambda gpu_id, log=None: object())
    w = Worker.__new__(Worker)
    w.gpu_id = -1
    w._busy = False
    w._adapter_cache = {}  # type: ignore[assignment]
    w._adapter_meta = {}
    w.min_free_vram_mib = min_free_mib
    w._torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    w._probe_vram = lambda: (free_mib, total_mib)  # type: ignore[assignment]
    w._log = SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return w


def _record_residency(w: Worker, desc: _Descriptor, vram_used_mib: int) -> None:
    """Populate the adapter-meta map as if ``_load_adapter`` had just measured
    ``vram_used_mib`` MiB of resident weights for ``(desc.name, desc.version)``.
    """
    w._adapter_meta[(desc.name, desc.version)] = AdapterMeta(
        vram_used_mib=vram_used_mib,
        last_used_ts=0.0,
        load_count=1,
        run_count=0,
        pin_warm=False,
    )


def test_admits_when_resident_residency_covers_most_of_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario from the spec.

    Worker total=110_000, free=30_000 (after loading a big model).
    Adapter is loaded; residency = 80_000 MiB.
    ``estimate_vram(params)`` returns 100_000 (total, including resident).

    Pre-fix: ``free 30_000 < 100_000 + 2_048`` → false-reject.
    Post-fix: incremental = 100_000 - 80_000 = 20_000; ``free 30_000 >= 22_048``
    → admit.
    """
    desc = _Descriptor(vram_min_gib=4)
    adapter = _TotalEstimateAdapter(total_mib=100_000)
    params = object()
    w = _bare_worker(monkeypatch, free_mib=30_000, total_mib=110_000, min_free_mib=2_048)
    _record_residency(w, desc, vram_used_mib=80_000)

    # Should NOT raise.
    w._check_params_admission(desc, adapter, params)


def test_still_rejects_when_incremental_need_exceeds_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: even with the residency subtracted, a too-tight machine
    still gets rejected with the new error envelope.
    """
    desc = _Descriptor(vram_min_gib=4)
    adapter = _TotalEstimateAdapter(total_mib=300_000)  # huge job
    params = object()
    w = _bare_worker(monkeypatch, free_mib=30_000, total_mib=110_000, min_free_mib=2_048)
    _record_residency(w, desc, vram_used_mib=80_000)

    with pytest.raises(VramExhaustedError) as exc_info:
        w._check_params_admission(desc, adapter, params)
    err = exc_info.value
    assert err.code == "E_VRAM_EXHAUSTED"
    assert err.details["params_dependent"] is True
    assert err.details["total_required_mib"] == 300_000
    assert err.details["loaded_residency_mib"] == 80_000
    # 300_000 - 80_000 = 220_000 incremental
    assert err.details["incremental_need_mib"] == 220_000
    assert err.details["free_mib"] == 30_000


def test_cold_load_path_residency_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the adapter is not yet in the meta map (cold path) the check
    falls back to ``total`` — same as pre-fix behaviour for the cold case.
    """
    desc = _Descriptor(vram_min_gib=4)
    adapter = _TotalEstimateAdapter(total_mib=10_000)
    params = object()
    w = _bare_worker(monkeypatch, free_mib=20_000, total_mib=40_000, min_free_mib=2_048)
    # No _record_residency() call — adapter meta absent.

    # 10_000 + 2_048 <= 20_000 → admit.
    w._check_params_admission(desc, adapter, params)

    # And the same adapter with a heavier total rejects:
    heavy = _TotalEstimateAdapter(total_mib=25_000)
    with pytest.raises(VramExhaustedError) as exc_info:
        w._check_params_admission(desc, heavy, params)
    assert exc_info.value.details["loaded_residency_mib"] == 0
    assert exc_info.value.details["incremental_need_mib"] == 25_000


def test_descriptor_default_treated_as_total(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ToolAdapter.estimate_vram`` default returns ``vram_min_gib * 1024``.

    Per the new contract that's the *total*. When the adapter is loaded the
    admission check subtracts its residency, so on warm second runs the
    descriptor-floor adapter sees ~0 incremental need.
    """
    from vibeproteinagent.adapters.base import ToolAdapter

    # Build a real ToolAdapter subclass instance without invoking torch.
    class _Stub(ToolAdapter):
        descriptor = SimpleNamespace(vram_min_gib=4, supports_batching=False)

        def __init__(self) -> None:  # type: ignore[no-untyped-def]
            pass

        def run(self, params, ctx):  # type: ignore[override]
            raise NotImplementedError

    adapter = _Stub()
    assert adapter.estimate_vram(SimpleNamespace()) == 4 * 1024  # MiB, total

    desc = _Descriptor(vram_min_gib=4)
    w = _bare_worker(monkeypatch, free_mib=3_000, total_mib=8_192, min_free_mib=2_048)
    # Adapter resident at exactly the descriptor floor (4 GiB).
    _record_residency(w, desc, vram_used_mib=4 * 1024)
    # incremental = max(0, 4*1024 - 4*1024) = 0; free 3_000 >= 0 + 2_048.
    w._check_params_admission(desc, adapter, SimpleNamespace())
