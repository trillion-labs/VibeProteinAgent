"""Codex round-3 RC1: worker gate (b) — param-aware VRAM admission.

Gate (a) (descriptor floor, pre-load) was wired in Phase 7. Gate (b) is
the per-call admission once the adapter is loaded and the live
``estimate_vram(params)`` can give a tighter bound. The worker now
calls :func:`workers.admission.required_vram_mib` with the loaded
adapter and the validated params model; if the result + headroom
exceeds free VRAM the worker rejects with ``E_VRAM_EXHAUSTED`` and the
supervisor's self-heal (RC5) re-queues + blocklists. See
:mod:`tests.test_vram_disagreement_self_heals` for the supervisor side.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from vibeproteinagent.core.errors import VramExhaustedError
from vibeproteinagent.workers import worker as worker_mod
from vibeproteinagent.workers.worker import Worker


@dataclass
class _StubDescriptor:
    vram_min_gib: int = 4
    name: str = "synthetic"
    version: str = "1.0.0"


class _HeavyParams(SimpleNamespace):
    heavy: bool


class _SyntheticAdapter:
    """Adapter whose VRAM estimate explodes when ``params.heavy`` is set."""

    def __init__(self, descriptor: _StubDescriptor) -> None:
        self._descriptor = descriptor

    def estimate_vram(self, params: Any) -> int:
        base = self._descriptor.vram_min_gib * 1024
        if getattr(params, "heavy", False):
            return base + 200_000
        return base


def _bare_worker(monkeypatch: pytest.MonkeyPatch, free_mib: int) -> Worker:
    """Construct a Worker without subprocess setup or torch dependency."""
    monkeypatch.setattr(worker_mod, "_setup_torch", lambda gpu_id, log=None: object())
    w = Worker.__new__(Worker)
    w.gpu_id = -1
    w._busy = False
    w._adapter_cache = {}  # type: ignore[assignment]
    w._adapter_meta = {}
    w.min_free_vram_mib = 2 * 1024
    w._torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    w._probe_vram = lambda: (free_mib, free_mib + 4096)  # type: ignore[assignment]
    w._log = SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return w


def test_admits_when_params_light(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate (b): heavy=False → estimate stays at descriptor floor → admit."""
    desc = _StubDescriptor(vram_min_gib=4)
    adapter = _SyntheticAdapter(desc)
    params = _HeavyParams(heavy=False)
    w = _bare_worker(monkeypatch, free_mib=180_000)
    # No raise.
    w._check_params_admission(desc, adapter, params)


def test_rejects_when_params_blow_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate (b): heavy=True → 4 GiB + 200_000 MiB > 180_000 free → reject."""
    desc = _StubDescriptor(vram_min_gib=4)
    adapter = _SyntheticAdapter(desc)
    params = _HeavyParams(heavy=True)
    w = _bare_worker(monkeypatch, free_mib=180_000)
    with pytest.raises(VramExhaustedError) as exc_info:
        w._check_params_admission(desc, adapter, params)
    err = exc_info.value
    assert err.code == "E_VRAM_EXHAUSTED"
    assert err.details["params_dependent"] is True
    # The estimate must reflect the heavy bump. Codex round-4 RC2 renamed
    # ``required_mib`` -> ``total_required_mib`` and added the residency
    # breakdown; both keys exist on the new envelope.
    assert err.details["total_required_mib"] >= 200_000
    assert err.details["loaded_residency_mib"] == 0   # adapter not in meta map
    assert err.details["incremental_need_mib"] >= 200_000


def test_helper_signature_matches_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker's gate (b) call uses ``required_vram_mib(desc, adapter, params)``."""
    from vibeproteinagent.workers.admission import required_vram_mib

    desc = _StubDescriptor(vram_min_gib=4)
    adapter = _SyntheticAdapter(desc)
    light = _HeavyParams(heavy=False)
    heavy = _HeavyParams(heavy=True)
    assert required_vram_mib(desc, adapter, light) == 4 * 1024
    assert required_vram_mib(desc, adapter, heavy) == 4 * 1024 + 200_000


def test_free_vram_mib_returns_first_probe_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared helper used by gate (b) reads ``_probe_vram()[0]``."""
    w = _bare_worker(monkeypatch, free_mib=12345)
    assert w.free_vram_mib() == 12345
