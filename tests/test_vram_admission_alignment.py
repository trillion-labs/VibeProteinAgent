"""Codex review R10: supervisor + worker must agree on VRAM admission.

Before Phase 7 the supervisor's placement code looked at
``descriptor.vram_min_gib`` directly while the worker's ``_load_adapter``
called a private ``_required_vram_mib`` helper. They computed the same
formula by hand twice — easy to drift.

The fix is a single shared helper :func:`workers.admission.required_vram_mib`
that both call. This test pins the contract: identical input → identical
output, and the worker's pre-load check uses the helper too.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibeproteinagent.workers.admission import required_vram_mib


@dataclass
class _StubDescriptor:
    vram_min_gib: int | None = 4


def test_returns_descriptor_minimum_when_no_adapter() -> None:
    """Scheduler-side path: no live adapter, no params → descriptor MiB."""
    d = _StubDescriptor(vram_min_gib=2)
    assert required_vram_mib(d) == 2 * 1024
    assert required_vram_mib(d, adapter=None, params=None) == 2 * 1024


def test_falls_back_to_4gib_default_when_unset() -> None:
    d = _StubDescriptor(vram_min_gib=None)
    assert required_vram_mib(d) == 4 * 1024


def test_adapter_estimate_raises_floor_above_descriptor() -> None:
    d = _StubDescriptor(vram_min_gib=2)

    class FakeAdapter:
        def estimate_vram(self, _params: object) -> int:
            return 8 * 1024

    assert required_vram_mib(d, FakeAdapter(), object()) == 8 * 1024


def test_adapter_estimate_clamped_to_descriptor_floor() -> None:
    d = _StubDescriptor(vram_min_gib=12)

    class CheapAdapter:
        def estimate_vram(self, _params: object) -> int:
            return 1024  # 1 GiB — descriptor declares more

    assert required_vram_mib(d, CheapAdapter(), object()) == 12 * 1024


def test_buggy_adapter_estimate_does_not_brick() -> None:
    d = _StubDescriptor(vram_min_gib=2)

    class BrokenAdapter:
        def estimate_vram(self, _params: object) -> int:
            raise RuntimeError("no")

    assert required_vram_mib(d, BrokenAdapter(), object()) == 2 * 1024


def test_supervisor_and_worker_paths_agree() -> None:
    """Both call sites must produce the same number for the same args.

    This is the integration assertion: the supervisor's scheduler path and
    the worker's pre-load check call ``required_vram_mib`` identically; a
    job whose required VRAM exceeds the candidate's free space is filtered
    by the scheduler and never reaches the worker.
    """
    from vibeproteinagent.workers import supervisor as sup_mod
    from vibeproteinagent.workers import worker as worker_mod

    # The two import the same helper from the same module.
    assert sup_mod.required_vram_mib is worker_mod.required_vram_mib

    d = _StubDescriptor(vram_min_gib=2)
    assert sup_mod.required_vram_mib(d) == worker_mod.required_vram_mib(d)
