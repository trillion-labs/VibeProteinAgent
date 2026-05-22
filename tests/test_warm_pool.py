"""LRU eviction + warm-pool unit tests for Worker.

We avoid spawning real worker subprocesses; instead a Worker is constructed with
``gpu_id=-1`` (CPU mode) and ``_probe_vram`` is monkey-patched to simulate VRAM
pressure. Adapter classes are tiny fakes that record load/unload activity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from vibeproteinagent.core.errors import VramExhaustedError
from vibeproteinagent.workers.worker import Worker
from vibeproteinagent.workspace import Workspace


@dataclass
class _FakeDescriptor:
    name: str
    version: str
    vram_min_gib: int
    pin_warm: bool = False
    gpu_required: bool = False
    gpus_required: int = 1


class _FakeAdapter:
    """Plain class — not a ToolAdapter subclass, but quacks like one for Worker."""

    instances_created = 0
    instances_unloaded = 0

    def __init__(self, config: dict, weights_dir: Path, device: Any) -> None:
        type(self).instances_created += 1
        self.config = config

    def run(self, params: Any, ctx: Any) -> Any:  # pragma: no cover - unused in unit tests
        return {}

    def unload(self) -> None:
        type(self).instances_unloaded += 1


def _make_fake_class(name: str) -> type[_FakeAdapter]:
    """Each adapter key needs its own subclass so per-class counters don't bleed."""
    cls = type(name, (_FakeAdapter,), {"instances_created": 0, "instances_unloaded": 0})
    return cls  # type: ignore[return-value]


def _make_worker(tmp_workspace: Workspace, *, total_free_mib: int = 32 * 1024,
                 min_free_vram_mib: int = 2 * 1024) -> Worker:
    """Construct a CPU-only Worker with a controllable VRAM probe.

    ``_probe_vram`` is monkey-patched to compute free = total - sum(adapter_meta).
    Because it inspects the live cache state, eviction during ``_load_adapter``
    is immediately reflected — exactly how the real CUDA path behaves.
    """
    worker = Worker(
        gpu_id=-1,
        workspace_name=tmp_workspace.name,
        min_free_vram_mib=min_free_vram_mib,
    )
    worker._sim_total_mib = total_free_mib  # type: ignore[attr-defined]

    def _probe() -> tuple[int, int]:
        used = sum(m.vram_used_mib for m in worker._adapter_meta.values())
        free = max(0, worker._sim_total_mib - used)  # type: ignore[attr-defined]
        return (free, worker._sim_total_mib)  # type: ignore[attr-defined]

    worker._probe_vram = _probe  # type: ignore[method-assign]
    return worker


def _load(worker: Worker, descriptor: _FakeDescriptor, cls: type) -> Any:
    return worker._load_adapter(descriptor.name, descriptor.version, descriptor, cls)


def test_load_then_hit_returns_same_instance(tmp_workspace: Workspace) -> None:
    worker = _make_worker(tmp_workspace)
    d = _FakeDescriptor(name="alpha", version="1", vram_min_gib=1)
    cls = _make_fake_class("AlphaAdapter")
    a1 = _load(worker, d, cls)
    a2 = _load(worker, d, cls)
    assert a1 is a2
    assert cls.instances_created == 1


def test_lru_eviction_kicks_in_when_vram_tight(tmp_workspace: Workspace) -> None:
    """With ~16 GiB free + 2 GiB headroom, three 5-GiB loads must evict the first."""
    worker = _make_worker(tmp_workspace, total_free_mib=16 * 1024, min_free_vram_mib=2 * 1024)
    descs = [_FakeDescriptor(name=f"a{i}", version="1", vram_min_gib=5) for i in range(3)]
    classes = [_make_fake_class(f"A{i}") for i in range(3)]

    # Loads 1 + 2 fit (10 GiB used, 6 GiB free, 5+2 headroom not satisfied for #3).
    _load(worker, descs[0], classes[0])
    _load(worker, descs[1], classes[1])
    assert len(worker._adapter_cache) == 2

    # Touch desc[1] so desc[0] becomes the LRU victim.
    worker._touch((descs[1].name, descs[1].version))

    _load(worker, descs[2], classes[2])
    # desc[0] should be evicted; desc[1] and desc[2] should remain.
    keys = [k[0] for k in worker._adapter_cache]
    assert "a0" not in keys, keys
    assert "a1" in keys and "a2" in keys
    assert classes[0].instances_unloaded == 1


def test_pin_warm_survives_eviction_sweep(tmp_workspace: Workspace) -> None:
    """A pin_warm adapter is skipped while a non-pinned LRU peer is evicted."""
    worker = _make_worker(tmp_workspace, total_free_mib=16 * 1024, min_free_vram_mib=2 * 1024)
    d_pin = _FakeDescriptor(name="pinme", version="1", vram_min_gib=5, pin_warm=True)
    d_norm = _FakeDescriptor(name="norm", version="1", vram_min_gib=5)
    d_new = _FakeDescriptor(name="new", version="1", vram_min_gib=5)
    cls_pin = _make_fake_class("PinAdapter")
    cls_norm = _make_fake_class("NormAdapter")
    cls_new = _make_fake_class("NewAdapter")

    _load(worker, d_pin, cls_pin)
    _load(worker, d_norm, cls_norm)
    # `pinme` is LRU (loaded first); `norm` is MRU. With 6 GiB free we cannot
    # fit `new` (5+2 = 7) — eviction must skip the pinned `pinme` and take `norm`.
    _load(worker, d_new, cls_new)

    keys = {k[0] for k in worker._adapter_cache}
    assert "pinme" in keys, "pin_warm adapter was evicted!"
    assert "new" in keys
    assert "norm" not in keys
    assert cls_norm.instances_unloaded == 1
    assert cls_pin.instances_unloaded == 0


def test_all_pinned_and_over_budget_raises(tmp_workspace: Workspace) -> None:
    """If every cached adapter is pinned and free VRAM cannot satisfy the load,
    we raise E_VRAM_EXHAUSTED."""
    worker = _make_worker(tmp_workspace, total_free_mib=12 * 1024, min_free_vram_mib=2 * 1024)
    d_pin1 = _FakeDescriptor(name="pa", version="1", vram_min_gib=5, pin_warm=True)
    d_pin2 = _FakeDescriptor(name="pb", version="1", vram_min_gib=5, pin_warm=True)
    d_new = _FakeDescriptor(name="pc", version="1", vram_min_gib=5)
    cls_pin1 = _make_fake_class("PA")
    cls_pin2 = _make_fake_class("PB")
    cls_new = _make_fake_class("PC")

    _load(worker, d_pin1, cls_pin1)
    _load(worker, d_pin2, cls_pin2)

    with pytest.raises(VramExhaustedError) as ei:
        _load(worker, d_new, cls_new)
    assert "pin_warm" in str(ei.value).lower()


def test_run_count_and_lru_promotion(tmp_workspace: Workspace) -> None:
    """`_touch` promotes the entry to MRU; load_count / run_count are stable."""
    worker = _make_worker(tmp_workspace)
    d1 = _FakeDescriptor(name="x", version="1", vram_min_gib=1)
    d2 = _FakeDescriptor(name="y", version="1", vram_min_gib=1)
    cls1 = _make_fake_class("X")
    cls2 = _make_fake_class("Y")

    _load(worker, d1, cls1)
    _load(worker, d2, cls2)
    # Order should be [x, y] (insertion order; y is MRU).
    assert list(worker._adapter_cache.keys())[-1] == ("y", "1")
    worker._touch(("x", "1"))
    assert list(worker._adapter_cache.keys())[-1] == ("x", "1")


def test_required_vram_defaults_to_4gib_when_descriptor_null(tmp_workspace: Workspace) -> None:
    worker = _make_worker(tmp_workspace, total_free_mib=4 * 1024,
                          min_free_vram_mib=2 * 1024)
    d = _FakeDescriptor(name="d", version="1", vram_min_gib=None)  # type: ignore[arg-type]
    cls = _make_fake_class("D")
    # Required = 4 GiB, headroom = 2 GiB; total 6 > 4 free -> should raise.
    with pytest.raises(VramExhaustedError):
        _load(worker, d, cls)


class _UnhealthyAdapter(_FakeAdapter):
    """Cold-load succeeds but healthcheck() returns False — Pi1 fixture."""

    def healthcheck(self) -> bool:  # type: ignore[override]
        return False


class _RaisingAdapter(_FakeAdapter):
    """Cold-load succeeds but healthcheck() raises."""

    def healthcheck(self) -> bool:  # type: ignore[override]
        raise RuntimeError("license file missing")


def test_warm_blocking_surfaces_healthy_true_when_default(tmp_workspace: Workspace) -> None:
    """Pi1: healthy=True / warning=None for vanilla adapters."""
    worker = _make_worker(tmp_workspace)
    d = _FakeDescriptor(name="alpha", version="1", vram_min_gib=1)
    cls = _make_fake_class("AlphaAdapter")

    class _FakeRegistry:
        def get(self, name: str, version_spec: str) -> tuple[Any, type]:
            return d, cls

    # Treat the fake adapter as not implementing healthcheck() -> defaults True
    cls.healthcheck = lambda self: True
    worker._registry = _FakeRegistry()  # type: ignore[assignment]

    result = worker._warm_blocking("alpha", "*")
    assert result["healthy"] is True
    assert result["warning"] is None


def test_warm_blocking_surfaces_warning_when_unhealthy(tmp_workspace: Workspace) -> None:
    """Pi1: healthy=False with a non-None warning when healthcheck returns False."""
    worker = _make_worker(tmp_workspace)
    d = _FakeDescriptor(name="gated", version="1", vram_min_gib=1)
    cls = _make_fake_class("GatedAdapter")
    cls.healthcheck = _UnhealthyAdapter.healthcheck

    class _FakeRegistry:
        def get(self, name: str, version_spec: str) -> tuple[Any, type]:
            return d, cls

    worker._registry = _FakeRegistry()  # type: ignore[assignment]
    result = worker._warm_blocking("gated", "*")
    assert result["healthy"] is False
    assert isinstance(result["warning"], str) and result["warning"]
    assert "healthcheck returned False" in result["warning"].lower() \
        or "gate" in result["warning"].lower()


def test_warm_blocking_surfaces_exception_warning(tmp_workspace: Workspace) -> None:
    """Pi1: healthcheck raising is captured as healthy=False + reason."""
    worker = _make_worker(tmp_workspace)
    d = _FakeDescriptor(name="boom", version="1", vram_min_gib=1)
    cls = _make_fake_class("BoomAdapter")
    cls.healthcheck = _RaisingAdapter.healthcheck

    class _FakeRegistry:
        def get(self, name: str, version_spec: str) -> tuple[Any, type]:
            return d, cls

    worker._registry = _FakeRegistry()  # type: ignore[assignment]
    result = worker._warm_blocking("boom", "*")
    assert result["healthy"] is False
    assert "RuntimeError" in (result["warning"] or "")
    assert "license file missing" in (result["warning"] or "")


def test_info_includes_vram_and_loaded_adapters(tmp_workspace: Workspace) -> None:
    worker = _make_worker(tmp_workspace, total_free_mib=16 * 1024)
    d = _FakeDescriptor(name="z", version="1", vram_min_gib=2, pin_warm=True)
    cls = _make_fake_class("Z")
    _load(worker, d, cls)
    info = worker._info()
    assert "free_vram_mib" in info
    assert "total_vram_mib" in info
    assert info["total_vram_mib"] == 16 * 1024
    assert info["loaded_adapters"]
    entry = info["loaded_adapters"][0]
    assert entry["name"] == "z"
    assert entry["version"] == "1"
    assert entry["pin_warm"] is True
    assert "last_used_ts" in entry
    assert "vram_used_mib" in entry
    assert entry["run_count"] == 0
