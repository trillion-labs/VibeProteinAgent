"""Adapter incompatibility tests for Worker._load_adapter.

Two synthetic adapters declared mutually incompatible. Loading one must evict
the other regardless of ``pin_warm`` state. Eviction is reciprocal: declaring
``incompatible_with`` on the descriptor of the *incoming* adapter is enough;
the cached adapter does not need to declare it. The reciprocal direction is
also tested — cached adapter Y whose own descriptor lists X causes Y to be
evicted when X loads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vibeproteinagent.workers.worker import Worker
from vibeproteinagent.workspace import Workspace


@dataclass
class _FakeDescriptor:
    name: str
    version: str
    vram_min_gib: int = 1
    pin_warm: bool = False
    incompatible_with: list[str] = field(default_factory=list)


class _FakeAdapter:
    """Bare-bones adapter substitute that records its own descriptor."""

    instances_created = 0
    instances_unloaded = 0

    def __init__(self, config: dict, weights_dir: Path, device: Any) -> None:
        type(self).instances_created += 1

    def run(self, params: Any, ctx: Any) -> Any:  # pragma: no cover - not exercised
        return {}

    def unload(self) -> None:
        type(self).instances_unloaded += 1


def _make_fake_class(name: str, descriptor: _FakeDescriptor) -> type[_FakeAdapter]:
    """Per-key subclass so counters and the bound descriptor don't bleed across tests."""
    cls = type(
        name,
        (_FakeAdapter,),
        {
            "instances_created": 0,
            "instances_unloaded": 0,
            "descriptor": descriptor,
        },
    )
    return cls  # type: ignore[return-value]


def _make_worker(tmp_workspace: Workspace) -> Worker:
    worker = Worker(
        gpu_id=-1,
        workspace_name=tmp_workspace.name,
        min_free_vram_mib=2 * 1024,
    )
    worker._sim_total_mib = 256 * 1024  # type: ignore[attr-defined]

    def _probe() -> tuple[int, int]:
        used = sum(m.vram_used_mib for m in worker._adapter_meta.values())
        free = max(0, worker._sim_total_mib - used)  # type: ignore[attr-defined]
        return (free, worker._sim_total_mib)  # type: ignore[attr-defined]

    worker._probe_vram = _probe  # type: ignore[method-assign]
    return worker


def _load(worker: Worker, descriptor: _FakeDescriptor, cls: type) -> Any:
    return worker._load_adapter(descriptor.name, descriptor.version, descriptor, cls)


def test_incompatible_with_evicts_cached_pinned_adapter(tmp_workspace: Workspace) -> None:
    """Loading B evicts A when B's descriptor lists A — even if A is pinned."""
    worker = _make_worker(tmp_workspace)
    d_a = _FakeDescriptor(name="alpha", version="1", pin_warm=True,
                          incompatible_with=["bravo"])
    d_b = _FakeDescriptor(name="bravo", version="1", pin_warm=False,
                          incompatible_with=["alpha"])
    cls_a = _make_fake_class("AlphaAdapter", d_a)
    cls_b = _make_fake_class("BravoAdapter", d_b)

    _load(worker, d_a, cls_a)
    assert ("alpha", "1") in worker._adapter_cache
    assert worker._adapter_meta[("alpha", "1")].pin_warm is True

    _load(worker, d_b, cls_b)
    # alpha is evicted even though pin_warm=True
    assert ("alpha", "1") not in worker._adapter_cache
    assert ("bravo", "1") in worker._adapter_cache
    assert cls_a.instances_unloaded == 1


def test_incompatibility_is_reciprocal_via_cached_descriptor(tmp_workspace: Workspace) -> None:
    """Cached adapter Y declared `incompatible_with: [X]` is evicted when X loads.

    Even if the incoming adapter X does NOT itself declare incompatibility with Y,
    Y's own declaration is enough to trigger reciprocal eviction.
    """
    worker = _make_worker(tmp_workspace)
    # Y declares the incompatibility; X does not.
    d_y = _FakeDescriptor(name="yankee", version="1", pin_warm=True,
                          incompatible_with=["xray"])
    d_x = _FakeDescriptor(name="xray", version="1", incompatible_with=[])
    cls_y = _make_fake_class("YankeeAdapter", d_y)
    cls_x = _make_fake_class("XrayAdapter", d_x)

    _load(worker, d_y, cls_y)
    assert ("yankee", "1") in worker._adapter_cache

    _load(worker, d_x, cls_x)
    # Yankee evicted by reciprocal rule: its descriptor lists xray.
    assert ("yankee", "1") not in worker._adapter_cache
    assert ("xray", "1") in worker._adapter_cache
    assert cls_y.instances_unloaded == 1


def test_reload_evicts_other_direction(tmp_workspace: Workspace) -> None:
    """After A->B eviction, loading A again evicts B (symmetry)."""
    worker = _make_worker(tmp_workspace)
    d_a = _FakeDescriptor(name="alpha", version="1", pin_warm=True,
                          incompatible_with=["bravo"])
    d_b = _FakeDescriptor(name="bravo", version="1", pin_warm=False,
                          incompatible_with=["alpha"])
    cls_a = _make_fake_class("AlphaAdapter2", d_a)
    cls_b = _make_fake_class("BravoAdapter2", d_b)

    _load(worker, d_a, cls_a)
    _load(worker, d_b, cls_b)
    assert ("alpha", "1") not in worker._adapter_cache
    assert ("bravo", "1") in worker._adapter_cache

    # Reload alpha — bravo must now be evicted.
    _load(worker, d_a, cls_a)
    assert ("alpha", "1") in worker._adapter_cache
    assert ("bravo", "1") not in worker._adapter_cache
    assert cls_b.instances_unloaded == 1


def test_compatible_adapters_coexist(tmp_workspace: Workspace) -> None:
    """No incompatibility declared → both adapters remain cached."""
    worker = _make_worker(tmp_workspace)
    d_p = _FakeDescriptor(name="papa", version="1")
    d_q = _FakeDescriptor(name="quebec", version="1")
    cls_p = _make_fake_class("PapaAdapter", d_p)
    cls_q = _make_fake_class("QuebecAdapter", d_q)

    _load(worker, d_p, cls_p)
    _load(worker, d_q, cls_q)
    assert ("papa", "1") in worker._adapter_cache
    assert ("quebec", "1") in worker._adapter_cache


def test_descriptor_schema_accepts_incompatible_with(tmp_path: Path) -> None:
    """The JSON Schema + pydantic round-trip accepts ``incompatible_with``."""
    from vibeproteinagent.adapters import load_descriptor

    body = (
        "name: alpha_tool\n"
        "version: \"1.0.0\"\n"
        "category: protein_design\n"
        "description: \"test tool\"\n"
        "runtime: python\n"
        "backend: pytorch\n"
        "gpu_required: true\n"
        "gpus_required: 1\n"
        "startup_cost: low\n"
        "estimated_runtime_s: 30\n"
        "warm_load_capable: true\n"
        "supported_tasks: [test]\n"
        "input_schema: \"x.input.json\"\n"
        "output_schema: \"x.output.json\"\n"
        "supported_precisions: [bf16]\n"
        "incompatible_with: [bravo_tool, charlie_tool]\n"
    )
    p = tmp_path / "descriptor.yaml"
    p.write_text(body)
    d = load_descriptor(p)
    assert d.incompatible_with == ["bravo_tool", "charlie_tool"]


def test_descriptor_schema_rejects_invalid_incompatible_name(tmp_path: Path) -> None:
    """An invalid tool name in incompatible_with is rejected by the schema."""
    import pytest

    from vibeproteinagent.adapters import load_descriptor
    from vibeproteinagent.core.errors import InvalidParamsError

    body = (
        "name: alpha_tool\n"
        "version: \"1.0.0\"\n"
        "category: protein_design\n"
        "description: \"test tool\"\n"
        "runtime: python\n"
        "backend: pytorch\n"
        "gpu_required: true\n"
        "gpus_required: 1\n"
        "startup_cost: low\n"
        "estimated_runtime_s: 30\n"
        "warm_load_capable: true\n"
        "supported_tasks: [test]\n"
        "input_schema: \"x.input.json\"\n"
        "output_schema: \"x.output.json\"\n"
        "supported_precisions: [bf16]\n"
        "incompatible_with: [\"Bad-Name\"]\n"
    )
    p = tmp_path / "descriptor.yaml"
    p.write_text(body)
    with pytest.raises(InvalidParamsError):
        load_descriptor(p)
