from __future__ import annotations

import torch

import vibeproteinagent.adapters.colabdesign_utils.adapter as colab_mod
from vibeproteinagent.adapters import ToolRegistry, null_context
from vibeproteinagent.adapters.colabdesign_utils.adapter import (
    ColabDesignUtilsAdapter,
    ColabDesignUtilsInput,
    ColabDesignUtilsOutput,
    colabdesign_runtime_skip_reason,
)


def test_descriptor_fields():
    d = ColabDesignUtilsAdapter.descriptor
    assert d.name == "colabdesign_utils"
    assert d.category.value == "protein_design"
    assert d.gpu_required is False
    assert "colabdesign_utilities" in d.supported_tasks
    assert "dependency_probe" in d.supported_tasks


def test_registered_in_builtin_registry():
    descriptor, cls = ToolRegistry().get("colabdesign_utils")
    assert descriptor.name == "colabdesign_utils"
    assert cls is ColabDesignUtilsAdapter


def test_run_capability_report_from_sidecar_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(
        colab_mod,
        "_sidecar_runtime_reasons",
        lambda: {"pxdesign_pipeline": None, "bindcraft": "not configured"},
    )
    adapter = ColabDesignUtilsAdapter(
        config={},
        weights_dir=tmp_path,
        device=torch.device("cpu"),
    )
    out = adapter.run(ColabDesignUtilsInput(), null_context(tmp_path))
    assert isinstance(out, ColabDesignUtilsOutput)
    assert out.action == "capabilities"
    assert out.package_available is None
    assert out.deferred is False
    assert {u.name for u in out.utilities} >= {
        "binder_design",
        "partial_sequence_design",
        "hallucination",
        "af2_validation",
    }
    assert {u.status for u in out.utilities} == {"sidecar_available"}


def test_runtime_skip_reason_accepts_any_ready_sidecar(monkeypatch):
    monkeypatch.setattr(colab_mod.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        colab_mod,
        "_sidecar_runtime_reasons",
        lambda: {"pxdesign_pipeline": "missing", "bindcraft": None},
    )
    assert colabdesign_runtime_skip_reason() is None


def test_runtime_skip_reason_reports_all_missing_sidecars(monkeypatch):
    monkeypatch.setattr(colab_mod.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(
        colab_mod,
        "_sidecar_runtime_reasons",
        lambda: {"pxdesign_pipeline": "missing", "bindcraft": "missing"},
    )
    reason = colabdesign_runtime_skip_reason()
    assert reason is not None
    assert "pxdesign_pipeline: missing" in reason
    assert "bindcraft: missing" in reason


def test_estimate_vram_zero(tmp_path):
    adapter = ColabDesignUtilsAdapter(
        config={},
        weights_dir=tmp_path,
        device=torch.device("cpu"),
    )
    assert adapter.estimate_vram(ColabDesignUtilsInput()) == 0
