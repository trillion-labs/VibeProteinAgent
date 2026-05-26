from __future__ import annotations

import pytest

from vibeproteinagent.adapters.rfdiffusion import adapter as rfdiffusion_mod
from vibeproteinagent.adapters.rfpeptides import adapter as rfpeptides_mod


class _NoAllocDGL:
    @staticmethod
    def graph(edges):  # pragma: no cover - failure path if default probe regresses
        del edges
        raise AssertionError("probe should not allocate")


class _OOMDGL:
    @staticmethod
    def graph(edges):
        del edges
        raise RuntimeError("CUDA OOM (simulated)")


def _prepare_rfdiffusion_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    dgl: object,
) -> None:
    tree = tmp_path / "rfdiffusion_tree"
    weights = tmp_path / "rfdiffusion_weights"
    (tree / "scripts").mkdir(parents=True)
    weights.mkdir()
    (tree / "scripts" / "run_inference.py").write_text("print('ok')\n")
    (weights / "Base_ckpt.pt").write_bytes(b"base")

    monkeypatch.setattr(rfdiffusion_mod, "RFDIFFUSION_TREE", tree)
    monkeypatch.setattr(rfdiffusion_mod, "RFDIFFUSION_WEIGHTS", weights)
    monkeypatch.setattr(rfdiffusion_mod, "import_dgl_with_graphbolt_guard", lambda: dgl)
    monkeypatch.setattr(rfdiffusion_mod.torch.cuda, "is_available", lambda: True)


def _prepare_rfpeptides_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    dgl: object,
) -> None:
    tree = tmp_path / "rfpeptides_tree"
    weights = tmp_path / "rfpeptides_weights"
    (tree / "scripts").mkdir(parents=True)
    weights.mkdir()
    (tree / "scripts" / "run_inference.py").write_text("print('ok')\n")
    (weights / "Base_ckpt.pt").write_bytes(b"base")

    monkeypatch.setattr(rfpeptides_mod, "RFPEPTIDES_TREE", tree)
    monkeypatch.setattr(rfpeptides_mod, "RFPEPTIDES_WEIGHTS", weights)
    monkeypatch.setattr(rfpeptides_mod, "_import_dgl_for_rfpeptides", lambda: dgl)
    monkeypatch.setattr(rfpeptides_mod.torch.cuda, "is_available", lambda: True)


def test_rfdiffusion_default_probe_does_not_allocate_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv("VPA_DGL_PROBE_CUDA_ALLOC", raising=False)
    _prepare_rfdiffusion_probe(monkeypatch, tmp_path, dgl=_NoAllocDGL)

    report = rfdiffusion_mod.availability_report()

    assert report["available"] is True
    assert report["checks"]["dgl_cuda"] == {"ok": "skipped_at_probe"}


def test_rfdiffusion_escape_hatch_restores_cuda_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_DGL_PROBE_CUDA_ALLOC", "1")
    _prepare_rfdiffusion_probe(monkeypatch, tmp_path, dgl=_OOMDGL)

    report = rfdiffusion_mod.availability_report()

    assert report["available"] is False
    assert "DGL CUDA graph ops unavailable" in str(report["reason"])


def test_rfpeptides_default_probe_does_not_allocate_cuda(monkeypatch, tmp_path):
    monkeypatch.delenv("VPA_DGL_PROBE_CUDA_ALLOC", raising=False)
    _prepare_rfpeptides_probe(monkeypatch, tmp_path, dgl=_NoAllocDGL)

    report = rfpeptides_mod.availability_report()

    assert report["available"] is True
    assert report["checks"]["dgl_cuda"] == {"ok": "skipped_at_probe"}


def test_rfpeptides_escape_hatch_restores_cuda_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_DGL_PROBE_CUDA_ALLOC", "1")
    _prepare_rfpeptides_probe(monkeypatch, tmp_path, dgl=_OOMDGL)

    report = rfpeptides_mod.availability_report()

    assert report["available"] is False
    assert "DGL CUDA graph ops unavailable" in str(report["reason"])
