"""Fast unit tests for the RFdiffusion3 Foundry adapter."""
from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.rfdiffusion3.adapter import (
    _FOUNDRY_AVAILABLE_MSG,
    RFDIFFUSION3_TREE,
    RFDIFFUSION3_WEIGHTS,
    RFdiffusion3Adapter,
    RFdiffusion3Design,
    RFdiffusion3Input,
    RFdiffusion3Output,
    _build_input_spec,
)


def test_descriptor_fields():
    d = RFdiffusion3Adapter.descriptor
    assert d.name == "rfdiffusion3"
    assert d.category.value == "protein_design"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.vram_min_gib == 24
    assert d.startup_cost.value == "high"
    assert d.warm_load_capable is False
    assert d.pin_warm is False
    assert d.execution_isolation.value == "subprocess"
    assert "foundry" in d.version
    assert d.support_status.value == "verified"


def test_foundry_release_is_documented():
    assert "rc-foundry" in _FOUNDRY_AVAILABLE_MSG


def test_default_paths_point_into_repo():
    assert RFDIFFUSION3_TREE.name == "rfdiffusion3"
    assert RFDIFFUSION3_WEIGHTS.parent == RFDIFFUSION3_TREE


def test_input_defaults():
    p = RFdiffusion3Input(target_pdb="/tmp/_t.pdb")
    assert p.target_chain == "A"
    assert p.target_hotspots == []
    assert p.binder_length == 60
    assert p.num_designs == 8
    assert p.num_diffusion_steps == 200
    assert p.seed is None


def test_input_validation_ranges():
    with pytest.raises(ValidationError):
        RFdiffusion3Input(target_pdb="/x", binder_length=29)
    with pytest.raises(ValidationError):
        RFdiffusion3Input(target_pdb="/x", binder_length=300)
    with pytest.raises(ValidationError):
        RFdiffusion3Input(target_pdb="/x", num_designs=0)
    with pytest.raises(ValidationError):
        RFdiffusion3Input(target_pdb="/x", num_designs=100)
    with pytest.raises(ValidationError):
        RFdiffusion3Input(target_pdb="/x", num_diffusion_steps=5)
    with pytest.raises(ValidationError):
        RFdiffusion3Input(target_pdb="")


def test_output_round_trip():
    out = RFdiffusion3Output(
        designs=[
            RFdiffusion3Design(pdb_path="/tmp/d_0.pdb", length=60),
            RFdiffusion3Design(
                pdb_path="/tmp/d_1.pdb", length=60, sequence="A" * 60, score=0.5
            ),
        ],
        runtime_ms=1234,
        num_designs_returned=2,
    )
    rebuilt = RFdiffusion3Output.model_validate(out.model_dump())
    assert rebuilt == out


def test_init_does_not_raise_and_health_matches_availability(tmp_path):
    adapter = RFdiffusion3Adapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    from vibeproteinagent.adapters.rfdiffusion3.adapter import availability_skip_reason

    assert adapter.healthcheck() is (availability_skip_reason() is None)


def test_run_raises_runtime_error_with_clear_missing_ckpt_blocker(monkeypatch, tmp_path):
    target = tmp_path / "target.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 10.00           C\n"
    )
    monkeypatch.setenv("VPA_RFDIFFUSION3_CKPT", str(tmp_path / "missing.ckpt"))
    adapter = RFdiffusion3Adapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    params = RFdiffusion3Input(target_pdb=str(target))
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    ctx = AdapterRunContext(
        job_id="rfd3_test",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: None,
    )
    with pytest.raises(RuntimeError, match="checkpoint missing"):
        adapter.run(params, ctx)


def test_estimate_vram_returns_descriptor_default(tmp_path):
    adapter = RFdiffusion3Adapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    p = RFdiffusion3Input(target_pdb="/x.pdb")
    assert adapter.estimate_vram(p) == 24 * 1024


def test_build_input_spec_uses_target_chain_span_and_hotspots(tmp_path):
    target = tmp_path / "target.pdb"
    target.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A  10      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY A  11      1.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  GLY A  12      1.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  SER B   1      2.000   0.000   0.000  1.00 10.00           C",
            ]
        )
        + "\n"
    )
    params = RFdiffusion3Input(
        target_pdb=str(target), target_chain="A", target_hotspots=[12], binder_length=55
    )
    spec = _build_input_spec(params, target)
    assert spec["contig"] == "55-55,/0,A10-12"
    assert spec["infer_ori_strategy"] == "hotspots"
    assert spec["select_hotspots"] == {"A12": "CA"}


def test_build_input_spec_uses_longest_contiguous_target_span(tmp_path):
    target = tmp_path / "gapped.pdb"
    target.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  ALA A   2      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  ALA A  10      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  ALA A  11      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      5  CA  ALA A  12      0.000   0.000   0.000  1.00 10.00           C",
            ]
        )
        + "\n"
    )
    spec = _build_input_spec(RFdiffusion3Input(target_pdb=str(target)), target)
    assert spec["contig"] == "60-60,/0,A10-12"


def test_build_input_spec_ignores_chain_matched_waters(tmp_path):
    target = tmp_path / "waters.pdb"
    target.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  ALA A   2      0.000   0.000   0.000  1.00 10.00           C",
                "HETATM    3  O   HOH A 101      0.000   0.000   0.000  1.00 10.00           O",
                "HETATM    4  O   HOH A 102      0.000   0.000   0.000  1.00 10.00           O",
                "HETATM    5  O   HOH A 103      0.000   0.000   0.000  1.00 10.00           O",
            ]
        )
        + "\n"
    )
    spec = _build_input_spec(RFdiffusion3Input(target_pdb=str(target)), target)
    assert spec["contig"] == "60-60,/0,A1-2"
