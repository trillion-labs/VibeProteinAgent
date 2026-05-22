"""Fast unit tests for the RosettaCommons RFdiffusion adapter."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.availability import descriptor_availability
from vibeproteinagent.adapters.rfdiffusion import adapter as adapter_mod
from vibeproteinagent.adapters.rfdiffusion.adapter import (
    RFDIFFUSION_TREE,
    RFDIFFUSION_WEIGHTS,
    RFdiffusionAdapter,
    RFdiffusionDesign,
    RFdiffusionInput,
    RFdiffusionOutput,
    _build_cmd,
    _build_contigs,
    _build_env,
    _collect_designs,
    _mode_checkpoint_checks,
    _required_weight,
    availability_report,
)


def _target_pdb(tmp_path):
    target = tmp_path / "target.pdb"
    target.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A  10      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  ALA A  11      1.000   0.000   0.000  1.00 10.00           C",
                "ATOM      3  CA  ALA A  12      2.000   0.000   0.000  1.00 10.00           C",
                "ATOM      4  CA  ALA B   1      3.000   0.000   0.000  1.00 10.00           C",
            ]
        )
        + "\n"
    )
    return target


def test_descriptor_fields():
    d = RFdiffusionAdapter.descriptor
    assert d.name == "rfdiffusion"
    assert d.category.value == "protein_design"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.vram_min_gib == 24
    assert d.execution_isolation.value == "subprocess"
    assert d.support_status.value == "verified"
    assert "backbone_generation" in d.supported_tasks
    assert "symmetric_motif_scaffolding" in d.supported_tasks
    assert "binder_backbone_design" in d.supported_tasks
    assert d.availability_probe == (
        "vibeproteinagent.adapters.rfdiffusion:availability_report"
    )


def test_default_paths_point_to_rfpeptides_checkout():
    assert RFDIFFUSION_TREE.name == "rfpeptides"
    assert RFDIFFUSION_WEIGHTS.parent == RFDIFFUSION_TREE


def test_input_defaults():
    p = RFdiffusionInput()
    assert p.mode == "unconditional"
    assert p.length_min == 100
    assert p.num_designs == 10
    assert p.num_diffusion_steps == 50
    assert p.noise_scale_ca == 1.0
    assert p.hydra_overrides == []


def test_mode_aliases_are_canonicalized(tmp_path):
    target = _target_pdb(tmp_path)
    assert RFdiffusionInput(mode="backbone_generation").mode == "unconditional"
    assert (
        RFdiffusionInput(mode="unconditional_generation").mode
        == "unconditional"
    )
    assert (
        RFdiffusionInput(mode="binder_backbone_design", target_pdb=str(target)).mode
        == "binder_design"
    )
    assert (
        RFdiffusionInput(mode="symmetric_oligomer_design", symmetry="C3").mode
        == "symmetric_oligomer"
    )


def test_validation_requires_mode_inputs(tmp_path):
    target = _target_pdb(tmp_path)
    with pytest.raises(ValidationError):
        RFdiffusionInput(mode="motif_scaffolding", target_pdb=str(target))
    with pytest.raises(ValidationError):
        RFdiffusionInput(mode="binder_design")
    with pytest.raises(ValidationError):
        RFdiffusionInput(mode="symmetric_oligomer")
    with pytest.raises(ValidationError):
        RFdiffusionInput(mode="partial_diffusion", target_pdb=str(target))
    with pytest.raises(ValidationError):
        RFdiffusionInput(mode="partial_diffusion", target_pdb=str(target), partial_T=80)
    with pytest.raises(ValidationError):
        RFdiffusionInput(num_diffusion_steps=14)


def test_output_round_trip():
    out = RFdiffusionOutput(
        mode="unconditional",
        contigs=["100-100"],
        designs=[
            RFdiffusionDesign(pdb_path="/tmp/design_0.pdb", length=100),
            RFdiffusionDesign(
                pdb_path="/tmp/design_1.pdb",
                trb_path="/tmp/design_1.trb",
                length=100,
            ),
        ],
        runtime_ms=50,
        num_designs_returned=2,
    )
    assert RFdiffusionOutput.model_validate(out.model_dump()) == out


def test_required_weight_routes_by_mode(tmp_path):
    weights = tmp_path / "weights"
    p_base = RFdiffusionInput(model_directory_path=str(weights))
    p_binder = RFdiffusionInput(
        mode="binder_design",
        target_pdb=str(tmp_path / "t.pdb"),
        model_directory_path=str(weights),
    )
    p_fold = RFdiffusionInput(
        mode="fold_conditioning",
        target_pdb=str(tmp_path / "t.pdb"),
        scaffold_dir=str(tmp_path),
        target_ss=str(tmp_path / "ss.pt"),
        target_adj=str(tmp_path / "adj.pt"),
        model_directory_path=str(weights),
    )
    assert _required_weight(p_base).name == "Base_ckpt.pt"
    assert _required_weight(p_binder).name == "Complex_base_ckpt.pt"
    assert _required_weight(p_fold).name == "Complex_Fold_base_ckpt.pt"

    override = tmp_path / "custom.pt"
    assert _required_weight(RFdiffusionInput(checkpoint_path=str(override))) == override


def test_mode_checkpoint_checks_are_mode_specific(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "Base_ckpt.pt").write_bytes(b"base")
    checks = _mode_checkpoint_checks(weights)

    assert checks["unconditional"]["ok"] is True
    assert checks["motif_scaffolding"]["ok"] is True
    assert checks["partial_diffusion"]["ok"] is True
    assert checks["symmetric_oligomer"]["ok"] is True
    assert checks["binder_design"]["ok"] is False
    assert checks["fold_conditioning"]["ok"] is False
    assert checks["binder_design"]["checkpoint"] == "Complex_base_ckpt.pt"
    assert checks["fold_conditioning"]["checkpoint"] == "Complex_Fold_base_ckpt.pt"


def test_availability_report_exposes_partial_checkpoint_readiness(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "Base_ckpt.pt").write_bytes(b"base")
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / "scripts" / "run_inference.py").write_text("print('ok')\n")

    class FakeGraph:
        def to(self, device):
            assert device == "cuda"
            return self

        def all_edges(self):
            return []

    class FakeDGL:
        @staticmethod
        def graph(edges):
            assert edges == ([0, 1], [1, 0])
            return FakeGraph()

    monkeypatch.setattr(adapter_mod, "RFDIFFUSION_TREE", tree)
    monkeypatch.setattr(adapter_mod, "RFDIFFUSION_WEIGHTS", weights)
    monkeypatch.setattr(adapter_mod, "import_dgl_with_graphbolt_guard", lambda: FakeDGL)
    monkeypatch.setattr(adapter_mod.torch.cuda, "is_available", lambda: True)

    report = availability_report()
    checks = report["checks"]
    assert report["available"] is True
    assert report["reason"] is None
    assert checks["mode_checkpoints"]["unconditional"]["ok"] is True
    assert checks["mode_checkpoints"]["binder_design"]["ok"] is False
    assert checks["mode_checkpoints"]["fold_conditioning"]["ok"] is False
    assert checks["missing_mode_checkpoints"] == [
        "binder_design",
        "fold_conditioning",
    ]
    assert checks["full_mode_set_available"] is False

    (weights / "Complex_base_ckpt.pt").write_bytes(b"complex")
    (weights / "Complex_Fold_base_ckpt.pt").write_bytes(b"fold")
    report = availability_report()
    checks = report["checks"]
    assert report["available"] is True
    assert checks["missing_mode_checkpoints"] == []
    assert checks["full_mode_set_available"] is True


def test_descriptor_availability_preserves_mode_checkpoint_details(
    monkeypatch, tmp_path
):
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "Base_ckpt.pt").write_bytes(b"base")
    tree = tmp_path / "tree"
    (tree / "scripts").mkdir(parents=True)
    (tree / "scripts" / "run_inference.py").write_text("print('ok')\n")

    class FakeGraph:
        def to(self, device):
            return self

        def all_edges(self):
            return []

    class FakeDGL:
        @staticmethod
        def graph(edges):
            return FakeGraph()

    monkeypatch.setattr(adapter_mod, "RFDIFFUSION_TREE", tree)
    monkeypatch.setattr(adapter_mod, "RFDIFFUSION_WEIGHTS", weights)
    monkeypatch.setattr(adapter_mod, "import_dgl_with_graphbolt_guard", lambda: FakeDGL)
    monkeypatch.setattr(adapter_mod.torch.cuda, "is_available", lambda: True)

    payload = descriptor_availability(RFdiffusionAdapter.descriptor)
    details = payload["availability_details"]
    assert payload["available"] is True
    assert details["mode_checkpoints"]["binder_design"]["ok"] is False
    assert details["missing_mode_checkpoints"] == [
        "binder_design",
        "fold_conditioning",
    ]


def test_build_contigs_unconditional():
    p = RFdiffusionInput(length_min=80, length_max=120)
    assert _build_contigs(p, target_pdb=None) == ["80-120"]


def test_build_contigs_binder_uses_target_span(tmp_path):
    target = _target_pdb(tmp_path)
    p = RFdiffusionInput(
        mode="binder_design",
        target_pdb=str(target),
        target_chain="A",
        binder_length_min=55,
        binder_length_max=65,
    )
    assert _build_contigs(p, target) == ["A10-12/0 55-65"]


def test_build_contigs_partial_uses_target_length(tmp_path):
    target = _target_pdb(tmp_path)
    p = RFdiffusionInput(mode="partial_diffusion", target_pdb=str(target), partial_T=10)
    assert _build_contigs(p, target) == ["3-3"]


def test_build_contigs_respects_explicit_motif_contigs(tmp_path):
    target = _target_pdb(tmp_path)
    p = RFdiffusionInput(
        mode="motif_scaffolding",
        target_pdb=str(target),
        contigs=["10-20/A10-12/10-20"],
    )
    assert _build_contigs(p, target) == ["10-20/A10-12/10-20"]


def test_build_cmd_unconditional_exposes_core_overrides(tmp_path):
    p = RFdiffusionInput(
        length_min=100,
        length_max=100,
        num_designs=2,
        num_diffusion_steps=30,
        hydra_overrides=["diffuser.crd_scale=0.5"],
    )
    cmd = _build_cmd(
        params=p,
        output_prefix=tmp_path / "design",
        contigs=["100-100"],
        target_pdb=None,
        target_ss=None,
        target_adj=None,
        checkpoint=tmp_path / "weights" / "Base_ckpt.pt",
    )
    payload = cmd[2]
    assert "--config-name', 'base" in payload
    assert "contigmap.contigs=[100-100]" in payload
    assert "inference.num_designs=2" in payload
    assert "diffuser.T=30" in payload
    assert "diffuser.crd_scale=0.5" in payload
    assert "allow_full_torch_load" in payload
    assert "inference.schedule_directory_path=" in payload


def test_build_cmd_binder_includes_target_hotspots_and_complex_ckpt(tmp_path):
    target = _target_pdb(tmp_path)
    p = RFdiffusionInput(
        mode="binder_design",
        target_pdb=str(target),
        target_hotspots=[10, "A12"],
        noise_scale_ca=0,
        noise_scale_frame=0,
    )
    cmd = _build_cmd(
        params=p,
        output_prefix=tmp_path / "design",
        contigs=["A10-12/0 70-70"],
        target_pdb=target,
        target_ss=None,
        target_adj=None,
        checkpoint=tmp_path / "weights" / "Complex_base_ckpt.pt",
    )
    payload = cmd[2]
    assert f"inference.input_pdb={target}" in payload
    assert "inference.ckpt_override_path=" in payload
    assert "ppi.hotspot_res=[A10,A12]" in payload
    assert "denoiser.noise_scale_ca=0.0" in payload


def test_build_cmd_symmetric_mode_uses_symmetry_config(tmp_path):
    target = _target_pdb(tmp_path)
    p = RFdiffusionInput(
        mode="symmetric_motif_scaffolding",
        target_pdb=str(target),
        symmetry="C6",
        contigs=["10-20/A10-12/10-20"],
    )
    cmd = _build_cmd(
        params=p,
        output_prefix=tmp_path / "design",
        contigs=["10-20/A10-12/10-20"],
        target_pdb=target,
        target_ss=None,
        target_adj=None,
        checkpoint=tmp_path / "weights" / "Base_ckpt.pt",
    )
    payload = cmd[2]
    assert "--config-name', 'symmetry" in payload
    assert "inference.symmetry=C6" in payload


def test_build_cmd_fold_conditioning_uses_scaffoldguided_fields(tmp_path):
    target = _target_pdb(tmp_path)
    ss = tmp_path / "target_ss.pt"
    adj = tmp_path / "target_adj.pt"
    ss.write_text("ss")
    adj.write_text("adj")
    p = RFdiffusionInput(
        mode="fold_conditioning",
        target_pdb=str(target),
        scaffold_dir=str(tmp_path / "scaffolds"),
        target_ss=str(ss),
        target_adj=str(adj),
    )
    cmd = _build_cmd(
        params=p,
        output_prefix=tmp_path / "design",
        contigs=[],
        target_pdb=target,
        target_ss=ss,
        target_adj=adj,
        checkpoint=tmp_path / "weights" / "Complex_Fold_base_ckpt.pt",
    )
    payload = cmd[2]
    assert "scaffoldguided.scaffoldguided=True" in payload
    assert f"scaffoldguided.target_path={target}" in payload
    assert f"scaffoldguided.target_ss={ss}" in payload
    assert f"scaffoldguided.target_adj={adj}" in payload
    assert "inference.input_pdb=" not in payload


def test_build_env_hygiene():
    env = _build_env(seed=17)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["DGLBACKEND"] == "pytorch"
    assert "SE3Transformer" in env["PYTHONPATH"]
    assert env["PYTHONHASHSEED"] == "17"


def test_collect_designs_reads_pdb_lengths(tmp_path):
    (tmp_path / "design_0.pdb").write_text(
        "\n".join(
            [
                "ATOM      1  CA  GLY A   1      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY A   2      1.000   0.000   0.000  1.00 10.00           C",
            ]
        )
        + "\n"
    )
    (tmp_path / "design_0.trb").write_bytes(b"trb")
    designs = _collect_designs(tmp_path / "design", ["100-100"])
    assert len(designs) == 1
    assert designs[0].length == 2
    assert designs[0].trb_path is not None
    assert designs[0].sequence is None


def test_collect_designs_raises_when_empty(tmp_path):
    with pytest.raises(RuntimeError, match="no PDB outputs"):
        _collect_designs(tmp_path / "design", ["100-100"])


def test_healthcheck_uses_availability_probe(monkeypatch):
    adapter = RFdiffusionAdapter.__new__(RFdiffusionAdapter)
    monkeypatch.setattr(
        adapter_mod,
        "availability_report",
        lambda: {"available": False, "reason": "missing", "checks": {}},
    )
    assert adapter.healthcheck() is False
    monkeypatch.setattr(
        adapter_mod,
        "availability_report",
        lambda: {"available": True, "reason": None, "checks": {}},
    )
    assert adapter.healthcheck() is True
