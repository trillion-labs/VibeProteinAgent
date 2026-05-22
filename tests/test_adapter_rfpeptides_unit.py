"""Fast unit tests for the RFpeptides adapter (no GPU, no rfdiffusion import)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.rfpeptides import adapter as adapter_mod
from vibeproteinagent.adapters.rfpeptides.adapter import (
    RFPEPTIDES_TREE,
    RFPEPTIDES_WEIGHTS,
    RFPeptidesAdapter,
    RFPeptidesDesign,
    RFPeptidesInput,
    RFPeptidesOutput,
    _build_cmd,
    _build_contigs,
    _build_env,
    _collect_designs,
    _required_weight,
)


def test_descriptor_fields():
    d = RFPeptidesAdapter.descriptor
    assert d.name == "rfpeptides"
    assert d.version == "1.1.0"
    assert d.category.value == "protein_design"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.startup_cost.value == "high"
    assert d.warm_load_capable is False
    assert d.execution_isolation.value == "subprocess"
    # We do not pin RFpeptides warm — every call is a fresh subprocess.
    assert d.pin_warm is False
    assert d.support_status.value == "verified"
    assert d.availability_probe == (
        "vibeproteinagent.adapters.rfpeptides:availability_skip_reason"
    )


def test_default_paths_point_into_repo():
    assert RFPEPTIDES_TREE.name == "rfpeptides"
    assert RFPEPTIDES_WEIGHTS.parent == RFPEPTIDES_TREE


def test_input_defaults():
    p = RFPeptidesInput()
    assert p.target_pdb is None
    assert p.target_chain == "A"
    assert p.peptide_length == 10
    assert p.num_designs == 10
    assert p.num_diffusion_steps == 50
    assert p.noise_scale == 1.0
    assert p.target_hotspots == []
    assert p.seed is None


def test_input_validation_ranges():
    with pytest.raises(ValidationError):
        RFPeptidesInput(peptide_length=4)  # < 5
    with pytest.raises(ValidationError):
        RFPeptidesInput(peptide_length=200)  # > 50
    with pytest.raises(ValidationError):
        RFPeptidesInput(num_designs=0)
    with pytest.raises(ValidationError):
        RFPeptidesInput(num_diffusion_steps=5)  # < 10
    with pytest.raises(ValidationError):
        RFPeptidesInput(noise_scale=2.0)


def test_input_target_residues_pattern():
    ok = RFPeptidesInput(target_residues="3-117")
    assert ok.target_residues == "3-117"
    with pytest.raises(ValidationError):
        RFPeptidesInput(target_residues="3..117")
    with pytest.raises(ValidationError):
        RFPeptidesInput(target_residues="not a range")


def test_output_round_trip():
    out = RFPeptidesOutput(
        designs=[
            RFPeptidesDesign(pdb_path="/tmp/design_0.pdb", length=10),
            RFPeptidesDesign(
                pdb_path="/tmp/design_1.pdb",
                trb_path="/tmp/design_1.trb",
                length=10,
            ),
        ],
        runtime_ms=1000,
        num_designs_returned=2,
    )
    blob = out.model_dump()
    rebuilt = RFPeptidesOutput.model_validate(blob)
    assert rebuilt == out


def test_required_weight_routes_by_target_mode():
    monomer = _required_weight(target_pdb=None)
    binder = _required_weight(target_pdb="/foo.pdb")
    assert monomer.name == "Base_ckpt.pt"
    assert binder.name == "Complex_base_ckpt.pt"
    assert monomer.parent == RFPEPTIDES_WEIGHTS
    assert binder.parent == RFPEPTIDES_WEIGHTS


def test_build_contigs_monomer():
    p = RFPeptidesInput(peptide_length=12)
    contigs = _build_contigs(p, staged_target=None)
    assert contigs == "[12-12]"


def test_build_contigs_binder_full_chain(tmp_path):
    target = tmp_path / "t.pdb"
    target.write_text("END\n")
    p = RFPeptidesInput(peptide_length=15, target_chain="A")
    contigs = _build_contigs(p, staged_target=target)
    assert contigs == "[15-15 A1-9999/0]"


def test_build_contigs_binder_explicit_range(tmp_path):
    target = tmp_path / "t.pdb"
    target.write_text("END\n")
    p = RFPeptidesInput(peptide_length=12, target_chain="B", target_residues="3-117")
    contigs = _build_contigs(p, staged_target=target)
    assert contigs == "[12-12 B3-117/0]"


def test_build_cmd_monomer_omits_input_pdb(tmp_path):
    p = RFPeptidesInput(peptide_length=10, num_designs=1, num_diffusion_steps=20)
    cmd = _build_cmd(
        params=p,
        output_prefix=tmp_path / "out" / "design",
        target_pdb=None,
        contigs="[10-10]",
        weights_dir=tmp_path / "weights",
    )
    assert cmd[0].endswith("python") or cmd[0].endswith("python3") or "python" in cmd[0]
    assert cmd[1] == "-c"
    payload = cmd[2]
    assert "inference.cyclic=True" in payload
    assert "inference.cyc_chains=a" in payload
    assert "diffuser.T=20" in payload
    assert "contigmap.contigs=[10-10]" in payload
    assert "inference.num_designs=1" in payload
    assert "inference.input_pdb=" not in payload


def test_build_cmd_binder_includes_input_pdb_and_hotspots(tmp_path):
    target = tmp_path / "t.pdb"
    target.write_text("END\n")
    p = RFPeptidesInput(
        target_pdb=str(target),
        peptide_length=12,
        num_designs=2,
        target_chain="A",
        target_hotspots=[50, 52],
    )
    cmd = _build_cmd(
        params=p,
        output_prefix=tmp_path / "out" / "design",
        target_pdb=target,
        contigs="[12-12 A1-9999/0]",
        weights_dir=tmp_path / "weights",
    )
    payload = cmd[2]
    assert f"inference.input_pdb={target}" in payload
    assert "ppi.hotspot_res=" in payload
    # Hotspot identifiers should mention both residues.
    assert "A50" in payload and "A52" in payload


def test_build_env_hygiene(tmp_path):
    env = _build_env(seed=None)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["DGLBACKEND"] == "pytorch"
    assert "PYTHONPATH" in env
    # Both rfdiffusion root and SE3Transformer subdir on the python path.
    assert "SE3Transformer" in env["PYTHONPATH"]
    seeded = _build_env(seed=11)
    assert seeded["PYTHONHASHSEED"] == "11"


def test_collect_designs_reads_indexed_pdbs(tmp_path):
    parent = tmp_path / "outputs"
    parent.mkdir()
    (parent / "design_0.pdb").write_text("REMARK design 0\nEND\n")
    (parent / "design_1.pdb").write_text("REMARK design 1\nEND\n")
    (parent / "design_0.trb").write_bytes(b"pickle")
    designs = _collect_designs(prefix=parent / "design", peptide_length=12)
    assert len(designs) == 2
    assert designs[0].pdb_path.endswith("design_0.pdb")
    assert designs[0].trb_path is not None
    assert designs[0].trb_path.endswith("design_0.trb")
    assert designs[1].trb_path is None
    assert all(d.length == 12 for d in designs)
    assert all(d.sequence is None for d in designs)
    assert all(d.score is None for d in designs)


def test_collect_designs_raises_when_empty(tmp_path):
    parent = tmp_path / "outputs"
    parent.mkdir()
    with pytest.raises(RuntimeError, match="no PDB outputs"):
        _collect_designs(prefix=parent / "design", peptide_length=10)


def test_healthcheck_includes_availability_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = RFPeptidesAdapter.__new__(RFPeptidesAdapter)

    monkeypatch.setattr(adapter_mod, "availability_skip_reason", lambda: "dgl missing")
    assert adapter.healthcheck() is False

    monkeypatch.setattr(adapter_mod, "availability_skip_reason", lambda: None)
    assert adapter.healthcheck() is True
