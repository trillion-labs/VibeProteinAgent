"""Fast unit tests for the BoltzGen adapter (no GPU, no boltzgen import)."""
from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from vibeproteinagent.adapters.boltzgen.adapter import (
    BOLTZGEN_CACHE,
    BOLTZGEN_TREE,
    BoltzGenAdapter,
    BoltzGenDesign,
    BoltzGenInput,
    BoltzGenOutput,
    _build_cmd,
    _build_env,
    _collect_designs,
    _safe_float,
    _safe_int,
    _stage_design_spec,
)


def test_descriptor_fields():
    d = BoltzGenAdapter.descriptor
    assert d.name == "boltzgen"
    assert d.version == "0.3.2"
    assert d.category.value == "protein_design"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.startup_cost.value == "high"
    assert d.warm_load_capable is False
    assert d.execution_isolation.value == "subprocess"
    # We do NOT pin BoltzGen warm — every call is a fresh subprocess.
    assert d.pin_warm is False


def test_pointer_paths_default_to_known_location():
    # The defaults match what the Phase 16a hand-off documents — verifying the
    # constants haven't drifted.
    assert str(BOLTZGEN_TREE).startswith("/")
    assert str(BOLTZGEN_CACHE).startswith("/")


def test_input_requires_exactly_one_target():
    with pytest.raises(ValidationError):
        BoltzGenInput()
    with pytest.raises(ValidationError):
        BoltzGenInput(target_pdb="/tmp/a.pdb", target_yaml="/tmp/b.yaml")
    ok = BoltzGenInput(target_pdb="/tmp/a.pdb")
    assert ok.target_pdb == "/tmp/a.pdb"
    assert ok.target_yaml is None
    ok2 = BoltzGenInput(target_yaml="/tmp/b.yaml")
    assert ok2.target_yaml == "/tmp/b.yaml"


def test_input_defaults():
    p = BoltzGenInput(target_pdb="/tmp/a.pdb")
    assert p.target_chain == "A"
    assert p.binder_length == 60
    assert p.num_designs == 10
    assert p.num_diffusion_steps == 500
    assert p.protocol == "protein-anything"
    assert p.design_only is True
    assert p.budget == 10
    assert p.seed is None


def test_input_rejects_out_of_range():
    with pytest.raises(ValidationError):
        BoltzGenInput(target_pdb="/tmp/a.pdb", binder_length=10)  # < 20
    with pytest.raises(ValidationError):
        BoltzGenInput(target_pdb="/tmp/a.pdb", binder_length=999)  # > 200
    with pytest.raises(ValidationError):
        BoltzGenInput(target_pdb="/tmp/a.pdb", num_designs=0)
    with pytest.raises(ValidationError):
        BoltzGenInput(target_pdb="/tmp/a.pdb", num_diffusion_steps=2)  # < 5


def test_output_round_trip():
    out = BoltzGenOutput(
        designs=[
            BoltzGenDesign(cif_path="/tmp/d_0.cif", length=60),
            BoltzGenDesign(
                cif_path="/tmp/d_1.cif",
                length=60,
                sequence="MKTAY",
                score=0.83,
                affinity_pred_value=2.1,
                affinity_probability_binary=0.42,
                confidence=0.71,
            ),
        ],
        runtime_ms=12_345,
        num_designs_returned=2,
        seeds_used=[42],
    )
    blob = out.model_dump()
    assert blob["num_designs_returned"] == 2
    assert blob["designs"][1]["sequence"] == "MKTAY"
    rebuilt = BoltzGenOutput.model_validate(blob)
    assert rebuilt == out


def test_stage_design_spec_from_pdb(tmp_path):
    pdb = tmp_path / "1g13.pdb"
    pdb.write_text("REMARK fake PDB for the unit test\nEND\n")
    params = BoltzGenInput(target_pdb=str(pdb), target_chain="A", binder_length=80)
    dest = tmp_path / "staged"
    dest.mkdir()
    written = _stage_design_spec(params, dest)
    assert written == dest / "design.yaml"
    assert (dest / "1g13.pdb").is_file()
    doc = yaml.safe_load(written.read_text())
    assert "entities" in doc
    # First entity is the binder protein, second the file-referenced target.
    binder = doc["entities"][0]["protein"]
    target = doc["entities"][1]["file"]
    assert binder["sequence"] == 80
    assert binder["id"] == "B"
    assert target["path"] == "1g13.pdb"
    assert target["include"] == [{"chain": {"id": "A"}}]


def test_stage_design_spec_uses_letter_b_when_target_is_a(tmp_path):
    pdb = tmp_path / "t.pdb"
    pdb.write_text("END\n")
    dest = tmp_path / "staged"
    dest.mkdir()
    p = BoltzGenInput(target_pdb=str(pdb), target_chain="A")
    written = _stage_design_spec(p, dest)
    binder_id = yaml.safe_load(written.read_text())["entities"][0]["protein"]["id"]
    assert binder_id == "B"


def test_stage_design_spec_uses_letter_a_when_target_is_not_a(tmp_path):
    pdb = tmp_path / "t.pdb"
    pdb.write_text("END\n")
    dest = tmp_path / "staged"
    dest.mkdir()
    p = BoltzGenInput(target_pdb=str(pdb), target_chain="B")
    written = _stage_design_spec(p, dest)
    binder_id = yaml.safe_load(written.read_text())["entities"][0]["protein"]["id"]
    assert binder_id == "A"


def test_stage_design_spec_from_yaml_copies_referenced_files(tmp_path):
    target_cif = tmp_path / "1g13.cif"
    target_cif.write_text("# fake cif\n")
    src_yaml = tmp_path / "design.yaml"
    src_yaml.write_text(
        yaml.safe_dump(
            {
                "entities": [
                    {"protein": {"id": "B", "sequence": "20..30"}},
                    {
                        "file": {
                            "path": "1g13.cif",
                            "include": [{"chain": {"id": "A"}}],
                        }
                    },
                ]
            },
            sort_keys=False,
        )
    )
    p = BoltzGenInput(target_yaml=str(src_yaml))
    dest = tmp_path / "staged"
    dest.mkdir()
    written = _stage_design_spec(p, dest)
    assert (dest / "1g13.cif").is_file()
    assert written.is_file()


def test_build_cmd_design_only_includes_steps_filter(tmp_path):
    p = BoltzGenInput(target_pdb="/tmp/a.pdb", num_designs=1, num_diffusion_steps=10)
    cmd = _build_cmd(
        design_spec=tmp_path / "design.yaml",
        output_dir=tmp_path / "out",
        params=p,
        steps=["design"],
    )
    assert "--no_subprocess" in cmd
    assert "--use_kernels" in cmd and "false" in cmd
    assert "--steps" in cmd
    assert "--protocol" in cmd
    # The diffusion sampling step count is forwarded via ``--config design sampling_steps=<n>``.
    joined = " ".join(cmd)
    assert "sampling_steps=10" in joined


def test_build_cmd_full_pipeline_omits_steps_filter(tmp_path):
    p = BoltzGenInput(target_pdb="/tmp/a.pdb", design_only=False)
    cmd = _build_cmd(
        design_spec=tmp_path / "design.yaml",
        output_dir=tmp_path / "out",
        params=p,
        steps=None,
    )
    assert "--steps" not in cmd


def test_build_env_hygiene():
    env = _build_env(seed=None)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env or env["PYTHONPATH"].endswith("/src")
    assert "HF_HUB_CACHE" in env
    env_seed = _build_env(seed=7)
    assert env_seed["PYTHONHASHSEED"] == "7"


def test_collect_designs_design_only_reads_intermediate(tmp_path):
    out = tmp_path / "out"
    inter = out / "intermediate_designs"
    inter.mkdir(parents=True)
    (inter / "1g13_0.cif").write_text("# fake")
    (inter / "1g13_1.cif").write_text("# fake")
    (inter / "1g13_native.cif").write_text("# native — should be skipped")
    params = BoltzGenInput(target_pdb="/tmp/a.pdb", binder_length=42)
    designs = _collect_designs(output_dir=out, params=params)
    assert len(designs) == 2
    assert all(d.length == 42 for d in designs)
    assert all(d.sequence is None for d in designs)
    assert all(not d.cif_path.endswith("_native.cif") for d in designs)


def test_collect_designs_raises_when_no_designs(tmp_path):
    out = tmp_path / "out"
    (out / "intermediate_designs").mkdir(parents=True)
    p = BoltzGenInput(target_pdb="/tmp/a.pdb")
    with pytest.raises(RuntimeError, match="no design CIFs"):
        _collect_designs(output_dir=out, params=p)


def test_safe_int_and_float():
    assert _safe_int(None) is None
    assert _safe_int("") is None
    assert _safe_int("42") == 42
    assert _safe_int("3.7") == 3
    assert _safe_int("not a num") is None
    assert _safe_float(None) is None
    assert _safe_float("") is None
    assert _safe_float("1.5") == 1.5
    assert _safe_float("nan") != _safe_float("nan")  # NaN is not equal to itself
    assert _safe_float("abc") is None
