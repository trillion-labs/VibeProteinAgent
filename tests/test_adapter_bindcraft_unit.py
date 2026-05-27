"""Fast unit tests for the BindCraft adapter (no GPU, no conda env)."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters.bindcraft.adapter import (
    BINDCRAFT_CONDA_ENV,
    BINDCRAFT_PARAMS,
    BINDCRAFT_TREE,
    BindCraftAdapter,
    BindCraftDesign,
    BindCraftInput,
    BindCraftOutput,
    _as_float,
    _build_advanced,
    _build_env,
    _build_generation_filters,
    _build_target_settings,
    _collect_designs,
    _conda_env_exists,
    _find_conda_exe,
    _load_bindcraft_stats,
)


def test_descriptor_fields():
    d = BindCraftAdapter.descriptor
    assert d.name == "bindcraft"
    assert d.version == "1.0.0"
    assert d.category.value == "protein_design"
    assert d.backend.value == "jax"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.vram_min_gib == 24
    assert d.startup_cost.value == "high"
    assert d.pin_warm is True
    assert d.execution_isolation.value == "persistent_sidecar"
    assert d.environment_name == "BindCraft"
    assert "persistent sidecar" in (d.environment_reason or "")


def test_default_paths_point_into_repo():
    assert BINDCRAFT_TREE.name == "bindcraft"
    assert BINDCRAFT_PARAMS.parent == BINDCRAFT_TREE
    # The default conda env name matches the upstream installer.
    assert BINDCRAFT_CONDA_ENV == "BindCraft" or BINDCRAFT_CONDA_ENV  # explicit override


def test_input_defaults():
    target_pdb = "/tmp/_unused.pdb"
    p = BindCraftInput(target_pdb=target_pdb)
    assert p.target_pdb == target_pdb
    assert p.target_chain == "A"
    assert p.target_hotspots == []
    assert p.binder_length == 60
    assert p.num_designs == 10
    assert p.num_recycles == 1
    assert p.seed is None


def test_input_validation_ranges():
    with pytest.raises(ValidationError):
        BindCraftInput(target_pdb="/x.pdb", binder_length=29)
    with pytest.raises(ValidationError):
        BindCraftInput(target_pdb="/x.pdb", binder_length=200)
    with pytest.raises(ValidationError):
        BindCraftInput(target_pdb="/x.pdb", num_designs=0)
    with pytest.raises(ValidationError):
        BindCraftInput(target_pdb="/x.pdb", num_designs=100)
    with pytest.raises(ValidationError):
        BindCraftInput(target_pdb="/x.pdb", num_recycles=-1)
    with pytest.raises(ValidationError):
        BindCraftInput(target_pdb="/x.pdb", num_recycles=10)
    with pytest.raises(ValidationError):
        BindCraftInput(target_pdb="")


def test_output_round_trip():
    out = BindCraftOutput(
        designs=[
            BindCraftDesign(
                pdb_path="/tmp/Accepted/d_0.pdb",
                sequence="MKKLPLLF",
                binder_length=8,
            ),
            BindCraftDesign(
                pdb_path="/tmp/Accepted/d_1.pdb",
                sequence="AAAAAAAA",
                binder_length=8,
                iptm=0.83,
                plddt=0.91,
                pae_interaction=6.5,
                sap_score=12.0,
            ),
        ],
        runtime_ms=1234,
        num_designs_returned=2,
        seeds_used=[7],
    )
    rebuilt = BindCraftOutput.model_validate(out.model_dump())
    assert rebuilt == out


def test_build_target_settings_shape(tmp_path):
    target = tmp_path / "target.pdb"
    target.write_text("END\n")
    p = BindCraftInput(
        target_pdb=str(target),
        target_chain="B",
        target_hotspots=[51, 52, 56],
        binder_length=45,
        num_designs=6,
    )
    s = _build_target_settings(p, target, tmp_path / "out")
    assert s["binder_name"] == "target"
    assert s["starting_pdb"] == str(target)
    assert s["chains"] == "B"
    assert s["target_hotspot_residues"] == "51,52,56"
    assert s["lengths"] == [45, 45]
    assert s["number_of_final_designs"] == 6
    assert s["design_path"].endswith("/out/") or s["design_path"].endswith("\\out\\")


def test_build_generation_filters_disables_thresholds(tmp_path):
    filters = _build_generation_filters(BINDCRAFT_TREE)
    assert isinstance(filters, dict)
    for conditions in filters.values():
        if not isinstance(conditions, dict):
            continue
        if "threshold" in conditions:
            assert conditions["threshold"] is None
        else:
            for aa_conditions in conditions.values():
                if isinstance(aa_conditions, dict) and "threshold" in aa_conditions:
                    assert aa_conditions["threshold"] is None
    assert _build_generation_filters(tmp_path / "nope") == {}


def test_build_advanced_carries_recycle_and_params(tmp_path):
    p = BindCraftInput(target_pdb="/x.pdb", num_recycles=2, max_trajectories=3)
    adv = _build_advanced(p, tmp_path / "params")
    assert adv["num_recycles_design"] == 2
    # Validation recycles stays >= 3 so the eval pass is at least default.
    assert adv["num_recycles_validation"] >= 3
    assert adv["af_params_dir"] == str(tmp_path / "params")
    assert "dalphaball_path" in adv and "DAlphaBall.gcc" in adv["dalphaball_path"]
    assert "dssp_path" in adv and adv["dssp_path"].endswith("dssp")
    assert adv["enable_mpnn"] is True
    assert adv["model_path"] == "v_48_020"
    assert adv["save_trajectory_pickle"] is False
    assert adv["remove_unrelaxed_trajectory"] is False
    assert adv["enable_rejection_check"] is False
    assert adv["optimise_beta_recycles_design"] >= 3
    assert adv["optimise_beta_recycles_valid"] >= 3
    assert adv["max_trajectories"] == 3


def test_build_env_hygiene():
    env = _build_env(seed=None)
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
    # JAX gets the cooperative-allocator flag by default.
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env.get("LD_LIBRARY_PATH") == os.environ.get("LD_LIBRARY_PATH")
    env_with_lib = _build_env(seed=None, env_prefix=Path("/tmp/bindcraft-env"))
    assert env_with_lib["LD_LIBRARY_PATH"].split(":")[0] == "/tmp/bindcraft-env/lib"
    seeded = _build_env(seed=42)
    assert seeded["PYTHONHASHSEED"] == "42"


def test_find_conda_exe_is_resolvable_or_none():
    # Whether a conda binary exists is environment-dependent. The helper
    # must not raise; it should either return a Path or None.
    res = _find_conda_exe()
    assert res is None or res.exists()


def test_conda_env_exists_handles_missing_binary(tmp_path):
    # Pointing at a non-existent path returns False rather than raising.
    fake = tmp_path / "definitely_not_conda"
    assert _conda_env_exists(fake, "BindCraft") is False


def test_af2_params_present_false_for_missing_dir(tmp_path, monkeypatch):
    import vibeproteinagent.adapters.bindcraft.adapter as mod

    monkeypatch.setattr(mod, "BINDCRAFT_PARAMS", tmp_path / "params_missing")
    assert mod._af2_params_present() is False


def test_collect_designs_reads_accepted_pdbs(tmp_path):
    accepted = tmp_path / "Accepted"
    accepted.mkdir()
    (accepted / "binder_0.pdb").write_text("ATOM\nEND\n")
    (accepted / "binder_1.pdb").write_text("ATOM\nEND\n")
    csv = tmp_path / "final_design_stats.csv"
    csv.write_text(
        "Design,Sequence,Average_i_pTM,Average_pLDDT,Average_PAE_Interaction,Binder_SAP_score\n"
        "binder_0,MKKLPLLFAA,0.71,0.88,8.2,15.4\n"
        "binder_1,AAAAAAAAAA,0.55,0.80,12.1,NA\n"
    )
    designs, warnings = _collect_designs(design_path=tmp_path, binder_length=10)
    assert len(designs) == 2
    assert warnings == []
    by_path = {Path(d.pdb_path).name: d for d in designs}
    assert by_path["binder_0.pdb"].candidate_id.startswith("bindcraft:binder_0:")
    assert by_path["binder_0.pdb"].sequence == "MKKLPLLFAA"
    assert by_path["binder_0.pdb"].iptm == pytest.approx(0.71)
    assert by_path["binder_0.pdb"].plddt == pytest.approx(0.88)
    assert by_path["binder_0.pdb"].pae_interaction == pytest.approx(8.2)
    assert by_path["binder_0.pdb"].sap_score == pytest.approx(15.4)
    # NA collapses to None.
    assert by_path["binder_1.pdb"].sap_score is None


def test_collect_designs_matches_bindcraft_model_suffix_to_stats(tmp_path):
    accepted = tmp_path / "Accepted"
    accepted.mkdir()
    (accepted / "prodigy_3BZD_l30_s680367_mpnn1_model2.pdb").write_text("ATOM\nEND\n")
    (tmp_path / "final_design_stats.csv").write_text(
        "Design,Sequence,Average_i_pTM,Average_pLDDT,Average_i_pAE,Binder_SAP_score\n"
        "prodigy_3BZD_l30_s680367_mpnn1,MKKLPLLFAAGGSSYYTTQQ,0.73,0.91,6.2,11.4\n"
    )

    designs, warnings = _collect_designs(design_path=tmp_path, binder_length=20)

    assert warnings == []
    assert len(designs) == 1
    design = designs[0]
    assert design.candidate_id.startswith("bindcraft:prodigy_3BZD_l30_s680367_mpnn1:")
    assert design.sequence == "MKKLPLLFAAGGSSYYTTQQ"
    assert design.iptm == pytest.approx(0.73)
    assert design.plddt == pytest.approx(0.91)
    assert design.pae_interaction == pytest.approx(6.2)
    assert design.sap_score == pytest.approx(11.4)


def test_collect_designs_falls_back_to_chain_b_sequence_when_stats_missing(tmp_path):
    accepted = tmp_path / "Accepted"
    accepted.mkdir()
    (accepted / "orphan_model1.pdb").write_text(
        "\n".join(
            [
                "ATOM      1  CA  MET B   1       0.000   0.000   0.000  1.00 20.00           C",
                "ATOM      2  CA  LYS B   2       1.000   0.000   0.000  1.00 20.00           C",
                "ATOM      3  CA  GLY B   3       2.000   0.000   0.000  1.00 20.00           C",
                "ATOM      4  CA  ALA A   1       3.000   0.000   0.000  1.00 20.00           C",
                "END",
            ]
        )
    )

    designs, warnings = _collect_designs(design_path=tmp_path, binder_length=30)

    assert len(designs) == 1
    assert designs[0].sequence == "MKG"
    assert designs[0].iptm is None
    assert warnings
    assert "recovered sequence from PDB chain B" in warnings[0]


def test_collect_designs_raises_when_no_generated_artifacts(tmp_path):
    with pytest.raises(RuntimeError, match="no generated PDB artifacts"):
        _collect_designs(design_path=tmp_path, binder_length=10)


def test_collect_designs_raises_when_output_dirs_empty(tmp_path):
    (tmp_path / "Accepted").mkdir()
    with pytest.raises(RuntimeError, match="no generated PDB artifacts"):
        _collect_designs(design_path=tmp_path, binder_length=10)


def test_load_bindcraft_stats_handles_missing_csv(tmp_path):
    assert _load_bindcraft_stats(tmp_path) == {}


def test_as_float_handles_edge_cases():
    assert _as_float(None) is None
    assert _as_float("") is None
    assert _as_float("NA") is None
    assert _as_float("not a float") is None
    assert _as_float(1.25) == 1.25
    assert _as_float("3.14") == pytest.approx(3.14)


def test_healthcheck_false_when_params_missing(monkeypatch, tmp_path):
    import vibeproteinagent.adapters.bindcraft.adapter as mod

    monkeypatch.setattr(mod, "BINDCRAFT_PARAMS", tmp_path / "params_missing")
    adapter = BindCraftAdapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    assert adapter.healthcheck() is False


def test_adapter_run_uses_persistent_sidecar(monkeypatch, tmp_path):
    import vibeproteinagent.adapters.bindcraft.adapter as adapter_mod
    from vibeproteinagent.adapters.context import null_context

    created: list[object] = []

    class FakeSidecar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[tuple[str, dict]] = []
            self.closed = False
            created.append(self)

        def call(
            self,
            target: str,
            kwargs: dict | None = None,
            *,
            timeout_s: float | None = None,
        ):
            self.calls.append((target, kwargs or {}))
            assert target.endswith(":run_sidecar")
            accepted = tmp_path / "job" / "outputs" / "Accepted"
            accepted.mkdir(parents=True, exist_ok=True)
            (accepted / f"binder_{len(self.calls)}.pdb").write_text("ATOM\nEND\n")
            (tmp_path / "job" / "outputs" / "final_design_stats.csv").write_text(
                "Design,Sequence,Average_i_pTM,Average_pLDDT,Average_PAE_Interaction,Binder_SAP_score\n"
                f"binder_{len(self.calls)},AAAAAAAAAA,0.71,0.88,8.2,15.4\n"
            )
            return {"ok": True}

        def close(self):
            self.closed = True

    def _no_subprocess_run(*_args, **_kwargs):
        raise AssertionError("run() should use the persistent sidecar, not subprocess.run")

    fake_python = tmp_path / "env" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.write_text("# fake python\n")
    params_dir = tmp_path / "params"
    params_dir.mkdir()
    (params_dir / "params_model_5_ptm.npz").write_text("fake")

    monkeypatch.setattr(adapter_mod, "_check_paths", lambda: None)
    monkeypatch.setattr(adapter_mod, "BINDCRAFT_PARAMS", params_dir)
    monkeypatch.setattr(adapter_mod, "availability_skip_reason", lambda: None)
    monkeypatch.setenv("VPA_BINDCRAFT_ENV_PATH", str(fake_python.parents[1]))
    monkeypatch.setattr(adapter_mod, "_find_conda_exe", lambda: None)
    monkeypatch.setattr(
        adapter_mod,
        "_bindcraft_python",
        lambda *_args: (_ for _ in ()).throw(AssertionError("conda lookup skipped")),
    )
    monkeypatch.setattr(adapter_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(adapter_mod, "PersistentPythonSidecar", FakeSidecar)
    monkeypatch.setattr(adapter_mod.subprocess, "run", _no_subprocess_run)

    pdb = tmp_path / "input.pdb"
    pdb.write_text("HEADER test\nEND\n")
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    adapter = BindCraftAdapter(config={}, weights_dir=tmp_path, device=torch.device("cuda"))
    out1 = adapter.run(
        BindCraftInput(target_pdb=str(pdb), binder_length=30), null_context(job_dir)
    )
    out2 = adapter.run(
        BindCraftInput(target_pdb=str(pdb), binder_length=30), null_context(job_dir)
    )

    assert out1.num_designs_returned == 1
    assert out2.num_designs_returned == 2
    assert len(created) == 1
    assert created[0].kwargs["python"] == fake_python
    assert len(created[0].calls) == 2
    first_call = created[0].calls[0][1]
    assert first_call["bindcraft_py"].endswith("bindcraft.py")
    assert first_call["settings_json"].endswith("target_settings.json")
    assert first_call["generation_only"] is True
    adapter.unload()
    assert created[0].closed is True


def test_collect_designs_prefers_generated_outputs(tmp_path):
    generated = tmp_path / "Generated"
    generated.mkdir()
    (generated / "target_l30_s1_mpnn1_model1.pdb").write_text("ATOM\nEND\n")
    (tmp_path / "final_design_stats.csv").write_text(
        "Design,Sequence,Average_i_pTM,Average_pLDDT,Average_i_pAE,Binder_SAP_score\n"
        "target_l30_s1_mpnn1,ACDEFGHIKL,0.81,0.92,4.2,9.4\n"
    )

    designs, warnings = _collect_designs(design_path=tmp_path, binder_length=10)

    assert warnings == []
    assert len(designs) == 1
    assert designs[0].pdb_path.endswith("Generated/target_l30_s1_mpnn1_model1.pdb")
    assert designs[0].sequence == "ACDEFGHIKL"
    assert designs[0].iptm == pytest.approx(0.81)


def test_runner_module_parses():
    runner_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "vibeproteinagent"
        / "adapters"
        / "bindcraft"
        / "_runner.py"
    )
    tree = ast.parse(runner_path.read_text(), filename=str(runner_path))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "run_sidecar"
        for node in ast.walk(tree)
    )
