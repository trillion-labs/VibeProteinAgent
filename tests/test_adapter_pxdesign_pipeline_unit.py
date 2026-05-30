"""Fast unit tests for the PXDesign adapter's generation pipeline dispatch.

These tests do NOT touch the real pipeline runtime — they validate:

* the ``top_k_designs`` / ``mpnn_temperature`` fields parse + reject bad input,
* ``availability_skip_reason()`` produces an actionable string when the
  ``pxdesign-pipeline-vpa`` env is absent,
* ``run()`` raises :class:`DependencyError` (``E_DEPENDENCY``)
  with the bootstrap pointer when the env is absent,
* ``run()`` uses the fixed generation pipeline path,
* vendor preview filtering/ranking flags are not part of the invocation,
* the synthetic pipeline output layout round-trips through
  ``_collect_pipeline_outputs``.

The slow real-run regression test lives in
``test_adapter_pxdesign_pipeline_real.py``.
"""
from __future__ import annotations

import csv
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.pxdesign import (
    PROTENIX_PXD_VENDOR,
    PXDBENCH_VENDOR,
    PXDESIGN_PIPELINE_ENV_DEFAULT,
    PXDESIGN_PIPELINE_ENV_NAME,
    PXDESIGN_VENDOR,
    PXDesignAdapter,
    PXDesignInput,
    _preview_fallback,
    availability_skip_reason,
    pipeline_env,
    pipeline_python,
)
from vibeproteinagent.adapters.pxdesign import adapter as adapter_mod
from vibeproteinagent.core.errors import DependencyError, ErrorCode

_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vibeproteinagent"
    / "adapters"
    / "pxdesign"
)
_INPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "pxdesign.input.json"
_DESCRIPTOR = _ADAPTER_DIR / "descriptor.yaml"


# --------------------------------------------------------------------------- schema


def test_input_schema_declares_pipeline_knobs() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    props = raw["properties"]
    assert "mode" not in props
    assert props["top_k_designs"]["default"] == 4
    assert props["mpnn_temperature"]["default"] == pytest.approx(0.1)
    assert "af2_init_guess" not in props
    assert "preset" not in props
    assert "use_msa" not in props
    assert "msa_backend" not in props


def test_input_schema_still_rejects_unknown_keys() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    assert raw["additionalProperties"] is False


def test_descriptor_declares_pipeline_subprocess_isolation() -> None:
    text = _DESCRIPTOR.read_text()
    assert "execution_isolation: subprocess" in text
    assert "environment_name: pxdesign-pipeline-vpa" in text
    assert "environment_reason:" in text


# --------------------------------------------------------------------------- pydantic


def test_pydantic_defaults_fixed_pipeline_fields() -> None:
    p = PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50)
    assert p.top_k_designs == 4
    assert p.mpnn_temperature == pytest.approx(0.1)
    assert not hasattr(p, "mode")
    assert not hasattr(p, "af2_init_guess")
    assert not hasattr(p, "use_msa")


def test_pydantic_accepts_pipeline_knobs() -> None:
    p = PXDesignInput(
        target_pdb="/tmp/x.pdb",
        target_chain="A",
        binder_length=50,
        top_k_designs=8,
        mpnn_temperature=0.2,
    )
    assert p.top_k_designs == 8
    assert p.mpnn_temperature == pytest.approx(0.2)


def test_pydantic_rejects_negative_mpnn_temperature() -> None:
    with pytest.raises(ValidationError):
        PXDesignInput(
            target_pdb="/tmp/x.pdb",
            target_chain="A",
            binder_length=50,
            mpnn_temperature=-0.1,
        )


def test_pydantic_rejects_top_k_zero() -> None:
    with pytest.raises(ValidationError):
        PXDesignInput(
            target_pdb="/tmp/x.pdb",
            target_chain="A",
            binder_length=50,
            top_k_designs=0,
        )


# --------------------------------------------------------------------------- env / availability


def test_pipeline_env_default_matches_constant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPA_PXDESIGN_PIPELINE_ENV", raising=False)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    monkeypatch.setattr(
        adapter_mod, "_discover_named_conda_env", lambda _name: None
    )
    assert pipeline_env() == PXDESIGN_PIPELINE_ENV_DEFAULT


def test_pipeline_env_ignores_default_dir_without_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default = tmp_path / "opt" / "pxdesign-pipeline-vpa"
    discovered = tmp_path / "home" / "pxdesign-pipeline-vpa"
    (default / "bin").mkdir(parents=True)
    (discovered / "bin").mkdir(parents=True)
    discovered_py = discovered / "bin" / "python"
    discovered_py.write_text("#!/bin/sh\n")

    monkeypatch.delenv("VPA_PXDESIGN_PIPELINE_ENV", raising=False)
    monkeypatch.setattr(adapter_mod, "PXDESIGN_PIPELINE_ENV_DEFAULT", default)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home-root")
    monkeypatch.setattr(Path, "is_file", lambda self: self == discovered_py)
    monkeypatch.setattr(
        adapter_mod, "_discover_named_conda_env", lambda _name: discovered
    )

    assert pipeline_env() == discovered


def test_pipeline_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(tmp_path / "fakeenv"))
    assert pipeline_env() == tmp_path / "fakeenv"


def test_pipeline_python_under_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(tmp_path / "fakeenv"))
    assert pipeline_python() == tmp_path / "fakeenv" / "bin" / "python"


def test_pipeline_env_name_constant() -> None:
    assert PXDESIGN_PIPELINE_ENV_NAME == "pxdesign-pipeline-vpa"


def test_availability_skip_reason_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(tmp_path / "no_such_env"))
    reason = availability_skip_reason()
    assert reason is not None
    assert "pxdesign-pipeline" in reason.lower() or "pipeline" in reason.lower()
    assert "VPA_PXDESIGN_PIPELINE_ENV" in reason
    assert "vendor/pxdesign/UPSTREAM.md" in reason


def test_availability_returns_none_when_python_and_smoke_import_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_env = tmp_path / "fakeenv"
    (fake_env / "bin").mkdir(parents=True)
    fake_py = fake_env / "bin" / "python"
    fake_py.write_text("#!/bin/sh\necho ok\n")
    fake_py.chmod(0o755)
    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(fake_env))
    monkeypatch.setattr(adapter_mod, "_pipeline_import_skip_reason", lambda _py: None)
    assert availability_skip_reason() is None


# --------------------------------------------------------------------------- dispatch


def _make_adapter_without_warm() -> PXDesignAdapter:
    """Construct a bare PXDesignAdapter for pipeline dispatch tests."""
    obj = PXDesignAdapter.__new__(PXDesignAdapter)
    return obj


def test_run_raises_dependency_error_when_env_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeproteinagent.adapters import null_context

    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(tmp_path / "no_such_env"))
    adapter = _make_adapter_without_warm()
    target_pdb = tmp_path / "target.pdb"
    target_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   9.291  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      12.560  13.207   9.291  1.00 20.00           C",
                "ATOM      3  C   ALA A   1      13.000  14.600   9.700  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    target = tmp_path / "target.cif"
    adapter_mod._convert_pdb_to_mmcif(target_pdb, target, target_chain="A")
    params = PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=50,
        num_samples=1,
        num_steps=5,
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    ctx = null_context(job_dir)
    with pytest.raises(DependencyError) as excinfo:
        adapter.run(params, ctx)
    err = excinfo.value
    assert err.code == ErrorCode.DEPENDENCY
    assert "pipeline" in err.message.lower()
    assert err.details.get("bootstrap") == "vendor/pxdesign/UPSTREAM.md#pipeline-mode"
    assert "env" in err.details

def test_run_dispatches_to_pipeline_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PXDesign has one runtime path: the fixed generation pipeline."""
    called = {"pipeline": 0}

    def _stub_pipeline(self: PXDesignAdapter, p: PXDesignInput, ctx: Any) -> Any:
        called["pipeline"] += 1
        return "PIPELINE_RESULT"

    monkeypatch.setattr(PXDesignAdapter, "_run_pipeline", _stub_pipeline)
    adapter = _make_adapter_without_warm()

    target = tmp_path / "target.cif"
    target.write_text("data_x\n#\n")
    params = PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=50,
    )
    out = adapter.run(params, ctx=None)
    assert out == "PIPELINE_RESULT"
    assert called == {"pipeline": 1}


def test_pipeline_subprocess_env_strips_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/leak/path")
    env = adapter_mod._pipeline_subprocess_env(seed=42)
    assert "/leak/path" not in env.get("PYTHONPATH", "")
    assert "src" in env.get("PYTHONPATH", "")
    for vendor_root in (PXDESIGN_VENDOR, PXDBENCH_VENDOR, PROTENIX_PXD_VENDOR):
        assert str(vendor_root) in env.get("PYTHONPATH", "")
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONHASHSEED"] == "42"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_pipeline_subprocess_env_seed_none_omits_hashseed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    env = adapter_mod._pipeline_subprocess_env(seed=None)
    assert env.get("PYTHONHASHSEED") is None


def test_run_pipeline_materializes_generated_sequence_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeproteinagent.adapters import null_context

    fake_env = tmp_path / "fakeenv"
    (fake_env / "bin").mkdir(parents=True)
    fake_py = fake_env / "bin" / "python"
    fake_py.write_text("#!/bin/sh\necho ok\n")
    fake_py.chmod(0o755)
    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(fake_env))
    monkeypatch.setattr(adapter_mod, "_pipeline_import_skip_reason", lambda _py: None)

    calls: list[list[str]] = []

    def _fake_preview_run(
        *,
        argv: list[str],
        cwd: Path,
        job_outputs: Path,
        task_name: str,
        seed: int | None,
        mpnn_temperature: float,
        timeout_s: int,
    ) -> None:
        del cwd, seed, mpnn_temperature, timeout_s
        calls.append(argv)
        out_dir = job_outputs / "design_outputs" / task_name / "orig"
        out_dir.mkdir(parents=True)
        (out_dir / "rank_0.cif").write_text("data_x\n#\n")
        with (out_dir.parent / "summary.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "rank", "task_name", "sequence", "af2_iptm",
                "chosen_struct_path",
            ])
            writer.writerow([
                "0", task_name, "GAVL", "0.77", "orig/rank_0.cif",
            ])

    monkeypatch.setattr(
        adapter_mod, "_run_preview_pipeline_until_artifacts", _fake_preview_run
    )

    adapter = _make_adapter_without_warm()
    target_pdb = tmp_path / "target.pdb"
    target_pdb.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   9.291  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      12.560  13.207   9.291  1.00 20.00           C",
                "ATOM      3  C   ALA A   1      13.000  14.600   9.700  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    target = tmp_path / "target.cif"
    adapter_mod._convert_pdb_to_mmcif(target_pdb, target, target_chain="A")
    params = PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=30,
        num_samples=1,
        num_steps=5,
        seed=42,
    )

    out1 = adapter.run(params, null_context(tmp_path / "job1"))
    out2 = adapter.run(params, null_context(tmp_path / "job2"))

    assert len(calls) == 2
    assert calls[0][calls[0].index("--preset") + 1] == "preview"
    assert [out1.designs[0].sequence, out2.designs[0].sequence] == ["GAVL", "GAVL"]
    assert out1.designs[0].score == pytest.approx(0.77)


def test_prepare_pipeline_invocation_normalizes_pdb_target_to_cif(tmp_path: Path) -> None:
    target = tmp_path / "target.pdb"
    target.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   9.291  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      12.560  13.207   9.291  1.00 20.00           C",
                "ATOM      3  C   ALA A   1      13.000  14.600   9.700  1.00 20.00           C",
                "ATOM      4  O   ALA A   1      12.300  15.500   9.600  1.00 20.00           O",
                "ATOM      5  N   GLY B   1      16.104  13.207   9.291  1.00 20.00           N",
                "ATOM      6  CA  GLY B   1      17.560  13.207   9.291  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    params = PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=30,
        num_samples=1,
        num_steps=5,
    )

    invocation = adapter_mod._prepare_pipeline_invocation(params, tmp_path / "job")
    input_json = next((tmp_path / "job" / "inputs").glob("*.json"))
    task = json.loads(input_json.read_text())[0]
    structure_file = Path(task["condition"]["structure_file"])

    assert structure_file.suffix == ".cif"
    assert structure_file.is_file()
    assert invocation.input_structure_state["converted_to_cif"] is True
    assert invocation.input_structure_state["pipeline_structure_file"] == str(structure_file)


def test_prepare_pipeline_invocation_records_defaulted_inputs(tmp_path: Path) -> None:
    target = tmp_path / "target.pdb"
    target.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   9.291  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      12.560  13.207   9.291  1.00 20.00           C",
                "ATOM      3  C   ALA A   1      13.000  14.600   9.700  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    params = PXDesignInput(target_pdb=str(target), num_samples=1, num_steps=5)

    invocation = adapter_mod._prepare_pipeline_invocation(params, tmp_path / "job")

    assert invocation.input_structure_state["defaulted_inputs"]["target_chain"]["value"] == "A"
    assert invocation.input_structure_state["defaulted_inputs"]["binder_length"]["value"] == 60


def test_run_pipeline_preflight_failure_writes_debug_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeproteinagent.adapters import null_context

    fake_env = tmp_path / "fakeenv"
    (fake_env / "bin").mkdir(parents=True)
    fake_py = fake_env / "bin" / "python"
    fake_py.write_text("#!/bin/sh\necho ok\n")
    fake_py.chmod(0o755)
    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(fake_env))
    monkeypatch.setattr(adapter_mod, "_pipeline_import_skip_reason", lambda _py: None)

    target = tmp_path / "target.pdb"
    target.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   9.291  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      12.560  13.207   9.291  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    adapter = _make_adapter_without_warm()
    job_dir = tmp_path / "job"

    with pytest.raises(Exception, match="preflight failed"):
        adapter.run(
            PXDesignInput(target_pdb=str(target), target_chain="B", binder_length=30),
            null_context(job_dir),
        )

    debug = json.loads((job_dir / "outputs" / "pxdesign_failure_debug.json").read_text())
    assert debug["command"]
    assert debug["cwd"] == str(job_dir)
    assert debug["stdout_path"]
    assert debug["stderr_path"]
    assert debug["checkpoint_state"]
    assert debug["input_structure_state"]["target_chain"] == "B"
    assert "raw_artifact_state" in debug


# --------------------------------------------------------------------------- output parsing


def test_collect_pipeline_outputs_parses_summary_csv(tmp_path: Path) -> None:
    """Synthetic pipeline output layout → _collect_pipeline_outputs returns rows."""
    task = "design_5o45_A_50aa"
    task_dir = tmp_path / "design_outputs" / task
    (task_dir / "orig").mkdir(parents=True)
    (task_dir / "af2").mkdir(parents=True)
    # CIF stubs (real-AA sequences are inside the upstream-injected file —
    # for unit purposes we only check `cif_path` resolves and `sequence` is
    # carried over from the CSV).
    for rank in (0, 1):
        (task_dir / "orig" / f"rank_{rank}.cif").write_text("data_x\n#\n")
        (task_dir / "af2" / f"rank_{rank}.cif").write_text("data_x\n#\n")

    summary_path = task_dir / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "task_name", "sequence", "af2_plddt", "af2_iptm",
            "chosen_struct_type", "chosen_struct_path",
        ])
        writer.writerow([
            "0", task, "GAVLIRTKEDQNS", "82.4", "0.71",
            "af2", "af2/rank_0.cif",
        ])
        writer.writerow([
            "1", task, "MDPSTYVKAGCHF", "79.1", "0.65",
            "orig", "orig/rank_1.cif",
        ])

    designs = PXDesignAdapter._collect_pipeline_outputs(tmp_path, task, 13, top_k=4)
    assert len(designs) == 2
    assert designs[0].sequence == "GAVLIRTKEDQNS"
    assert "X" not in designs[0].sequence
    assert designs[0].cif_path.endswith("af2/rank_0.cif")
    assert designs[0].score == pytest.approx(0.71)
    assert designs[1].sequence == "MDPSTYVKAGCHF"
    assert designs[1].score == pytest.approx(0.65)


def test_collect_pipeline_outputs_truncates_to_top_k(tmp_path: Path) -> None:
    task = "design_only_one"
    task_dir = tmp_path / "design_outputs" / task
    (task_dir / "orig").mkdir(parents=True)
    (task_dir / "orig" / "rank_0.cif").write_text("data_x\n#\n")
    (task_dir / "orig" / "rank_1.cif").write_text("data_x\n#\n")
    summary_path = task_dir / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "task_name", "sequence", "chosen_struct_path"])
        for r in range(3):
            writer.writerow([str(r), task, "GAGA", f"orig/rank_{r}.cif"])
    # Only rank_0 and rank_1 CIFs exist; rank_2 row is silently dropped.
    designs = PXDesignAdapter._collect_pipeline_outputs(tmp_path, task, 4, top_k=1)
    assert len(designs) == 1


def test_collect_pipeline_outputs_returns_empty_when_summary_missing(
    tmp_path: Path,
) -> None:
    designs = PXDesignAdapter._collect_pipeline_outputs(
        tmp_path, "absent_task", 50, top_k=4,
    )
    assert designs == []


def test_summary_score_from_row_prefers_iptm() -> None:
    row = {"af2_plddt": "80.0", "af2_iptm": "0.84", "af2_ptm": "0.66"}
    assert adapter_mod._summary_score_from_row(row) == pytest.approx(0.84)


def test_summary_score_from_row_falls_back_to_plddt() -> None:
    row = {"af2_plddt": "78.2"}
    assert adapter_mod._summary_score_from_row(row) == pytest.approx(78.2)


def test_summary_score_from_row_returns_none_when_unparseable() -> None:
    assert adapter_mod._summary_score_from_row({"af2_iptm": ""}) is None
    assert adapter_mod._summary_score_from_row({"af2_iptm": "nan-not-float"}) is None


# --------------------------------------------------------------------------- exports


def test_pipeline_helpers_exported_from_package() -> None:
    import vibeproteinagent.adapters.pxdesign as pkg

    for name in (
        "PXDESIGN_PIPELINE_ENV_DEFAULT",
        "PXDESIGN_PIPELINE_ENV_NAME",
        "availability_skip_reason",
        "pipeline_env",
        "pipeline_python",
    ):
        assert hasattr(pkg, name), f"missing export: {name}"


# --------------------------------------------------------------------------- phase 20


def test_build_pipeline_cmd_threads_mpnn_temperature(tmp_path: Path) -> None:
    """Pi2: upstream argparse MPNN temperature flag is included."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
        mpnn_temperature=0.2,
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=17,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    flag_idx = cmd.index("--eval.binder.tools.mpnn.temperature")
    assert cmd[flag_idx + 1] == "0.2"


def test_build_pipeline_cmd_default_mpnn_temperature(tmp_path: Path) -> None:
    """Pi2: the default 0.1 still lands as an explicit flag."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=17,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    flag_idx = cmd.index("--eval.binder.tools.mpnn.temperature")
    assert cmd[flag_idx + 1] == "0.1"


def test_build_pipeline_cmd_always_uses_preview_preset(tmp_path: Path) -> None:
    """The adapter fixes PXDesign to the raw-diffusion preview path."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=17,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    idx = cmd.index("--preset")
    assert cmd[idx + 1] == "preview"


def test_build_pipeline_cmd_omits_vendor_filter_selection_flags(tmp_path: Path) -> None:
    """top_k_designs is a VPA collection limit, not a vendor filter/rank knob."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
        top_k_designs=12,
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=42,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    assert "--min_total_return" not in cmd
    assert "--max_success_return" not in cmd
    i_seed = cmd.index("--seeds")
    assert cmd[i_seed + 1] == "42"
    i_msa = cmd.index("--use_msa")
    assert cmd[i_msa + 1] == "false"


def test_preview_pipeline_terminates_at_raw_cif_before_vendor_filtering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = "design_target_A_30aa"
    raw_dir = tmp_path / "out" / "global_run_0" / task / "seed_42" / "predictions"
    raw_dir.mkdir(parents=True)
    (raw_dir.parent / "SUCCESS_FILE").write_text("ok\n")
    raw_cif = raw_dir / f"{task}_sample_0.cif"
    raw_cif.write_text("data_x\n#\n")

    events: list[str] = []

    class FakeProc:
        pid = 12345

        def poll(self) -> int | None:
            events.append("poll")
            return None

    def fake_popen(*_args: Any, **_kwargs: Any) -> FakeProc:
        events.append("popen")
        return FakeProc()

    def fake_terminate(proc: FakeProc, *, kill: bool = False) -> None:
        del proc, kill
        events.append("terminate")

    def fake_materialize(**kwargs: Any) -> None:
        assert kwargs["raw_cif"] == raw_cif
        events.append("materialize")

    monkeypatch.setattr(adapter_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(adapter_mod, "_terminate_process_group", fake_terminate)
    monkeypatch.setattr(adapter_mod, "_materialize_preview_sequence_outputs", fake_materialize)
    monkeypatch.setattr(adapter_mod, "pipeline_python", lambda: Path("/bin/python"))
    monkeypatch.setattr(adapter_mod, "_pipeline_subprocess_env", lambda seed: {})

    adapter_mod._run_preview_pipeline_until_artifacts(
        argv=["--preset", "preview"],
        cwd=tmp_path,
        job_outputs=tmp_path / "out",
        task_name=task,
        seed=42,
        mpnn_temperature=0.1,
        timeout_s=30,
    )

    assert events == ["popen", "terminate", "materialize"]


def test_preview_fallback_mpnn_pdb_is_single_model(tmp_path: Path) -> None:
    raw_cif = tmp_path / "design_sample_0.cif"
    raw_cif.write_text(
        "\n".join(
            [
                "data_design",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "ATOM N N ALA A 1 1.0 2.0 3.0",
                "ATOM C CA ALA A 1 2.0 2.0 3.0",
                "ATOM C C ALA A 1 3.0 2.0 3.0",
                "ATOM N N XPB B 1 6.0 2.0 3.0",
                "ATOM C CA XPB B 1 7.0 2.0 3.0",
                "#",
            ]
        )
        + "\n"
    )
    pdb_path = tmp_path / "mpnn_input.pdb"

    metadata = _preview_fallback._write_mpnn_pdb(raw_cif, pdb_path)

    lines = pdb_path.read_text().splitlines()
    assert lines[0] == "MODEL        1"
    assert lines[-2] == "ENDMDL"
    assert lines[-1] == "END"
    assert sum(1 for line in lines if line.startswith("MODEL")) == 1
    assert sum(1 for line in lines if line.startswith("ENDMDL")) == 1
    assert any(" GLY B" in line for line in lines if line.startswith("ATOM"))
    assert metadata.raw_to_pdb_chain == {"A": "A", "B": "B"}
    assert metadata.binder_chains == ["B"]
    assert metadata.cond_chains == ["A"]


def test_preview_fallback_binder_only_b0_uses_existing_chain_only(tmp_path: Path) -> None:
    raw_cif = tmp_path / "design_sample_0.cif"
    raw_cif.write_text(
        "\n".join(
            [
                "data_design",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "ATOM N N xpb B0 1 1.0 2.0 3.0",
                "ATOM C CA xpb B0 1 2.0 2.0 3.0",
                "ATOM C C xpb B0 1 3.0 2.0 3.0",
                "#",
            ]
        )
        + "\n"
    )
    pdb_path = tmp_path / "mpnn_input.pdb"

    metadata = _preview_fallback._write_mpnn_pdb(raw_cif, pdb_path)

    assert metadata.raw_to_pdb_chain == {"B0": "B"}
    assert metadata.binder_chains == ["B"]
    assert metadata.cond_chains == []
    assert all(line[21] == "B" for line in pdb_path.read_text().splitlines()
               if line.startswith("ATOM"))


def test_preview_fallback_requires_xpb_binder_chain(tmp_path: Path) -> None:
    raw_cif = tmp_path / "design_sample_0.cif"
    raw_cif.write_text(
        "\n".join(
            [
                "data_design",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "ATOM N N ALA A 1 1.0 2.0 3.0",
                "ATOM C CA ALA A 1 2.0 2.0 3.0",
                "#",
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="XPB binder chain"):
        _preview_fallback._write_mpnn_pdb(raw_cif, tmp_path / "mpnn_input.pdb")


def test_preview_fallback_materialize_uses_parsed_chains(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_cif = tmp_path / "design_sample_0.cif"
    raw_cif.write_text(
        "\n".join(
            [
                "data_design",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "ATOM N N xpb B0 1 1.0 2.0 3.0",
                "ATOM C CA xpb B0 1 2.0 2.0 3.0",
                "ATOM C C xpb B0 1 3.0 2.0 3.0",
                "#",
            ]
        )
        + "\n"
    )
    calls: list[dict[str, Any]] = []

    def fake_design_binder(*args: Any, **kwargs: Any) -> list[dict[str, str]]:
        calls.append({"args": args, "kwargs": kwargs})
        return [{"sequence": "GAG"}]

    def fake_replace(_raw_cif: str, final_cif: str, sequence: str) -> None:
        Path(final_cif).write_text(f"data_final\n# {sequence}\n")

    ml_collections = types.ModuleType("ml_collections")
    ml_collections.ConfigDict = dict
    main_mpnn = types.ModuleType("pxdbench.tools.protmpnn.main_mpnn")
    main_mpnn.design_binder = fake_design_binder
    pipeline = types.ModuleType("pxdesign.utils.pipeline")
    pipeline.replace_last_xpb_chain_sequence = fake_replace
    for name in [
        "pxdbench",
        "pxdbench.tools",
        "pxdbench.tools.protmpnn",
        "pxdesign",
        "pxdesign.utils",
    ]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "ml_collections", ml_collections)
    monkeypatch.setitem(sys.modules, "pxdbench.tools.protmpnn.main_mpnn", main_mpnn)
    monkeypatch.setitem(sys.modules, "pxdesign.utils.pipeline", pipeline)

    final_cif = _preview_fallback.materialize_preview_outputs(
        raw_cif=raw_cif,
        task_name="design_target_A_60aa",
        output_root=tmp_path,
        temperature=0.1,
        seed=101,
    )

    assert final_cif.is_file()
    assert calls[0]["kwargs"]["binder_chains"] == ["B"]
    assert calls[0]["kwargs"]["cond_chains"] == []
    task_dir = tmp_path / "design_outputs" / "design_target_A_60aa"
    metadata = json.loads((task_dir / "preview_fallback_chain_metadata.json").read_text())
    assert metadata == {
        "binder_chains": ["B"],
        "cond_chains": [],
        "raw_to_pdb_chain": {"B0": "B"},
    }
    rows = list(csv.DictReader((task_dir / "summary.csv").open()))
    assert rows[0]["sequence"] == "GAG"


def test_collect_pipeline_outputs_handles_renamed_columns(tmp_path: Path) -> None:
    """Pi4: parser is robust against upstream column renames — it returns []
    when no parseable row is found, not a crash."""
    task = "renamed_columns"
    task_dir = tmp_path / "design_outputs" / task
    task_dir.mkdir(parents=True)
    summary_path = task_dir / "summary.csv"
    # Upstream renamed `chosen_struct_path` → `selected_path`; parser must
    # silently drop the row (no chosen_struct_path) and return [].
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "task_name", "sequence", "selected_path"])
        writer.writerow(["0", task, "GAGA", "orig/rank_0.cif"])
    designs = PXDesignAdapter._collect_pipeline_outputs(tmp_path, task, 4, top_k=4)
    assert designs == []


def test_collect_pipeline_outputs_full_layout_with_score_fallbacks(tmp_path: Path) -> None:
    """Pi4: real-shaped summary with multiple metrics chooses iptm > plddt > ptm > Rg."""
    task = "fallback_layout"
    task_dir = tmp_path / "design_outputs" / task
    (task_dir / "orig").mkdir(parents=True)
    (task_dir / "orig" / "rank_0.cif").write_text("data_x\n#\n")
    (task_dir / "orig" / "rank_1.cif").write_text("data_x\n#\n")
    (task_dir / "orig" / "rank_2.cif").write_text("data_x\n#\n")
    summary_path = task_dir / "summary.csv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["rank", "sequence", "af2_iptm", "af2_plddt", "af2_ptm", "Rg",
             "chosen_struct_path"],
        )
        # row 0: has iptm → iptm wins
        writer.writerow(["0", "MMMM", "0.91", "70.0", "0.5", "12.0", "orig/rank_0.cif"])
        # row 1: iptm blank → plddt
        writer.writerow(["1", "AAAA", "", "85.5", "0.6", "10.5", "orig/rank_1.cif"])
        # row 2: only Rg
        writer.writerow(["2", "GGGG", "", "", "", "8.7", "orig/rank_2.cif"])
    designs = PXDesignAdapter._collect_pipeline_outputs(tmp_path, task, 4, top_k=4)
    assert len(designs) == 3
    assert designs[0].score == pytest.approx(0.91)
    assert designs[1].score == pytest.approx(85.5)
    assert designs[2].score == pytest.approx(8.7)
    for d in designs:
        assert "X" not in d.sequence


def test_registry_pxdesign_count_unchanged() -> None:
    """Phase 19 extends the existing adapter — total registry stays at 26."""
    from vibeproteinagent.adapters import ToolRegistry

    r = ToolRegistry()
    r.discover()
    names = [d.name for d in r.list()]
    assert names.count("pxdesign") == 1
