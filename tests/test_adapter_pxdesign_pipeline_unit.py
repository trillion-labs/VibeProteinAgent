"""Fast unit tests for the PXDesign adapter's `pipeline` mode dispatch.

These tests do NOT touch the real pipeline runtime — they validate:

* the new ``mode`` / ``top_k_designs`` / ``mpnn_temperature`` / ``af2_init_guess``
  fields parse + reject bad input,
* ``availability_skip_reason()`` produces an actionable string when the
  ``pxdesign-pipeline-vpa`` env is absent,
* ``run(mode="pipeline")`` raises :class:`DependencyError` (``E_DEPENDENCY``)
  with the bootstrap pointer when the env is absent,
* ``run(mode="pipeline")`` uses and reuses a persistent sidecar instead of a
  cold subprocess per call,
* the synthetic pipeline output layout round-trips through
  ``_collect_pipeline_outputs``.

The slow real-run regression test lives in
``test_adapter_pxdesign_pipeline_real.py``.
"""
from __future__ import annotations

import csv
import json
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


def test_input_schema_declares_mode_enum() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    mode = raw["properties"]["mode"]
    assert mode["type"] == "string"
    assert sorted(mode["enum"]) == ["infer", "pipeline"]
    assert mode["default"] == "infer"


def test_input_schema_declares_pipeline_knobs() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    props = raw["properties"]
    assert props["top_k_designs"]["default"] == 4
    assert props["mpnn_temperature"]["default"] == pytest.approx(0.1)
    assert props["af2_init_guess"]["default"] is True
    assert "use_msa" not in props
    assert "msa_backend" not in props


def test_input_schema_still_rejects_unknown_keys() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    assert raw["additionalProperties"] is False


def test_descriptor_declares_pipeline_sidecar_isolation() -> None:
    text = _DESCRIPTOR.read_text()
    assert "execution_isolation: persistent_sidecar" in text
    assert "environment_name: pxdesign-pipeline-vpa" in text
    assert "environment_reason:" in text


# --------------------------------------------------------------------------- pydantic


def test_pydantic_defaults_default_to_infer_mode() -> None:
    p = PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50)
    assert p.mode == "infer"
    assert p.top_k_designs == 4
    assert p.mpnn_temperature == pytest.approx(0.1)
    assert p.af2_init_guess is True
    assert not hasattr(p, "use_msa")


def test_pydantic_accepts_pipeline_mode() -> None:
    p = PXDesignInput(
        target_pdb="/tmp/x.pdb",
        target_chain="A",
        binder_length=50,
        mode="pipeline",
        top_k_designs=8,
        mpnn_temperature=0.2,
        af2_init_guess=False,
    )
    assert p.mode == "pipeline"
    assert p.top_k_designs == 8
    assert p.mpnn_temperature == pytest.approx(0.2)
    assert p.af2_init_guess is False


def test_pydantic_rejects_unknown_mode() -> None:
    with pytest.raises(ValidationError):
        PXDesignInput(
            target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50, mode="magic"
        )


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
    """Construct a PXDesignAdapter skipping the heavy warm load.

    The pipeline path doesn't use ``self.runner`` / ``self.ckpt_path``; we
    build a bare instance that has just enough attributes to call
    ``_run_pipeline``.
    """
    obj = PXDesignAdapter.__new__(PXDesignAdapter)
    return obj


def test_run_pipeline_raises_dependency_error_when_env_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from vibeproteinagent.adapters import null_context

    monkeypatch.setenv("VPA_PXDESIGN_PIPELINE_ENV", str(tmp_path / "no_such_env"))
    adapter = _make_adapter_without_warm()
    target = tmp_path / "target.cif"
    target.write_text("data_x\n#\n")
    params = PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=50,
        num_samples=1,
        num_steps=5,
        mode="pipeline",
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


def test_run_infer_mode_does_not_consult_pipeline_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Default-mode call must not even probe the pipeline env."""
    # Force pipeline env probe to blow up if accidentally hit.
    def _fail(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("infer mode must not call availability_skip_reason()")

    monkeypatch.setattr(adapter_mod, "availability_skip_reason", _fail)
    adapter = _make_adapter_without_warm()

    # Provide a minimal _run_infer stub on the instance so we can verify
    # dispatch lands there (the real _run_infer needs a fully warm runner).
    called = {"infer": 0}

    def _stub_infer(self: PXDesignAdapter, p: PXDesignInput, ctx: Any) -> Any:
        called["infer"] += 1
        return "INFER_RESULT"

    monkeypatch.setattr(PXDesignAdapter, "_run_infer", _stub_infer)
    target = tmp_path / "target.cif"
    target.write_text("data_x\n#\n")
    params = PXDesignInput(
        target_pdb=str(target), target_chain="A", binder_length=50,
    )
    out = adapter.run(params, ctx=None)
    assert out == "INFER_RESULT"
    assert called["infer"] == 1


def test_run_pipeline_dispatches_to_pipeline_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pipeline mode must route through _run_pipeline (not _run_infer)."""
    called = {"pipeline": 0, "infer": 0}

    def _stub_pipeline(self: PXDesignAdapter, p: PXDesignInput, ctx: Any) -> Any:
        called["pipeline"] += 1
        return "PIPELINE_RESULT"

    def _stub_infer(self: PXDesignAdapter, p: PXDesignInput, ctx: Any) -> Any:
        called["infer"] += 1
        return "INFER_RESULT"

    monkeypatch.setattr(PXDesignAdapter, "_run_pipeline", _stub_pipeline)
    monkeypatch.setattr(PXDesignAdapter, "_run_infer", _stub_infer)
    adapter = _make_adapter_without_warm()

    target = tmp_path / "target.cif"
    target.write_text("data_x\n#\n")
    params = PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=50,
        mode="pipeline",
    )
    out = adapter.run(params, ctx=None)
    assert out == "PIPELINE_RESULT"
    assert called == {"pipeline": 1, "infer": 0}


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


def test_run_pipeline_uses_and_reuses_persistent_sidecar(
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

    def _fail_subprocess_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("pipeline mode must use the persistent sidecar")

    monkeypatch.setattr(adapter_mod.subprocess, "run", _fail_subprocess_run)

    instances: list[Any] = []

    class FakeSidecar:
        def __init__(
            self,
            *,
            python: Path,
            env: dict[str, str],
            name: str,
            startup_timeout_s: float,
        ) -> None:
            self.python = python
            self.env = env
            self.name = name
            self.startup_timeout_s = startup_timeout_s
            self.calls: list[tuple[str, dict[str, Any], float | None]] = []
            instances.append(self)

        def call(
            self,
            target: str,
            kwargs: dict[str, Any] | None = None,
            *,
            timeout_s: float | None = None,
        ) -> dict[str, Any]:
            payload = kwargs or {}
            self.calls.append((target, payload, timeout_s))
            argv = payload["argv"]
            assert argv[0] == "--dump_dir"
            assert "pxdesign.runner.pipeline" not in argv
            dump_dir = Path(argv[argv.index("--dump_dir") + 1])
            input_json_path = Path(argv[argv.index("--input_json_path") + 1])
            task = json.loads(input_json_path.read_text())[0]
            task_name = task["name"]
            out_dir = dump_dir / "design_outputs" / task_name / "orig"
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
            return {"ok": True}

        def close(self) -> None:
            pass

    monkeypatch.setattr(adapter_mod, "PersistentPythonSidecar", FakeSidecar)

    adapter = _make_adapter_without_warm()
    target = tmp_path / "target.cif"
    target.write_text("data_x\n#\n")
    params = PXDesignInput(
        target_pdb=str(target),
        target_chain="A",
        binder_length=30,
        num_samples=1,
        num_steps=5,
        mode="pipeline",
        preset="extended",
        seed=42,
    )

    out1 = adapter.run(params, null_context(tmp_path / "job1"))
    out2 = adapter.run(params, null_context(tmp_path / "job2"))

    assert len(instances) == 1
    sidecar = instances[0]
    assert sidecar.python == fake_py
    assert sidecar.name == PXDESIGN_PIPELINE_ENV_NAME
    assert sidecar.env["PYTHONHASHSEED"] == "42"
    assert len(sidecar.calls) == 2
    assert [out1.designs[0].sequence, out2.designs[0].sequence] == ["GAVL", "GAVL"]
    assert out1.designs[0].score == pytest.approx(0.77)


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


def test_input_schema_declares_preset_enum() -> None:
    """Pi3: schema exposes preset as preview/extended enum, default preview."""
    raw = json.loads(_INPUT_SCHEMA.read_text())
    preset = raw["properties"]["preset"]
    assert sorted(preset["enum"]) == ["extended", "preview"]
    assert preset["default"] == "preview"


def test_pydantic_accepts_preset_extended() -> None:
    """Pi3: pydantic round-trips the extended preset."""
    p = PXDesignInput(
        target_pdb="/tmp/x.pdb",
        target_chain="A",
        binder_length=50,
        mode="pipeline",
        preset="extended",
    )
    assert p.preset == "extended"


def test_pydantic_rejects_unknown_preset() -> None:
    """Pi3: bad preset value rejected at pydantic boundary."""
    with pytest.raises(ValidationError):
        PXDesignInput(
            target_pdb="/tmp/x.pdb",
            target_chain="A",
            binder_length=50,
            preset="ultra",
        )


def test_build_pipeline_cmd_threads_mpnn_temperature(tmp_path: Path) -> None:
    """Pi2: upstream argparse MPNN temperature flag is included."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
        mode="pipeline",
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
        mode="pipeline",
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=17,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    flag_idx = cmd.index("--eval.binder.tools.mpnn.temperature")
    assert cmd[flag_idx + 1] == "0.1"


def test_build_pipeline_cmd_threads_preset_extended(tmp_path: Path) -> None:
    """Pi3: preset=extended is passed through to --preset."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
        mode="pipeline",
        preset="extended",
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=17,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    assert "--preset" in cmd
    idx = cmd.index("--preset")
    assert cmd[idx + 1] == "extended"


def test_build_pipeline_cmd_preset_default_preview(tmp_path: Path) -> None:
    """Pi3: omitted preset defaults to preview at the CLI level."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
        mode="pipeline",
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=17,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    idx = cmd.index("--preset")
    assert cmd[idx + 1] == "preview"


def test_build_pipeline_cmd_legacy_af2_init_guess_false_falls_back_to_custom(
    tmp_path: Path,
) -> None:
    """Pi3: callers that pre-date the preset enum can still disable AF2 init guess."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
        mode="pipeline",
        af2_init_guess=False,
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=17,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    idx = cmd.index("--preset")
    assert cmd[idx + 1] == "custom"


def test_build_pipeline_cmd_carries_top_k_and_seed(tmp_path: Path) -> None:
    """Pi4 (regression): top_k_designs threaded into --min_total_return + --max_success_return."""
    params = PXDesignInput(
        target_pdb=str(tmp_path / "t.cif"),
        target_chain="A",
        binder_length=50,
        mode="pipeline",
        top_k_designs=12,
    )
    cmd = PXDesignAdapter._build_pipeline_cmd(
        params=params,
        seed_val=42,
        input_json_path=tmp_path / "in.json",
        job_outputs=tmp_path / "out",
    )
    i_min = cmd.index("--min_total_return")
    i_max = cmd.index("--max_success_return")
    assert cmd[i_min + 1] == "12"
    assert cmd[i_max + 1] == "12"
    i_seed = cmd.index("--seeds")
    assert cmd[i_seed + 1] == "42"
    i_msa = cmd.index("--use_msa")
    assert cmd[i_msa + 1] == "false"


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
