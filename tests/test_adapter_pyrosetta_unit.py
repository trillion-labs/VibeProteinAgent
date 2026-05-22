"""Fast unit tests for the PyRosetta adapter.

These all run in vpa — no PyRosetta install, no subprocess, no GPU.
They cover the registry-side contract (descriptor validity, schema
round-trips, input validation, helper hygiene) and the negative path
of ``availability_skip_reason()`` when ``$VPA_PYROSETTA_ENV`` points
nowhere.
"""
from __future__ import annotations

import ast
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters.pyrosetta.adapter import (
    PYROSETTA_ENV_DEFAULT,
    PYROSETTA_ENV_NAME,
    PyRosettaAdapter,
    PyRosettaInput,
    PyRosettaInterfaceMetrics,
    PyRosettaOutput,
    _build_cmd,
    _find_conda_exe,
    _safe_read_json,
    _subprocess_env,
    availability_skip_reason,
    env_present,
    pyrosetta_env,
    pyrosetta_python,
)

# ----------------------------------------------------------------- descriptor


def test_descriptor_fields():
    d = PyRosettaAdapter.descriptor
    assert d.name == "pyrosetta"
    assert d.version == "1.0.0"
    assert d.category.value == "scoring"
    assert d.backend.value == "binary"
    assert d.gpu_required is False
    assert d.vram_min_gib is None
    assert d.startup_cost.value == "low"
    assert d.pin_warm is True
    assert d.warm_load_capable is True
    assert d.execution_isolation.value == "persistent_sidecar"
    assert d.environment_name == "pyrosetta-vpa"
    assert d.support_status.value == "verified"
    assert "scoring" in d.supported_tasks
    assert "relaxation" in d.supported_tasks


def test_descriptor_registered_in_builtins():
    from vibeproteinagent.adapters.base import _BUILTIN_ADAPTERS

    assert "pyrosetta" in _BUILTIN_ADAPTERS
    descriptor, cls = _BUILTIN_ADAPTERS["pyrosetta"]
    assert cls is PyRosettaAdapter
    assert descriptor.name == "pyrosetta"


# ----------------------------------------------------------------- input model


def test_input_defaults():
    p = PyRosettaInput(pdb_path="/tmp/_unused.pdb")
    assert p.pdb_path == "/tmp/_unused.pdb"
    assert p.mode == "score"
    assert p.score_fn == "ref2015"
    assert p.chains_a is None
    assert p.chains_b is None
    assert p.max_relax_cycles == 5
    assert p.seed is None


def test_input_mode_enum():
    for ok in ("score", "interface", "relax", "pack"):
        p = PyRosettaInput(pdb_path="/x.pdb", mode=ok, chains_a="A", chains_b="B")
        assert p.mode == ok
    with pytest.raises(ValidationError):
        PyRosettaInput(pdb_path="/x.pdb", mode="minimize")


def test_input_interface_requires_chains():
    with pytest.raises(ValidationError, match="chains_a and chains_b"):
        PyRosettaInput(pdb_path="/x.pdb", mode="interface")
    with pytest.raises(ValidationError, match="chains_a and chains_b"):
        PyRosettaInput(pdb_path="/x.pdb", mode="interface", chains_a="A")
    # Both present is fine.
    PyRosettaInput(pdb_path="/x.pdb", mode="interface", chains_a="A", chains_b="B")


def test_input_relax_cycles_range():
    with pytest.raises(ValidationError):
        PyRosettaInput(pdb_path="/x.pdb", max_relax_cycles=0)
    with pytest.raises(ValidationError):
        PyRosettaInput(pdb_path="/x.pdb", max_relax_cycles=100)
    PyRosettaInput(pdb_path="/x.pdb", max_relax_cycles=5)


def test_input_rejects_empty_pdb_path():
    with pytest.raises(ValidationError):
        PyRosettaInput(pdb_path="")


# ----------------------------------------------------------------- output model


def test_output_round_trip():
    out = PyRosettaOutput(
        mode="interface",
        total_score=-123.45,
        interface_metrics=PyRosettaInterfaceMetrics(
            dG_separated=-12.3,
            dSASA=850.5,
            sc=0.71,
            packstat=0.62,
            nres_int=23,
            interface_hbonds=4,
            dG_separated_per_int_area=-12.3 / 850.5,
        ),
        relaxed_pdb_path=None,
        runtime_ms=4321,
        seed_used=42,
    )
    rebuilt = PyRosettaOutput.model_validate(out.model_dump())
    assert rebuilt == out


def test_output_score_only():
    out = PyRosettaOutput(mode="score", total_score=-99.0, runtime_ms=10)
    assert out.interface_metrics is None
    assert out.relaxed_pdb_path is None


# ----------------------------------------------------------------- schemas (JSON)


def test_input_schema_well_formed():
    schema_path = (
        Path(PyRosettaAdapter.descriptor.input_schema)
        if Path(str(PyRosettaAdapter.descriptor.input_schema)).is_absolute()
        else Path(__file__).resolve().parent.parent
        / "src" / "vibeproteinagent" / "adapters" / "pyrosetta"
        / str(PyRosettaAdapter.descriptor.input_schema)
    )
    data = json.loads(schema_path.read_text())
    assert data["type"] == "object"
    assert "pdb_path" in data["required"]
    assert data["properties"]["mode"]["enum"] == ["score", "interface", "relax", "pack"]


def test_output_schema_well_formed():
    schema_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "vibeproteinagent" / "adapters" / "pyrosetta"
        / str(PyRosettaAdapter.descriptor.output_schema)
    )
    data = json.loads(schema_path.read_text())
    assert data["type"] == "object"
    assert "total_score" in data["required"]
    assert "mode" in data["required"]


# ----------------------------------------------------------------- env / paths


def test_pyrosetta_env_default():
    # Without VPA_PYROSETTA_ENV set, we should fall through to either
    # the canonical default path or a discovered env. Either way the
    # return is a Path, not an Optional.
    p = pyrosetta_env()
    assert isinstance(p, Path)


def test_pyrosetta_env_var_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_PYROSETTA_ENV", str(tmp_path / "alt-env"))
    assert pyrosetta_env() == tmp_path / "alt-env"
    assert pyrosetta_python() == tmp_path / "alt-env" / "bin" / "python"


def test_env_present_false_when_path_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_PYROSETTA_ENV", str(tmp_path / "definitely_not_here"))
    assert env_present() is False


def test_availability_skip_reason_missing_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_PYROSETTA_ENV", str(tmp_path / "definitely_not_here"))
    reason = availability_skip_reason()
    assert reason is not None
    assert "PyRosetta env not present" in reason
    assert "vendor/pyrosetta/UPSTREAM.md" in reason


def test_env_name_constants():
    assert PYROSETTA_ENV_NAME == "pyrosetta-vpa"
    assert PYROSETTA_ENV_DEFAULT.name == "pyrosetta-vpa"


# ----------------------------------------------------------------- helpers


def test_subprocess_env_strips_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/should/not/leak")
    env = _subprocess_env(seed=None)
    assert "/should/not/leak" not in env["PYTHONPATH"]
    assert env["PYTHONPATH"].endswith("/src")
    assert env["PYTHONNOUSERSITE"] == "1"

    seeded = _subprocess_env(seed=7)
    assert seeded["PYTHONHASHSEED"] == "7"


def test_build_cmd_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_PYROSETTA_ENV", str(tmp_path / "envroot"))
    spec = tmp_path / "spec.json"
    res = tmp_path / "res.json"
    spec.write_text("{}")
    cmd = _build_cmd(spec_path=spec, result_path=res)
    assert cmd[0] == str(tmp_path / "envroot" / "bin" / "python")
    assert cmd[1] == "-m"
    assert cmd[2] == "vibeproteinagent.adapters.pyrosetta._runner"
    assert "--input" in cmd and str(spec) in cmd
    assert "--output" in cmd and str(res) in cmd


def test_safe_read_json_handles_missing(tmp_path):
    assert _safe_read_json(tmp_path / "nope.json") is None


def test_safe_read_json_handles_garbage(tmp_path):
    p = tmp_path / "broken.json"
    p.write_text("not json {")
    assert _safe_read_json(p) is None


def test_find_conda_exe_is_resolvable_or_none():
    # Whether a conda binary exists is environment-dependent; just
    # verify the helper doesn't raise and returns None or a real path.
    res = _find_conda_exe()
    assert res is None or res.exists()


# ----------------------------------------------------------------- adapter wiring


def test_adapter_constructs_without_env(monkeypatch, tmp_path):
    """Adapter constructor must not require the pyrosetta env to exist —
    registry discovery happens on hosts that may never provision it."""
    monkeypatch.setenv("VPA_PYROSETTA_ENV", str(tmp_path / "no-env"))
    adapter = PyRosettaAdapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    assert adapter.healthcheck() is False


def test_adapter_run_raises_when_unavailable(monkeypatch, tmp_path):
    from vibeproteinagent.adapters.context import null_context

    monkeypatch.setenv("VPA_PYROSETTA_ENV", str(tmp_path / "no-env"))
    adapter = PyRosettaAdapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    params = PyRosettaInput(pdb_path=str(tmp_path / "nope.pdb"))
    with pytest.raises(RuntimeError, match="PyRosetta unavailable"):
        adapter.run(params, null_context(job_dir))


def test_adapter_run_uses_persistent_sidecar(monkeypatch, tmp_path):
    import vibeproteinagent.adapters.pyrosetta.adapter as adapter_mod
    from vibeproteinagent.adapters.context import null_context

    created: list[object] = []

    class FakeSidecar:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.calls: list[tuple[str, dict]] = []
            self.closed = False
            created.append(self)

        def call(self, target: str, kwargs: dict | None = None, *, timeout_s: float | None = None):
            self.calls.append((target, kwargs or {}))
            if target.endswith(":warm_sidecar"):
                return {"initialized": True, "seed": None}
            assert target.endswith(":run_sidecar")
            return {
                "mode": "score",
                "total_score": -12.5,
                "interface_metrics": None,
                "relaxed_pdb_path": None,
                "runtime_ms": 7,
                "seed_used": None,
            }

        def close(self):
            self.closed = True

    def _no_subprocess_run(*_args, **_kwargs):
        raise AssertionError("run() should use the persistent sidecar, not subprocess.run")

    monkeypatch.setattr(adapter_mod, "availability_skip_reason", lambda: None)
    monkeypatch.setattr(adapter_mod, "PersistentPythonSidecar", FakeSidecar)
    monkeypatch.setattr(adapter_mod.subprocess, "run", _no_subprocess_run)

    pdb = tmp_path / "input.pdb"
    pdb.write_text("HEADER test\nEND\n")
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    adapter = PyRosettaAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out1 = adapter.run(PyRosettaInput(pdb_path=str(pdb)), null_context(job_dir))
    out2 = adapter.run(PyRosettaInput(pdb_path=str(pdb)), null_context(job_dir))

    assert out1.total_score == -12.5
    assert out2.total_score == -12.5
    assert len(created) == 1
    assert len(created[0].calls) == 2
    assert (job_dir / "outputs" / "_result.json").is_file()
    adapter.unload()
    assert created[0].closed is True


def test_estimate_vram_returns_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_PYROSETTA_ENV", str(tmp_path / "no-env"))
    adapter = PyRosettaAdapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    p = PyRosettaInput(pdb_path="/x.pdb")
    assert adapter.estimate_vram(p) == 0


# ----------------------------------------------------------------- runner module syntax


def test_runner_module_parses():
    """The runner ships in vpa but never runs there — at minimum the
    file must parse as valid Python, so a stray syntax error doesn't
    only surface when an operator finally provisions the env."""
    runner_path = (
        Path(__file__).resolve().parent.parent
        / "src" / "vibeproteinagent" / "adapters" / "pyrosetta" / "_runner.py"
    )
    src = runner_path.read_text()
    tree = ast.parse(src, filename=str(runner_path))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "run"
        for node in ast.walk(tree)
    )


def test_runner_does_not_import_pyrosetta_at_module_load():
    """``import pyrosetta`` MUST be deferred (lazy) — otherwise the runner
    module can't even be syntax-checked from vpa, breaking the unit test."""
    import importlib

    mod = importlib.import_module("vibeproteinagent.adapters.pyrosetta._runner")
    # If the module imported pyrosetta at top-level, we'd already have
    # failed (vpa does not have pyrosetta). Reaching this point proves
    # the lazy-import contract.
    assert mod is not None
    assert not hasattr(mod, "pyrosetta")


def test_runner_sidecar_score_path_with_fake_pyrosetta(monkeypatch, tmp_path):
    import importlib

    mod = importlib.import_module("vibeproteinagent.adapters.pyrosetta._runner")
    monkeypatch.setattr(mod, "_INITIALIZED", False)
    monkeypatch.setattr(mod, "_INITIALIZED_SEED", None)

    init_calls: list[str] = []
    fake_pyrosetta = types.SimpleNamespace(
        init=lambda extra_options: init_calls.append(extra_options),
        pose_from_pdb=lambda _path: object(),
        create_score_function=lambda _tag: (lambda _pose: -7.25),
    )
    monkeypatch.setitem(sys.modules, "pyrosetta", fake_pyrosetta)

    pdb = tmp_path / "pose.pdb"
    pdb.write_text("HEADER fake\nEND\n")
    out = mod.run_sidecar(
        {
            "pdb_path": str(pdb),
            "mode": "score",
            "score_fn": "ref2015",
            "seed": 123,
        }
    )

    assert out["mode"] == "score"
    assert out["total_score"] == -7.25
    assert out["seed_used"] == 123
    assert len(init_calls) == 1


# ----------------------------------------------------------------- registry roundtrip


def test_registry_lists_pyrosetta():
    from vibeproteinagent.adapters.registry import ToolRegistry

    r = ToolRegistry()
    r.discover()
    names = {d.name for d in r.list()}
    assert "pyrosetta" in names


def test_registry_get_pyrosetta():
    from vibeproteinagent.adapters.registry import ToolRegistry

    r = ToolRegistry()
    r.discover()
    descriptor, cls = r.get("pyrosetta")
    assert descriptor.name == "pyrosetta"
    assert cls is PyRosettaAdapter
