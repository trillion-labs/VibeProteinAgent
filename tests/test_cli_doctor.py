from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app


def test_doctor_returns_env_info(runner: CliRunner, isolated_vpa_home: Path) -> None:
    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    assert data["python_version"] == sys.version.split()[0]
    assert data["executable"] == sys.executable
    # daemon is not yet implemented
    assert data["daemon_running"] is False
    # registry should at minimum contain noop
    assert "noop" in data["registry_tools"]
    assert "install_readiness" in data
    assert "tools" in data["install_readiness"]


def test_doctor_reports_torch_and_cuda_or_skips_cleanly(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    res = runner.invoke(app, ["doctor", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)["data"]
    # Schema: torch_version may be None on a torch-less host but the keys exist.
    assert "torch_version" in data
    assert "cuda_available" in data
    assert "num_gpus" in data
    assert "gpus" in data
    assert isinstance(data["gpus"], list)
    if data["cuda_available"]:
        assert data["num_gpus"] >= 1
        for g in data["gpus"]:
            assert "index" in g
            assert "name" in g
            assert "compute_capability" in g
            assert "total_memory_gib" in g
            major, _, minor = g["compute_capability"].partition(".")
            assert major.isdigit() and minor.isdigit()


def test_doctor_includes_workspace_and_env(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    res = runner.invoke(app, ["doctor", "--json"])
    data = json.loads(res.stdout)["data"]
    assert data["active_workspace"] == "default"
    assert "workspace_root" in data
    assert "env_vars" in data
    # VPA_HOME points at the isolated tmp dir
    assert data["env_vars"]["VPA_HOME"] == str(isolated_vpa_home)
    assert data["model_cache_root"].endswith("/models")


def test_doctor_filters_install_readiness_tools(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["doctor", "--tools", "rfdiffusion", "--json"])
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    readiness = data["install_readiness"]
    assert [tool["name"] for tool in readiness["tools"]] == ["rfdiffusion"]
    assert "rfdiffusion_modes" in readiness["tools"][0]
