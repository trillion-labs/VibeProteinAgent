from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app


def test_cli_install_selective_dry_run(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(
        app,
        ["install", "--tools", "rfdiffusion,proteinmpnn", "--dry-run", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["dry_run"] is True
    assert data["profile"] is None
    assert data["selected_tools"] == ["rfdiffusion", "proteinmpnn"]
    assert any(step["unit"] == "weight:rfdiffusion:rfdiffusion_base" for step in data["steps"])
    assert data["cuda_profile"] in {"cu128", "cu126"}
    assert any(step["unit"] == "cuda:profile" for step in data["steps"])


def test_cli_install_boltz_tools_fetch_sources_and_caches(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(
        app,
        ["install", "--tools", "boltz2_affinity,boltzgen", "--dry-run", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    units = {step["unit"] for step in data["steps"]}

    assert "source:boltz2_affinity:boltz_source" in units
    assert "source:boltzgen:boltzgen_source" in units
    assert "weight:boltz2_affinity:boltz_mols" in units
    assert "weight:boltzgen:boltzgen_model_cache" in units
    assert data["env_exports"]["VPA_BOLTZ_TREE"].endswith("/sidecars/boltz")
    assert data["env_exports"]["VPA_BOLTZGEN_TREE"].endswith("/sidecars/boltzgen")


def test_cli_install_full_profile_includes_all_tools(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["install", "--profile", "full", "--dry-run", "--json"])
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert "rfdiffusion" in data["selected_tools"]
    assert "proteinmpnn" in data["selected_tools"]
    assert "bindcraft" in data["selected_tools"]
    assert "proteina_complexa" in data["selected_tools"]
    assert "pxdesign" in data["selected_tools"]
    assert not any(
        name.startswith("proteina_complexa_") for name in data["selected_tools"]
    )
    assert "noop" not in data["selected_tools"]
    assert "mpnn_to_protenix_input" not in data["selected_tools"]
    assert "sequence_to_protenix_input" not in data["selected_tools"]
    assert "pxdesign_pipeline_multi_gpu" not in data["selected_tools"]
    assert any(step["unit"] == "sidecar:pxdesign" for step in data["steps"])
    warning_types = {warning["type"] for warning in data["warnings"]}
    assert "credential_optional_for_esm3" in warning_types


def test_cli_install_default_selects_full_profile(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["install", "--dry-run", "--json"])

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["profile"] == "full"
    assert "protenix_v2" in data["selected_tools"]
    assert "pxdesign" in data["selected_tools"]
    assert "capabilities" in data


def test_cli_install_lint_manifest(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["install", "lint-manifest", "--json"])

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["ok"] is True
    assert data["error_count"] == 0


def test_cli_install_docker_plan(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["install", "--docker", "--dry-run", "--json"])

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["status"] == "docker-plan"
    assert data["profile"] == "full"
    assert data["docker_image"].startswith("vibeproteinagent/vpa-")


def test_cli_install_unknown_tool_errors_with_suggestion(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["install", "--tools", "rfdifusion", "--dry-run", "--json"])
    assert res.exit_code == 1
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_UNKNOWN_TOOL"
    assert "rfdiffusion" in payload["error"]["details"]["suggestions"]


def test_cli_install_doctor_reports_rfdiffusion_modes(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(app, ["install", "doctor", "--tools", "rfdiffusion", "--json"])
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["tools"][0]["name"] == "rfdiffusion"
    assert "rfdiffusion_modes" in data["tools"][0]
    assert "missing_mode_checkpoints" in data["tools"][0]["rfdiffusion_modes"]


def test_cli_install_preflight_reports_cuda_contract(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(
        app,
        [
            "install",
            "preflight",
            "--tools",
            "rfdiffusion,proteinmpnn",
            "--cuda-profile",
            "cu128",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["requested_cuda_profile"] == "cu128"
    assert data["cuda_profile"] == "cu128"
    assert data["cuda_version"] == "12.8"
    assert "checks" in data
    assert any(check["name"] == "rfdiffusion_dgl" for check in data["checks"])


def test_cli_install_dry_run_accepts_cpu_cuda_profile(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    res = runner.invoke(
        app,
        ["install", "--profile", "agent", "--cuda-profile", "cpu", "--dry-run", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["cuda_profile"] == "cpu"
    assert data["env_exports"]["VPA_CUDA_PROFILE"] == "cpu"
