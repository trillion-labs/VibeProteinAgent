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
