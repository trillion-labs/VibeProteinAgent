from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app


def test_tools_lists_both_builtins(runner: CliRunner, isolated_vpa_home: Path) -> None:
    result = runner.invoke(app, ["tools", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    names = {t["name"] for t in payload["data"]["tools"]}
    assert "noop" not in names
    assert "mpnn_to_protenix_input" not in names
    assert "sequence_to_protenix_input" not in names
    assert "pxdesign_pipeline_multi_gpu" not in names
    assert "proteinmpnn" in names


def test_tools_include_internal_is_explicit(
    runner: CliRunner,
    isolated_vpa_home: Path,
) -> None:
    result = runner.invoke(app, ["tools", "--include-internal", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    names = {t["name"] for t in payload["data"]["tools"]}
    assert "noop" in names
    assert "mpnn_to_protenix_input" in names
    assert "sequence_to_protenix_input" in names
    assert "pxdesign_pipeline_multi_gpu" not in names


def test_tools_filter_by_category_misc(runner: CliRunner, isolated_vpa_home: Path) -> None:
    result = runner.invoke(app, ["tools", "-c", "misc", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    names = {t["name"] for t in payload["data"]["tools"]}
    # Internal benchmark/glue adapters stay out of the public catalog by default.
    expected = {
        "dssp",
        "mmseqs2_gpu",
        "foldseek",
    }
    assert expected.issubset(names)
    # Anything else in misc would be a regression — keep the set tight.
    assert names == expected


def test_tools_category_accepts_hyphenated_alias(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    """`-c protein-design` and `-c protein_design` must both match.

    Regression: the descriptor enum stores ``protein_design`` (underscored);
    users naturally type the hyphenated form from the CLI. Codex review R5.
    """
    res_hyphen = runner.invoke(app, ["tools", "-c", "protein-design", "--json"])
    res_under = runner.invoke(app, ["tools", "-c", "protein_design", "--json"])
    assert res_hyphen.exit_code == 0
    assert res_under.exit_code == 0
    hy = {t["name"] for t in json.loads(res_hyphen.stdout)["data"]["tools"]}
    un = {t["name"] for t in json.loads(res_under.stdout)["data"]["tools"]}
    assert hy == un
    assert "pxdesign" in hy
    assert hy  # non-empty


def test_tool_info_returns_full_descriptor(runner: CliRunner, isolated_vpa_home: Path) -> None:
    result = runner.invoke(app, ["tool", "info", "noop", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    desc = payload["data"]["tool"]
    assert desc["name"] == "noop"
    assert desc["version"] == "1.0.0"
    assert desc["category"] == "misc"
    assert desc["entry_point"].endswith(":NoopAdapter")


def test_tool_info_unknown_tool_returns_error(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["tool", "info", "no_such_tool", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_UNKNOWN_TOOL"


def test_tool_info_version_mismatch(runner: CliRunner, isolated_vpa_home: Path) -> None:
    result = runner.invoke(app, ["tool", "info", "noop", "--version", ">=2.0.0", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_UNKNOWN_TOOL"
