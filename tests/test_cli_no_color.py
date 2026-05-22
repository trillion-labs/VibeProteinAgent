from __future__ import annotations

import re
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def test_tools_with_no_color_env_emits_no_ansi(
    runner: CliRunner, isolated_vpa_home: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    result = runner.invoke(app, ["tools"])
    assert result.exit_code == 0
    assert not _ANSI_RE.search(result.stdout), result.stdout


def test_tools_with_no_color_flag_emits_no_ansi(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["--no-color", "tools"])
    assert result.exit_code == 0
    assert not _ANSI_RE.search(result.stdout), result.stdout


def test_doctor_no_color(runner: CliRunner, isolated_vpa_home: Path) -> None:
    result = runner.invoke(app, ["--no-color", "doctor"])
    assert result.exit_code == 0
    assert not _ANSI_RE.search(result.stdout), result.stdout


def test_version_no_color(runner: CliRunner, isolated_vpa_home: Path) -> None:
    result = runner.invoke(app, ["--no-color", "version"])
    assert result.exit_code == 0
    assert "vpa " in result.stdout
    assert not _ANSI_RE.search(result.stdout)


def test_json_mode_unchanged_by_no_color(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    """JSON output must remain bit-exact whether or not --no-color is set."""
    a = runner.invoke(app, ["tools", "--json"])
    b = runner.invoke(app, ["--no-color", "tools", "--json"])
    assert a.exit_code == 0 and b.exit_code == 0
    assert a.stdout == b.stdout
