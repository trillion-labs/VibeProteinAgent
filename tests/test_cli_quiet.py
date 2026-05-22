from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app


def test_quiet_version_emits_only_version(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["-q", "version"])
    assert result.exit_code == 0
    out = result.stdout.strip()
    # Quiet: just the bare version string.
    assert out
    # Should NOT contain explanatory chrome.
    assert "python" not in out.lower()
    assert "torch" not in out.lower()


def test_quiet_tools_still_lists_tools_but_no_title(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["-q", "tools"])
    assert result.exit_code == 0
    # Tool names still appear (the table body).
    assert "proteinmpnn" in result.stdout
    # Quiet → title chrome suppressed.
    assert "Registered tools" not in result.stdout


def test_quiet_does_not_change_exit_code_on_success(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["-q", "doctor"])
    assert result.exit_code == 0


def test_quiet_does_not_affect_json(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    """JSON must stay bit-exact whether -q is passed or not."""
    a = runner.invoke(app, ["tools", "--json"])
    b = runner.invoke(app, ["-q", "tools", "--json"])
    assert a.exit_code == 0 and b.exit_code == 0
    assert a.stdout == b.stdout


def test_verbose_flag_does_not_break_normal_commands(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["-v", "tools"])
    assert result.exit_code == 0
    assert "proteinmpnn" in result.stdout
