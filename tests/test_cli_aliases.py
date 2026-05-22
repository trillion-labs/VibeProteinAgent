"""Codex review C16: ``vibeproteinagent`` and ``vpa`` are both installed.

The brief uses the long form (``vibeproteinagent <verb>``); we shipped the
short ``vpa`` from day one. Phase 7 adds the long form as a parallel
console-script alias in ``pyproject.toml`` so both work.

Note: this test invokes the installed script via ``subprocess`` because we
specifically want to verify the entry-point wiring, not the Typer app object.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


@pytest.mark.skipif(not _have("vibeproteinagent"), reason="package not installed")
def test_vibeproteinagent_console_script_version() -> None:
    result = subprocess.run(
        ["vibeproteinagent", "version", "--json"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert "version" in payload["data"]


@pytest.mark.skipif(
    not (_have("vpa") and _have("vibeproteinagent")),
    reason="both scripts must be installed",
)
def test_vpa_and_vibeproteinagent_emit_same_envelope() -> None:
    a = subprocess.run(
        ["vpa", "version", "--json"], capture_output=True, text=True, timeout=30,
    )
    b = subprocess.run(
        ["vibeproteinagent", "version", "--json"],
        capture_output=True, text=True, timeout=30,
    )
    assert a.returncode == 0
    assert b.returncode == 0
    # Bit-exact: same Typer app, same render path.
    assert a.stdout == b.stdout
