from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.workspace import Workspace


@pytest.fixture
def runner() -> CliRunner:
    """Typer test runner. Stdout/stderr are kept separate in modern click."""
    return CliRunner()


@pytest.fixture
def app_obj() -> object:
    return app


@pytest.fixture
def isolated_vpa_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point $VPA_HOME (and $VPA_RUNTIME_DIR) at temp dirs."""
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)
    yield home


@pytest.fixture
def tmp_workspace(isolated_vpa_home: Path) -> Workspace:
    """A freshly-created, schema-applied workspace named ``test`` under the isolated home."""
    return Workspace.create("test")
