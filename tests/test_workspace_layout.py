from __future__ import annotations

from pathlib import Path

import pytest

from vibeproteinagent.core.errors import NotFoundError
from vibeproteinagent.workspace import Workspace


def test_create_workspace_makes_layout(isolated_vpa_home: Path) -> None:
    ws = Workspace.create("ws1")
    assert ws.name == "ws1"
    assert ws.root.is_dir()
    assert ws.db_path.is_file()
    for sub in (ws.jobs_dir, ws.artifacts_dir, ws.workflows_dir, ws.cache_dir):
        assert sub.is_dir()


def test_create_is_idempotent(isolated_vpa_home: Path) -> None:
    ws1 = Workspace.create("dup")
    ws2 = Workspace.create("dup")
    assert ws1.root == ws2.root
    assert ws2.db_path.is_file()


def test_resolve_defaults_when_nothing_set(
    isolated_vpa_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)
    ws = Workspace.resolve()
    assert ws.name == "default"


def test_resolve_uses_arg_first(isolated_vpa_home: Path) -> None:
    ws = Workspace.resolve("explicit")
    assert ws.name == "explicit"


def test_resolve_falls_back_to_env(
    isolated_vpa_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VPA_WORKSPACE", "from_env")
    ws = Workspace.resolve()
    assert ws.name == "from_env"


def test_use_then_resolve_reads_active(isolated_vpa_home: Path) -> None:
    Workspace.create("alpha")
    Workspace.use("alpha")
    ws = Workspace.resolve()
    assert ws.name == "alpha"


def test_list_returns_only_initialized_workspaces(isolated_vpa_home: Path) -> None:
    Workspace.create("a")
    Workspace.create("b")
    # Place a non-workspace directory in the workspaces root to ensure list() ignores it.
    (isolated_vpa_home / "workspaces" / "garbage").mkdir()
    names = Workspace.list()
    assert "a" in names
    assert "b" in names
    assert "garbage" not in names


def test_delete_clears_active_when_deleting_active(isolated_vpa_home: Path) -> None:
    Workspace.create("zeta")
    Workspace.use("zeta")
    Workspace.delete("zeta")
    assert "zeta" not in Workspace.list()
    # Active file removed, so resolve falls back to default.
    ws = Workspace.resolve()
    assert ws.name == "default"


def test_delete_missing_raises(isolated_vpa_home: Path) -> None:
    with pytest.raises(NotFoundError):
        Workspace.delete("nope")


def test_use_missing_raises(isolated_vpa_home: Path) -> None:
    with pytest.raises(NotFoundError):
        Workspace.use("missing")
