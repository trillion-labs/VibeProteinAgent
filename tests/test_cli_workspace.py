from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app


def test_workspace_lifecycle(runner: CliRunner, isolated_vpa_home: Path) -> None:
    create = runner.invoke(app, ["workspace", "create", "alpha", "--json"])
    assert create.exit_code == 0, create.stdout
    cdata = json.loads(create.stdout)["data"]
    assert cdata["workspace"] == "alpha"
    assert "alpha" in cdata["root"]

    use = runner.invoke(app, ["workspace", "use", "alpha", "--json"])
    assert use.exit_code == 0, use.stdout
    assert json.loads(use.stdout)["data"]["workspace"] == "alpha"

    status = runner.invoke(app, ["workspace", "status", "--json"])
    assert status.exit_code == 0
    sdata = json.loads(status.stdout)["data"]
    assert sdata["active"] == "alpha"
    assert sdata["via"] in ("file", "env")

    listed = runner.invoke(app, ["workspace", "list", "--json"])
    assert listed.exit_code == 0
    ldata = json.loads(listed.stdout)["data"]
    assert "alpha" in ldata["workspaces"]
    assert ldata["active"] == "alpha"

    deleted = runner.invoke(app, ["workspace", "delete", "alpha", "--json"])
    assert deleted.exit_code == 0
    ddata = json.loads(deleted.stdout)["data"]
    assert ddata["workspace"] == "alpha"
    assert ddata["deleted"] is True


def test_workspace_status_default_when_nothing_set(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    res = runner.invoke(app, ["workspace", "status", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)["data"]
    assert data["active"] == "default"
    assert data["via"] == "default"


def test_workspace_delete_missing_is_idempotent(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    res = runner.invoke(app, ["workspace", "delete", "ghost", "--json"])
    assert res.exit_code == 0
    data = json.loads(res.stdout)["data"]
    assert data["deleted"] is True


def test_workspace_create_idempotent(runner: CliRunner, isolated_vpa_home: Path) -> None:
    a = runner.invoke(app, ["workspace", "create", "twice", "--json"])
    b = runner.invoke(app, ["workspace", "create", "twice", "--json"])
    assert a.exit_code == 0 and b.exit_code == 0
    assert json.loads(a.stdout)["data"] == json.loads(b.stdout)["data"]
