from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.workflows import Workflow, WorkflowNode, WorkflowStore, serialize_workflow
from vibeproteinagent.workspace import Workspace

_JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")


def test_run_noop_returns_queued_envelope(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    result = runner.invoke(app, ["run", "noop", "payload=hello", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    data = payload["data"]
    assert _JOB_ID.match(data["job_id"]), data["job_id"]
    assert data["state"] == "queued"
    assert data["tool"] == "noop"
    assert data["version"] == "1.0.0"
    assert data["workspace"] == "default"


def test_run_writes_to_specified_workspace(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("alpha")
    result = runner.invoke(
        app, ["run", "noop", "payload=hi", "--workspace", "alpha", "--json"]
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)["data"]
    assert data["workspace"] == "alpha"


def test_run_invalid_params_returns_error_envelope(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    # noop schema requires "payload"; sending only "x" violates it.
    result = runner.invoke(app, ["run", "noop", "x=1", "--json"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_INVALID_PARAMS"
    assert "field" in payload["error"]["details"]
    assert "errors" in payload["error"]["details"]


def test_run_call_id_is_idempotent(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    a = runner.invoke(app, ["run", "noop", "payload=x", "--call-id", "k1", "--json"])
    b = runner.invoke(app, ["run", "noop", "payload=x", "--call-id", "k1", "--json"])
    assert a.exit_code == 0 and b.exit_code == 0
    assert json.loads(a.stdout)["data"]["job_id"] == json.loads(b.stdout)["data"]["job_id"]


def test_run_unknown_tool(runner: CliRunner, isolated_vpa_home: Path) -> None:
    Workspace.create("default")
    result = runner.invoke(app, ["run", "no_such_tool", "payload=hi", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_UNKNOWN_TOOL"


def test_workflow_resume_without_daemon_returns_immediately(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    ws = Workspace.create("default")
    wf = Workflow(
        name="resume-no-daemon",
        version="1.0.0",
        nodes=[WorkflowNode(id="n", tool="noop", inputs={"payload": "x"})],
    )
    workflow_id = WorkflowStore(ws).create(
        name=wf.name,
        version=wf.version,
        descriptor_yaml=serialize_workflow(wf),
        params={},
    )

    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            wf.name,
            "--resume",
            workflow_id,
            "--json",
            "--max-wait-s",
            "0",
        ],
    )

    assert result.exit_code == 3, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_DAEMON_UNAVAILABLE"
    assert payload["error"]["details"]["workflow_id"] == workflow_id


def test_workflow_list_includes_iterative_examples(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["workflow", "list", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    names = {row["name"] for row in payload["data"]["workflows"]}
    assert {
        "semantic_constraint_to_family_sequences",
        "template_guided_design_prep",
        "rfdiffusion_backbone_design_round",
        "mutation_repair_round",
    }.issubset(names)
