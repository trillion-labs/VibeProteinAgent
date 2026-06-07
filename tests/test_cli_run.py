from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.core.errors import DaemonWorkspaceMismatchError
from vibeproteinagent.workflows import Workflow, WorkflowNode, WorkflowStore, serialize_workflow
from vibeproteinagent.workspace import JobStore, TrajectoryStore, Workspace

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
    assert data["daemon_running"] is False
    assert data["execution_ready"] is False


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


def test_run_tags_job_with_trajectory(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    ws = Workspace.create("alpha")
    traj = TrajectoryStore(ws).create(user_query="design one binder")
    result = runner.invoke(
        app,
        [
            "run",
            "noop",
            "payload=hi",
            "--workspace",
            "alpha",
            "--trajectory",
            traj.trajectory_id,
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)["data"]
    assert data["trajectory_id"] == traj.trajectory_id
    rec = JobStore(ws).get(data["job_id"])
    assert rec.trajectory_id == traj.trajectory_id


def test_wait_trajectory_waits_tagged_jobs(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    ws = Workspace.create("alpha")
    traj = TrajectoryStore(ws).create(user_query="wait on this query")
    store = JobStore(ws)
    job = store.submit(
        "noop",
        "1.0.0",
        {"payload": "hi"},
        trajectory_id=traj.trajectory_id,
    )
    store.transition(job.job_id, "queued")
    store.transition(job.job_id, "cancelled")

    result = runner.invoke(
        app,
        [
            "wait",
            "--trajectory",
            traj.trajectory_id,
            "--workspace",
            "alpha",
            "--timeout-ms",
            "0",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)["data"]
    assert data["trajectory_id"] == traj.trajectory_id
    assert data["all_terminal"] is True
    assert data["execution_all_terminal"] is True
    assert data["trajectory_complete"] is False
    assert [job["job_id"] for job in data["jobs"]] == [job.job_id]


def test_wait_trajectory_accounts_for_workflow_only_trajectory(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    ws = Workspace.create("alpha")
    traj = TrajectoryStore(ws).create(user_query="workflow only")
    wf_id = WorkflowStore(ws).create(
        name="recorded_block",
        version="1.0",
        descriptor_yaml="name: recorded_block\nversion: '1.0'\nnodes: []\n",
        params={},
        trajectory_id=traj.trajectory_id,
    )

    result = runner.invoke(
        app,
        [
            "wait",
            "--trajectory",
            traj.trajectory_id,
            "--workspace",
            "alpha",
            "--timeout-ms",
            "0",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)["data"]
    assert data["trajectory_id"] == traj.trajectory_id
    assert data["execution_all_terminal"] is False
    assert data["trajectory_complete"] is False
    assert data["unfinished_job_ids"] == []
    assert data["unfinished_workflow_ids"] == [wf_id]
    assert data["timed_out"] is True


def test_status_without_job_id_returns_actionable_hint(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("alpha")
    result = runner.invoke(app, ["status", "--workspace", "alpha", "--json"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "E_INVALID_PARAMS"
    assert "vpa trajectory status" in payload["error"]["details"]["hint"]


def test_run_fails_on_daemon_workspace_mismatch(
    runner: CliRunner,
    isolated_vpa_home: Path,
    monkeypatch,
) -> None:
    Workspace.create("alpha")

    def _mismatch(self, workspace_name: str):
        raise DaemonWorkspaceMismatchError(
            "running supervisor is attached to a different workspace",
            details={
                "requested_workspace": workspace_name,
                "supervisor_workspace": "other",
            },
        )

    monkeypatch.setattr(
        "vibeproteinagent.workers.supervisor_client.SupervisorClient.require_workspace",
        _mismatch,
    )

    result = runner.invoke(
        app,
        ["run", "noop", "payload=hi", "--workspace", "alpha", "--json"],
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "E_DAEMON_WORKSPACE_MISMATCH"
    assert payload["error"]["details"]["requested_workspace"] == "alpha"


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


def test_run_protenix_confusable_msa_backend_hint(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    result = runner.invoke(
        app,
        ["run", "protenix_v2", "input_json=/tmp/input.json", "msa_backend=mmseqs2_gpu", "--json"],
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    messages = " ".join(e["message"] for e in payload["error"]["details"]["errors"])
    assert "mmseqs2_gpu" in messages
    assert "tool name" in messages
    assert "auto" in messages


def test_run_proteina_ligand_binder_conditional_required_hint(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    result = runner.invoke(
        app,
        [
            "run",
            "proteina_complexa",
            "mode=ligand_binder",
            "target_pdb=/tmp/target.pdb",
            "ligand_smiles=CCO",
            "--json",
        ],
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    messages = " ".join(e["message"] for e in payload["error"]["details"]["errors"])
    assert "mode='ligand_binder'" in messages
    assert "target_pocket_residues" in messages
    assert "target_pocket_residue_ids" in messages


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


def test_workflow_run_tags_workflow_with_trajectory(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    ws = Workspace.create("default")
    traj = TrajectoryStore(ws).create(user_query="workflow block")
    result = runner.invoke(
        app,
        [
            "workflow",
            "run",
            "noop_chain",
            "--workspace",
            ws.name,
            "--trajectory",
            traj.trajectory_id,
            "--json",
        ],
    )
    assert result.exit_code == 3, result.stdout
    payload = json.loads(result.stdout)
    workflow_id = payload["error"]["details"]["workflow_id"]
    wf = WorkflowStore(ws).get(workflow_id)
    assert wf.trajectory_id == traj.trajectory_id


def test_workflow_run_fails_before_submit_on_daemon_workspace_mismatch(
    runner: CliRunner,
    isolated_vpa_home: Path,
    monkeypatch,
) -> None:
    ws = Workspace.create("default")

    def _mismatch(self, workspace_name: str):
        raise DaemonWorkspaceMismatchError(
            "running supervisor is attached to a different workspace",
            details={
                "requested_workspace": workspace_name,
                "supervisor_workspace": "other",
            },
        )

    monkeypatch.setattr(
        "vibeproteinagent.workers.supervisor_client.SupervisorClient.require_workspace",
        _mismatch,
    )

    result = runner.invoke(
        app,
        ["workflow", "run", "noop_chain", "--workspace", ws.name, "--json"],
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "E_DAEMON_WORKSPACE_MISMATCH"


def test_workflow_list_omits_task_specific_bundled_recipes(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    result = runner.invoke(app, ["workflow", "list", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    names = {row["name"] for row in payload["data"]["workflows"]}
    assert "noop_chain" in names
    assert not {
        "semantic_constraint_to_family_sequences",
        "template_guided_design_prep",
        "rfdiffusion_backbone_design_round",
        "mutation_repair_round",
        "target_sequence_binder_design_round",
        "mpnn_then_refold",
    } & names
