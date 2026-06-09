"""Unit tests for the orchestration tool functions in vibeproteinagent.mcp.tools.

Each function is called directly (no MCP transport) so we can assert the returned
dict / list shape and confirm the thin facade preserves the underlying JobStore /
SupervisorClient / ToolRegistry semantics.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.core.errors import DaemonWorkspaceMismatchError, TimeoutError_
from vibeproteinagent.mcp.tools import (
    SessionCtx,
    cancel_job,
    current_workspace,
    daemon_status,
    get_logs,
    get_results,
    get_status,
    get_trajectory_status,
    get_workflow_status,
    list_artifacts,
    list_tools,
    list_workflows,
    list_workspaces,
    render_report,
    run_workflow,
    start_trajectory,
    submit_job,
    tool_info,
    wait_trajectory,
    wait_job,
    warm_tool,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace


@pytest.fixture
def ctx(tmp_workspace: Workspace) -> SessionCtx:
    return SessionCtx(workspace=tmp_workspace, registry=ToolRegistry())


def test_list_tools_returns_descriptors(ctx: SessionCtx) -> None:
    out = list_tools(ctx)
    assert isinstance(out, list)
    names = {d["name"] for d in out}
    assert "noop" not in names
    assert "protein_qc" in names


def test_list_tools_can_include_internal_descriptors(ctx: SessionCtx) -> None:
    out = list_tools(ctx, include_internal=True)
    names = {d["name"] for d in out}
    assert "noop" in names


def test_list_tools_schema_only_skips_availability_and_keeps_merge_candidates(
    ctx: SessionCtx,
    monkeypatch,
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("availability probe should not run")

    monkeypatch.setattr("vibeproteinagent.mcp.tools.attach_availability", _boom)

    out = list_tools(ctx, schema_only=True)

    names = {d["name"] for d in out}
    assert "merge_candidates" in names
    assert "noop" not in names
    merge = next(d for d in out if d["name"] == "merge_candidates")
    assert "input_schema_resolved" in merge
    assert "availability_status" not in merge


def test_list_tools_category_filter(ctx: SessionCtx) -> None:
    out = list_tools(ctx, category="protein_design")
    cats = {d["category"] for d in out}
    assert cats == {"protein_design"} or out == []  # may be empty if no design tool


def test_tool_info_known(ctx: SessionCtx) -> None:
    out = tool_info(ctx, "noop")
    assert out["name"] == "noop"
    assert out["version"] == "1.0.0"


def test_tool_info_schema_only_skips_availability(ctx: SessionCtx, monkeypatch) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("availability probe should not run")

    monkeypatch.setattr("vibeproteinagent.mcp.tools.attach_availability", _boom)

    out = tool_info(ctx, "noop", schema_only=True)

    assert out["name"] == "noop"
    assert "input_schema_resolved" in out
    assert "availability_status" not in out


def test_tool_info_unknown_returns_error_dict(ctx: SessionCtx) -> None:
    out = tool_info(ctx, "no_such_tool")
    assert out["ok"] is False
    assert out["code"] == "E_UNKNOWN_TOOL"


def test_submit_job_validates_params(ctx: SessionCtx) -> None:
    # noop schema requires {"payload": str}; sending {"x": 1} must fail.
    out = submit_job(ctx, tool="noop", params={"x": 1})
    assert out["ok"] is False
    assert out["code"] == "E_INVALID_PARAMS"


def test_submit_job_unknown_tool(ctx: SessionCtx) -> None:
    out = submit_job(ctx, tool="no_such_tool", params={})
    assert out["ok"] is False
    assert out["code"] == "E_UNKNOWN_TOOL"


def test_submit_job_returns_queued(ctx: SessionCtx) -> None:
    out = submit_job(ctx, tool="noop", params={"payload": "hi"})
    assert out["state"] == "queued"
    assert out["tool"] == "noop"
    assert out["workspace"] == ctx.workspace.name
    assert out["job_id"].startswith("job_")
    assert out["daemon_running"] is False
    assert out["execution_ready"] is False


def test_submit_job_reports_daemon_workspace_mismatch(
    ctx: SessionCtx,
    monkeypatch,
) -> None:
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

    out = submit_job(ctx, tool="noop", params={"payload": "hi"})

    assert out["ok"] is False
    assert out["code"] == "E_DAEMON_WORKSPACE_MISMATCH"
    assert out["details"]["requested_workspace"] == ctx.workspace.name


def test_get_status_unknown(ctx: SessionCtx) -> None:
    out = get_status(ctx, "job_deadbeef" * 4)
    assert out["ok"] is False


def test_wait_job_timeout_returns_record(ctx: SessionCtx) -> None:
    rec = submit_job(ctx, tool="noop", params={"payload": "x"})
    out = wait_job(ctx, rec["job_id"], timeout_ms=0)
    # Without a supervisor the job stays queued; wait_job returns the record
    # with `timed_out=True`.
    assert out["job_id"] == rec["job_id"]
    assert out.get("timed_out") is True


def test_get_results_unfinished_returns_error(ctx: SessionCtx) -> None:
    rec = submit_job(ctx, tool="noop", params={"payload": "y"})
    out = get_results(ctx, rec["job_id"])
    assert out["ok"] is False
    assert out["code"] == "E_NOT_FOUND"


def test_cancel_job_sets_state(ctx: SessionCtx) -> None:
    rec = submit_job(ctx, tool="noop", params={"payload": "c"})
    out = cancel_job(ctx, rec["job_id"])
    assert out["job_id"] == rec["job_id"]
    assert out["state"] in {"cancelled", "queued"}  # queued + cancel_requested flag


def test_list_artifacts_empty(ctx: SessionCtx) -> None:
    rec = submit_job(ctx, tool="noop", params={"payload": "a"})
    out = list_artifacts(ctx, rec["job_id"])
    assert isinstance(out, list)
    assert out == []


def test_get_logs_returns_string(ctx: SessionCtx) -> None:
    rec = submit_job(ctx, tool="noop", params={"payload": "l"})
    out = get_logs(ctx, rec["job_id"])
    assert isinstance(out, str)  # may be empty before the worker writes anything


def test_get_logs_unknown_job_returns_error_dict(ctx: SessionCtx) -> None:
    out = get_logs(ctx, "job_deadbeef" * 4)
    assert isinstance(out, dict)
    assert out["ok"] is False
    assert out["code"] == "E_NOT_FOUND"


def test_daemon_status_not_running(ctx: SessionCtx) -> None:
    # No supervisor in unit tests; expect `running: False`.
    out = daemon_status(ctx)
    assert out["running"] is False


def test_warm_tool_no_daemon(ctx: SessionCtx) -> None:
    # No supervisor: warm_tool reports E_DAEMON_UNAVAILABLE without blocking 180s.
    out = warm_tool(ctx, "noop", poll_timeout_s=0)
    assert out["ok"] is False
    assert out["code"] == "E_DAEMON_UNAVAILABLE"


def test_current_workspace(ctx: SessionCtx, tmp_workspace: Workspace) -> None:
    out = current_workspace(ctx)
    assert out["name"] == tmp_workspace.name
    assert Path(out["root"]) == tmp_workspace.root


def test_mcp_trajectory_lifecycle_and_wait(ctx: SessionCtx) -> None:
    started = start_trajectory(ctx, "design one")
    trajectory_id = started["trajectory_id"]
    rec = submit_job(
        ctx,
        tool="noop",
        params={"payload": "hi"},
        trajectory_id=trajectory_id,
    )
    JobStore(ctx.workspace).transition(rec["job_id"], JobState.CANCELLED)

    waited = wait_trajectory(ctx, trajectory_id, timeout_ms=0)
    status = get_trajectory_status(ctx, trajectory_id)["trajectory"]

    assert waited["trajectory_id"] == trajectory_id
    assert waited["execution_all_terminal"] is True
    assert waited["trajectory_complete"] is False
    assert status["job_count"] == 1
    assert status["workflow_count"] == 0


def test_mcp_render_report_records_trajectory_report(ctx: SessionCtx, tmp_path: Path) -> None:
    started = start_trajectory(ctx, "report one")
    trajectory_id = started["trajectory_id"]
    store = JobStore(ctx.workspace)
    job = store.submit(
        "submit_candidate",
        "1.0.0",
        {},
        trajectory_id=trajectory_id,
    )
    store.transition(job.job_id, JobState.QUEUED)
    store.transition(job.job_id, JobState.RUNNING)
    store.transition(
        job.job_id,
        JobState.COMPLETED,
        output={
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
    )

    out = render_report(ctx, trajectory_id, out=str(tmp_path / "report.html"))

    assert out["report_html"].endswith("report.html")
    assert out["trajectory"]["report_path"] == str(tmp_path / "report.html")
    status = get_trajectory_status(ctx, trajectory_id)["trajectory"]
    assert status["report_count"] == 1
    assert status["report_matches_final_submit"] is True

    duplicate = render_report(ctx, trajectory_id, out=str(tmp_path / "duplicate.html"))
    assert duplicate["ok"] is False
    assert duplicate["code"] == "E_CONFLICT"
    assert not (tmp_path / "duplicate.html").exists()


def test_mcp_render_report_relative_path_is_idempotent(
    ctx: SessionCtx,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = start_trajectory(ctx, "relative report")
    trajectory_id = started["trajectory_id"]
    store = JobStore(ctx.workspace)
    job = store.submit(
        "submit_candidate",
        "1.0.0",
        {},
        trajectory_id=trajectory_id,
    )
    store.transition(job.job_id, JobState.QUEUED)
    store.transition(job.job_id, JobState.RUNNING)
    store.transition(
        job.job_id,
        JobState.COMPLETED,
        output={
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
    )
    monkeypatch.chdir(tmp_path)

    first = render_report(ctx, trajectory_id, out="rel.html")
    second = render_report(ctx, trajectory_id, out="rel.html")

    assert first["report_html"] == str((tmp_path / "rel.html").resolve())
    assert second["report_html"] == str((tmp_path / "rel.html").resolve())
    status = get_trajectory_status(ctx, trajectory_id)["trajectory"]
    assert status["report_path"] == str((tmp_path / "rel.html").resolve())
    assert status["complete"] is True


def test_mcp_render_report_default_path_is_workspace_root(
    ctx: SessionCtx,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = start_trajectory(ctx, "default report path")
    trajectory_id = started["trajectory_id"]
    store = JobStore(ctx.workspace)
    job = store.submit(
        "submit_candidate",
        "1.0.0",
        {},
        trajectory_id=trajectory_id,
    )
    store.transition(job.job_id, JobState.QUEUED)
    store.transition(job.job_id, JobState.RUNNING)
    store.transition(
        job.job_id,
        JobState.COMPLETED,
        output={
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
    )
    monkeypatch.chdir(tmp_path)

    out = render_report(ctx, trajectory_id)

    expected = ctx.workspace.root / f"{trajectory_id}_report.html"
    assert out["report_html"] == str(expected)
    assert expected.exists()
    assert not (tmp_path / f"{trajectory_id}_report.html").exists()
    status = get_trajectory_status(ctx, trajectory_id)["trajectory"]
    assert status["report_path"] == str(expected)
    assert status["complete"] is True


def test_mcp_render_report_enriches_sequence_analysis_by_default(
    ctx: SessionCtx,
    tmp_path: Path,
) -> None:
    store = JobStore(ctx.workspace)
    job = store.submit(
        "boltzgen",
        "1.0.0",
        {},
    )
    store.transition(job.job_id, JobState.QUEUED)
    store.transition(job.job_id, JobState.RUNNING)
    store.transition(
        job.job_id,
        JobState.COMPLETED,
        output={
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_1",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "scores": {"boltzgen": {"confidence": 0.73}},
                }
            ]
        },
    )

    out = render_report(ctx, job.job_id, out=str(tmp_path / "report.html"))

    assert out["enrichment_enabled"] is True
    assert len(out["enrichment_job_ids"]) == 1
    html = Path(out["report_html"]).read_text()
    assert "GRAVY" in html


def test_mcp_render_bundle_report_default_path_is_workspace_root(
    ctx: SessionCtx,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = start_trajectory(ctx, "default bundle path")
    trajectory_id = started["trajectory_id"]
    store = JobStore(ctx.workspace)
    job = store.submit(
        "submit_candidate",
        "1.0.0",
        {},
        trajectory_id=trajectory_id,
    )
    store.transition(job.job_id, JobState.QUEUED)
    store.transition(job.job_id, JobState.RUNNING)
    store.transition(
        job.job_id,
        JobState.COMPLETED,
        output={
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
    )
    monkeypatch.chdir(tmp_path)

    out = render_report(ctx, trajectory_id, format="bundle")

    expected = ctx.workspace.root / f"{trajectory_id}_report_bundle" / "report.html"
    assert out["report_html"] == str(expected)
    assert expected.exists()
    assert not (tmp_path / f"{trajectory_id}_report_bundle").exists()
    status = get_trajectory_status(ctx, trajectory_id)["trajectory"]
    assert status["report_path"] == str(expected)
    assert status["complete"] is True


def test_list_workspaces_includes_session(ctx: SessionCtx) -> None:
    out = list_workspaces(ctx)
    assert isinstance(out, list)
    assert ctx.workspace.name in out


def test_list_workflows_returns_only_generic_bundled_examples(ctx: SessionCtx) -> None:
    items = list_workflows(ctx)
    assert isinstance(items, list)
    names = {it.get("name") for it in items}
    assert "noop_chain" in names
    assert not {
        "mpnn_then_refold",
        "semantic_constraint_to_family_sequences",
        "template_guided_design_prep",
        "rfdiffusion_backbone_design_round",
        "mutation_repair_round",
        "target_sequence_binder_design_round",
    } & names


def test_run_workflow_unknown_returns_not_found(ctx: SessionCtx) -> None:
    out = run_workflow(ctx, "definitely_not_a_workflow_xyz", {"x": 1})
    assert out.get("ok") is False
    assert out.get("code") == "E_NOT_FOUND"


def test_run_workflow_daemon_unavailable_preserves_workflow_context(
    ctx: SessionCtx,
) -> None:
    out = run_workflow(ctx, "noop_chain", {})

    assert out["ok"] is False
    assert out["code"] == "E_DAEMON_UNAVAILABLE"
    assert out["workflow_id"].startswith("wf_")
    assert out["state"] == "submitted"
    assert out["name"] == "noop_chain"
    assert [node["state"] for node in out["nodes"]] == ["pending", "pending", "pending"]
    assert [node["tool"] for node in out["nodes"]] == ["noop", "noop", "noop"]


def test_run_workflow_timeout_preserves_workflow_context(
    ctx: SessionCtx,
    monkeypatch,
) -> None:
    class FakeSupervisor:
        def is_running(self):
            return True

        def require_workspace(self, workspace_name):
            return {"running": True, "workspace": workspace_name}

        def runtime_info(self):
            return {"running": True, "workspace": ctx.workspace.name}

    monkeypatch.setattr(
        SessionCtx,
        "supervisor",
        staticmethod(lambda: FakeSupervisor()),
    )

    def _timeout(self, workflow_id, *, max_wait_s=3600, poll_s=0.25):
        self.store.update_state(workflow_id, "failed", finished=True)
        raise TimeoutError_("workflow timed out", details={"workflow_id": workflow_id})

    monkeypatch.setattr(
        "vibeproteinagent.workflows.WorkflowExecutor.run_sync",
        _timeout,
    )

    out = run_workflow(ctx, "noop_chain", {}, max_wait_s=0)

    assert out["ok"] is False
    assert out["code"] == "E_TIMEOUT"
    assert out["workflow_id"].startswith("wf_")
    assert out["state"] == "failed"
    assert out["nodes"]


def test_get_workflow_status_includes_node_failure_cause(
    tmp_workspace: Workspace,
) -> None:
    from vibeproteinagent.workflows import Workflow, WorkflowExecutor

    registry = ToolRegistry()
    registry.discover()
    wf = Workflow(name="bad_input", version="1.0.0")
    wf.node("design", tool="noop", inputs={"payload": 123})

    ex = WorkflowExecutor(tmp_workspace, registry, object())
    wf_id = ex.submit(wf, {})
    rec = ex.run_sync(wf_id, max_wait_s=5, poll_s=0.05)
    assert rec.state == "failed"

    out = get_workflow_status(
        SessionCtx(workspace=tmp_workspace, registry=registry), wf_id
    )
    node = out["nodes"][0]
    assert node["state"] == "failed"
    assert node["error_code"] == "E_INVALID_PARAMS"
    assert node["error_message"]
    assert isinstance(node["error_details"], dict)


def test_get_workflow_status_includes_completed_node_output(
    tmp_workspace: Workspace,
) -> None:
    from vibeproteinagent.workflows import Workflow, WorkflowExecutor

    registry = ToolRegistry()
    registry.discover()
    wf = Workflow(name="done_output", version="1.0.0")
    wf.node("design", tool="noop", inputs={"payload": "x"})

    ex = WorkflowExecutor(tmp_workspace, registry, None)
    wf_id = ex.submit(wf, {})
    node = ex.store.get_node(wf_id, "design")
    job = ex.jobs.submit(
        "noop",
        "1.0.0",
        {"payload": "x"},
        workspace_name=tmp_workspace.name,
        workflow_id=wf_id,
        workflow_node_id=node.node_id,
    )
    ex.jobs.transition(job.job_id, JobState.QUEUED)
    ex.jobs.claim_next_queued(worker_id=0, gpu_ids=[], job_id=job.job_id)
    ex.jobs.transition(job.job_id, JobState.COMPLETED, output={"payload": "x"})
    ex.store.update_node_state(wf_id, "design", "completed", job_id=job.job_id)

    out = get_workflow_status(
        SessionCtx(workspace=tmp_workspace, registry=registry), wf_id
    )

    assert out["nodes"][0]["output"] == {"payload": "x"}


def test_get_workflow_status_includes_node_row_output_without_job(
    tmp_workspace: Workspace,
) -> None:
    from vibeproteinagent.workflows import Workflow, WorkflowExecutor

    registry = ToolRegistry()
    registry.discover()
    wf = Workflow(name="aggregate_output", version="1.0.0")
    wf.fanout("fan", tool="noop", over="${params.items}", inputs={"payload": "${item}"})

    ex = WorkflowExecutor(tmp_workspace, registry, None)
    wf_id = ex.submit(wf, {"items": []})
    ex.store.update_node_state(wf_id, "fan", "completed", output=[])

    out = get_workflow_status(
        SessionCtx(workspace=tmp_workspace, registry=registry), wf_id
    )

    assert out["nodes"][0]["output"] == []
