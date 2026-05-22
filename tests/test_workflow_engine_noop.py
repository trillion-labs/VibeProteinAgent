from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.workflows import (
    Workflow,
    WorkflowExecutor,
    load_workflow,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _run_pending_jobs(store: JobStore, registry: ToolRegistry, stop: threading.Event) -> None:
    """Process queued jobs inline — minimal stand-in for the real supervisor."""
    import typing

    import torch

    from vibeproteinagent.adapters.context import null_context

    device = torch.device("cpu")
    weights_dir = Path("/tmp/vpa_test_weights")
    weights_dir.mkdir(exist_ok=True)

    while not stop.is_set():
        recs = store.list(state=JobState.QUEUED, limit=8)
        if not recs:
            time.sleep(0.02)
            continue
        for rec in recs:
            try:
                claimed = store.claim_next_queued(worker_id=0, gpu_ids=[],
                                                  job_id=rec.job_id)
                if claimed is None:
                    continue
                _, cls = registry.get(rec.tool_name, "*")
                adapter = cls(config={}, weights_dir=weights_dir, device=device)
                # Resolve the actual `params` type via get_type_hints to defeat
                # `from __future__ import annotations` stringification.
                try:
                    hints = typing.get_type_hints(adapter.run)
                except Exception:
                    hints = {}
                input_model = hints.get("params")
                params = (
                    input_model(**rec.params)
                    if input_model is not None
                    and hasattr(input_model, "model_validate")
                    else rec.params
                )
                out = adapter.run(params, null_context(rec.job_id))
                payload = (
                    out.model_dump()
                    if hasattr(out, "model_dump")
                    else dict(out) if hasattr(out, "__dict__") else out
                )
                store.transition(rec.job_id, JobState.COMPLETED, output=payload)
            except Exception as exc:  # pragma: no cover - log and mark failed
                store.transition(
                    rec.job_id, JobState.FAILED,
                    error_code="E_INTERNAL", error_message=str(exc),
                )


@pytest.fixture
def inline_supervisor(tmp_workspace: Workspace):
    """Background thread that simulates a supervisor for noop jobs."""
    store = JobStore(tmp_workspace)
    registry = ToolRegistry()
    registry.discover()
    stop = threading.Event()
    t = threading.Thread(target=_run_pending_jobs, args=(store, registry, stop),
                         daemon=True)
    t.start()
    try:
        yield object()  # truthy supervisor stand-in
    finally:
        stop.set()
        t.join(timeout=2.0)


def test_three_node_noop_workflow(
    tmp_workspace: Workspace,
    tmp_path: Path,
    inline_supervisor: object,
) -> None:
    """A 3-noop chain runs end-to-end and the second submission hits cache for all 3."""
    wf = Workflow(name="noop3", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "${params.msg}"})
    wf.node("b", tool="noop", inputs={"payload": "${outputs.a.payload}-b"})
    wf.node("c", tool="noop", inputs={"payload": "${outputs.b.payload}-c"})

    p = tmp_path / "noop3.yaml"
    wf.save(p)
    loaded = load_workflow(p)

    registry = ToolRegistry()
    registry.discover()

    # First run — fresh.
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(loaded, {"msg": "hello"})
    rec = ex.run_sync(wf_id, max_wait_s=30, poll_s=0.05)
    assert rec.state == "completed"

    nodes = ex.store.list_nodes(wf_id)
    by_id = {n.node_id: n for n in nodes}
    assert {n.state for n in nodes} == {"completed"}, [n.state for n in nodes]

    # Verify chained payloads.
    job_c = ex.jobs.get(by_id["c"].job_id)
    assert job_c.output["payload"] == "hello-b-c"

    # Second run — should hit cache for all three nodes.
    ex2 = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id2 = ex2.submit(loaded, {"msg": "hello"})
    assert wf_id2 != wf_id
    rec2 = ex2.run_sync(wf_id2, max_wait_s=30, poll_s=0.05)
    assert rec2.state == "completed"

    nodes2 = ex2.store.list_nodes(wf_id2)
    assert {n.state for n in nodes2} == {"cached"}, [n.state for n in nodes2]

    # Cached nodes reuse the same upstream job_ids, so outputs round-trip.
    by_id2 = {n.node_id: n for n in nodes2}
    cached_c = ex2.jobs.get(by_id2["c"].job_id)
    assert cached_c.output["payload"] == "hello-b-c"


def test_resume_failed_node_reattempt(
    tmp_workspace: Workspace,
    inline_supervisor: object,
) -> None:
    """Resuming a workflow whose nodes already completed is a no-op (all cached)."""
    wf = Workflow(name="noop1", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "resume-me"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)
    assert rec.state == "completed"

    # Re-run with the same workflow_id (resume). It should observe the existing
    # completed row and finish immediately.
    rec2 = ex.run_sync(wf_id, max_wait_s=5, poll_s=0.05)
    assert rec2.state == "completed"


def test_resume_ready_node_reattempts_submission(
    tmp_workspace: Workspace,
    inline_supervisor: object,
) -> None:
    """A crash after marking ready but before submitting must not spin forever."""
    wf = Workflow(name="noop-ready-resume", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "resume-ready"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {})
    ex.store.update_node_state(wf_id, "a", "ready")

    rec = ex.run_sync(wf_id, max_wait_s=10, poll_s=0.05)
    assert rec.state == "completed"
    node = ex.store.get_node(wf_id, "a")
    assert node.state == "completed"


def test_resume_preserves_failed_foreach_child(
    tmp_workspace: Workspace,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="noop-foreach-resume", version="1.0.0")
    wf.fanout("fan", tool="noop", over="${params.items}", inputs={"payload": "${item}"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {"items": ["bad"]})
    ex.store.update_node_state(wf_id, "fan", "skipped")
    ex.store.add_node(
        workflow_id=wf_id,
        node_id="fan[0]",
        tool_name="noop",
        tool_version="*",
        upstream=[],
        params_hash="",
        input_hash="",
        cache_key="",
        state="pending",
    )
    ex.store.update_node_state(
        wf_id,
        "fan[0]",
        "failed",
        error_code="E_INTERNAL",
        error_message="previous child failure",
    )

    rec = ex.run_sync(wf_id, max_wait_s=10, poll_s=0.05)

    assert rec.state == "failed"
    child = ex.store.get_node(wf_id, "fan[0]")
    assert child.state == "failed"
    assert child.error_message == "previous child failure"


def test_foreach_outputs_gate_aggregate_downstream(
    tmp_workspace: Workspace,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="noop-foreach-aggregate", version="1.0.0")
    wf.fanout("fan", tool="noop", over="${params.items}", inputs={"payload": "${item}"})
    wf.node(
        "aggregate",
        tool="noop",
        inputs={"payload": "${outputs.fan[0].payload}-${outputs.fan[1].payload}"},
    )

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {"items": ["a", "b"]})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)

    assert rec.state == "completed"
    aggregate = ex.store.get_node(wf_id, "aggregate")
    assert aggregate.state == "completed"
    job = ex.jobs.get(aggregate.job_id)
    assert job.output["payload"] == "a-b"


def test_empty_foreach_outputs_allow_downstream_aggregate(
    tmp_workspace: Workspace,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="noop-empty-foreach", version="1.0.0")
    wf.fanout("fan", tool="noop", over="${params.items}", inputs={"payload": "${item}"})
    wf.node("aggregate", tool="noop", inputs={"payload": "items=${outputs.fan}"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {"items": []})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)

    assert rec.state == "completed"
    aggregate = ex.store.get_node(wf_id, "aggregate")
    job = ex.jobs.get(aggregate.job_id)
    assert job.output["payload"] == "items=[]"


def test_resume_empty_foreach_preserves_durable_aggregate_output(
    tmp_workspace: Workspace,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="noop-empty-foreach-resume", version="1.0.0")
    wf.fanout("fan", tool="noop", over="${params.items}", inputs={"payload": "${item}"})
    wf.node("aggregate", tool="noop", inputs={"payload": "items=${outputs.fan}"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {"items": []})
    ex.store.update_node_state(wf_id, "fan", "completed", output=[])

    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)

    assert rec.state == "completed"
    fan = ex.store.get_node(wf_id, "fan")
    aggregate = ex.store.get_node(wf_id, "aggregate")
    assert fan.output == []
    job = ex.jobs.get(aggregate.job_id)
    assert job.output["payload"] == "items=[]"


def test_hyphenated_node_id_static_upstream_runs(
    tmp_workspace: Workspace,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="noop-hyphen-id", version="1.0.0")
    wf.node("prep-node", tool="noop", inputs={"payload": "x"})
    wf.node("use", tool="noop", inputs={"payload": "${outputs.prep-node.payload}-ok"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)

    assert rec.state == "completed"
    job = ex.jobs.get(ex.store.get_node(wf_id, "use").job_id)
    assert job.output["payload"] == "x-ok"


def test_stop_policy_cancels_running_sibling_jobs(
    tmp_workspace: Workspace,
) -> None:
    wf = Workflow(name="noop-stop-cancels-sibling", version="1.0.0")
    wf.node("running_sibling", tool="noop", inputs={"payload": "queued"})
    wf.node("bad", tool="noop", inputs={"payload": 123}, on_failure="stop")

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, object())
    wf_id = ex.submit(wf, {})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)

    assert rec.state == "failed"
    sibling = ex.store.get_node(wf_id, "running_sibling")
    bad = ex.store.get_node(wf_id, "bad")
    assert sibling.state == "skipped"
    assert bad.state == "failed"
    job = ex.jobs.get(sibling.job_id)
    assert job.state is JobState.CANCELLED


def test_run_sync_timeout_marks_workflow_failed_durably(
    tmp_workspace: Workspace,
) -> None:
    wf = Workflow(name="noop-timeout", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "x"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, object())
    wf_id = ex.submit(wf, {})

    from vibeproteinagent.core.errors import TimeoutError_

    with pytest.raises(TimeoutError_):
        ex.run_sync(wf_id, max_wait_s=0, poll_s=0.01)

    wf_rec = ex.store.get(wf_id)
    node = ex.store.get_node(wf_id, "a")
    assert wf_rec.state == "failed"
    assert wf_rec.finished_at is not None
    assert node.state == "failed"
    assert node.error_code == "E_TIMEOUT"


def test_run_without_supervisor_marks_workflow_pending(
    tmp_workspace: Workspace,
) -> None:
    """No supervisor → submit() still writes rows but nodes can't run."""
    wf = Workflow(name="noop-pending", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "x"})
    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, None)
    wf_id = ex.submit(wf, {})
    nodes = ex.store.list_nodes(wf_id)
    assert len(nodes) == 1
    assert nodes[0].state == "pending"
