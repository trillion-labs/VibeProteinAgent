"""Codex review R8: workflow nodes must run their inputs through the same
JSON-Schema validator the CLI uses before submitting a job.

The old executor would happily submit a node with malformed inputs; the
worker would then raise ``E_INVALID_PARAMS`` from inside the worker
process and the workflow would fail awkwardly. After Phase 7 the executor
catches the mismatch up-front via :func:`adapters.descriptor.validate_params`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.workflows import Workflow, WorkflowExecutor, load_workflow
from vibeproteinagent.workflows.store import WorkflowStore
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _run_pending_jobs(
    store: JobStore, registry: ToolRegistry, stop: threading.Event
) -> None:
    """Inline noop runner shared with test_workflow_engine_noop."""
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
            claimed = store.claim_next_queued(
                worker_id=0, gpu_ids=[], job_id=rec.job_id
            )
            if claimed is None:
                continue
            _descriptor, cls = registry.get(rec.tool_name, "*")
            adapter = cls(config={}, weights_dir=weights_dir, device=device)
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
                out.model_dump() if hasattr(out, "model_dump") else dict(out)
            )
            store.transition(rec.job_id, JobState.COMPLETED, output=payload)


@pytest.fixture
def inline_supervisor(tmp_workspace: Workspace):
    store = JobStore(tmp_workspace)
    registry = ToolRegistry()
    registry.discover()
    stop = threading.Event()
    t = threading.Thread(
        target=_run_pending_jobs, args=(store, registry, stop), daemon=True
    )
    t.start()
    try:
        yield object()
    finally:
        stop.set()
        t.join(timeout=2.0)


def test_workflow_rejects_wrong_typed_input_before_dispatch(
    tmp_workspace: Workspace, tmp_path: Path, inline_supervisor: object,
) -> None:
    """A workflow node declaring ``noop`` with int payload (schema wants str)
    must fail validation in the executor, never reach the worker."""
    wf = Workflow(name="bad_input", version="1.0.0")
    wf.node("design", tool="noop", inputs={"payload": 123})  # int, schema says str

    p = tmp_path / "bad.yaml"
    wf.save(p)
    loaded = load_workflow(p)
    registry = ToolRegistry()
    registry.discover()

    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(loaded, {})
    rec = ex.run_sync(wf_id, max_wait_s=10, poll_s=0.05)
    assert rec.state == "failed"

    nodes = ex.store.list_nodes(wf_id)
    by_id = {n.node_id: n for n in nodes}
    assert by_id["design"].state == "failed"
    assert by_id["design"].error_code == "E_INVALID_PARAMS"
    assert by_id["design"].error_message
    assert isinstance(by_id["design"].error_details, dict)
    # And critically: no job was ever submitted for it, because validation
    # is the executor's responsibility now (not the worker's).
    jobs = JobStore(tmp_workspace).list(workflow_id=wf_id)
    assert not any(j.tool_name == "noop" for j in jobs), (
        f"workflow node leaked into job table despite validation failure: {jobs}"
    )


def test_workflow_resolution_failure_preserves_node_cause(
    tmp_workspace: Workspace, inline_supervisor: object,
) -> None:
    wf = Workflow(name="bad_resolution", version="1.0.0")
    wf.node("design", tool="noop", inputs={"payload": "${params.missing}"})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {})
    rec = ex.run_sync(wf_id, max_wait_s=10, poll_s=0.05)
    assert rec.state == "failed"

    node = ex.store.get_node(wf_id, "design")
    assert node.state == "failed"
    assert node.error_code == "E_INVALID_PARAMS"
    assert "params.missing" in node.error_message
    assert node.error_details["expr"] == "params.missing"
    assert node.error_details["missing"] == "missing"


def test_workflow_unknown_tool_preserves_node_cause(
    tmp_workspace: Workspace, inline_supervisor: object,
) -> None:
    wf = Workflow(name="bad_tool", version="1.0.0")
    wf.node("design", tool="no_such_tool", inputs={})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {})
    rec = ex.run_sync(wf_id, max_wait_s=10, poll_s=0.05)
    assert rec.state == "failed"

    node = ex.store.get_node(wf_id, "design")
    assert node.state == "failed"
    assert node.error_code == "E_UNKNOWN_TOOL"
    assert node.error_message


def test_cli_workflow_status_json_includes_node_failure_cause(
    tmp_workspace: Workspace, inline_supervisor: object, runner,
) -> None:
    from vibeproteinagent.cli.main import app

    wf = Workflow(name="bad_input_cli", version="1.0.0")
    wf.node("design", tool="noop", inputs={"payload": 123})

    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(wf, {})
    rec = ex.run_sync(wf_id, max_wait_s=10, poll_s=0.05)
    assert rec.state == "failed"

    result = runner.invoke(
        app,
        ["workflow", "status", wf_id, "--workspace", tmp_workspace.name, "--json"],
    )
    assert result.exit_code == 0, result.stdout
    node = json.loads(result.stdout)["data"]["nodes"][0]
    assert node["state"] == "failed"
    assert node["error_code"] == "E_INVALID_PARAMS"
    assert node["error_message"]
    assert isinstance(node["error_details"], dict)


def test_workflow_store_backfills_node_error_columns_for_legacy_db(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy"
    root.mkdir()
    ws = Workspace("legacy", root)
    conn = sqlite3.connect(ws.db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE workflows (
              workflow_id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              version TEXT NOT NULL,
              descriptor TEXT NOT NULL,
              params_json TEXT NOT NULL,
              state TEXT NOT NULL,
              submitted_at INTEGER NOT NULL,
              finished_at INTEGER
            );
            CREATE TABLE workflow_nodes (
              workflow_id TEXT NOT NULL,
              node_id TEXT NOT NULL,
              tool_name TEXT NOT NULL,
              tool_version TEXT NOT NULL,
              upstream TEXT NOT NULL,
              params_hash TEXT NOT NULL,
              input_hash TEXT NOT NULL,
              cache_key TEXT NOT NULL,
              state TEXT NOT NULL,
              job_id TEXT,
              PRIMARY KEY (workflow_id, node_id)
            );
            INSERT INTO workflows VALUES (
              'wf_legacy', 'legacy', '1.0.0', 'name: legacy', '{}',
              'failed', 1, 2
            );
            INSERT INTO workflow_nodes VALUES (
              'wf_legacy', 'n1', 'noop', '1.0.0', '[]', '', '', '',
              'failed', NULL
            );
            """
        )
    finally:
        conn.close()

    store = WorkflowStore(ws)
    node = store.get_node("wf_legacy", "n1")
    assert node.error_code is None
    store.update_node_state(
        "wf_legacy",
        "n1",
        "failed",
        error_code="E_INVALID_PARAMS",
        error_message="legacy cause",
        error_details={"x": 1},
    )
    node = store.get_node("wf_legacy", "n1")
    assert node.error_code == "E_INVALID_PARAMS"
    assert node.error_details == {"x": 1}


def test_workflow_accepts_well_typed_input(
    tmp_workspace: Workspace, tmp_path: Path, inline_supervisor: object,
) -> None:
    """Sanity check: a well-typed node still runs to completion through the
    shared validator path."""
    wf = Workflow(name="good_input", version="1.0.0")
    wf.node("design", tool="noop", inputs={"payload": "ok"})

    p = tmp_path / "ok.yaml"
    wf.save(p)
    loaded = load_workflow(p)
    registry = ToolRegistry()
    registry.discover()

    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(loaded, {})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)
    assert rec.state == "completed", rec.state
