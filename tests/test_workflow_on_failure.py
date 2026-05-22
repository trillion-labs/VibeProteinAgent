"""Codex review R9: ``on_failure: stop|continue|skip`` cascade semantics.

The workflow descriptor has accepted the field since Phase 4.4, but only
``stop`` was actually implemented. Phase 7 adds the cascade logic:

* ``stop`` (default): on a node failure, all unstarted nodes are skipped
  and the workflow halts with ``state=failed``.
* ``continue``: only direct dependents of the failed node are skipped;
  independent branches keep running. Workflow ends ``failed`` because
  *any* failed node poisons the final state.
* ``skip``: transitive descendants of the failed node are skipped;
  independent branches keep running. Workflow ends ``failed``.

Each test uses a node whose inputs fail JSON Schema validation, which
triggers ``_node_failed`` without needing a real worker.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.workflows import Workflow, WorkflowExecutor, load_workflow
from vibeproteinagent.workspace import JobState, JobStore, Workspace

# -- inline supervisor (copied tightly from test_workflow_engine_noop) --


def _run_pending_jobs(store: JobStore, registry: ToolRegistry, stop: threading.Event) -> None:
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
            payload = out.model_dump() if hasattr(out, "model_dump") else dict(out)
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


def _run_workflow(
    tmp_workspace: Workspace,
    tmp_path: Path,
    inline_supervisor: object,
    wf: Workflow,
) -> tuple[str, dict[str, str], str]:
    p = tmp_path / f"{wf.name}.yaml"
    wf.save(p)
    loaded = load_workflow(p)
    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(loaded, {})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)
    nodes = {n.node_id: n.state for n in ex.store.list_nodes(wf_id)}
    return wf_id, nodes, rec.state


def test_stop_skips_downstream(
    tmp_workspace: Workspace, tmp_path: Path, inline_supervisor: object,
) -> None:
    """Three-node chain a → b → c: b fails validation, c is skipped.

    "b" carries an explicit ``${outputs.a.payload}`` reference so the executor
    doesn't try "b" until "a" is in a success state — that pins the race the
    naïve "b fails before a completes" test would otherwise expose.
    """
    wf = Workflow(name="stop_chain", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "ok"})
    # b: noop schema has additionalProperties=false; an extra key plus a real
    # ${outputs.a.payload} reference forces the executor to wait for "a"
    # before validating, then fail cleanly.
    wf.node(
        "b", tool="noop",
        inputs={"payload": "${outputs.a.payload}", "extra": "boom"},
    )
    wf.node("c", tool="noop", inputs={"payload": "${outputs.b.payload}"})
    _, nodes, state = _run_workflow(tmp_workspace, tmp_path, inline_supervisor, wf)
    assert state == "failed"
    assert nodes["a"] == "completed"
    assert nodes["b"] == "failed"
    assert nodes["c"] == "skipped"


def test_continue_skips_only_direct_dependents(
    tmp_workspace: Workspace, tmp_path: Path, inline_supervisor: object,
) -> None:
    """Parallel branches: failing-branch only skips its direct dependent.

    DAG:    a — b(fail, on_failure=continue) — c(depends on b)
                                                d(independent of b)
    Expect: a completed, b failed, c skipped, d completed.
    """
    wf = Workflow(name="continue_branches", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "ok"})
    wf.node(
        "b", tool="noop",
        inputs={"payload": 0},  # validation fails
        on_failure="continue",
    )
    wf.node("c", tool="noop", inputs={"payload": "${outputs.b.payload}"})
    wf.node("d", tool="noop", inputs={"payload": "${outputs.a.payload}-d"})
    _, nodes, state = _run_workflow(tmp_workspace, tmp_path, inline_supervisor, wf)
    assert state == "failed"  # any failed node ⇒ failed workflow
    assert nodes["a"] == "completed"
    assert nodes["b"] == "failed"
    assert nodes["c"] == "skipped"
    assert nodes["d"] == "completed"


def test_skip_cascades_transitive_descendants(
    tmp_workspace: Workspace, tmp_path: Path, inline_supervisor: object,
) -> None:
    """DAG where the failed node has two dependents (one transitive).

    Topology:
        a — b(fail, on_failure=skip)
              └─ c (depends on b)
                   └─ e (depends on c, transitive on b)
        d (independent)
    Expect: a, d completed; b failed; c, e skipped.
    """
    wf = Workflow(name="skip_transitive", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "ok"})
    wf.node(
        "b", tool="noop",
        inputs={"payload": 0},
        on_failure="skip",
    )
    wf.node("c", tool="noop", inputs={"payload": "${outputs.b.payload}"})
    wf.node("e", tool="noop", inputs={"payload": "${outputs.c.payload}"})
    wf.node("d", tool="noop", inputs={"payload": "${outputs.a.payload}"})
    _, nodes, state = _run_workflow(tmp_workspace, tmp_path, inline_supervisor, wf)
    assert state == "failed"
    assert nodes["a"] == "completed"
    assert nodes["b"] == "failed"
    assert nodes["c"] == "skipped"
    assert nodes["e"] == "skipped"
    assert nodes["d"] == "completed"
