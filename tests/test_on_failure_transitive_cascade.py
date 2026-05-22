"""Codex round-2 RC7: ``on_failure: continue`` cascades to transitive descendants.

Phase 7 implemented ``continue`` as "skip only direct dependents", which left
second-order descendants ``pending`` forever and the workflow would time out.
The Phase 8 cascade is transitive for both ``continue`` and ``skip`` — they
differ from ``stop`` only in that independent branches keep running.

DAG:    a → b → c           branch with the failure
        d → e               independent branch
b fails with ``on_failure: continue``. Expect a completed; b failed; c skipped
(transitive); d and e completed.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.workflows import Workflow, WorkflowExecutor, load_workflow
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _run_pending_jobs(
    store: JobStore, registry: ToolRegistry, stop: threading.Event
) -> None:
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


def test_continue_cascades_to_transitive_descendants(
    tmp_workspace: Workspace, tmp_path: Path, inline_supervisor: object,
) -> None:
    """Three-node failing chain plus an independent branch.

    DAG:
        a → b(fail, on_failure=continue) → c       (chain rooted at b)
        d → e                                       (independent)

    Phase 8 contract: c is the *transitive* dependent of b (via the
    explicit ``${outputs.b.payload}`` reference) and must be skipped, even
    though ``continue`` would historically have only skipped direct dependents.
    """
    wf = Workflow(name="continue_transitive", version="1.0.0")
    wf.node("a", tool="noop", inputs={"payload": "ok"})
    wf.node(
        "b", tool="noop",
        inputs={"payload": 0},  # validation fails (noop wants string)
        on_failure="continue",
    )
    # c depends directly on b.
    wf.node("c", tool="noop", inputs={"payload": "${outputs.b.payload}"})
    # Independent branch.
    wf.node("d", tool="noop", inputs={"payload": "ok-d"})
    wf.node("e", tool="noop", inputs={"payload": "${outputs.d.payload}-e"})

    p = tmp_path / f"{wf.name}.yaml"
    wf.save(p)
    loaded = load_workflow(p)
    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(loaded, {})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)
    nodes = {n.node_id: n.state for n in ex.store.list_nodes(wf_id)}

    assert rec.state == "failed"
    assert nodes["a"] == "completed"
    assert nodes["b"] == "failed"
    assert nodes["c"] == "skipped"
    assert nodes["d"] == "completed"
    assert nodes["e"] == "completed"


def test_continue_cascades_through_two_hops(
    tmp_workspace: Workspace, tmp_path: Path, inline_supervisor: object,
) -> None:
    """The cascade is forward-transitive over the full DAG.

    DAG: x → y(fail, continue) → z → w
    Expect: x completed; y failed; z and w skipped.
    """
    wf = Workflow(name="continue_two_hops", version="1.0.0")
    wf.node("x", tool="noop", inputs={"payload": "ok"})
    wf.node(
        "y", tool="noop",
        inputs={"payload": 0},
        on_failure="continue",
    )
    wf.node("z", tool="noop", inputs={"payload": "${outputs.y.payload}"})
    wf.node("w", tool="noop", inputs={"payload": "${outputs.z.payload}"})

    p = tmp_path / f"{wf.name}.yaml"
    wf.save(p)
    loaded = load_workflow(p)
    registry = ToolRegistry()
    registry.discover()
    ex = WorkflowExecutor(tmp_workspace, registry, inline_supervisor)
    wf_id = ex.submit(loaded, {})
    rec = ex.run_sync(wf_id, max_wait_s=15, poll_s=0.05)
    nodes = {n.node_id: n.state for n in ex.store.list_nodes(wf_id)}

    assert rec.state == "failed"
    assert nodes["x"] == "completed"
    assert nodes["y"] == "failed"
    assert nodes["z"] == "skipped"
    assert nodes["w"] == "skipped"
