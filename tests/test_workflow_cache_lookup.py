"""Codex round-2 C12: ``find_cached_node`` orders by ``jobs.finished_at DESC``.

Phase 7's lookup did ``WHERE cache_key = ? AND state IN (...)`` with no
``ORDER BY``, leaving SQLite to return rows in undefined order. When two
nodes shared a cache_key (same key reused across workflows) the lookup
could return the older row. Phase 8 joins ``jobs`` and orders by
``finished_at DESC`` so the most recent successful node always wins.
"""

from __future__ import annotations

import time

from vibeproteinagent.workflows.store import WorkflowStore
from vibeproteinagent.workspace import (
    JobState,
    JobStore,
    Workspace,
)


def test_find_cached_node_returns_most_recent_match(
    tmp_workspace: Workspace,
) -> None:
    store = WorkflowStore(tmp_workspace)
    jobs = JobStore(tmp_workspace)

    # Two completed jobs with the same cache_key but different finished_at.
    older = jobs.submit("noop", "1.0.0", {"payload": "p"},
                        workspace_name=tmp_workspace.name)
    jobs.transition(older.job_id, JobState.QUEUED)
    jobs.transition(older.job_id, JobState.RUNNING)
    jobs.transition(older.job_id, JobState.COMPLETED, output={"k": "older"})

    # Inject the older one first.
    wf_id_1 = store.create(
        name="wf_a", version="1.0.0", descriptor_yaml="x: y", params={},
    )
    store.add_node(
        workflow_id=wf_id_1, node_id="n1",
        tool_name="noop", tool_version="1.0.0",
        upstream=[], params_hash="ph", input_hash="ih",
        cache_key="ck_shared", state="completed",
    )
    store.update_node_state(wf_id_1, "n1", "completed", job_id=older.job_id)

    # Force a perceptible time gap on the finished_at column.
    time.sleep(0.005)

    newer = jobs.submit("noop", "1.0.0", {"payload": "q"},
                        workspace_name=tmp_workspace.name)
    jobs.transition(newer.job_id, JobState.QUEUED)
    jobs.transition(newer.job_id, JobState.RUNNING)
    jobs.transition(newer.job_id, JobState.COMPLETED, output={"k": "newer"})

    wf_id_2 = store.create(
        name="wf_b", version="1.0.0", descriptor_yaml="x: y", params={},
    )
    store.add_node(
        workflow_id=wf_id_2, node_id="n2",
        tool_name="noop", tool_version="1.0.0",
        upstream=[], params_hash="ph", input_hash="ih",
        cache_key="ck_shared", state="completed",
    )
    store.update_node_state(wf_id_2, "n2", "completed", job_id=newer.job_id)

    found = store.find_cached_node("ck_shared")
    assert found is not None
    # Ordering by jobs.finished_at DESC → newer wins.
    assert found.job_id == newer.job_id
    assert found.node_id == "n2"


def test_find_cached_node_returns_none_when_missing(
    tmp_workspace: Workspace,
) -> None:
    store = WorkflowStore(tmp_workspace)
    assert store.find_cached_node("never_inserted") is None
