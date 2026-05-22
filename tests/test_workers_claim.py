from __future__ import annotations

import pytest

from vibeproteinagent.workspace import JobState, JobStore, Workspace


@pytest.fixture
def store(tmp_workspace: Workspace) -> JobStore:
    return JobStore(tmp_workspace)


def test_claim_returns_none_when_no_queued(store: JobStore) -> None:
    assert store.claim_next_queued(worker_id=0, gpu_ids=[0]) is None


def test_claim_transitions_to_running(store: JobStore) -> None:
    a = store.submit("noop", "1.0.0", {"i": 1}, workspace_name=store.workspace.name)
    store.transition(a.job_id, JobState.QUEUED)

    claimed = store.claim_next_queued(worker_id=3, gpu_ids=[5])
    assert claimed is not None
    assert claimed.job_id == a.job_id
    assert claimed.state is JobState.RUNNING
    assert claimed.worker_id == 3
    assert claimed.gpu_ids == [5]
    assert claimed.started_at is not None


def test_claim_returns_none_after_only_queued_consumed(store: JobStore) -> None:
    a = store.submit("noop", "1.0.0", {"i": 1}, workspace_name=store.workspace.name)
    store.transition(a.job_id, JobState.QUEUED)
    first = store.claim_next_queued(worker_id=0, gpu_ids=[0])
    assert first is not None
    second = store.claim_next_queued(worker_id=1, gpu_ids=[1])
    assert second is None


def test_claim_respects_priority_then_fifo(store: JobStore) -> None:
    a = store.submit("noop", "1.0.0", {"i": 1}, priority="low",
                     workspace_name=store.workspace.name)
    b = store.submit("noop", "1.0.0", {"i": 2}, priority="high",
                     workspace_name=store.workspace.name)
    c = store.submit("noop", "1.0.0", {"i": 3}, priority="normal",
                     workspace_name=store.workspace.name)
    for r in (a, b, c):
        store.transition(r.job_id, JobState.QUEUED)
    first = store.claim_next_queued(worker_id=0, gpu_ids=[0])
    assert first is not None and first.job_id == b.job_id
    second = store.claim_next_queued(worker_id=1, gpu_ids=[1])
    assert second is not None and second.job_id == c.job_id
    third = store.claim_next_queued(worker_id=2, gpu_ids=[2])
    assert third is not None and third.job_id == a.job_id
    fourth = store.claim_next_queued(worker_id=3, gpu_ids=[3])
    assert fourth is None


def test_bump_retry_increments(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, max_retries=2,
                       workspace_name=store.workspace.name)
    n1 = store.bump_retry(rec.job_id)
    assert n1 == 1
    n2 = store.bump_retry(rec.job_id)
    assert n2 == 2


def test_bump_retry_over_max_raises(store: JobStore) -> None:
    from vibeproteinagent.core.errors import ConflictError

    rec = store.submit("noop", "1.0.0", {}, max_retries=0,
                       workspace_name=store.workspace.name)
    with pytest.raises(ConflictError):
        store.bump_retry(rec.job_id)
