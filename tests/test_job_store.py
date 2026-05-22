from __future__ import annotations

import pytest

from vibeproteinagent.core.errors import ConflictError, NotFoundError
from vibeproteinagent.workspace import JobState, JobStore, Workspace


@pytest.fixture
def store(tmp_workspace: Workspace) -> JobStore:
    return JobStore(tmp_workspace)


def test_submit_creates_submitted_job(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {"x": 1, "y": 2}, workspace_name=store.workspace.name)
    assert rec.state is JobState.SUBMITTED
    assert rec.tool_name == "noop"
    assert rec.tool_version == "1.0.0"
    assert rec.params == {"x": 1, "y": 2}
    assert rec.params_hash and len(rec.params_hash) == 64
    assert rec.submitted_at > 0
    assert rec.workspace == store.workspace.name


def test_submit_idempotent_on_call_id(store: JobStore) -> None:
    """Equivalent re-submission with the same ``call_id`` returns the original.

    Per Architecture v1 §5: idempotency requires that the tool, version,
    params, and workflow binding all match. A mismatched re-use is the
    job of :func:`test_submit_call_id_conflict_on_mismatch` (Codex review R6).
    """
    a = store.submit("noop", "1.0.0", {"a": 1}, call_id="idem-1",
                     workspace_name=store.workspace.name)
    b = store.submit("noop", "1.0.0", {"a": 1}, call_id="idem-1",
                     workspace_name=store.workspace.name)
    assert a.job_id == b.job_id
    assert b.params == {"a": 1}


def test_get_by_job_id_and_call_id(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, call_id="alias-1",
                       workspace_name=store.workspace.name)
    by_jid = store.get(rec.job_id)
    by_call = store.get("alias-1")
    assert by_jid.job_id == by_call.job_id == rec.job_id


def test_get_missing_raises(store: JobStore) -> None:
    with pytest.raises(NotFoundError):
        store.get("never-existed")


def test_legal_transitions(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    r1 = store.transition(rec.job_id, JobState.QUEUED)
    assert r1.state is JobState.QUEUED
    assert r1.queued_at is not None
    r2 = store.transition(rec.job_id, JobState.RUNNING, worker_id=3, gpu_ids=[2, 3])
    assert r2.state is JobState.RUNNING
    assert r2.worker_id == 3
    assert r2.gpu_ids == [2, 3]
    assert r2.started_at is not None
    r3 = store.transition(rec.job_id, JobState.COMPLETED,
                          output={"score": 0.42}, cache_hit=True)
    assert r3.state is JobState.COMPLETED
    assert r3.output == {"score": 0.42}
    assert r3.finished_at is not None
    assert r3.cache_hit is True


def test_illegal_transition_raises(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    with pytest.raises(ConflictError):
        store.transition(rec.job_id, JobState.RUNNING)  # must go through queued

    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED)
    with pytest.raises(ConflictError):
        # COMPLETED is terminal — no transitions out.
        store.transition(rec.job_id, JobState.QUEUED)


def test_failed_then_retry_increments(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, max_retries=2,
                       workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.FAILED,
                     error_code="E_INTERNAL", error_message="boom")
    r = store.transition(rec.job_id, JobState.QUEUED)
    assert r.state is JobState.QUEUED
    assert r.retry_count == 1


def test_list_filters(store: JobStore) -> None:
    a = store.submit("noop", "1.0.0", {"i": 1}, workspace_name=store.workspace.name)
    b = store.submit("foo", "0.1.0", {"i": 2}, workspace_name=store.workspace.name)
    store.transition(a.job_id, JobState.QUEUED)
    store.transition(b.job_id, JobState.QUEUED)
    store.transition(a.job_id, JobState.RUNNING)
    store.transition(a.job_id, JobState.COMPLETED)

    completed = store.list(state=JobState.COMPLETED)
    assert {r.job_id for r in completed} == {a.job_id}

    only_foo = store.list(tool="foo")
    assert {r.job_id for r in only_foo} == {b.job_id}

    all_jobs = store.list()
    assert len(all_jobs) == 2
    # default order: submitted_at_desc — b submitted after a.
    assert all_jobs[0].job_id == b.job_id


def test_cancel_from_queued(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    r = store.cancel(rec.job_id)
    assert r.state is JobState.CANCELLED


def test_cancel_from_running_sets_flag(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    r = store.cancel(rec.job_id)
    # State stays running until worker honors it.
    assert r.state is JobState.RUNNING
    assert store.is_cancel_requested(rec.job_id) is True
    # Worker now reports it terminated as cancelled.
    final = store.transition(rec.job_id, JobState.CANCELLED)
    assert final.state is JobState.CANCELLED
    # Cancel-request flag is durable (DB-backed); it persists past terminal state.
    assert store.is_cancel_requested(rec.job_id) is True


def test_cancel_terminal_raises(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED)
    with pytest.raises(ConflictError):
        store.cancel(rec.job_id)
