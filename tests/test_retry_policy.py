"""Retry-policy tests for JobStore.requeue_failed + RETRYABLE_ERRORS."""
from __future__ import annotations

import pytest

from vibeproteinagent.core.errors import (
    RETRYABLE_ERRORS,
    ConflictError,
    ErrorCode,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace


@pytest.fixture
def store(tmp_workspace: Workspace) -> JobStore:
    return JobStore(tmp_workspace)


# ---------------------------------------------------------------- constants


def test_retryable_errors_set_matches_spec() -> None:
    """RETRYABLE_ERRORS must include exactly the four codes from Architecture v1 §5."""
    assert RETRYABLE_ERRORS == frozenset({
        ErrorCode.TIMEOUT,
        ErrorCode.NO_GPU,
        ErrorCode.VRAM_EXHAUSTED,
        ErrorCode.DAEMON_UNAVAILABLE,
    })
    # Domain errors must NOT be retryable.
    assert ErrorCode.INVALID_PARAMS not in RETRYABLE_ERRORS
    assert ErrorCode.UNKNOWN_TOOL not in RETRYABLE_ERRORS


# ---------------------------------------------------------------- requeue_failed


def _make_failed(store: JobStore, *, max_retries: int = 0,
                 error_code: str = ErrorCode.TIMEOUT,
                 payload: str = "x") -> str:
    """Submit, queue, run, and fail a job; return its job_id.

    ``payload`` is part of the deterministic-id input (Architecture v1 —
    Errata 2). Callers that need multiple distinct failed jobs in the
    same workspace must vary it so each submission resolves to its own
    job_id rather than collapsing onto an existing row.
    """
    rec = store.submit("noop", "1.0.0", {"payload": payload},
                       max_retries=max_retries,
                       workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    # Claim the specific job (works regardless of how many other queued jobs exist).
    store.claim_next_queued(worker_id=0, gpu_ids=[0], job_id=rec.job_id)
    store.transition(rec.job_id, JobState.FAILED,
                     error_code=error_code, error_message="simulated")
    return rec.job_id


def test_requeue_failed_clears_stale_fields(store: JobStore) -> None:
    """A retry must scrub worker_id, gpu_ids, started_at, finished_at, error_*."""
    job_id = _make_failed(store, max_retries=2)
    before = store.get(job_id)
    assert before.state is JobState.FAILED
    assert before.error_code == ErrorCode.TIMEOUT
    assert before.worker_id is not None
    assert before.gpu_ids == [0]
    assert before.started_at is not None
    assert before.finished_at is not None

    updated = store.requeue_failed(job_id)
    assert updated.state is JobState.QUEUED
    assert updated.retry_count == 1
    assert updated.error_code is None
    assert updated.error_message is None
    assert updated.worker_id is None
    assert updated.gpu_ids == []
    assert updated.started_at is None
    assert updated.finished_at is None
    assert updated.queued_at is not None  # fresh queued_at on re-queue


def test_requeue_failed_increments_retry_count(store: JobStore) -> None:
    """retry_count goes 0 → 1 → 2; on the 3rd attempt with max_retries=2 it conflicts."""
    job_id = _make_failed(store, max_retries=2)
    store.requeue_failed(job_id)  # 1
    # Re-fail (would happen in a real run) and try again.
    store.claim_next_queued(worker_id=0, gpu_ids=[0], job_id=job_id)
    store.transition(job_id, JobState.FAILED, error_code=ErrorCode.TIMEOUT)
    store.requeue_failed(job_id)  # 2 — equals max_retries
    store.claim_next_queued(worker_id=0, gpu_ids=[0], job_id=job_id)
    store.transition(job_id, JobState.FAILED, error_code=ErrorCode.TIMEOUT)
    with pytest.raises(ConflictError) as ei:
        store.requeue_failed(job_id)
    assert ei.value.details.get("reason") == "max_retries_exceeded"


def test_requeue_failed_rejects_when_max_retries_zero(store: JobStore) -> None:
    """The default max_retries=0 means failed jobs are NOT re-queueable."""
    job_id = _make_failed(store, max_retries=0)
    with pytest.raises(ConflictError) as ei:
        store.requeue_failed(job_id)
    assert ei.value.details.get("reason") == "max_retries_exceeded"


def test_requeue_failed_rejects_non_failed(store: JobStore) -> None:
    """Calling requeue_failed on a completed/queued job raises E_CONFLICT."""
    rec = store.submit("noop", "1.0.0", {"i": 9}, max_retries=2,
                       workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    with pytest.raises(ConflictError) as ei:
        store.requeue_failed(rec.job_id)
    assert ei.value.details.get("reason") == "not_failed"
    assert ei.value.details.get("state") == "queued"


def test_supervisor_retry_logic_only_for_retryable_codes(
    tmp_workspace: Workspace,
) -> None:
    """The supervisor's auto-retry path must skip non-retryable error codes.

    Exercises ``Supervisor._maybe_auto_retry`` directly against a real workspace
    without spawning workers.
    """
    from vibeproteinagent.workers.supervisor import Supervisor

    store = JobStore(tmp_workspace)

    # Build a Supervisor instance without invoking _async_main (no workers).
    sup = Supervisor.__new__(Supervisor)
    sup.workspace_name = tmp_workspace.name
    sup.workspace = tmp_workspace

    # Case 1: retryable error code (E_TIMEOUT) → re-queued.
    job_id = _make_failed(store, max_retries=2,
                          error_code=ErrorCode.TIMEOUT, payload="case1")
    events_path = tmp_workspace.job_dir(job_id) / "events.ndjson"
    sup._maybe_auto_retry(job_id, events_path)
    after = store.get(job_id)
    assert after.state is JobState.QUEUED, after.state
    assert after.retry_count == 1
    assert events_path.exists()
    body = events_path.read_text()
    assert '"retrying"' in body
    assert '"E_TIMEOUT"' in body

    # Case 2: domain error code (E_INVALID_PARAMS) → must NOT retry.
    # Distinct payload so it doesn't collapse onto case-1's deterministic id.
    job_id2 = _make_failed(store, max_retries=2,
                           error_code=ErrorCode.INVALID_PARAMS,
                           payload="case2")
    events2 = tmp_workspace.job_dir(job_id2) / "events.ndjson"
    sup._maybe_auto_retry(job_id2, events2)
    after2 = store.get(job_id2)
    assert after2.state is JobState.FAILED  # unchanged
    assert after2.retry_count == 0


def test_supervisor_retry_respects_max_retries(tmp_workspace: Workspace) -> None:
    """When retry_count == max_retries, _maybe_auto_retry leaves the job failed."""
    from vibeproteinagent.workers.supervisor import Supervisor

    store = JobStore(tmp_workspace)
    sup = Supervisor.__new__(Supervisor)
    sup.workspace_name = tmp_workspace.name
    sup.workspace = tmp_workspace

    job_id = _make_failed(store, max_retries=0, error_code=ErrorCode.TIMEOUT)
    events = tmp_workspace.job_dir(job_id) / "events.ndjson"
    sup._maybe_auto_retry(job_id, events)
    after = store.get(job_id)
    assert after.state is JobState.FAILED
    assert after.retry_count == 0
