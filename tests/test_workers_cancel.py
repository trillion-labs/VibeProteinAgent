from __future__ import annotations

import pytest

from vibeproteinagent.core.errors import ConflictError
from vibeproteinagent.workspace import JobState, JobStore, Workspace


@pytest.fixture
def store(tmp_workspace: Workspace) -> JobStore:
    return JobStore(tmp_workspace)


def test_request_cancel_sets_db_flag(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {"x": 1}, workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    assert store.is_cancel_requested(rec.job_id) is False
    store.request_cancel(rec.job_id)
    assert store.is_cancel_requested(rec.job_id) is True


def test_request_cancel_queued_transitions_to_cancelled(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    r = store.request_cancel(rec.job_id)
    assert r.state is JobState.CANCELLED
    assert store.is_cancel_requested(rec.job_id) is True


def test_request_cancel_submitted_also_transitions(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    r = store.request_cancel(rec.job_id)
    assert r.state is JobState.CANCELLED


def test_request_cancel_terminal_raises(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED)
    with pytest.raises(ConflictError):
        store.request_cancel(rec.job_id)


def test_is_cancel_requested_missing_job_returns_false(store: JobStore) -> None:
    assert store.is_cancel_requested("job_" + "0" * 32) is False


def test_cancel_alias_calls_request_cancel(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {}, workspace_name=store.workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    r = store.cancel(rec.job_id)
    assert r.state is JobState.CANCELLED
