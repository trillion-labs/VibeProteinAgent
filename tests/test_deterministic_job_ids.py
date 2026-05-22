"""Codex round-4 RC1: deterministic job IDs end-to-end through JobStore.

Per **Architecture v1 — Errata 2** the framework derives ``job_id`` from
``sha256(canonical_submission)`` so the same submission resolves to the
same id. Side-effects on the submit lifecycle:

* identical re-submission returns the existing job (no new row);
* a failed/cancelled job is re-queued *in place* under the same job_id
  (history preserved, ``retry_count`` bumped);
* ``salt`` (the CLI's ``--force-new``) yields a fresh id even when params
  are identical;
* ``call_id`` collisions against a mismatched submission still raise
  ``E_CONFLICT`` (Phase 7 contract preserved).
"""

from __future__ import annotations

import pytest

from vibeproteinagent.core.errors import ConflictError
from vibeproteinagent.workspace import JobState, JobStore, Workspace
from vibeproteinagent.workspace.ids import derive_job_id


@pytest.fixture
def store(tmp_workspace: Workspace) -> JobStore:
    return JobStore(tmp_workspace)


def _submit(store: JobStore, **overrides) -> object:
    args = {
        "tool": "noop", "version": "1.0.0",
        "params": {"payload": "x"},
        "workspace_name": store.workspace.name,
    }
    args.update(overrides)
    return store.submit(**args)


def test_same_submission_returns_same_job_id_without_call_id(store: JobStore) -> None:
    a = _submit(store)
    b = _submit(store)
    assert a.job_id == b.job_id
    # And the id is the canonical sha256-derived form.
    expected = derive_job_id(
        tool="noop", version="1.0.0", params={"payload": "x"},
        workspace=store.workspace.name,
    )
    assert a.job_id == expected


def test_different_params_yield_different_job_ids(store: JobStore) -> None:
    a = _submit(store, params={"payload": "a"})
    b = _submit(store, params={"payload": "b"})
    assert a.job_id != b.job_id


def test_force_new_salt_yields_a_fresh_job_id(store: JobStore) -> None:
    a = _submit(store)
    b = _submit(store, salt="rerun-001")
    assert a.job_id != b.job_id
    # Re-using the same salt collapses to the same id again — salt is part of
    # the canonical input, not a randomiser baked into submit().
    b2 = _submit(store, salt="rerun-001")
    assert b.job_id == b2.job_id


def test_failed_resubmission_requeues_same_job_id(store: JobStore) -> None:
    a = _submit(store, max_retries=0)
    store.transition(a.job_id, JobState.QUEUED)
    store.transition(a.job_id, JobState.RUNNING)
    failed = store.transition(
        a.job_id, JobState.FAILED,
        error_code="E_INTERNAL", error_message="boom",
        error_details={"trace": "stub"},
    )
    assert failed.state is JobState.FAILED
    assert failed.retry_count == 0

    # Re-submission with the same inputs re-queues in place.
    b = _submit(store)
    assert b.job_id == a.job_id
    assert b.state is JobState.QUEUED
    assert b.retry_count == 1
    assert b.error_code is None
    assert b.error_message is None
    assert b.error_details is None
    assert b.queued_at is not None


def test_cancelled_resubmission_also_requeues_same_job_id(store: JobStore) -> None:
    a = _submit(store)
    store.transition(a.job_id, JobState.QUEUED)
    cancelled = store.request_cancel(a.job_id)
    assert cancelled.state is JobState.CANCELLED
    assert store.is_cancel_requested(a.job_id) is True

    # Re-submission clears cancel-requested and re-queues.
    b = _submit(store)
    assert b.job_id == a.job_id
    assert b.state is JobState.QUEUED
    assert b.retry_count == 1
    assert store.is_cancel_requested(a.job_id) is False


def test_completed_resubmission_returns_cached(store: JobStore) -> None:
    a = _submit(store)
    store.transition(a.job_id, JobState.QUEUED)
    store.transition(a.job_id, JobState.RUNNING)
    store.transition(a.job_id, JobState.COMPLETED, output={"score": 0.5})

    b = _submit(store)
    assert b.job_id == a.job_id
    assert b.state is JobState.COMPLETED
    assert b.output == {"score": 0.5}
    assert b.retry_count == 0  # not bumped — no new run started


def test_call_id_conflict_still_raised_against_deterministic_match(
    store: JobStore,
) -> None:
    """Phase 7 behaviour preserved: same call_id with a different submission
    still raises E_CONFLICT even though the deterministic id derivation now
    catches identical submissions on its own.
    """
    _submit(store, call_id="k_pin", params={"payload": "a"})
    with pytest.raises(ConflictError) as exc_info:
        _submit(store, call_id="k_pin", params={"payload": "b"})
    err = exc_info.value
    assert err.code == "E_CONFLICT"
    assert "params_hash" in err.details["diff"]


def test_workflow_binding_changes_deterministic_id(store: JobStore) -> None:
    a = _submit(store, workflow_id="wf_a", workflow_node_id="n1")
    b = _submit(store, workflow_id="wf_b", workflow_node_id="n1")
    c = _submit(store, workflow_id="wf_a", workflow_node_id="n2")
    assert len({a.job_id, b.job_id, c.job_id}) == 3
