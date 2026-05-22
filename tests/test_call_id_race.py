"""Codex round-2 RC6: ``call_id`` race handler routes through the same comparison
helper as the pre-check path.

Phase 7 added field-level diff to the normal pre-insert path. The SQLite
``IntegrityError`` race handler (concurrent submitters racing on the
``UNIQUE(call_id)`` constraint) re-fetched the existing row and *unconditionally*
returned it — letting a mismatched submission silently shadow the original.

The fix extracts the comparison into ``_assert_call_id_match`` and calls it
from both paths.
"""

from __future__ import annotations

import pytest

from vibeproteinagent.core.errors import ConflictError
from vibeproteinagent.workspace import JobStore, Workspace
from vibeproteinagent.workspace.store import _assert_call_id_match, params_hash


@pytest.fixture
def store(tmp_workspace: Workspace) -> JobStore:
    return JobStore(tmp_workspace)


# ---------------------------------------------------------------- helper unit


def test_assert_call_id_match_returns_existing_on_match(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {"payload": "p"}, call_id="k_match",
                       workspace_name=store.workspace.name)
    out = _assert_call_id_match(
        rec,
        call_id="k_match",
        tool=rec.tool_name, version=rec.tool_version, params_hash=rec.params_hash,
        workflow_id=None, workflow_node_id=None,
    )
    assert out.job_id == rec.job_id


def test_assert_call_id_match_raises_on_mismatch(store: JobStore) -> None:
    rec = store.submit("noop", "1.0.0", {"payload": "p"}, call_id="k_diff",
                       workspace_name=store.workspace.name)
    other_hash = params_hash({"payload": "different"})
    with pytest.raises(ConflictError) as exc:
        _assert_call_id_match(
            rec,
            call_id="k_diff",
            tool=rec.tool_name, version=rec.tool_version, params_hash=other_hash,
            workflow_id=None, workflow_node_id=None,
        )
    assert "params_hash" in exc.value.details["diff"]
    assert exc.value.details["existing_job_id"] == rec.job_id


# ---------------------------------------------------------------- race-handler path


def _patch_lying_pre_checks(
    store: JobStore, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    """Make both pre-checks (call_id + job_id) lie on first call, real on second.

    Codex round-4 RC1 added a deterministic ``job_id`` pre-check to
    :meth:`JobStore.submit`. Without lying about both, the deterministic
    pre-check would short-circuit before the call_id pre-check is even
    reached and the INSERT path under test would never run.
    """
    real_call = store._get_by_call_id
    real_job = store._get_by_job_id
    seen = {"call_calls": 0, "job_calls": 0}

    def lying_call(call_id: str):
        seen["call_calls"] += 1
        if seen["call_calls"] == 1:
            return None
        return real_call(call_id)

    def lying_job(job_id: str):
        seen["job_calls"] += 1
        if seen["job_calls"] == 1:
            return None
        return real_job(job_id)

    monkeypatch.setattr(store, "_get_by_call_id", lying_call)
    monkeypatch.setattr(store, "_get_by_job_id", lying_job)
    return seen


def test_race_handler_returns_existing_when_submission_matches(
    store: JobStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate the race: pre-check sees no row, INSERT collides, recovery wins.

    To exercise the IntegrityError branch we monkeypatch the two pre-checks
    so they return ``None`` (simulating two submitters racing on the UNIQUE
    constraint). The INSERT then collides naturally because the row already
    exists, and the recovery branch must fetch + compare.
    """
    rec = store.submit("noop", "1.0.0", {"payload": "p"}, call_id="k_race1",
                       workspace_name=store.workspace.name)
    seen = _patch_lying_pre_checks(store, monkeypatch)

    # Same submission → race handler must compare and return the original.
    out = store.submit("noop", "1.0.0", {"payload": "p"}, call_id="k_race1",
                       workspace_name=store.workspace.name)
    assert out.job_id == rec.job_id
    # Pre-check lied once and recovery branch fetched the truth — exactly
    # two reads against whichever helper the recovery uses.
    assert seen["call_calls"] >= 2 or seen["job_calls"] >= 2


def test_race_handler_raises_conflict_when_submission_differs(
    store: JobStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race handler must detect a mismatched re-submission and raise E_CONFLICT.

    Before Phase 8, the race handler returned the existing row regardless of
    whether the submission matched — silently shadowing.
    """
    rec = store.submit("noop", "1.0.0", {"payload": "p"}, call_id="k_race2",
                       workspace_name=store.workspace.name)
    _patch_lying_pre_checks(store, monkeypatch)

    with pytest.raises(ConflictError) as exc:
        store.submit("noop", "1.0.0", {"payload": "different"},
                     call_id="k_race2",
                     workspace_name=store.workspace.name)
    assert "diff" in exc.value.details
    assert "params_hash" in exc.value.details["diff"]
    assert exc.value.details["existing_job_id"] == rec.job_id
