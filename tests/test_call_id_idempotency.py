"""Regression tests for ``call_id`` idempotency vs. conflict (Codex review R6).

Architecture v1 §5 promises that a resubmission with the same ``call_id``
returns the original ``job_id``. The framework previously interpreted this
*regardless of submission contents* — so a buggy caller could re-use a
``call_id`` with different params and silently see stale results.

Phase 7 tightens the contract:

* Identical re-submission (same tool, version, params, workflow binding)
  is idempotent: original ``JobRecord`` returned.
* Any mismatch raises :class:`ConflictError` (``E_CONFLICT``) with a
  ``details.diff`` map naming each differing field.
"""

from __future__ import annotations

import pytest

from vibeproteinagent.core.errors import ConflictError
from vibeproteinagent.workspace import JobStore, Workspace


@pytest.fixture
def store(tmp_workspace: Workspace) -> JobStore:
    return JobStore(tmp_workspace)


def test_identical_resubmission_is_idempotent(store: JobStore) -> None:
    a = store.submit("noop", "1.0.0", {"payload": "a"}, call_id="k1",
                     workspace_name=store.workspace.name)
    b = store.submit("noop", "1.0.0", {"payload": "a"}, call_id="k1",
                     workspace_name=store.workspace.name)
    assert a.job_id == b.job_id


def test_mismatched_params_raises_conflict_with_diff(store: JobStore) -> None:
    store.submit("noop", "1.0.0", {"payload": "a"}, call_id="k2",
                 workspace_name=store.workspace.name)
    with pytest.raises(ConflictError) as exc_info:
        store.submit("noop", "1.0.0", {"payload": "b"}, call_id="k2",
                     workspace_name=store.workspace.name)
    err = exc_info.value
    assert err.code == "E_CONFLICT"
    assert "diff" in err.details
    assert "params_hash" in err.details["diff"]
    assert "existing_job_id" in err.details


def test_mismatched_tool_raises_conflict(store: JobStore) -> None:
    store.submit("noop", "1.0.0", {"payload": "a"}, call_id="k3",
                 workspace_name=store.workspace.name)
    with pytest.raises(ConflictError) as exc_info:
        # Same call_id, different tool name. (Even if no such adapter is loaded,
        # JobStore only persists the strings — registry validation happens upstream.)
        store.submit("proteinmpnn", "1.0.1", {"payload": "a"}, call_id="k3",
                     workspace_name=store.workspace.name)
    assert "tool_name" in exc_info.value.details["diff"]


def test_mismatched_workflow_binding_raises_conflict(store: JobStore) -> None:
    store.submit("noop", "1.0.0", {"payload": "a"}, call_id="k4",
                 workspace_name=store.workspace.name,
                 workflow_id="wf_a", workflow_node_id="n1")
    with pytest.raises(ConflictError) as exc_info:
        store.submit("noop", "1.0.0", {"payload": "a"}, call_id="k4",
                     workspace_name=store.workspace.name,
                     workflow_id="wf_b", workflow_node_id="n1")
    assert "workflow_id" in exc_info.value.details["diff"]
