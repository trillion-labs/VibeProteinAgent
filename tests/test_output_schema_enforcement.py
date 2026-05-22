"""Codex round-2 RC3: enforce ``descriptor.output_schema`` post-adapter.

Architecture v1 §6: every tool descriptor declares an ``output_schema``.
Phase 7 only validated *inputs*; the worker's output dict went straight
into ``output_json`` without checks. A buggy adapter could persist the
wrong shape and the failure surfaced only when a downstream node tried
to consume it.

Phase 8 wires :func:`validate_output` into the supervisor's
``_finalize_completed_job`` path: a schema violation marks the job
``failed`` with ``E_INVALID_OUTPUT`` and ``details.violations``.

Two layers of test:

* unit-level — :func:`validate_output` rejects/accepts directly;
* finalize-level — drive the supervisor's helper without an event loop
  and observe the JobStore transition to ``failed``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibeproteinagent.adapters import (
    InvalidOutputError,
    validate_output,
)
from vibeproteinagent.workers.supervisor import Supervisor
from vibeproteinagent.workspace import (
    ArtifactStore,
    JobState,
    JobStore,
    Workspace,
)

# ---------------------------------------------------------------- unit


def _desc(schema: object) -> SimpleNamespace:
    """Stand-in descriptor for the validator helpers."""
    return SimpleNamespace(name="synthetic", output_schema=schema)


def test_validate_output_passes_when_required_field_present() -> None:
    desc = _desc({"type": "object", "required": ["x"]})
    validate_output(desc, {"x": 1})  # no raise


def test_validate_output_raises_with_violations_on_missing_required() -> None:
    desc = _desc({"type": "object", "required": ["x"]})
    with pytest.raises(InvalidOutputError) as exc_info:
        validate_output(desc, {"y": 1})
    err = exc_info.value
    assert err.code == "E_INVALID_OUTPUT"
    assert "violations" in err.details
    fields = [v["field"] for v in err.details["violations"]]
    # JSON Schema reports missing required as a root-level error mentioning 'x'.
    assert any("x" in v["message"] for v in err.details["violations"]) or "x" in fields


def test_validate_output_skips_when_schema_is_wildcard() -> None:
    desc = _desc("*")
    validate_output(desc, {"anything": "goes"})


def test_validate_output_skips_when_schema_empty() -> None:
    desc = _desc({})
    validate_output(desc, {"anything": "goes"})


# ---------------------------------------------------------------- supervisor finalize


def test_finalize_marks_failed_on_output_schema_violation(
    tmp_workspace: Workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive Supervisor._finalize_completed_job with a synthetic descriptor.

    No spawn, no event loop — we cheat by constructing a Supervisor
    instance and patching ``_descriptor_for`` to hand back a synthetic
    descriptor with a strict ``output_schema``.
    """
    store = JobStore(tmp_workspace)
    artifact_store = ArtifactStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "x"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)

    sup = Supervisor.__new__(Supervisor)  # bypass __init__
    sup.workspace = tmp_workspace
    sup._log = SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    monkeypatch.setattr(
        sup, "_descriptor_for",
        lambda name, version: SimpleNamespace(
            name=name,
            output_schema={"type": "object", "required": ["x"]},
        ),
    )

    sup._finalize_completed_job(
        store, artifact_store, rec.job_id,
        {"output": {"y": 1}, "artifacts": []},
    )

    final = store.get(rec.job_id)
    assert final.state is JobState.FAILED
    assert final.error_code == "E_INVALID_OUTPUT"
    # The error message should name the offending field.
    assert "x" in (final.error_message or "")


def test_finalize_completes_when_output_matches_schema(
    tmp_workspace: Workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JobStore(tmp_workspace)
    artifact_store = ArtifactStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "x"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)

    sup = Supervisor.__new__(Supervisor)
    sup.workspace = tmp_workspace
    sup._log = SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    monkeypatch.setattr(
        sup, "_descriptor_for",
        lambda name, version: SimpleNamespace(
            name=name,
            output_schema={"type": "object", "required": ["x"]},
        ),
    )

    sup._finalize_completed_job(
        store, artifact_store, rec.job_id,
        {"output": {"x": 42}, "artifacts": []},
    )

    final = store.get(rec.job_id)
    assert final.state is JobState.COMPLETED
    assert final.output == {"x": 42}
