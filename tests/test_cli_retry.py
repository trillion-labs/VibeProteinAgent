"""CLI surface tests for --retry option and `vpa retry <job_id>` command."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.core.errors import ErrorCode
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def test_run_with_retry_persists_max_retries(
    runner: CliRunner, isolated_vpa_home: Path,
) -> None:
    """``vpa run noop --retry 2 ...`` must persist max_retries=2 on the job row."""
    Workspace.create("default")
    result = runner.invoke(
        app, ["run", "noop", "payload=hi", "--retry", "2", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)["data"]
    # Verify by reading the row directly from the workspace DB.
    store = JobStore(Workspace.resolve("default"))
    rec = store.get(data["job_id"])
    assert rec.max_retries == 2
    assert rec.retry_count == 0


def test_run_unknown_tool_with_retry_does_not_create_job(
    runner: CliRunner, isolated_vpa_home: Path,
) -> None:
    """``vpa run nope --retry 2`` must fail with E_UNKNOWN_TOOL before submit.

    Acceptance: domain errors never trigger auto-retry; no job is created.
    """
    Workspace.create("default")
    result = runner.invoke(app, ["run", "nope", "--retry", "2", "--json"])
    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == ErrorCode.UNKNOWN_TOOL
    # And no job rows for that tool in the workspace.
    store = JobStore(Workspace.resolve("default"))
    rows = store.list(tool="nope")
    assert rows == []


def test_retry_command_requeues_failed_job(
    runner: CliRunner, isolated_vpa_home: Path,
) -> None:
    """``vpa retry <job_id>`` flips a failed job back to queued and clears stale fields."""
    Workspace.create("default")
    store = JobStore(Workspace.resolve("default"))
    rec = store.submit("noop", "1.0.0", {"payload": "x"}, max_retries=3,
                       workspace_name="default")
    store.transition(rec.job_id, JobState.QUEUED)
    store.claim_next_queued(worker_id=0, gpu_ids=[0], job_id=rec.job_id)
    store.transition(rec.job_id, JobState.FAILED,
                     error_code=ErrorCode.TIMEOUT, error_message="oops")

    result = runner.invoke(app, ["retry", rec.job_id, "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    job = payload["data"]["job"]
    assert job["state"] == "queued"
    assert job["retry_count"] == 1
    assert job["error_code"] is None
    assert job["worker_id"] is None


def test_retry_command_rejects_when_max_retries_exhausted(
    runner: CliRunner, isolated_vpa_home: Path,
) -> None:
    """``vpa retry`` against a job with max_retries=0 must fail with E_CONFLICT."""
    Workspace.create("default")
    store = JobStore(Workspace.resolve("default"))
    rec = store.submit("noop", "1.0.0", {"payload": "x"}, max_retries=0,
                       workspace_name="default")
    store.transition(rec.job_id, JobState.QUEUED)
    store.claim_next_queued(worker_id=0, gpu_ids=[0], job_id=rec.job_id)
    store.transition(rec.job_id, JobState.FAILED, error_code=ErrorCode.TIMEOUT)

    result = runner.invoke(app, ["retry", rec.job_id, "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"]["code"] == ErrorCode.CONFLICT
    assert payload["error"]["details"].get("reason") == "max_retries_exceeded"
