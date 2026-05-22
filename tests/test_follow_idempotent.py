"""Codex round-2 Re9: ``vpa run --follow`` on an idempotent re-submission emits
``idempotent_hit`` instead of fabricating fresh ``submitted``/``queued`` events.

Phase 7 wired ``--follow`` to always emit ``submitted`` and ``queued`` for any
returned ``JobRecord`` — including idempotent hits where no new job was
actually queued. The contract violation: a downstream agent watching the
stream would see "queued at T+0" for a job that's actually been running for
ten minutes.

Phase 8 detects the idempotent hit (``rec.submitted_at`` predates the
caller's pre-submit timestamp) and emits a single ``idempotent_hit`` event
with the cached job's current state.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.workspace import Workspace


def test_follow_emits_idempotent_hit_on_resubmit(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    Workspace.create("default")
    # First submission — fresh queue path.
    a = runner.invoke(app, ["run", "noop", "payload=x", "--call-id", "k_idem", "--json"])
    assert a.exit_code == 0, a.stdout
    job_id = json.loads(a.stdout)["data"]["job_id"]

    # Second submission, same call_id, with --follow.
    # No supervisor running → the follow loop heart-beats and exits at deadline.
    b = runner.invoke(
        app,
        [
            "run", "noop", "payload=x",
            "--call-id", "k_idem",
            "--follow", "--json",
            "--timeout-ms", "200",
        ],
    )
    assert b.exit_code == 0, b.stdout
    lines = [line for line in b.stdout.splitlines() if line.strip()]
    events = [json.loads(line) for line in lines]
    types_seen = [e["type"] for e in events]

    assert events, "expected at least one stream event"
    # The very first event for an idempotent hit must be `idempotent_hit`,
    # NOT `submitted` or `queued`.
    first = events[0]
    assert first["type"] == "idempotent_hit", first
    assert first["job_id"] == job_id
    assert first["payload"]["call_id"] == "k_idem"
    # No ``submitted``/``queued`` fabrications.
    assert "submitted" not in types_seen
    assert "queued" not in types_seen


def test_follow_fresh_submission_still_emits_submitted_and_queued(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    """Sanity: the fresh-submit path is unchanged."""
    Workspace.create("default")
    result = runner.invoke(
        app,
        [
            "run", "noop", "payload=fresh",
            "--call-id", "k_fresh",
            "--follow", "--json",
            "--timeout-ms", "200",
        ],
    )
    assert result.exit_code == 0, result.stdout
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    types_seen = [e["type"] for e in events]
    assert "submitted" in types_seen
    assert "queued" in types_seen
    assert "idempotent_hit" not in types_seen


def test_run_envelope_marks_idempotent_hit_in_data(
    runner: CliRunner, isolated_vpa_home: Path
) -> None:
    """Non-follow path: ``data.idempotent_hit: true`` on the second submit."""
    Workspace.create("default")
    a = runner.invoke(app, ["run", "noop", "payload=x", "--call-id", "k_dup", "--json"])
    b = runner.invoke(app, ["run", "noop", "payload=x", "--call-id", "k_dup", "--json"])
    assert a.exit_code == 0 and b.exit_code == 0
    a_data = json.loads(a.stdout)["data"]
    b_data = json.loads(b.stdout)["data"]
    # Fresh submit has no idempotent_hit field.
    assert "idempotent_hit" not in a_data
    # Re-submit flips it on.
    assert b_data.get("idempotent_hit") is True
    assert a_data["job_id"] == b_data["job_id"]
