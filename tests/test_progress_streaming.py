"""End-to-end + unit tests for progress streaming via IPC.

The e2e test spawns a real CPU-only supervisor + 1 worker, submits a noop job,
and verifies the worker emitted at least 3 progress events that the supervisor
forwarded to ``events.ndjson`` in the §10 NDJSON shape.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import JobState, JobStore, Workspace

pytestmark = pytest.mark.slow


def _supervisor_subprocess_target(
    workspace_name: str,
    runtime_dir_s: str,
    vpa_home_s: str,
    num_workers: int,
) -> None:
    """Module-level so it's picklable by spawn."""
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vibeproteinagent.workers.supervisor import supervisor_main

    gpu_ids = [-1] * num_workers
    supervisor_main(workspace_name, len(gpu_ids), gpu_ids)


def _wait_socket(socket_path: Path, *, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            return True
        time.sleep(0.1)
    return False


def _stop(proc: mp.Process) -> None:
    if not proc.is_alive():
        return
    try:
        proc.terminate()
        proc.join(timeout=10.0)
    except Exception:
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def test_noop_run_emits_progress_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Submit a noop job; assert events.ndjson contains >=3 progress events."""
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("progress")
    store = JobStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=("progress", str(runtime), str(home), 1),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"

        rec = store.submit("noop", "1.0.0", {"payload": "hello"},
                           workspace_name=ws.name)
        store.transition(rec.job_id, JobState.QUEUED)

        deadline = time.time() + 15.0
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state.value in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.1)
        assert store.get(rec.job_id).state is JobState.COMPLETED

        # The supervisor should have persisted progress frames to events.ndjson.
        events_path = ws.job_dir(rec.job_id) / "events.ndjson"
        assert events_path.exists(), "events.ndjson was never written"
        lines = [
            json.loads(ln)
            for ln in events_path.read_text().splitlines() if ln.strip()
        ]
        progress_events = [e for e in lines if e.get("type") == "progress"]
        assert len(progress_events) >= 3, (
            f"expected >=3 progress events, got {len(progress_events)}: {lines}"
        )
        fractions = [e["payload"]["fraction"] for e in progress_events]
        # The final emission must be 1.0 (either from noop or the drainer flush).
        assert fractions[-1] == pytest.approx(1.0), fractions
        # Every event must be in [0.0, 1.0] and have job_id + ts.
        for e in progress_events:
            assert e["job_id"] == rec.job_id
            assert "ts" in e
            f = e["payload"]["fraction"]
            assert 0.0 <= float(f) <= 1.0
    finally:
        _stop(proc)


def test_stream_body_to_ndjson_translation() -> None:
    """Unit test the §10 NDJSON shape translator the supervisor uses."""
    from vibeproteinagent.workers.supervisor import _stream_body_to_ndjson

    out = _stream_body_to_ndjson(
        {"event": "progress", "job_id": "job_x", "fraction": 0.5, "ts": 12345},
        "job_x",
    )
    assert out["type"] == "progress"
    assert out["job_id"] == "job_x"
    assert out["payload"] == {"fraction": 0.5}
    assert out["ts"] == 12.345  # ms -> s

    out2 = _stream_body_to_ndjson(
        {"event": "retrying", "attempt": 2, "reason": "E_TIMEOUT", "ts": 1000},
        "job_y",
    )
    assert out2["type"] == "retrying"
    assert out2["payload"] == {"attempt": 2, "reason": "E_TIMEOUT"}

    # Garbage in, None out.
    assert _stream_body_to_ndjson({"no_event": True}, "job_z") is None
