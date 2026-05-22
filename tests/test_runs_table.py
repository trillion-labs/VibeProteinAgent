"""Codex review Re12: the ``runs`` table records per-attempt run metadata.

It has been in the schema since Phase 3.1 but no code wrote to it until
Phase 7. The supervisor now opens a row on :meth:`_dispatch_to_worker`
and updates it after the job reaches a terminal state.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import JobState, JobStore, RunStore, Workspace

pytestmark = pytest.mark.slow


def _supervisor_target(
    workspace_name: str, runtime_dir_s: str, vpa_home_s: str
) -> None:
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vibeproteinagent.workers.supervisor import supervisor_main

    supervisor_main(workspace_name, 1, [-1])


def _wait_socket(path: Path, *, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
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


def test_noop_job_produces_runs_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("runs")
    store = JobStore(ws)
    runs = RunStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_target,
        args=("runs", str(runtime), str(home)),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"
        rec = store.submit("noop", "1.0.0", {"payload": "hi"},
                           workspace_name=ws.name)
        store.transition(rec.job_id, JobState.QUEUED)

        deadline = time.time() + 15.0
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                break
            time.sleep(0.1)
        terminal = store.get(rec.job_id)
        assert terminal.state is JobState.COMPLETED, terminal.error_message

        # Wait briefly for the supervisor's post-completion run-close.
        deadline = time.time() + 5.0
        rows = runs.for_job(rec.job_id)
        while time.time() < deadline and (not rows or rows[0].finished_at is None):
            time.sleep(0.1)
            rows = runs.for_job(rec.job_id)
        assert len(rows) == 1, f"expected exactly one runs row, got: {rows}"
        r = rows[0]
        assert r.job_id == rec.job_id
        assert r.attempt == 1
        assert r.worker_id == -1  # CPU worker
        assert r.exit_state == "completed"
        assert r.duration_ms is not None
        assert r.duration_ms > 0
    finally:
        _stop(proc)
