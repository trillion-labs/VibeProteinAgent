"""Codex review B2: when a worker dies mid-job the supervisor must transition
the in-flight job to ``failed`` with ``E_INTERNAL``.

Before Phase 7 the supervisor logged ``worker_closed_mid_job`` and walked
away. The job sat in ``running`` forever and any workflow waiting on it
hung until ``max_wait_s``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import JobState, JobStore, Workspace

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


def _read_worker_pid(runtime: Path) -> int | None:
    pid_file = runtime / "worker_-1.pid"
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None


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


def test_worker_killed_mid_job_marks_job_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("crash")
    store = JobStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_target,
        args=("crash", str(runtime), str(home)),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"
        # Submit noop and queue it.
        rec = store.submit("noop", "1.0.0", {"payload": "victim"},
                           workspace_name=ws.name)
        store.transition(rec.job_id, JobState.QUEUED)

        # Wait until the job is dispatched (running) AND the worker pid is known.
        deadline = time.time() + 10.0
        worker_pid: int | None = None
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state is JobState.RUNNING:
                worker_pid = _read_worker_pid(runtime)
                if worker_pid is not None:
                    break
            time.sleep(0.05)
        assert worker_pid is not None, (
            "never observed running state + worker pid file"
        )

        # SIGKILL the worker process from outside.
        os.kill(worker_pid, signal.SIGKILL)

        # Within a few seconds the supervisor must mark the job failed.
        deadline = time.time() + 10.0
        terminal = None
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state is JobState.FAILED:
                terminal = cur
                break
            time.sleep(0.1)
        assert terminal is not None, (
            f"worker crash did not produce a terminal state: {store.get(rec.job_id).state}"
        )
        assert terminal.error_code == "E_INTERNAL"
        assert "worker process died mid-job" in (terminal.error_message or "")
    finally:
        _stop(proc)
