from __future__ import annotations

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
    cpu_only: bool,
) -> None:
    """Module-level so it's picklable by spawn."""
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vibeproteinagent.workers.supervisor import (
        resolve_default_gpu_ids,
        supervisor_main,
    )
    gpu_ids = resolve_default_gpu_ids(num_workers)
    if cpu_only:
        gpu_ids = [-1] * max(1, num_workers)
    supervisor_main(workspace_name, len(gpu_ids), gpu_ids)


def _spawn_supervisor(workspace_name: str, runtime_dir: Path, vpa_home: Path,
                      *, num_workers: int = 1, cpu_only: bool = True) -> mp.Process:
    """Spawn the supervisor in a child process with isolated env."""
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=(workspace_name, str(runtime_dir), str(vpa_home), num_workers, cpu_only),
        daemon=False,
    )
    proc.start()
    return proc


def _wait_socket(socket_path: Path, *, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            return True
        time.sleep(0.1)
    return False


def _stop_supervisor(proc: mp.Process, socket_path: Path) -> None:
    if not proc.is_alive():
        return
    # Best-effort polite shutdown via signal.
    try:
        proc.terminate()
        proc.join(timeout=10.0)
    except Exception:
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def test_daemon_e2e_noop_run_and_cancel(tmp_path: Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """Full pipeline: supervisor starts, runs a noop job, then cancels another."""
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("e2e")
    store = JobStore(ws)

    socket_path = runtime / "supervisor.sock"
    proc = _spawn_supervisor("e2e", runtime, home, num_workers=1, cpu_only=True)
    try:
        assert _wait_socket(socket_path, timeout_s=20.0), "supervisor never came up"

        # ---------------- happy path: job should complete via worker ----
        rec = store.submit("noop", "1.0.0", {"payload": "hello"},
                           workspace_name=ws.name)
        store.transition(rec.job_id, JobState.QUEUED)

        deadline = time.time() + 15.0
        terminal = None
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state.value in ("completed", "failed", "cancelled"):
                terminal = cur
                break
            time.sleep(0.2)
        assert terminal is not None, f"job stuck at {store.get(rec.job_id).state}"
        assert terminal.state is JobState.COMPLETED, terminal.error_message
        assert terminal.output is not None
        assert terminal.output["payload"] == "hello"

        # ---------------- cancel path: cancel a queued job ----
        rec2 = store.submit("noop", "1.0.0", {"payload": "bye"},
                            workspace_name=ws.name)
        store.transition(rec2.job_id, JobState.QUEUED)
        # Immediate cancel — likely catches it queued or running.
        store.request_cancel(rec2.job_id)

        deadline = time.time() + 10.0
        terminal2 = None
        while time.time() < deadline:
            cur = store.get(rec2.job_id)
            if cur.state.value in ("completed", "failed", "cancelled"):
                terminal2 = cur
                break
            time.sleep(0.1)
        assert terminal2 is not None
        assert terminal2.state in (JobState.CANCELLED, JobState.COMPLETED)
        # The race is OK: if it completed first (the noop adapter is very fast),
        # the test still demonstrates request_cancel flips the DB flag.
        assert store.is_cancel_requested(rec2.job_id) is True
    finally:
        _stop_supervisor(proc, socket_path)


def test_daemon_supervisor_client_status_and_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SupervisorClient.is_running / worker_info reflect a live supervisor."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))

    Workspace.create("e2e2")
    socket_path = runtime / "supervisor.sock"
    proc = _spawn_supervisor("e2e2", runtime, home, num_workers=1, cpu_only=True)
    try:
        assert _wait_socket(socket_path, timeout_s=20.0)
        client = SupervisorClient()
        assert client.is_running()
        info = client.worker_info()
        assert "workers" in info
        assert isinstance(info["workers"], list)
        assert len(info["workers"]) >= 1
        w0 = info["workers"][0]
        assert "gpu_id" in w0
        assert "pid" in w0

        # Shutdown via client; ensure stop is observed.
        client.shutdown()
        for _ in range(50):
            if not client.is_running():
                break
            time.sleep(0.1)
        assert not client.is_running()
    finally:
        _stop_supervisor(proc, socket_path)


def test_daemon_multiple_cpu_workers_have_distinct_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy ``[-1, -1]`` CPU layout starts distinct workers instead of colliding."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))

    Workspace.create("e2e_cpu2")
    socket_path = runtime / "supervisor.sock"
    proc = _spawn_supervisor("e2e_cpu2", runtime, home, num_workers=2, cpu_only=True)
    try:
        assert _wait_socket(socket_path, timeout_s=20.0)
        client = SupervisorClient()
        info = client.worker_info()
        workers = info.get("workers", [])
        assert len(workers) == 2
        assert sorted(w["worker_id"] for w in workers) == [-2, -1]
        assert all(w["gpu_id"] == -1 for w in workers)
        assert len({w["socket"] for w in workers}) == 2
    finally:
        _stop_supervisor(proc, socket_path)
