"""End-to-end: ``vpa daemon warm noop`` loads the adapter into every worker."""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import Workspace

pytestmark = pytest.mark.slow


def _supervisor_subprocess_target(
    workspace_name: str,
    runtime_dir_s: str,
    vpa_home_s: str,
) -> None:
    """Module-level so spawn can pickle it. Single CPU worker."""
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vibeproteinagent.workers.supervisor import supervisor_main

    supervisor_main(workspace_name, 1, [-1])


def _spawn_supervisor(workspace_name: str, runtime_dir: Path, vpa_home: Path) -> mp.Process:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=(workspace_name, str(runtime_dir), str(vpa_home)),
        daemon=False,
    )
    proc.start()
    return proc


def _wait_socket(socket_path: Path, *, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            return True
        time.sleep(0.1)
    return False


def _stop(proc: mp.Process) -> None:
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10.0)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def test_daemon_warm_loads_adapter_visible_in_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """warm `noop` -> status shows it in `loaded_adapters` for every worker."""
    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    Workspace.create("warm_ws")
    socket_path = runtime / "supervisor.sock"
    proc = _spawn_supervisor("warm_ws", runtime, home)
    try:
        assert _wait_socket(socket_path, timeout_s=20.0)
        client = SupervisorClient()
        assert client.is_running()

        # Warm noop on every worker.
        result = client.warm("noop", version_spec="*", timeout_s=30.0)
        assert "warmed" in result
        warmed = result["warmed"]
        assert len(warmed) == 1
        assert warmed[0]["success"] is True

        # Allow one poll cycle so the supervisor's worker_info cache observes
        # the new loaded_adapters from the worker.
        deadline = time.time() + 5.0
        info = {}
        while time.time() < deadline:
            info = client.worker_info()
            workers = info.get("workers", [])
            if workers and workers[0].get("loaded_adapters"):
                break
            time.sleep(0.3)
        workers = info.get("workers", [])
        assert workers, "no workers reported"
        loaded = workers[0].get("loaded_adapters", [])
        names = [a.get("name") for a in loaded]
        assert "noop" in names, f"noop not in loaded adapters: {loaded}"
        assert "free_vram_mib" in workers[0]
        assert "total_vram_mib" in workers[0]
    finally:
        try:
            client = SupervisorClient()
            if client.is_running():
                client.shutdown()
        except Exception:
            pass
        _stop(proc)
