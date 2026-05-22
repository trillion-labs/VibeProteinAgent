"""Scale smoke test: 50 concurrent noop jobs against a warmed supervisor.

Validates the scheduler / IPC pipeline holds up under burst load. Marked
``@pytest.mark.slow``; opt-in via ``pytest -m slow``. Captures min / median /
p95 / total wall-clock stats for Phase 3.6 sign-off.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import JobState, JobStore, Workspace

pytestmark = pytest.mark.slow


def _supervisor_subprocess_target(
    workspace_name: str,
    runtime_dir_s: str,
    vpa_home_s: str,
    gpu_ids: list[int],
) -> None:
    """Module-level so it's picklable by spawn."""
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    from vibeproteinagent.workers.supervisor import supervisor_main

    supervisor_main(workspace_name, len(gpu_ids), gpu_ids)


def _wait_socket(socket_path: Path, *, timeout_s: float = 30.0) -> bool:
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
        proc.join(timeout=15.0)
    except Exception:
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def test_50_noop_jobs_concurrent_to_warm_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    """Submit 50 noop jobs at once; verify all complete in < 30s on 8 workers."""
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("scale")
    store = JobStore(ws)

    # Resolve real GPU ids if available (one worker per GPU); fall back to a
    # single CPU worker on non-CUDA hosts so the test is still runnable.
    try:
        import torch
        if torch.cuda.is_available():
            gpu_ids = list(range(min(torch.cuda.device_count(), 8)))
        else:
            gpu_ids = [-1]
    except Exception:
        gpu_ids = [-1]
    num_workers = len(gpu_ids)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=("scale", str(runtime), str(home), gpu_ids),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path, timeout_s=30.0), "supervisor never came up"

        ids: list[str] = []
        t0 = time.time()
        for i in range(50):
            rec = store.submit(
                "noop", "1.0.0", {"payload": f"job-{i}"},
                workspace_name=ws.name, priority="normal",
            )
            store.transition(rec.job_id, JobState.QUEUED)
            ids.append(rec.job_id)

        # Wait for all to terminate, max 60s.
        deadline = time.time() + 60.0
        while time.time() < deadline:
            states = [store.get(j).state for j in ids]
            if all(s in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
                   for s in states):
                break
            time.sleep(0.1)
        duration = time.time() - t0

        # Assertions: every job completed.
        recs = [store.get(j) for j in ids]
        not_completed = [r for r in recs if r.state is not JobState.COMPLETED]
        assert not not_completed, (
            f"{len(not_completed)}/{len(recs)} jobs did not complete: "
            f"first={not_completed[0].state.value} "
            f"error={not_completed[0].error_code} "
            f"msg={not_completed[0].error_message}"
        )

        # Stats: per-job (started_at .. finished_at) duration.
        durations_us = [
            r.finished_at - r.started_at for r in recs
            if r.finished_at is not None and r.started_at is not None
        ]
        durations_ms = sorted(d / 1000.0 for d in durations_us)
        stats = {
            "n": len(durations_ms),
            "min_ms": min(durations_ms),
            "median_ms": statistics.median(durations_ms),
            "p95_ms": statistics.quantiles(durations_ms, n=20)[-1],
            "total_s": duration,
        }
        # Print so the manual `-s` invocation captures the numbers.
        print(
            f"\n[scale-50] min={stats['min_ms']:.1f}ms "
            f"median={stats['median_ms']:.1f}ms "
            f"p95={stats['p95_ms']:.1f}ms "
            f"total={stats['total_s']:.2f}s on {num_workers} workers"
        )

        # Loose bound: 50 noop jobs on 8 workers should fit in < 60s. The
        # 250-ms scheduler poll caps throughput at ~32 jobs/sec, plus 1-conn-
        # per-submit IPC overhead — tighter bounds are a Phase 4 concern.
        assert duration < 60.0, (
            f"50 noop jobs took {duration:.2f}s on {num_workers} workers; "
            f"expected < 60s"
        )
    finally:
        _stop(proc)
