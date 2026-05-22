"""End-to-end: supervisor + 1 worker, warm proteinmpnn, submit a real job.

Verifies that the Phase-4.1 adapter drives all the Phase-3 plumbing
(scheduler, warm pool, IPC, NDJSON events, artifact paths) on a real
pretrained model. Slow: opt in with ``pytest -m slow``.
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

_FIXTURE = Path(__file__).resolve().parents[0] / "fixtures" / "6MRR.pdb"


def _supervisor_subprocess_target(
    workspace_name: str,
    runtime_dir_s: str,
    vpa_home_s: str,
    num_workers: int,
    gpu_ids: list[int],
) -> None:
    """Module-level so spawn can pickle it."""
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    if all(g < 0 for g in gpu_ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(g) for g in gpu_ids if g >= 0
        )
    from vibeproteinagent.workers.supervisor import supervisor_main

    supervisor_main(workspace_name, num_workers, gpu_ids)


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
        proc.join(timeout=15.0)
    except Exception:
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def _wait_state(store: JobStore, job_id: str, *, timeout_s: float) -> JobState:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        cur = store.get(job_id)
        last = cur.state
        if cur.state.value in ("completed", "failed", "cancelled"):
            return cur.state
        time.sleep(0.1)
    raise AssertionError(
        f"job {job_id} never reached terminal state within {timeout_s}s "
        f"(last state={last})"
    )


def test_proteinmpnn_e2e_with_warm_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm proteinmpnn on one GPU worker, submit two jobs.

    Verifies the Phase 3 supervisor + worker pipeline drives the real adapter
    end-to-end. Also re-submits the same input and asserts the second job
    completes within an envelope of the first (warm-load advantage).
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required: ProteinMPNN descriptor sets gpu_required=true")

    assert _FIXTURE.exists(), f"missing fixture {_FIXTURE}"

    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("mpnn_e2e")
    store = JobStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=("mpnn_e2e", str(runtime), str(home), 1, [0]),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"

        from vibeproteinagent.workers.supervisor_client import (
            SupervisorClient,
        )

        client = SupervisorClient()
        assert client.is_running()

        warm_result = client.warm("proteinmpnn", version_spec="*", timeout_s=120.0)
        assert "warmed" in warm_result, warm_result
        assert all(w.get("success") for w in warm_result["warmed"]), warm_result

        params = {
            "pdb_path": str(_FIXTURE),
            "num_designs": 4,
            "temperature": 0.1,
            "seed": 17,
        }

        # First job (warm)
        rec1 = store.submit("proteinmpnn", "1.0.1", params, workspace_name=ws.name)
        store.transition(rec1.job_id, JobState.QUEUED)
        t1_start = time.time()
        state1 = _wait_state(store, rec1.job_id, timeout_s=120.0)
        t1_elapsed = time.time() - t1_start
        cur1 = store.get(rec1.job_id)
        assert state1 is JobState.COMPLETED, (
            f"first run failed: state={state1} err={cur1.error_code} "
            f"msg={cur1.error_message}"
        )
        out1 = cur1.output
        assert isinstance(out1, dict), out1
        assert len(out1["designs"]) == 4
        L = out1["num_residues"]
        assert L >= 30
        for d in out1["designs"]:
            assert len(d["sequence"]) == L
        fasta = Path(out1["designs_fasta"])
        assert fasta.exists()
        header_count = sum(
            1 for ln in fasta.read_text().splitlines() if ln.startswith(">")
        )
        # 1 input header + 4 design headers
        assert header_count == 5

        # Second job — same params, different salt: identical submissions
        # now collapse to the same deterministic job_id (Architecture v1 —
        # Errata 2), so we pass a salt to force a fresh row + lifecycle.
        # Adapter is already warm-cached, which is the property under test.
        rec2 = store.submit(
            "proteinmpnn", "1.0.1", params,
            workspace_name=ws.name, salt="rerun-warm-check",
        )
        store.transition(rec2.job_id, JobState.QUEUED)
        t2_start = time.time()
        state2 = _wait_state(store, rec2.job_id, timeout_s=120.0)
        t2_elapsed = time.time() - t2_start
        assert state2 is JobState.COMPLETED

        # Verify NDJSON event stream contains progress events.
        events_path = ws.job_dir(rec1.job_id) / "events.ndjson"
        assert events_path.exists()
        lines = [
            json.loads(ln)
            for ln in events_path.read_text().splitlines()
            if ln.strip()
        ]
        progress = [e for e in lines if e.get("type") == "progress"]
        assert progress, "expected progress events"
        assert progress[-1]["payload"]["fraction"] == pytest.approx(1.0)

        # Warm-run should be at least as fast as the first; permit a small
        # tolerance so per-run jitter (CPU loaded, GC pause) doesn't flap.
        assert t2_elapsed <= t1_elapsed + 5.0, (
            f"second run unexpectedly slower: first={t1_elapsed:.2f}s "
            f"second={t2_elapsed:.2f}s"
        )
    finally:
        try:
            client = SupervisorClient()
            if client.is_running():
                client.shutdown()
        except Exception:
            pass
        _stop(proc)
