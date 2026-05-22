"""Slow: supervisor + 1 GPU worker, warm protenix_v2, submit a real job.

Verifies the Phase-4.2 adapter drives all the Phase-3 plumbing (scheduler,
warm pool, IPC, NDJSON events, artifact paths) on the real Protenix model.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from vibeproteinagent.adapters.protenix_v2.adapter import (
    PROTENIX_CKPT_DIR,
    PROTENIX_MODEL_NAME,
    PROTENIX_TREE,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace

pytestmark = pytest.mark.slow

_FIXTURE = Path(__file__).resolve().parents[0] / "fixtures" / "protenix_v2_input_tiny.json"


def _supervisor_subprocess_target(
    workspace_name: str,
    runtime_dir_s: str,
    vpa_home_s: str,
    num_workers: int,
    gpu_ids: list[int],
) -> None:
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
        proc.join(timeout=20.0)
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
        time.sleep(0.2)
    raise AssertionError(
        f"job {job_id} never reached terminal state within {timeout_s}s "
        f"(last state={last})"
    )


def test_protenix_v2_e2e_with_warm_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required: Protenix v2 descriptor sets gpu_required=true")
    if not PROTENIX_TREE.is_dir():
        pytest.skip(f"Protenix tree not found at {PROTENIX_TREE}")
    ckpt = PROTENIX_CKPT_DIR / f"{PROTENIX_MODEL_NAME}.pt"
    if not ckpt.is_file():
        pytest.skip(f"Protenix checkpoint not found at {ckpt}")
    assert _FIXTURE.exists(), f"missing fixture {_FIXTURE}"

    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("protenix_v2_e2e")
    store = JobStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("fork")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=("protenix_v2_e2e", str(runtime), str(home), 1, [0]),
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

        # Trigger warm. The supervisor's internal worker-RPC timeout
        # (workers/supervisor.py:_op_warm) is 240 s. Protenix v1 cold load
        # is ~100-110 s on B200 (model construction dominates), so this
        # usually returns success. If the load races slightly past the
        # ceiling, the *worker* continues loading; we then poll status()
        # and wait for the adapter to appear in loaded_adapters.
        from vibeproteinagent.core.errors import VPAError

        try:
            client.warm("protenix_v2", version_spec="*", timeout_s=600.0)
        except VPAError as exc:
            assert "timeout" in str(exc).lower(), exc

        def _adapter_loaded() -> bool:
            info = client.worker_info()
            for w in info.get("workers", []) or []:
                for entry in w.get("loaded_adapters", []) or []:
                    if isinstance(entry, str) and entry == "protenix_v2":
                        return True
                    if isinstance(entry, dict) and entry.get("name") == "protenix_v2":
                        return True
            return False

        deadline = time.time() + 240.0
        while time.time() < deadline:
            if _adapter_loaded():
                break
            time.sleep(2.0)
        else:
            pytest.fail("protenix_v2 never appeared in loaded_adapters within 240s")

        params = {
            "input_json": str(_FIXTURE),
            "seeds": [101],
            "use_msa": False,
            "use_template": False,
            "use_rna_msa": False,
            "n_sample": 1,
            "n_step": 20,
            "n_cycle": 1,
        }
        rec = store.submit(
            "protenix_v2", "1.0.0", params, workspace_name=ws.name
        )
        store.transition(rec.job_id, JobState.QUEUED)
        t_start = time.time()
        state = _wait_state(store, rec.job_id, timeout_s=240.0)
        elapsed = time.time() - t_start
        cur = store.get(rec.job_id)
        assert state is JobState.COMPLETED, (
            f"run failed: state={state} err={cur.error_code} msg={cur.error_message}"
        )
        out = cur.output
        assert isinstance(out, dict), out

        predicted_cif = Path(out["predicted_cif"])
        assert predicted_cif.is_file(), predicted_cif
        assert predicted_cif.stat().st_size > 0
        assert out["seeds_used"] == [101]
        assert out["runtime_ms"] > 0
        metrics = out["metrics"]
        for k in ("plddt", "iptm", "ptm", "ranking_score"):
            assert k in metrics
            assert isinstance(metrics[k], (int, float))

        # Verify the NDJSON event stream contains progress events.
        events_path = ws.job_dir(rec.job_id) / "events.ndjson"
        assert events_path.exists()
        lines = [
            json.loads(ln)
            for ln in events_path.read_text().splitlines()
            if ln.strip()
        ]
        progress = [e for e in lines if e.get("type") == "progress"]
        assert progress, "expected progress events"
        assert progress[-1]["payload"]["fraction"] == pytest.approx(1.0)

        # The adapter is warm by the time we submit, so the job itself only
        # bears the diffusion + dumping cost (~15 s on B200 for tiny inputs).
        # Allow a generous envelope to absorb scheduler ticks + IPC slack.
        assert elapsed < 120.0, f"warm-path job too slow: {elapsed:.1f}s"
    finally:
        try:
            client = SupervisorClient()
            if client.is_running():
                client.shutdown()
        except Exception:
            pass
        _stop(proc)
