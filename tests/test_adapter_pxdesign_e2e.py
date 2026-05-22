"""Slow: supervisor + 1 GPU worker, warm pxdesign, submit a real job.

Also verifies the incompatibility framework: in a separate supervisor session,
warming protenix_v2 then pxdesign must evict protenix_v2.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import pytest

from vibeproteinagent.adapters.pxdesign.adapter import (
    PROTENIX_PXD_VENDOR,
    PXDBENCH_VENDOR,
    PXDESIGN_CKPT_DIR,
    PXDESIGN_MODEL_NAME,
    PXDESIGN_VENDOR,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace

pytestmark = pytest.mark.slow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_CIF = _PROJECT_ROOT / "tests" / "fixtures" / "5o45.cif"


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


def _skip_if_unavailable() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required: pxdesign descriptor sets gpu_required=true")
    for root in (PXDESIGN_VENDOR, PXDBENCH_VENDOR, PROTENIX_PXD_VENDOR):
        if not root.is_dir():
            pytest.skip(f"vendor tree missing: {root}")
    ckpt = PXDESIGN_CKPT_DIR / f"{PXDESIGN_MODEL_NAME}.pt"
    if not ckpt.is_file():
        pytest.skip(f"PXDesign checkpoint not found at {ckpt}")
    if not _FIXTURE_CIF.is_file():
        pytest.skip(f"fixture CIF missing: {_FIXTURE_CIF}")


def _adapter_loaded(client: Any, name: str) -> bool:
    info = client.worker_info()
    for w in info.get("workers", []) or []:
        for entry in w.get("loaded_adapters", []) or []:
            if isinstance(entry, str) and entry == name:
                return True
            if isinstance(entry, dict) and entry.get("name") == name:
                return True
    return False


def test_pxdesign_e2e_with_warm_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _skip_if_unavailable()

    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("pxdesign_e2e")
    store = JobStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("fork")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=("pxdesign_e2e", str(runtime), str(home), 1, [0]),
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

        from vibeproteinagent.core.errors import VPAError

        try:
            client.warm("pxdesign", version_spec="*", timeout_s=600.0)
        except VPAError as exc:
            assert "timeout" in str(exc).lower(), exc

        deadline = time.time() + 240.0
        while time.time() < deadline:
            if _adapter_loaded(client, "pxdesign"):
                break
            time.sleep(2.0)
        else:
            pytest.fail("pxdesign never appeared in loaded_adapters within 240s")

        params = {
            "target_pdb": str(_FIXTURE_CIF),
            "target_chain": "A",
            "binder_length": 50,
            "num_samples": 1,
            "num_steps": 5,
            "seed": 101,
            "crop": "1-116",
        }
        rec = store.submit("pxdesign", "1.0.0", params, workspace_name=ws.name)
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
        assert out["num_samples"] == 1
        assert len(out["designs"]) == 1
        cif = Path(out["designs"][0]["cif_path"])
        assert cif.is_file(), cif
        assert cif.stat().st_size > 1024

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

        assert elapsed < 120.0, f"warm-path job too slow: {elapsed:.1f}s"
    finally:
        try:
            client = SupervisorClient()
            if client.is_running():
                client.shutdown()
        except Exception:
            pass
        _stop(proc)


def test_pxdesign_evicts_protenix_v2_on_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh supervisor: warm protenix_v2, then warm pxdesign; expect eviction."""
    _skip_if_unavailable()
    from vibeproteinagent.adapters.protenix_v2.adapter import (
        PROTENIX_CKPT_DIR,
        PROTENIX_MODEL_NAME,
        PROTENIX_TREE,
    )

    if not PROTENIX_TREE.is_dir():
        pytest.skip(f"protenix_v2 tree missing: {PROTENIX_TREE}")
    if not (PROTENIX_CKPT_DIR / f"{PROTENIX_MODEL_NAME}.pt").is_file():
        pytest.skip("protenix_v2 checkpoint missing")

    home = tmp_path / "vpa_home_evict"
    home.mkdir()
    runtime = tmp_path / "vpa_run_evict"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    Workspace.create("pxdesign_evict")
    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("fork")
    proc = ctx.Process(
        target=_supervisor_subprocess_target,
        args=("pxdesign_evict", str(runtime), str(home), 1, [0]),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"

        from vibeproteinagent.workers.supervisor_client import (
            SupervisorClient,
        )

        client = SupervisorClient()
        from vibeproteinagent.core.errors import VPAError

        # Warm protenix_v2 first.
        try:
            client.warm("protenix_v2", version_spec="*", timeout_s=600.0)
        except VPAError:
            pass
        deadline = time.time() + 240.0
        while time.time() < deadline:
            if _adapter_loaded(client, "protenix_v2"):
                break
            time.sleep(2.0)
        else:
            pytest.fail("protenix_v2 never warmed within 240s")

        # Now warm pxdesign — should evict protenix_v2.
        try:
            client.warm("pxdesign", version_spec="*", timeout_s=600.0)
        except VPAError:
            pass
        deadline = time.time() + 240.0
        while time.time() < deadline:
            if _adapter_loaded(client, "pxdesign"):
                break
            time.sleep(2.0)
        else:
            pytest.fail("pxdesign never warmed within 240s")

        # Now protenix_v2 must NOT be in loaded_adapters.
        assert not _adapter_loaded(client, "protenix_v2"), (
            "protenix_v2 should have been evicted by pxdesign incompatibility"
        )
    finally:
        try:
            client = SupervisorClient()
            if client.is_running():
                client.shutdown()
        except Exception:
            pass
        _stop(proc)
