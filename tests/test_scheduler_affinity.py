"""Unit tests for GPU-aware placement in Supervisor (warm-pool + workflow affinity)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vibeproteinagent.workers.supervisor import Supervisor, _WorkerHandle
from vibeproteinagent.workspace import JobState, JobStore, Workspace


@dataclass
class _FakeProcess:
    pid: int
    alive: bool = True

    def is_alive(self) -> bool:
        return self.alive


def _make_handle(gpu_id: int, pid: int = 1000) -> _WorkerHandle:
    h = _WorkerHandle.__new__(_WorkerHandle)
    h.gpu_id = gpu_id
    h.process = _FakeProcess(pid=pid + gpu_id)  # type: ignore[assignment]
    from vibeproteinagent.workers import paths
    h.socket_path = paths.worker_socket(gpu_id)
    h.busy = False
    h.current_job_id = None
    return h


def _make_supervisor(tmp_workspace: Workspace, gpu_ids: list[int]) -> Supervisor:
    s = Supervisor(tmp_workspace.name, num_workers=len(gpu_ids))
    s.workers = [_make_handle(g) for g in gpu_ids]
    for g in gpu_ids:
        s._submit_locks[g] = asyncio.Lock()
        s._worker_state[g] = {
            "gpu_id": g,
            "busy": False,
            "current_job_id": None,
            "alive": True,
            "free_vram_mib": 32 * 1024,
            "total_vram_mib": 32 * 1024,
            "loaded_adapters": [],
        }
    return s


def test_warm_pool_boost_picks_warmed_worker(tmp_workspace: Workspace) -> None:
    """Two idle workers; only one has noop loaded — that one must win."""
    s = _make_supervisor(tmp_workspace, [0, 1])
    s._worker_state[1]["loaded_adapters"] = [
        {"name": "noop", "version": "1.0.0", "vram_used_mib": 0,
         "pin_warm": False, "run_count": 0, "last_used_ts": 0}
    ]
    store = JobStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "x"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)

    decisions = s._place_one_pass()
    assert len(decisions) == 1
    handle, job_id, slaves = decisions[0]
    assert handle.gpu_id == 1, "scheduler did not prefer the warmed worker"
    assert job_id == rec.job_id
    assert slaves == []


def test_workflow_affinity_overrides_warm_when_both_set(tmp_workspace: Workspace) -> None:
    """Affinity outranks warm-pool: the prior worker for this workflow wins."""
    s = _make_supervisor(tmp_workspace, [0, 1])
    # Workflow ran previously on gpu 0; gpu 1 happens to be warmed.
    s._workflow_affinity["wf-A"] = 0
    s._worker_state[1]["loaded_adapters"] = [
        {"name": "noop", "version": "1.0.0", "vram_used_mib": 0,
         "pin_warm": False, "run_count": 0, "last_used_ts": 0}
    ]
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop", "1.0.0", {"payload": "x"},
        workspace_name=tmp_workspace.name,
        workflow_id="wf-A",
        workflow_node_id="node-1",
    )
    store.transition(rec.job_id, JobState.QUEUED)
    decisions = s._place_one_pass()
    assert len(decisions) == 1
    chosen, _, _ = decisions[0]
    assert chosen.gpu_id == 0


def test_free_vram_tiebreak_when_no_warmth(tmp_workspace: Workspace) -> None:
    """Without affinity or warm hits, prefer the worker with the most free VRAM."""
    s = _make_supervisor(tmp_workspace, [0, 1])
    s._worker_state[0]["free_vram_mib"] = 10 * 1024
    s._worker_state[1]["free_vram_mib"] = 20 * 1024
    store = JobStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "x"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    decisions = s._place_one_pass()
    chosen, _, _ = decisions[0]
    assert chosen.gpu_id == 1


def test_busy_worker_is_not_a_candidate(tmp_workspace: Workspace) -> None:
    """A busy worker is excluded even if it would otherwise win on warmth."""
    s = _make_supervisor(tmp_workspace, [0, 1])
    s.workers[1].busy = True  # was the warm one
    s._worker_state[1]["loaded_adapters"] = [
        {"name": "noop", "version": "1.0.0", "vram_used_mib": 0,
         "pin_warm": False, "run_count": 0, "last_used_ts": 0}
    ]
    store = JobStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "x"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    decisions = s._place_one_pass()
    assert len(decisions) == 1
    chosen, _, _ = decisions[0]
    assert chosen.gpu_id == 0


def test_after_dispatch_records_workflow_affinity(tmp_workspace: Workspace) -> None:
    """Once a workflow node runs on a worker, that mapping should persist."""
    s = _make_supervisor(tmp_workspace, [0, 1])
    s._worker_state[0]["free_vram_mib"] = 30 * 1024
    s._worker_state[1]["free_vram_mib"] = 10 * 1024
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop", "1.0.0", {"payload": "x"},
        workspace_name=tmp_workspace.name,
        workflow_id="wf-X",
        workflow_node_id="step-1",
    )
    store.transition(rec.job_id, JobState.QUEUED)
    s._place_one_pass()
    assert s._workflow_affinity.get("wf-X") == 0


def test_gpu_required_job_fails_on_cpu_only_daemon(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A GPU job should not stay queued forever on a CPU-only daemon."""
    s = _make_supervisor(tmp_workspace, [-1])
    monkeypatch.setattr(
        s,
        "_descriptor_for",
        lambda name, version: SimpleNamespace(
            gpu_required=True,
            gpus_required=1,
            vram_min_gib=2,
            name=name,
            version=version,
        ),
    )
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop",
        "1.0.0",
        {"payload": "x"},
        workspace_name=tmp_workspace.name,
    )
    store.transition(rec.job_id, JobState.QUEUED)

    decisions = s._place_one_pass()

    assert decisions == []
    final = store.get(rec.job_id)
    assert final.state is JobState.FAILED
    assert final.error_code == "E_NO_GPU"
