"""Priority + anti-starvation tests for the Supervisor scheduler."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from vibeproteinagent.workers.supervisor import (
    _ANTI_STARVATION_LIMIT,
    Supervisor,
    _WorkerHandle,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace


@dataclass
class _FakeProcess:
    pid: int
    alive: bool = True

    def is_alive(self) -> bool:
        return self.alive


def _make_handle(gpu_id: int) -> _WorkerHandle:
    h = _WorkerHandle.__new__(_WorkerHandle)
    h.gpu_id = gpu_id
    h.process = _FakeProcess(pid=2000 + gpu_id)  # type: ignore[assignment]
    from vibeproteinagent.workers import paths
    h.socket_path = paths.worker_socket(gpu_id)
    h.busy = False
    h.current_job_id = None
    return h


def _make_supervisor_single_worker(tmp_workspace: Workspace) -> Supervisor:
    s = Supervisor(tmp_workspace.name, num_workers=1)
    s.workers = [_make_handle(0)]
    s._submit_locks[0] = asyncio.Lock()
    s._worker_state[0] = {
        "gpu_id": 0,
        "busy": False,
        "current_job_id": None,
        "alive": True,
        "free_vram_mib": 32 * 1024,
        "total_vram_mib": 32 * 1024,
        "loaded_adapters": [],
    }
    return s


def _dispatch_one(s: Supervisor, store: JobStore) -> str:
    """Run one scheduler pass on a single-worker supervisor and return the
    dispatched job's priority string. Resets the worker to idle after."""
    decisions = s._place_one_pass()
    assert len(decisions) == 1, f"expected exactly 1 decision, got {decisions}"
    _, job_id, _ = decisions[0]
    rec = store.get(job_id)
    # Worker would normally be flipped back to idle by _dispatch_to_worker's
    # finally clause; do it manually here for the test.
    s.workers[0].busy = False
    s.workers[0].current_job_id = None
    s._worker_state[0]["busy"] = False
    return rec.priority.value


def test_anti_starvation_limit_is_eight() -> None:
    """The constant is part of the contract — guard against silent drift."""
    assert _ANTI_STARVATION_LIMIT == 8


def test_priority_high_runs_before_normal(tmp_workspace: Workspace) -> None:
    s = _make_supervisor_single_worker(tmp_workspace)
    store = JobStore(tmp_workspace)
    # Submit normal first, then high. Highs must win regardless of arrival order.
    n = store.submit("noop", "1.0.0", {"payload": "n"}, priority="normal",
                     workspace_name=tmp_workspace.name)
    h = store.submit("noop", "1.0.0", {"payload": "h"}, priority="high",
                     workspace_name=tmp_workspace.name)
    for r in (n, h):
        store.transition(r.job_id, JobState.QUEUED)
    p = _dispatch_one(s, store)
    assert p == "high"


def test_eight_consecutive_highs_then_anti_starvation_forces_normal(
    tmp_workspace: Workspace,
) -> None:
    """Submit 10 high + 1 normal; expect ``high*8, normal, high, high``."""
    s = _make_supervisor_single_worker(tmp_workspace)
    store = JobStore(tmp_workspace)
    high_ids = []
    for i in range(10):
        r = store.submit("noop", "1.0.0", {"i": i}, priority="high",
                         workspace_name=tmp_workspace.name)
        store.transition(r.job_id, JobState.QUEUED)
        high_ids.append(r.job_id)
    norm = store.submit("noop", "1.0.0", {"k": "n"}, priority="normal",
                        workspace_name=tmp_workspace.name)
    store.transition(norm.job_id, JobState.QUEUED)

    observed = []
    for _ in range(11):
        observed.append(_dispatch_one(s, store))
    # First 8 are high (counter 1..8), the 9th must be normal (anti-starvation),
    # then highs resume.
    assert observed[:8] == ["high"] * 8, observed
    assert observed[8] == "normal", observed
    assert observed[9:11] == ["high", "high"], observed


def test_counter_resets_after_non_high(tmp_workspace: Workspace) -> None:
    s = _make_supervisor_single_worker(tmp_workspace)
    store = JobStore(tmp_workspace)
    # Dispatch one normal; counter should be 0.
    n = store.submit("noop", "1.0.0", {"x": 1}, priority="normal",
                     workspace_name=tmp_workspace.name)
    store.transition(n.job_id, JobState.QUEUED)
    _dispatch_one(s, store)
    assert s._consecutive_high_dispatched == 0
    # Two highs: counter -> 2.
    for i in range(2):
        r = store.submit("noop", "1.0.0", {"i": i}, priority="high",
                         workspace_name=tmp_workspace.name)
        store.transition(r.job_id, JobState.QUEUED)
    for _ in range(2):
        _dispatch_one(s, store)
    assert s._consecutive_high_dispatched == 2
