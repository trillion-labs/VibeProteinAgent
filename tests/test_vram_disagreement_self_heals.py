"""Codex round-2 RC5: scheduler/worker VRAM disagreement self-heals.

The shared :func:`workers.admission.required_vram_mib` helper closed Phase 7's
gap when both sides have the same view of the job. But the worker can call
``adapter.estimate_vram(params)`` (after the adapter is loaded) and discover
the job needs *more* VRAM than the descriptor's static minimum implies. When
that happens the scheduler is wrong and the worker rejects the dispatch.

Phase 8 makes the disagreement self-healing:

* Worker rejects with ``E_VRAM_EXHAUSTED``.
* Supervisor catches the response, calls
  :meth:`JobStore.requeue_after_worker_reject` (no retry-count bump — this
  is *our* bug, not the user's), and blocklists the rejecting worker for
  this specific job for K=2 placement ticks.
* Scheduler tries another worker (or waits). After K ticks the blocklist
  decays and the scheduler can try the same worker again.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibeproteinagent.core.errors import ConflictError, ErrorCode
from vibeproteinagent.workers.supervisor import Supervisor
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _bare_supervisor(tmp_workspace: Workspace) -> Supervisor:
    sup = Supervisor.__new__(Supervisor)
    sup.workspace = tmp_workspace
    sup._gpu_wait_log_ts = {}
    sup._vram_reject_blocklist = {}
    sup._vram_reject_gpus = {}
    sup.workers = []
    sup._registry = None
    sup._log = SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return sup


def _failed_job(store: JobStore, ws: Workspace) -> str:
    rec = store.submit(
        "noop",
        "1.0.0",
        {"payload": "p"},
        workspace_name=ws.name,
        max_retries=3,
    )
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.FAILED,
                     error_code=ErrorCode.VRAM_EXHAUSTED,
                     error_message="adapter requires 200000 MiB; free=80000 MiB")
    return rec.job_id


def test_requeue_after_worker_reject_does_not_bump_retry_count(
    tmp_workspace: Workspace,
) -> None:
    """The disagreement is *our* bug — don't burn the user's retry budget."""
    store = JobStore(tmp_workspace)
    job_id = _failed_job(store, tmp_workspace)
    before = store.get(job_id)
    assert before.state is JobState.FAILED
    assert before.retry_count == 0
    store.requeue_after_worker_reject(job_id)
    after = store.get(job_id)
    assert after.state is JobState.QUEUED
    assert after.retry_count == 0
    assert after.error_code is None
    assert after.error_message is None


def test_requeue_after_worker_reject_only_from_failed(
    tmp_workspace: Workspace,
) -> None:
    store = JobStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "p"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    with pytest.raises(ConflictError):
        store.requeue_after_worker_reject(rec.job_id)


def test_handle_vram_reject_requeues_and_blocklists_worker(
    tmp_workspace: Workspace,
) -> None:
    """A rejection requeues when another eligible worker remains."""
    store = JobStore(tmp_workspace)
    sup = _bare_supervisor(tmp_workspace)
    sup.workers = [
        SimpleNamespace(gpu_id=0, process=SimpleNamespace(is_alive=lambda: True)),
        SimpleNamespace(gpu_id=1, process=SimpleNamespace(is_alive=lambda: True)),
    ]
    job_id = _failed_job(store, tmp_workspace)

    sup._handle_vram_reject(store, job_id, gpu_id=0)

    after = store.get(job_id)
    assert after.state is JobState.QUEUED
    assert after.retry_count == 0
    assert sup._vram_reject_gpus[job_id] == {0}
    assert 0 in sup._vram_reject_blocklist[job_id]


def test_vram_reject_stays_failed_after_all_workers_reject(
    tmp_workspace: Workspace,
) -> None:
    """Once every eligible worker rejected the params, do not requeue forever."""
    store = JobStore(tmp_workspace)
    sup = _bare_supervisor(tmp_workspace)
    sup.workers = [
        SimpleNamespace(gpu_id=0, process=SimpleNamespace(is_alive=lambda: True)),
    ]
    job_id = _failed_job(store, tmp_workspace)

    sup._handle_vram_reject(store, job_id, gpu_id=0)

    after = store.get(job_id)
    assert after.state is JobState.FAILED
    assert after.error_code == ErrorCode.VRAM_EXHAUSTED


def test_terminal_vram_reject_is_not_auto_retried(
    tmp_workspace: Workspace,
) -> None:
    """Generic retry policy must not undo all-worker VRAM terminalization."""
    store = JobStore(tmp_workspace)
    sup = _bare_supervisor(tmp_workspace)
    sup.workers = [
        SimpleNamespace(gpu_id=0, process=SimpleNamespace(is_alive=lambda: True)),
    ]
    job_id = _failed_job(store, tmp_workspace)
    sup._vram_reject_gpus[job_id] = {0}

    sup._maybe_auto_retry(job_id, tmp_workspace.job_dir(job_id) / "events.ndjson")

    after = store.get(job_id)
    assert after.state is JobState.FAILED
    assert after.error_code == ErrorCode.VRAM_EXHAUSTED
    assert after.retry_count == 0
