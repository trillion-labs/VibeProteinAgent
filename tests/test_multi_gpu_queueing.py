"""Multi-GPU jobs must use an explicit distributed adapter contract."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from vibeproteinagent.workers.supervisor import Supervisor, _WorkerHandle
from vibeproteinagent.workers.worker import DistributedRunContext, Worker
from vibeproteinagent.workspace import JobState, JobStore, Workspace

_DISTRIBUTED_CALLS: list[dict[str, object]] = []


def fake_distributed_entrypoint(params, ctx: DistributedRunContext) -> dict[str, object]:
    _DISTRIBUTED_CALLS.append(
        {
            "payload": params.payload,
            "gpu_ids": list(ctx.gpu_ids),
            "rank": ctx.rank,
            "world_size": ctx.world_size,
        }
    )
    ctx.emit_progress(1.0)
    return {"payload": params.payload, "echoed_at": 1.0}


def _make_supervisor(tmp_workspace: Workspace, n_gpu_workers: int) -> Supervisor:
    sup = Supervisor.__new__(Supervisor)
    sup.workspace = tmp_workspace
    sup.workspace_name = tmp_workspace.name
    sup.workers = []
    sup._submit_locks = {}
    sup._worker_state = {}
    sup._workflow_affinity = {}
    sup._consecutive_high_dispatched = 0
    sup._registry = None
    sup._draining = False
    sup._gpu_wait_log_ts = {}
    sup._vram_reject_blocklist = {}
    sup._vram_reject_gpus = {}
    sup._log = SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    for gid in range(n_gpu_workers):
        proc = SimpleNamespace(
            pid=10_000 + gid,
            is_alive=lambda: True,
        )
        h = _WorkerHandle(gpu_id=gid, process=proc)
        h.socket_path = tmp_workspace.root / f"fake_worker_{gid}.sock"
        sup.workers.append(h)
        sup._worker_state[gid] = {
            "gpu_id": gid, "free_vram_mib": 80_000, "total_vram_mib": 80_000,
            "loaded_adapters": [], "busy": False, "current_job_id": None,
        }
    return sup


def _patch_descriptor(monkeypatch: pytest.MonkeyPatch, sup: Supervisor,
                      *, gpus_required: int,
                      distributed_entrypoint: str | None = None) -> None:
    monkeypatch.setattr(
        sup, "_descriptor_for",
        lambda name, version: SimpleNamespace(
            gpus_required=gpus_required,
            gpu_required=True,
            vram_min_gib=2,
            name=name,
            output_schema={},
            distributed_entrypoint=distributed_entrypoint,
        ),
    )


def test_multi_gpu_job_with_more_workers_than_host_fails_permanently(
    tmp_workspace: Workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """g > total_worker_count is a permanent E_NO_GPU."""
    sup = _make_supervisor(tmp_workspace, n_gpu_workers=2)
    _patch_descriptor(monkeypatch, sup, gpus_required=8)
    store = JobStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "p"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)

    sup._place_one_pass()
    final = store.get(rec.job_id)
    assert final.state is JobState.FAILED
    assert final.error_code == "E_NO_GPU"


def test_multi_gpu_job_fails_when_no_distributed_entrypoint(
    tmp_workspace: Workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """g <= total still fails honestly until real multi-rank execution exists."""
    sup = _make_supervisor(tmp_workspace, n_gpu_workers=2)
    _patch_descriptor(monkeypatch, sup, gpus_required=2)

    store = JobStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "p"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)

    sup._place_one_pass()
    after = store.get(rec.job_id)
    assert after.state is JobState.FAILED
    assert after.error_code == "E_NO_GPU"
    assert "distributed worker entrypoint" in (after.error_message or "")


def test_multi_gpu_job_claims_contiguous_workers_when_entrypoint_declared(
    tmp_workspace: Workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler may reserve multiple workers only for explicit contracts."""
    sup = _make_supervisor(tmp_workspace, n_gpu_workers=3)
    _patch_descriptor(
        monkeypatch,
        sup,
        gpus_required=2,
        distributed_entrypoint="tests.fake_multi_gpu:run_distributed",
    )

    store = JobStore(tmp_workspace)
    rec = store.submit("noop", "1.0.0", {"payload": "p"},
                       workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)

    decisions = sup._place_one_pass()
    assert len(decisions) == 1
    rank0, job_id, slaves = decisions[0]
    assert job_id == rec.job_id
    assert rank0.gpu_id == 0
    assert [s.gpu_id for s in slaves] == [1]
    after = store.get(rec.job_id)
    assert after.state is JobState.RUNNING
    assert after.gpu_ids == [0, 1]


def test_distributed_submit_invokes_declared_entrypoint(
    tmp_workspace: Workspace,
) -> None:
    _DISTRIBUTED_CALLS.clear()
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop",
        "1.0.0",
        {"payload": "dist"},
        workspace_name=tmp_workspace.name,
    )
    store.transition(rec.job_id, JobState.QUEUED)
    store.claim_next_queued(worker_id=0, gpu_ids=[0, 1], job_id=rec.job_id)

    worker = Worker(gpu_id=-1, workspace_name=tmp_workspace.name)
    body = {
        "distributed_entrypoint": (
            "tests.test_multi_gpu_queueing:fake_distributed_entrypoint"
        ),
        "gpu_ids": [0, 1],
        "rank": 0,
        "world_size": 2,
    }
    out = worker._run_job_blocking(rec.job_id, lambda _f: None, body)

    assert out["completed"] is True
    assert out["output"] == {"payload": "dist", "echoed_at": 1.0}
    assert _DISTRIBUTED_CALLS == [
        {"payload": "dist", "gpu_ids": [0, 1], "rank": 0, "world_size": 2}
    ]


def test_distributed_nonzero_rank_reports_rank_completion(
    tmp_workspace: Workspace,
) -> None:
    _DISTRIBUTED_CALLS.clear()
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop",
        "1.0.0",
        {"payload": "dist"},
        workspace_name=tmp_workspace.name,
    )
    store.transition(rec.job_id, JobState.QUEUED)
    store.claim_next_queued(worker_id=1, gpu_ids=[0, 1], job_id=rec.job_id)

    worker = Worker(gpu_id=-1, workspace_name=tmp_workspace.name)
    body = {
        "distributed_entrypoint": (
            "tests.test_multi_gpu_queueing:fake_distributed_entrypoint"
        ),
        "gpu_ids": [0, 1],
        "rank": 1,
        "world_size": 2,
    }
    out = worker._run_job_blocking(rec.job_id, lambda _f: None, body)

    assert out["rank_completed"] is True
    assert "completed" not in out
    assert _DISTRIBUTED_CALLS == [
        {"payload": "dist", "gpu_ids": [0, 1], "rank": 1, "world_size": 2}
    ]


def test_supervisor_distributed_dispatch_submits_every_rank(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sup = _make_supervisor(tmp_workspace, n_gpu_workers=2)
    _patch_descriptor(
        monkeypatch,
        sup,
        gpus_required=2,
        distributed_entrypoint="tests.fake_multi_gpu:run_distributed",
    )
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop",
        "1.0.0",
        {"payload": "dist"},
        workspace_name=tmp_workspace.name,
    )
    store.transition(rec.job_id, JobState.QUEUED)
    store.claim_next_queued(worker_id=0, gpu_ids=[0, 1], job_id=rec.job_id)
    calls: list[tuple[int, dict[str, object]]] = []

    async def fake_submit_rpc(handle, body):
        calls.append((handle.gpu_id, dict(body)))
        rank = int(body["rank"])
        if rank == 0:
            return {"body": {"completed": True, "output": {"ok": True}, "artifacts": []}}
        return {"body": {"rank_completed": True, "rank": rank}}

    monkeypatch.setattr(sup, "_submit_rpc", fake_submit_rpc)

    asyncio.run(
        sup._dispatch_distributed_job(
            store,
            artifact_store=SimpleNamespace(register=lambda *a, **k: None),
            job_id=rec.job_id,
            rank0=sup.workers[0],
            slaves=[sup.workers[1]],
            gpu_ids=[0, 1],
        )
    )

    assert [(gid, body["rank"]) for gid, body in calls] == [(0, 0), (1, 1)]
    assert {body["master_addr"] for _, body in calls} == {"127.0.0.1"}
    ports = {body["master_port"] for _, body in calls}
    assert len(ports) == 1
    assert {body["world_size"] for _, body in calls} == {2}
    assert {body["local_world_size"] for _, body in calls} == {2}
    final = store.get(rec.job_id)
    assert final.state is JobState.COMPLETED
    assert final.output == {"ok": True}


def test_multi_gpu_placement_respects_descriptor_vram(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sup = _make_supervisor(tmp_workspace, n_gpu_workers=2)
    sup._worker_state[1]["free_vram_mib"] = 1024
    _patch_descriptor(
        monkeypatch,
        sup,
        gpus_required=2,
        distributed_entrypoint="tests.fake_multi_gpu:run_distributed",
    )

    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop",
        "1.0.0",
        {"payload": "p"},
        workspace_name=tmp_workspace.name,
    )
    store.transition(rec.job_id, JobState.QUEUED)

    decisions = sup._place_one_pass()
    assert decisions == []
    assert store.get(rec.job_id).state is JobState.QUEUED


def test_submit_error_response_marks_running_job_failed(
    tmp_workspace: Workspace,
) -> None:
    sup = _make_supervisor(tmp_workspace, n_gpu_workers=1)
    store = JobStore(tmp_workspace)
    rec = store.submit(
        "noop",
        "1.0.0",
        {"payload": "p"},
        workspace_name=tmp_workspace.name,
    )
    store.transition(rec.job_id, JobState.QUEUED)
    store.claim_next_queued(worker_id=0, gpu_ids=[0], job_id=rec.job_id)
    resp = {
        "error": {
            "code": "E_INTERNAL",
            "message": "submit exploded",
            "details": {"phase": "submit"},
        }
    }

    completed = sup._handle_submit_response(
        store,
        artifact_store=SimpleNamespace(register=lambda *a, **k: None),
        job_id=rec.job_id,
        handle=sup.workers[0],
        resp=resp,
        distributed=False,
    )

    final = store.get(rec.job_id)
    assert completed is False
    assert final.state is JobState.FAILED
    assert final.error_code == "E_INTERNAL"
    assert final.error_details["phase"] == "submit"
