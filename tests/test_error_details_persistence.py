from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from vibeproteinagent.core.errors import VPAError
from vibeproteinagent.workers.worker import Worker
from vibeproteinagent.workspace import JobState, JobStore, Workspace


class _AdapterError(VPAError):
    code = "E_INTERNAL"


class _Input(BaseModel):
    payload: str


class _FailingAdapter:
    def run(self, params: _Input, ctx) -> None:
        del params, ctx
        raise _AdapterError("structured failure", details={"phase": "adapter_run"})

    def estimate_vram(self, params: _Input) -> int:
        del params
        return 0


class _AdapterClass:
    def run(self, params: _Input, ctx) -> None:
        del params, ctx


def test_adapter_run_vpaerror_details_persisted_to_job(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    store = JobStore(tmp_workspace)
    rec = store.submit("fake", "1.0.0", {"payload": "x"}, workspace_name=tmp_workspace.name)
    store.transition(rec.job_id, JobState.QUEUED)
    store.claim_next_queued(worker_id=-1, gpu_ids=[], job_id=rec.job_id)

    worker = Worker(gpu_id=-1, workspace_name=tmp_workspace.name)
    descriptor = SimpleNamespace(
        name="fake",
        version="1.0.0",
        vram_min_gib=None,
        pin_warm=False,
        gpu_required=False,
        gpus_required=1,
    )
    monkeypatch.setattr(
        worker,
        "_ensure_registry",
        lambda: SimpleNamespace(get=lambda *_args: (descriptor, _AdapterClass)),
    )
    monkeypatch.setattr(worker, "_load_adapter", lambda *_args: _FailingAdapter())
    monkeypatch.setattr(worker, "free_vram_mib", lambda: 1 << 20)

    out = worker._run_job_blocking(rec.job_id)

    final = store.get(rec.job_id)
    assert out["state"] == "failed"
    assert out["error_details"] == {"phase": "adapter_run"}
    assert final.state is JobState.FAILED
    assert final.error_details == {"phase": "adapter_run"}
