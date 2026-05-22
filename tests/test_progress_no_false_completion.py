"""Codex review Re11: don't inject a final ``progress=1.0`` on failure.

Previously the worker's ``_op_submit`` always flushed a 1.0 frame as the
job-submit response was being built — including when the adapter raised
mid-run. That made the events stream lie about completion.

This test uses a synthetic adapter (dropped into ``~/.vibeproteinagent/tools/``
via the existing drop-in path) that emits one 0.4 progress event and then
raises ``RuntimeError``. The supervisor must mark the job failed and the
events stream must end at 0.4 — never 1.0.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import textwrap
import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import JobState, JobStore, Workspace

pytestmark = pytest.mark.slow


_DROPIN_DIR_NAME = "tools"
_ADAPTER_NAME = "fail_after_progress"


def _write_dropin(home: Path) -> Path:
    """Write a fail-mid-run adapter under VPA_HOME/tools/<name>/."""
    tool_dir = home / _DROPIN_DIR_NAME / _ADAPTER_NAME
    tool_dir.mkdir(parents=True, exist_ok=True)

    descriptor = textwrap.dedent(
        """
        name: fail_after_progress
        version: "1.0.0"
        category: misc
        description: "Synthetic adapter that emits progress then raises (Phase 7 Re11 test)."
        runtime: python
        backend: pytorch
        gpu_required: false
        startup_cost: low
        warm_load_capable: true
        supported_tasks: [test]
        input_schema: {"$schema": "https://json-schema.org/draft/2020-12/schema",
                       "type": "object", "additionalProperties": false}
        output_schema: {"$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object"}
        supported_precisions: [fp32]
        entry_point: "vibeproteinagent_user_tools.fail_after_progress:FailAdapter"
        """
    ).strip() + "\n"
    (tool_dir / "descriptor.yaml").write_text(descriptor)

    adapter = textwrap.dedent(
        '''
        from __future__ import annotations
        from pathlib import Path
        import torch
        from pydantic import BaseModel
        from vibeproteinagent.adapters.base import ToolAdapter
        from vibeproteinagent.adapters.context import AdapterRunContext

        class FailIn(BaseModel):
            pass

        class FailOut(BaseModel):
            pass

        class FailAdapter(ToolAdapter):
            def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
                self.device = device
            def run(self, params: FailIn, ctx: AdapterRunContext) -> FailOut:
                ctx.emit_progress(0.4)
                import time as _t
                _t.sleep(0.1)
                raise RuntimeError("synthetic mid-run failure")
        '''
    ).strip() + "\n"
    (tool_dir / "adapter.py").write_text(adapter)
    return tool_dir


def _supervisor_target(
    workspace_name: str, runtime_dir_s: str, vpa_home_s: str
) -> None:
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Drop-in adapters live under ~/.vibeproteinagent/tools by default; rather
    # than pollute the user's HOME we expose the same path via $HOME = VPA_HOME.
    os.environ["HOME"] = vpa_home_s
    from vibeproteinagent.workers.supervisor import supervisor_main

    supervisor_main(workspace_name, 1, [-1])


def _wait_socket(path: Path, *, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def _stop(proc: mp.Process) -> None:
    if not proc.is_alive():
        return
    try:
        proc.terminate()
        proc.join(timeout=10.0)
    except Exception:
        pass
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5.0)


def test_failed_job_does_not_end_at_progress_1_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "vpa_home"
    home.mkdir()
    # Drop-in must live under HOME/.vibeproteinagent/tools (registry contract).
    dotvpa = home / ".vibeproteinagent"
    dotvpa.mkdir()
    _write_dropin(dotvpa)

    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("re11")
    store = JobStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_target,
        args=("re11", str(runtime), str(home)),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"

        rec = store.submit(_ADAPTER_NAME, "1.0.0", {},
                           workspace_name=ws.name)
        store.transition(rec.job_id, JobState.QUEUED)

        deadline = time.time() + 15.0
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                break
            time.sleep(0.1)
        terminal = store.get(rec.job_id)
        assert terminal.state is JobState.FAILED, (
            f"expected failed, got {terminal.state.value}: {terminal.error_message}"
        )

        events_path = ws.job_dir(rec.job_id) / "events.ndjson"
        # On a failure the only progress event should be the 0.4 the adapter
        # emitted; the worker must NOT inject a final 1.0.
        if events_path.exists():
            lines = [
                json.loads(ln)
                for ln in events_path.read_text().splitlines() if ln.strip()
            ]
            progress = [
                e for e in lines if e.get("type") == "progress"
            ]
            for e in progress:
                f = e["payload"]["fraction"]
                assert f < 1.0, (
                    f"failed job has a 1.0 progress frame in its stream: {progress}"
                )
    finally:
        _stop(proc)
