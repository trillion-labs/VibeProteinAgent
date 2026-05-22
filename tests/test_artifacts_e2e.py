"""Codex review B1: the supervisor must register adapter outputs as artifacts.

End-to-end: spin up a real supervisor + 1 worker, run a synthetic adapter
that writes a file under ``ctx.job_dir/outputs/`` and returns the path, then
assert that :class:`ArtifactStore.for_job` returns one row whose ``pool_path``
points into the workspace artifact pool.

Marked slow because it spawns a subprocess supervisor; the equivalent
unit-level test of the worker's introspection helper lives in
:mod:`test_worker_artifact_introspection`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from vibeproteinagent.workspace import (
    ArtifactStore,
    JobState,
    JobStore,
    Workspace,
)

pytestmark = pytest.mark.slow


def _supervisor_target(
    workspace_name: str, runtime_dir_s: str, vpa_home_s: str
) -> None:
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vibeproteinagent.workers.supervisor import supervisor_main

    supervisor_main(workspace_name, 1, [-1])


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


def _wait_socket(path: Path, *, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return False


def test_noop_does_not_register_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A baseline: noop returns no file paths → zero artifact rows."""
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("artifacts")
    store = JobStore(ws)
    art = ArtifactStore(ws)

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_target,
        args=("artifacts", str(runtime), str(home)),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"
        rec = store.submit("noop", "1.0.0", {"payload": "hi"},
                           workspace_name=ws.name)
        store.transition(rec.job_id, JobState.QUEUED)
        deadline = time.time() + 15.0
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state.value in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.1)
        terminal = store.get(rec.job_id)
        assert terminal.state is JobState.COMPLETED, terminal.error_message
        # noop emits no paths → no artifacts.
        assert art.for_job(rec.job_id) == {}
    finally:
        _stop(proc)


def test_mpnn_to_protenix_input_registers_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adapter that writes a JSON output file must produce an artifact row.

    ``mpnn_to_protenix_input`` is the CPU-only glue adapter from Phase 5.4 —
    it writes per-design Protenix JSON files under ``ctx.job_dir/outputs/``
    and returns their paths in ``input_jsons``. After B1 the supervisor
    registers each into :class:`ArtifactStore`.
    """
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    monkeypatch.setenv("VPA_HOME", str(home))
    monkeypatch.setenv("VPA_RUNTIME_DIR", str(runtime))
    monkeypatch.delenv("VPA_WORKSPACE", raising=False)

    ws = Workspace.create("artifacts2")
    store = JobStore(ws)
    art = ArtifactStore(ws)

    # Stage a tiny FASTA the adapter can parse.
    fasta = tmp_path / "designs.fa"
    fasta.write_text(
        ">design_1, score=1.23, seq_recovery=0.5\nGACMQ\n"
        ">design_2, score=2.34, seq_recovery=0.7\nKLMNP\n"
    )

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_supervisor_target,
        args=("artifacts2", str(runtime), str(home)),
        daemon=False,
    )
    proc.start()
    try:
        assert _wait_socket(socket_path), "supervisor never came up"

        rec = store.submit(
            "mpnn_to_protenix_input", "1.0.1",
            {"designs_fasta": str(fasta), "top_k": 1, "name_prefix": "demo"},
            workspace_name=ws.name,
        )
        store.transition(rec.job_id, JobState.QUEUED)
        deadline = time.time() + 30.0
        while time.time() < deadline:
            cur = store.get(rec.job_id)
            if cur.state.value in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.2)
        terminal = store.get(rec.job_id)
        assert terminal.state is JobState.COMPLETED, terminal.error_message

        # Wait a beat for the supervisor's post-completion artifact register.
        records = art.for_job(rec.job_id)
        # Expect at least one artifact for input_jsons / fasta path.
        assert records, (
            f"no artifacts registered; output keys={list(terminal.output or {})}"
        )
        for field, ar in records.items():
            assert ar.sha256 and len(ar.sha256) == 64
            pool = Path(ar.pool_path)
            assert pool.is_file()
            assert pool.is_relative_to(ws.artifacts_dir), (
                f"artifact {field} not in pool: {pool}"
            )
    finally:
        _stop(proc)
