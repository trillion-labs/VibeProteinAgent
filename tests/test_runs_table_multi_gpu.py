"""Codex round-2 Re10: ``runs.gpu_ids`` records the full GPU set for multi-GPU jobs.

Phase 7 wrote only rank0 (``[handle.gpu_id]``) to the ``runs`` table even when
the supervisor leased ``g`` adjacent workers. The schema's ``gpu_ids`` column
is comma-separated and ``RunStore.start_run`` already accepts a list.

This test calls :meth:`RunStore.start_run` with a multi-GPU set and reads the
row back to assert the comma-joined string round-trips.
"""

from __future__ import annotations

from vibeproteinagent.workspace import JobStore, RunStore, Workspace


def _real_job_id(tmp_workspace: Workspace, payload: str = "p") -> str:
    """Insert a row in ``jobs`` so the FK on ``runs.job_id`` is satisfied."""
    rec = JobStore(tmp_workspace).submit(
        "noop", "1.0.0", {"payload": payload},
        workspace_name=tmp_workspace.name,
    )
    return rec.job_id


def test_start_run_persists_multi_gpu_gpu_ids(tmp_workspace: Workspace) -> None:
    job_id = _real_job_id(tmp_workspace, payload="multi")
    rs = RunStore(tmp_workspace)
    run_id = rs.start_run(job_id, attempt=0, worker_id=0, gpu_ids=[0, 1, 2])
    assert run_id.startswith("run_")
    conn = tmp_workspace.connect()
    try:
        row = conn.execute(
            "SELECT gpu_ids FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["gpu_ids"] == "0,1,2"


def test_start_run_single_gpu_still_works(tmp_workspace: Workspace) -> None:
    job_id = _real_job_id(tmp_workspace, payload="single")
    rs = RunStore(tmp_workspace)
    run_id = rs.start_run(job_id, attempt=0, worker_id=3, gpu_ids=[3])
    conn = tmp_workspace.connect()
    try:
        row = conn.execute(
            "SELECT gpu_ids FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row["gpu_ids"] == "3"


def test_start_run_empty_gpu_list_persists_empty_string(
    tmp_workspace: Workspace,
) -> None:
    """CPU-only jobs (gpu_id == -1 path) write an empty gpu_ids cell."""
    job_id = _real_job_id(tmp_workspace, payload="cpu")
    rs = RunStore(tmp_workspace)
    run_id = rs.start_run(job_id, attempt=0, worker_id=-1, gpu_ids=[])
    conn = tmp_workspace.connect()
    try:
        row = conn.execute(
            "SELECT gpu_ids FROM runs WHERE run_id = ?", (run_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row["gpu_ids"] == ""


def test_for_job_returns_multi_gpu_records(tmp_workspace: Workspace) -> None:
    """The pydantic record should round-trip the gpu_ids list."""
    job_id = _real_job_id(tmp_workspace, payload="query")
    rs = RunStore(tmp_workspace)
    rs.start_run(job_id, attempt=0, worker_id=0, gpu_ids=[0, 1])
    runs = rs.for_job(job_id)
    assert len(runs) == 1
    assert runs[0].gpu_ids == [0, 1]
