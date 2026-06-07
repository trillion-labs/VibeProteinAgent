from __future__ import annotations

import time
from typing import Any

import typer

from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_err,
    print_envelope_ok,
    print_ndjson_event,
)
from vibeproteinagent.cli.render import render_job, render_job_list
from vibeproteinagent.core.errors import ErrorCode
from vibeproteinagent.workers.supervisor_client import SupervisorClient
from vibeproteinagent.workspace import (
    TERMINAL_STATES,
    JobRecord,
    JobState,
    JobStore,
    LogFileManager,
    TrajectoryStore,
    Workspace,
    now_us,
)

jobs_app = typer.Typer(
    name="jobs", help="Inspect jobs in a workspace.", invoke_without_command=True
)


@jobs_app.callback(invoke_without_command=True)
def jobs_root(
    ctx: typer.Context,
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    state: str | None = typer.Option(None, "--state", help="Filter by job state."),
    tool: str | None = typer.Option(None, "--tool", help="Filter by tool name."),
    workflow: str | None = typer.Option(None, "--workflow", help="Filter by workflow id."),
    trajectory: str | None = typer.Option(None, "--trajectory", help="Filter by trajectory id."),
    limit: int = typer.Option(100, "--limit", help="Maximum number of jobs."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Alias bare `vpa jobs --workspace X` to `vpa jobs list --workspace X`."""
    if ctx.invoked_subcommand is not None:
        return
    list_jobs(
        workspace=workspace,
        state=state,
        tool=tool,
        workflow=workflow,
        trajectory=trajectory,
        limit=limit,
        json_=json_,
    )

# ---------------------------------------------------------------- helpers


def _store(workspace: str | None) -> JobStore:
    ws = Workspace.resolve(workspace)
    return JobStore(ws)


def _job_summary(rec: JobRecord) -> dict[str, Any]:
    """Light per-job record for enumeration (no params/output/log bodies)."""
    return {
        "job_id": rec.job_id,
        "call_id": rec.call_id,
        "tool": rec.tool_name,
        "version": rec.tool_version,
        "state": rec.state.value,
        "workflow_id": rec.workflow_id,
        "workflow_node_id": rec.workflow_node_id,
        "trajectory_id": rec.trajectory_id,
        "params_hash": rec.params_hash,
        "cache_hit": rec.cache_hit,
        "error_code": rec.error_code,
        "submitted_at": rec.submitted_at,
        "started_at": rec.started_at,
        "finished_at": rec.finished_at,
        "worker_id": rec.worker_id,
    }


def _job_to_dict(rec: JobRecord) -> dict[str, Any]:
    return {
        "job_id": rec.job_id,
        "call_id": rec.call_id,
        "state": rec.state.value,
        "tool": rec.tool_name,
        "version": rec.tool_version,
        "params": rec.params,
        "params_hash": rec.params_hash,
        "output": rec.output,
        "error_code": rec.error_code,
        "error_message": rec.error_message,
        "error_details": rec.error_details,
        "log_path": rec.log_path,
        "workspace": rec.workspace,
        "workflow_id": rec.workflow_id,
        "workflow_node_id": rec.workflow_node_id,
        "trajectory_id": rec.trajectory_id,
        "worker_id": rec.worker_id,
        "gpu_ids": rec.gpu_ids,
        "priority": rec.priority.value,
        "retry_count": rec.retry_count,
        "max_retries": rec.max_retries,
        "submitted_at": rec.submitted_at,
        "queued_at": rec.queued_at,
        "started_at": rec.started_at,
        "finished_at": rec.finished_at,
        "cache_hit": rec.cache_hit,
    }


# ---------------------------------------------------------------- status


def status(
    id: str | None = typer.Argument(None, help="Job id or call_id."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show one job's record."""
    if id is None:
        print_envelope_err(
            ErrorCode.INVALID_PARAMS,
            "status requires a job id or call_id",
            {
                "hint": (
                    "use `vpa status <job_id> --workspace <name>` for one job, "
                    "`vpa jobs list --workspace <name>` for workspace jobs, or "
                    "`vpa trajectory status <traj_id> --workspace <name>` for a query"
                )
            },
            json_mode=json_,
        )
        return
    try:
        store = _store(workspace)
        rec = store.get(id)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"job": _job_to_dict(rec)}, json_mode=json_, render=render_job,
    )


# ---------------------------------------------------------------- wait


def wait(
    ids: list[str] | None = typer.Argument(None, help="One or more job ids."),  # noqa: B008
    timeout_ms: int = typer.Option(30000, "--timeout-ms", help="Maximum wait."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    trajectory: str | None = typer.Option(None, "--trajectory", help="Trajectory id (traj_*)."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Poll until all jobs are terminal or the timeout elapses."""
    try:
        store = _store(workspace)
        if trajectory is not None:
            trajectory_store = TrajectoryStore(store.workspace)
            deadline = time.time() + (timeout_ms / 1000.0)
            timed_out = False
            status = trajectory_store.status(trajectory)
            while True:
                execution_all_terminal = bool(status["execution_all_terminal"])
                if execution_all_terminal:
                    break
                if timeout_ms == 0 or time.time() >= deadline:
                    timed_out = not execution_all_terminal
                    break
                time.sleep(0.25)
                status = trajectory_store.status(trajectory)
            print_envelope_ok(
                {
                    "trajectory_id": trajectory,
                    "jobs": status["jobs"],
                    "workflows": status["workflows"],
                    "execution_all_terminal": status["execution_all_terminal"],
                    "trajectory_complete": status["complete"],
                    "unfinished_job_ids": status["unfinished_job_ids"],
                    "unfinished_workflow_ids": status["unfinished_workflow_ids"],
                    "final_submit_job_id": status["final_submit_job_id"],
                    "report_path": status["report_path"],
                    "all_terminal": status["execution_all_terminal"],
                    "timed_out": timed_out,
                },
                json_mode=json_,
                render=render_job_list,
            )
            return
        ids = ids or []
        if not ids:
            print_envelope_err(
                ErrorCode.INVALID_PARAMS,
                "wait requires at least one job id or --trajectory",
                {"hint": "use `vpa wait --trajectory traj_* --workspace <name>` or pass job ids"},
                json_mode=json_,
            )
            return
        deadline = time.time() + (timeout_ms / 1000.0)
        recs: list[JobRecord]
        timed_out = False
        while True:
            recs = [store.get(i) for i in ids]
            all_terminal = all(r.state in TERMINAL_STATES for r in recs)
            if all_terminal:
                break
            if timeout_ms == 0 or time.time() >= deadline:
                timed_out = not all_terminal
                break
            time.sleep(0.25)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {
            "jobs": [_job_to_dict(r) for r in recs],
            "trajectory_id": trajectory,
            "all_terminal": all(r.state in TERMINAL_STATES for r in recs),
            "timed_out": timed_out,
        },
        json_mode=json_,
        render=render_job_list,
    )


# ---------------------------------------------------------------- logs


def logs(
    id: str = typer.Argument(..., help="Job id."),
    tail: int | None = typer.Option(None, "--tail", help="Tail N lines."),
    follow: bool = typer.Option(False, "--follow", help="Follow new lines."),
    stream: str = typer.Option(
        "stdout", "--stream", help="Which stream to read: 'stdout' or 'stderr'."
    ),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Print job logs."""
    if stream not in ("stdout", "stderr"):
        print_envelope_err(
            ErrorCode.INVALID_PARAMS,
            f"--stream must be 'stdout' or 'stderr' (got {stream!r})",
            {"got": stream, "allowed": ["stdout", "stderr"]},
            json_mode=json_,
        )
    try:
        store = _store(workspace)
        rec = store.get(id)
        ws = store.workspace
        log_mgr = LogFileManager()
        job_dir = ws.job_dir(rec.job_id)

        if not follow:
            text = log_mgr.read(job_dir, stream, tail=tail)
            lines = text.splitlines(keepends=True)
            if json_:
                print_envelope_ok(
                    {"job_id": rec.job_id, "stream": stream, "lines": lines},
                    json_mode=True,
                )
                return
            import sys
            sys.stdout.write(text)
            sys.stdout.flush()
            return

        # follow: poll <stream>.log + events.ndjson; emit NDJSON events; exit
        # when terminal. Synchronous polling; an asyncio rewrite is for later.
        import json as _json
        from pathlib import Path

        path = Path(job_dir) / f"{stream}.log"
        path.touch(exist_ok=True)
        events_path = Path(job_dir) / "events.ndjson"
        events_path.touch(exist_ok=True)
        events_f = events_path.open("r", encoding="utf-8")
        try:
            with path.open("rb") as f:
                while True:
                    # Drain progress/retrying events from events.ndjson.
                    while True:
                        line = events_f.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        if json_:
                            try:
                                evt = _json.loads(line)
                            except Exception:
                                continue
                            print_ndjson_event(evt)

                    chunk = f.read()
                    if chunk:
                        for raw in chunk.splitlines(keepends=True):
                            line = raw.decode("utf-8", errors="replace").rstrip("\n")
                            if json_:
                                print_ndjson_event(
                                    {
                                        "type": "log",
                                        "job_id": rec.job_id,
                                        "ts": now_us() / 1_000_000,
                                        "payload": {"stream": stream, "line": line},
                                    }
                                )
                            else:
                                import sys
                                sys.stdout.write(raw.decode("utf-8", errors="replace"))
                                sys.stdout.flush()
                        continue
                    # No new bytes: check terminal state.
                    rec = store.get(id)
                    if rec.state in TERMINAL_STATES:
                        if json_:
                            print_ndjson_event(
                                {
                                    "type": rec.state.value,
                                    "job_id": rec.job_id,
                                    "ts": now_us() / 1_000_000,
                                    "payload": {"error_code": rec.error_code},
                                }
                            )
                        return
                    time.sleep(0.25)
        finally:
            events_f.close()
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


# ---------------------------------------------------------------- results


def results(
    id: str = typer.Argument(..., help="Job id."),
    fields: str | None = typer.Option(None, "--fields", help="Comma-separated field list."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show a completed job's output."""
    try:
        store = _store(workspace)
        rec = store.get(id)
        if rec.output is None:
            print_envelope_err(
                ErrorCode.NOT_FOUND,
                f"job {rec.job_id} has no output (state={rec.state.value})",
                {"job_id": rec.job_id, "state": rec.state.value},
                json_mode=json_,
            )
        out = rec.output
        if fields:
            keys = [k.strip() for k in fields.split(",") if k.strip()]
            out = {k: out.get(k) for k in keys}
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok({"job_id": rec.job_id, "output": out}, json_mode=json_)


# ---------------------------------------------------------------- cancel


def cancel(
    id: str = typer.Argument(..., help="Job id."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Cancel a job, stopping the owning VPA worker when it is already running."""
    try:
        store = _store(workspace)
        rec = store.get(id)
        client = SupervisorClient()
        if client.is_running():
            client.require_workspace(store.workspace.name)
            result = client.cancel(rec.job_id)
            print_envelope_ok(result, json_mode=json_)
            return
        rec = store.request_cancel(id)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok({"job_id": rec.job_id, "state": rec.state.value}, json_mode=json_)


# ---------------------------------------------------------------- retry


def retry(
    id: str = typer.Argument(..., help="Job id or call_id of a failed job."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Manually re-queue a ``failed`` job (sub-supervisor; honours max_retries)."""
    try:
        store = _store(workspace)
        rec = store.get(id)
        updated = store.requeue_failed(rec.job_id)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"job": _job_to_dict(updated)}, json_mode=json_, render=render_job,
    )


# ---------------------------------------------------------------- list


@jobs_app.command("list")
def list_jobs(
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    state: str | None = typer.Option(None, "--state", help="Filter by job state."),
    tool: str | None = typer.Option(None, "--tool", help="Filter by tool name."),
    workflow: str | None = typer.Option(None, "--workflow", help="Filter by workflow id."),
    trajectory: str | None = typer.Option(None, "--trajectory", help="Filter by trajectory id."),
    since: int | None = typer.Option(
        None, "--since", help="Only jobs with submitted_at >= this (epoch microseconds)."
    ),
    limit: int = typer.Option(100, "--limit", help="Maximum number of jobs."),
    order: str = typer.Option("asc", "--order", help="Chronological order: 'asc' or 'desc'."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List jobs in one workspace (read-only).

    Workspace-scoped: each workspace has its own store, so one workspace per
    objective makes this read as that objective's trajectory.
    """
    if order not in ("asc", "desc"):
        print_envelope_err(
            ErrorCode.INVALID_PARAMS,
            f"--order must be 'asc' or 'desc' (got {order!r})",
            {"got": order, "allowed": ["asc", "desc"]},
            json_mode=json_,
        )
    if state is not None and state not in {s.value for s in JobState}:
        print_envelope_err(
            ErrorCode.INVALID_PARAMS,
            f"unknown --state {state!r}",
            {"got": state, "allowed": [s.value for s in JobState]},
            json_mode=json_,
        )
    try:
        store = _store(workspace)
        recs = store.list(
            state=state,
            tool=tool,
            workflow_id=workflow,
            trajectory_id=trajectory,
            since=since,
            limit=limit,
            order=f"submitted_at_{order}",
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"jobs": [_job_summary(r) for r in recs], "count": len(recs)},
        json_mode=json_,
        render=render_job_list,
    )


__all__ = ["cancel", "jobs_app", "list_jobs", "logs", "results", "retry", "status", "wait"]
