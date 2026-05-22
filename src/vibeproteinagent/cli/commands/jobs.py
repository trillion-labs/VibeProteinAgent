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
from vibeproteinagent.workspace import (
    TERMINAL_STATES,
    JobRecord,
    JobStore,
    LogFileManager,
    Workspace,
    now_us,
)

# ---------------------------------------------------------------- helpers


def _store(workspace: str | None) -> JobStore:
    ws = Workspace.resolve(workspace)
    return JobStore(ws)


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
    id: str = typer.Argument(..., help="Job id or call_id."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show one job's record."""
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
    ids: list[str] = typer.Argument(..., help="One or more job ids."),  # noqa: B008
    timeout_ms: int = typer.Option(30000, "--timeout-ms", help="Maximum wait."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Poll until all jobs are terminal or the timeout elapses."""
    try:
        store = _store(workspace)
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
    """Cancel a job (sets the durable cancel_requested flag the worker polls)."""
    try:
        store = _store(workspace)
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


__all__ = ["cancel", "logs", "results", "retry", "status", "wait"]
