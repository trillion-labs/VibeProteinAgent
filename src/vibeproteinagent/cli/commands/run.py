from __future__ import annotations

import contextlib
import time
from typing import Any

import typer

from vibeproteinagent.adapters import ToolDescriptor, ToolRegistry, validate_params
from vibeproteinagent.adapters.descriptor import resolve_input_schema_dict
from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_ok,
    print_ndjson_event,
)
from vibeproteinagent.core.errors import ErrorCode, VPAError
from vibeproteinagent.core.types import (
    iter_batch_records,
    load_params_file,
    parse_kv_pairs,
    resolve_salt,
)
from vibeproteinagent.workspace import (
    JobRecord,
    JobState,
    JobStore,
    Workspace,
    now_us,
)


def run(
    tool: str = typer.Argument(..., help="Tool name."),
    params: list[str] = typer.Argument(None, help="KEY=VALUE positional params."),  # noqa: B008
    params_file: str | None = typer.Option(None, "--params-file", help="JSON/YAML params file."),
    batch: str | None = typer.Option(None, "--batch", help="JSONL file or '-' for stdin."),
    follow: bool = typer.Option(False, "--follow", help="Stream NDJSON events."),
    priority: str = typer.Option("normal", "--priority", help="high | normal | low."),
    call_id: str | None = typer.Option(
        None,
        "--call-id",
        help=(
            "User-controlled alias. Re-using the same --call-id with a "
            "different submission raises E_CONFLICT. Job idempotency itself "
            "is built into the deterministic job_id (sha256 of the "
            "submission) — re-submitting an identical request without "
            "--call-id still returns the existing job. Use --force-new for "
            "explicit re-runs."
        ),
    ),
    force_new: bool = typer.Option(
        False,
        "--force-new",
        help=(
            "Force a new job even if an identical submission already exists. "
            "Use for explicit re-run scenarios (e.g. seed-reproducibility "
            "tests). Implementation: passes a random salt so the deterministic "
            "job_id no longer collides."
        ),
    ),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    timeout_ms: int | None = typer.Option(None, "--timeout-ms", help="Server-side timeout."),
    retry: int = typer.Option(0, "--retry", help="Max auto-retries on retryable errors."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Submit a tool run."""
    try:
        registry = ToolRegistry()
        registry.discover()
        descriptor, _ = registry.get(tool, "*")
        ws = _ensure_workspace(workspace)
        store = JobStore(ws)

        salt = resolve_salt(force_new)

        if batch is not None:
            base_params = _build_params(params_file, params)
            records = list(iter_batch_records(batch))
            results = [
                _submit_one(
                    store,
                    descriptor,
                    {**base_params, **rec},
                    priority=priority,
                    call_id=None,  # call_id is per-job; leave unset for batch
                    workspace_name=ws.name,
                    max_retries=retry,
                    salt=salt,
                )
                for rec in records
            ]
            jobs = [r[0] for r in results]
            if follow:
                _follow_many(
                    store, jobs, timeout_ms=timeout_ms, json_mode=json_,
                    idempotent_hits={r.job_id: idem for r, idem in results},
                )
                return
            print_envelope_ok(
                {"jobs": [_job_to_dict(j) for j in jobs]},
                json_mode=json_,
            )
            return

        merged = _build_params(params_file, params)
        rec, idempotent_hit = _submit_one(
            store,
            descriptor,
            merged,
            priority=priority,
            call_id=call_id,
            workspace_name=ws.name,
            max_retries=retry,
            salt=salt,
        )

        if follow:
            _follow_many(
                store, [rec], timeout_ms=timeout_ms, json_mode=json_,
                idempotent_hits={rec.job_id: idempotent_hit},
            )
            return

        payload = {
            "job_id": rec.job_id,
            "call_id": rec.call_id,
            "state": rec.state.value,
            "tool": rec.tool_name,
            "version": rec.tool_version,
            "params_hash": rec.params_hash,
            "workspace": rec.workspace,
            "submitted_at": rec.submitted_at,
            "queued_at": rec.queued_at,
        }
        if idempotent_hit:
            payload["idempotent_hit"] = True
        print_envelope_ok(payload, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


# ---------------------------------------------------------------- helpers


def _ensure_workspace(name: str | None) -> Workspace:
    """Resolve and (lazily) create a workspace if its dir does not yet exist."""
    ws = Workspace.resolve(name)
    if not ws.db_path.exists():
        ws = Workspace.create(ws.name)
    return ws


def _build_params(params_file: str | None, kv_pairs: list[str] | None) -> dict[str, Any]:
    """Merge ``--params-file`` (base) with positional KEY=VALUE pairs (later wins)."""
    base: dict[str, Any] = {}
    if params_file:
        base.update(load_params_file(params_file))
    base.update(parse_kv_pairs(kv_pairs))
    return base


def _submit_one(
    store: JobStore,
    descriptor: ToolDescriptor,
    params: dict[str, Any],
    *,
    priority: str,
    call_id: str | None,
    workspace_name: str,
    max_retries: int = 0,
    salt: str | None = None,
) -> tuple[JobRecord, bool]:
    """Validate params, submit + transition to queued.

    Returns ``(rec, idempotent_hit)``. ``idempotent_hit`` is True iff the
    JobStore returned an already-existing job (either via the deterministic
    ``job_id`` derived from the submission per Architecture v1 — Errata 2,
    or via the ``call_id`` alias). The flag drives the ``--follow`` stream's
    event order: an idempotent hit emits a single ``idempotent_hit`` event
    with the cached job's current state, not a fresh ``submitted``/``queued``
    pair. A failed/cancelled job that was re-queued in place under the same
    job_id is *not* an idempotent hit — its ``submitted_at`` predates the
    pre-submit timestamp but its ``queued_at`` does not, and the rec.state
    is QUEUED (the new run lifecycle is starting).
    """
    validate_params(descriptor, params)
    pre_submit_us = now_us()
    rec = store.submit(
        descriptor.name,
        descriptor.version,
        params,
        call_id=call_id,
        priority=priority,
        workspace_name=workspace_name,
        max_retries=int(max_retries),
        salt=salt,
        input_schema=resolve_input_schema_dict(descriptor),
    )
    # An idempotent hit is a row whose lifecycle was unchanged by this
    # submit — i.e. it predates our pre_submit_us AND we did not just
    # re-queue it. ``queued_at`` updates on every re-queue, so a fresh
    # queued_at signals the re-queue path.
    idempotent_hit = (
        int(rec.submitted_at) < pre_submit_us
        and (rec.queued_at is None or int(rec.queued_at) < pre_submit_us)
    )
    if not idempotent_hit and rec.state is JobState.SUBMITTED:
        rec = store.transition(rec.job_id, JobState.QUEUED)
    return rec, idempotent_hit


def _job_to_dict(rec: JobRecord) -> dict[str, Any]:
    """Compact JSON view of a JobRecord."""
    return {
        "job_id": rec.job_id,
        "call_id": rec.call_id,
        "state": rec.state.value,
        "tool": rec.tool_name,
        "version": rec.tool_version,
        "params_hash": rec.params_hash,
        "workspace": rec.workspace,
        "submitted_at": rec.submitted_at,
        "queued_at": rec.queued_at,
    }


def _follow_many(
    store: JobStore,
    jobs: list[JobRecord],
    *,
    timeout_ms: int | None,
    json_mode: bool = True,
    idempotent_hits: dict[str, bool] | None = None,
) -> None:
    """Follow NDJSON events for each job until they reach a terminal state.

    If the supervisor is running, jobs progress through running -> completed/failed.
    Otherwise (queue-only mode), state never changes and we emit periodic heartbeats
    until the timeout elapses.
    Also tails each job's ``events.ndjson`` (progress, retrying) when present.
    """
    import json as _json
    from pathlib import Path

    from vibeproteinagent.workers.supervisor_client import SupervisorClient

    supervisor_running = SupervisorClient().is_running()

    deadline = time.time() + (timeout_ms / 1000.0 if timeout_ms else 60.0)
    last_states: dict[str, str] = {}

    # Per-job events.ndjson tail handles (best-effort).
    event_files: dict[str, Any] = {}
    progress_bar = None
    progress_tasks: dict[str, int] = {}
    if not json_mode:
        try:
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
            )
            progress_bar = Progress(
                TextColumn("[bold]{task.fields[job_id]}"),
                BarColumn(),
                TaskProgressColumn(),
                transient=False,
            )
            progress_bar.start()
        except Exception:
            progress_bar = None

    def _tail_path(job_id: str) -> Path:
        return Path(store.workspace.job_dir(job_id)) / "events.ndjson"

    hits = idempotent_hits or {}
    for j in jobs:
        ev = _tail_path(j.job_id)
        ev.touch(exist_ok=True)
        with contextlib.suppress(Exception):
            event_files[j.job_id] = ev.open("r", encoding="utf-8")
        if progress_bar is not None:
            tid = progress_bar.add_task("", total=1.0, job_id=j.job_id[-12:])
            progress_tasks[j.job_id] = tid
        if hits.get(j.job_id):
            # Re9: idempotent re-submission. Don't fabricate fresh
            # `submitted`/`queued` events for a job we didn't actually
            # queue. Emit a single `idempotent_hit` event with the
            # cached job's CURRENT state (it may already be running,
            # completed, or failed).
            print_ndjson_event(
                {
                    "type": "idempotent_hit",
                    "job_id": j.job_id,
                    "ts": now_us() / 1_000_000,
                    "payload": {
                        "call_id": j.call_id,
                        "state": j.state.value,
                        "tool": j.tool_name,
                        "version": j.tool_version,
                    },
                }
            )
            last_states[j.job_id] = j.state.value
            continue
        print_ndjson_event(
            {
                "type": "submitted",
                "job_id": j.job_id,
                "ts": now_us() / 1_000_000,
                "payload": {"tool": j.tool_name, "version": j.tool_version},
            }
        )
        print_ndjson_event(
            {
                "type": "queued",
                "job_id": j.job_id,
                "ts": now_us() / 1_000_000,
                "payload": {"call_id": j.call_id},
            }
        )
        last_states[j.job_id] = "queued"

    poll = 0.05 if supervisor_running else 5.0
    try:
        while time.time() < deadline:
            # Drain events.ndjson for every job.
            for j in jobs:
                f = event_files.get(j.job_id)
                if f is None:
                    continue
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        evt = _json.loads(line)
                    except Exception:
                        continue
                    # Pass-through to NDJSON stdout.
                    print_ndjson_event(evt)
                    # Drive Rich progress bar (non-JSON mode).
                    if (progress_bar is not None
                            and evt.get("type") == "progress"
                            and j.job_id in progress_tasks):
                        payload = evt.get("payload") or {}
                        fraction = payload.get("fraction")
                        if isinstance(fraction, (int, float)):
                            progress_bar.update(
                                progress_tasks[j.job_id],
                                completed=float(fraction),
                            )

            recs = [_safe_get(store, j.job_id) for j in jobs]
            # Emit state-change events.
            for r in recs:
                if r is None:
                    continue
                new = r.state.value
                if new != last_states.get(r.job_id):
                    print_ndjson_event(
                        {
                            "type": new,
                            "job_id": r.job_id,
                            "ts": now_us() / 1_000_000,
                            "payload": {
                                "state": new,
                                "output": r.output,
                                "error_code": r.error_code,
                            },
                        }
                    )
                    last_states[r.job_id] = new

            terminal = {"completed", "failed", "cancelled"}
            if all(r is not None and r.state.value in terminal for r in recs):
                return

            if not supervisor_running:
                for r in recs:
                    if r is None:
                        continue
                    print_ndjson_event(
                        {
                            "type": "heartbeat",
                            "job_id": r.job_id,
                            "ts": now_us() / 1_000_000,
                            "payload": {
                                "state": r.state.value,
                                "hint": "supervisor not running; "
                                        "start it with `vpa daemon start`",
                            },
                        }
                    )

            sleep_for = min(poll, max(0.0, deadline - time.time()))
            if sleep_for <= 0:
                break
            time.sleep(sleep_for)

        for j in jobs:
            cur = last_states.get(j.job_id, "queued")
            print_ndjson_event(
                {
                    "type": "heartbeat",
                    "job_id": j.job_id,
                    "ts": now_us() / 1_000_000,
                    "payload": {
                        "state": cur,
                        "hint": "follow timed out",
                    },
                }
            )
    finally:
        if progress_bar is not None:
            with contextlib.suppress(Exception):
                progress_bar.stop()
        for f in event_files.values():
            with contextlib.suppress(Exception):
                f.close()


def _safe_get(store: JobStore, job_id: str) -> JobRecord | None:
    try:
        return store.get(job_id)
    except VPAError:
        return None


# Surface error code reference so callers can introspect.
__all__ = ["ErrorCode", "run"]
