"""Rich-based rendering for the CLI's pretty/plain modes.

Each ``render_*`` function takes a ``data`` argument (the same dict the JSON
envelope would carry), a :class:`LogStyle`, and an optional :class:`Console`.
Pretty mode emits coloured Rich output. Plain mode emits the same content as
a fixed-width text block with no ANSI escapes.

These functions never touch JSON output: JSON-mode callers route through
:mod:`vibeproteinagent.cli.output` instead. Verbosity is consulted only for
``info``-level chrome (titles, headers); a ``-q`` invocation skips the chrome
and emits the bare result row.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterable
from io import StringIO
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vibeproteinagent.core.log import (
    LogStyle,
    Verbosity,
    get_style,
    get_verbosity,
    should_emit,
)

# ---------------------------------------------------------------- console helpers


def _console_for(style: LogStyle | None, console: Console | None) -> Console:
    """Resolve the Console to use. Plain → no_color + no_terminal; Pretty → Rich default.

    Highlighting (Rich's automatic number/string colouring) is disabled in both
    modes: it produces visually noisy output on plain text like ``version`` and
    is irrelevant to the table-heavy rendering we do here.
    """
    if console is not None:
        return console
    style = style or get_style()
    if style == LogStyle.PLAIN:
        return Console(no_color=True, force_terminal=False, highlight=False, soft_wrap=False)
    return Console(highlight=False)


def _is_pretty(style: LogStyle | None) -> bool:
    return (style or get_style()) == LogStyle.PRETTY


# Status glyphs: rich (pretty) vs ASCII (plain).
_GLYPHS_PRETTY = {
    "ok": "[green]✓[/green]",
    "warn": "[yellow]⚠[/yellow]",
    "err": "[red]✗[/red]",
    "run": "[cyan]⋯[/cyan]",
}
_GLYPHS_PLAIN = {"ok": "[ok]", "warn": "[warn]", "err": "[err]", "run": "[..]"}


def _g(kind: str, style: LogStyle | None) -> str:
    return (_GLYPHS_PRETTY if _is_pretty(style) else _GLYPHS_PLAIN)[kind]


def _table(title: str | None, style: LogStyle | None) -> Table:
    if _is_pretty(style):
        return Table(title=title, header_style="bold cyan", show_lines=False)
    # Plain: simple, no colour, ASCII box.
    return Table(
        title=title,
        show_header=True,
        header_style=None,
        show_lines=False,
        box=None,
        pad_edge=False,
    )


def _human_size(n: int | None) -> str:
    if n is None:
        return "-"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.0f}{u}" if u == "B" else f"{f:.1f}{u}"
        f /= 1024
    return f"{f:.1f}{units[-1]}"


def _short(s: str | None, n: int = 12) -> str:
    if not s:
        return "-"
    return s if len(s) <= n else s[-n:]


def _clip(value: Any, limit: int = 96) -> str:
    text = "-" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _state_style(state: str) -> str:
    normalized = state.lower()
    if normalized in {"completed", "success", "succeeded", "cached", "valid"}:
        return "green"
    if normalized in {"failed", "error", "cancelled", "invalid"}:
        return "red"
    if normalized in {"running", "submitted", "queued"}:
        return "cyan"
    if normalized in {"skipped", "pending"}:
        return "yellow"
    return "white"


def _state_icon(state: str, style: LogStyle | None) -> str:
    normalized = state.lower()
    if normalized in {"completed", "success", "succeeded", "cached", "valid"}:
        return "✅" if _is_pretty(style) else "[ok]"
    if normalized in {"failed", "error", "cancelled", "invalid"}:
        return "❌" if _is_pretty(style) else "[err]"
    if normalized in {"running", "submitted", "queued"}:
        return "⏳" if _is_pretty(style) else "[..]"
    if normalized in {"skipped", "pending"}:
        return "⚠️" if _is_pretty(style) else "[warn]"
    return "•" if _is_pretty(style) else "-"


# ---------------------------------------------------------------- tools


def render_tool_list(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    """Render the output of ``vpa tools``."""
    style = style or get_style()
    c = _console_for(style, console)
    tools = data.get("tools", []) or []
    category = data.get("category")
    if not tools:
        if should_emit("info"):
            c.print("No tools found." + (f" (category={category})" if category else ""))
        return
    title = "Registered tools" + (f"  ·  category={category}" if category else "")
    table = _table(title if get_verbosity() != Verbosity.QUIET else None, style)
    table.add_column("name")
    table.add_column("version")
    table.add_column("category")
    table.add_column("gpu", justify="right")
    table.add_column("vram", justify="right")
    table.add_column("description", overflow="fold")
    for t in tools:
        gpu = "yes" if t.get("gpu_required") else "no"
        vram = t.get("vram_min_gib")
        vram_s = f"{vram} GiB" if vram is not None else "-"
        table.add_row(
            str(t.get("name", "")),
            str(t.get("version", "")),
            str(t.get("category", "")),
            gpu,
            vram_s,
            (t.get("description") or "")[:80],
        )
    c.print(table)


def render_tool_info(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    """Render the output of ``vpa tool info``."""
    style = style or get_style()
    c = _console_for(style, console)
    d = data.get("tool") or {}
    if not d:
        c.print("No tool data.")
        return
    name = d.get("name", "?")
    version = d.get("version", "?")
    header = f"{name} v{version}"
    if _is_pretty(style) and should_emit("info"):
        c.print(Panel.fit(header, style="bold cyan", border_style="cyan"))
    elif should_emit("info"):
        c.print(header)
        c.print("=" * len(header))
    rows = [
        ("category", d.get("category")),
        ("backend", d.get("backend")),
        ("runtime", d.get("runtime")),
        ("gpu_required", d.get("gpu_required")),
        ("gpus_required", d.get("gpus_required")),
        ("vram_min_gib", d.get("vram_min_gib")),
        ("pin_warm", d.get("pin_warm")),
        ("warm_load_capable", d.get("warm_load_capable")),
        ("execution_isolation", d.get("execution_isolation")),
        ("environment_name", d.get("environment_name") or "-"),
        ("environment_reason", d.get("environment_reason") or "-"),
        ("startup_cost", d.get("startup_cost")),
        ("supported_tasks", ", ".join(d.get("supported_tasks") or [])),
        ("incompatible_with", ", ".join(d.get("incompatible_with") or []) or "-"),
        ("entry_point", d.get("entry_point") or "-"),
    ]
    t = _table(None, style)
    t.add_column("field", style="bold" if _is_pretty(style) else None)
    t.add_column("value")
    for k, v in rows:
        t.add_row(str(k), "-" if v is None else str(v))
    c.print(t)
    desc = d.get("description")
    if desc:
        c.print(f"\n{desc}")


# ---------------------------------------------------------------- workspaces


def render_workspace_list(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    active = data.get("active")
    workspaces = data.get("workspaces") or []
    if not workspaces:
        c.print("No workspaces.")
        return
    title = "Workspaces" if get_verbosity() != Verbosity.QUIET else None
    t = _table(title, style)
    t.add_column("name")
    t.add_column("active", justify="center")
    for name in workspaces:
        marker = (_g("ok", style) if name == active else "")
        t.add_row(str(name), marker)
    c.print(t)


def render_workspace_status(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    active = data.get("active") or "-"
    via = data.get("via") or "-"
    root = data.get("root") or "-"
    if should_emit("info"):
        c.print(f"{_g('ok', style)} active workspace: {active}  via={via}")
        c.print(f"  root={root}")
    else:
        c.print(active)


# ---------------------------------------------------------------- jobs


def render_job(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    """Render a single ``{job: {...}}`` payload."""
    style = style or get_style()
    c = _console_for(style, console)
    job = data.get("job") or data
    state = job.get("state", "?")
    glyph = _state_glyph(state, style)
    short = _short(job.get("job_id"), 24)
    line = (
        f"{glyph}  {short}  {job.get('tool','?')} v{job.get('version','?')}  "
        f"state={state}"
    )
    if should_emit("info"):
        c.print(line)
        if job.get("error_code"):
            c.print(f"   error: [red]{job['error_code']}[/red] {job.get('error_message','')}")
            details = job.get("error_details") or None
            if details:
                _render_error_details(c, details, style)
        if job.get("workflow_id"):
            workflow_id = _short(job["workflow_id"], 16)
            node_id = job.get("workflow_node_id") or "-"
            c.print(f"   workflow={workflow_id}  node={node_id}")
    else:
        c.print(line)


def _render_error_details(
    c: Console,
    details: dict[str, Any],
    style: LogStyle | None,
) -> None:
    """Pretty-print structured error details under the error line.

    For schema-violation failures the details carry a ``violations`` list
    of ``{field, message}``; render up to the first 8 entries. Other
    detail shapes (E_CONFLICT diffs, E_NO_GPU details, ...) fall back to
    a key/value dump. (Codex round-3 Re1.)
    """
    violations = details.get("violations") if isinstance(details, dict) else None
    if isinstance(violations, list) and violations:
        if _is_pretty(style):
            t = _table(None, style)
            t.add_column("field")
            t.add_column("message", overflow="fold")
            for v in violations[:8]:
                t.add_row(
                    str(v.get("field", "?")),
                    str(v.get("message", "")),
                )
            c.print(Panel.fit(t, title="violations", border_style="red"))
        else:
            c.print("   violations:")
            for v in violations[:8]:
                c.print(f"     - {v.get('field', '?')}: {v.get('message', '')}")
        if len(violations) > 8:
            c.print(f"   ... and {len(violations) - 8} more")
        return
    # Fall-through: dump non-violation details one key per line.
    for k, v in details.items():
        if k == "violations":
            continue
        c.print(f"   {k}: {v}")


def render_job_list(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    jobs = data.get("jobs") or []
    if not jobs:
        c.print("No jobs.")
        return
    t = _table("Jobs" if get_verbosity() != Verbosity.QUIET else None, style)
    t.add_column("job_id")
    t.add_column("tool")
    t.add_column("state")
    t.add_column("worker", justify="right")
    t.add_column("err")
    for j in jobs:
        t.add_row(
            _short(j.get("job_id"), 24),
            f"{j.get('tool','?')} v{j.get('version','?')}",
            str(j.get("state", "?")),
            str(j.get("worker_id") if j.get("worker_id") is not None else "-"),
            j.get("error_code") or "-",
        )
    c.print(t)


def _state_glyph(state: str, style: LogStyle | None) -> str:
    s = (state or "").lower()
    if s in {"completed", "cached", "ok"}:
        return _g("ok", style)
    if s in {"failed", "cancelled"}:
        return _g("err", style)
    if s in {"running", "ready", "submitted", "queued"}:
        return _g("run", style)
    return _g("warn", style)


# ---------------------------------------------------------------- daemon


def render_daemon_status(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    running = bool(data.get("running"))
    if not running:
        c.print(f"{_g('warn', style)} supervisor not running")
        return
    pid = data.get("pid")
    socket = data.get("socket")
    log_p = data.get("log")
    requested_workspace = data.get("requested_workspace")
    supervisor_workspace = data.get("supervisor_workspace") or data.get("workspace")
    workspace_match = data.get("workspace_match")
    if should_emit("info"):
        c.print(
            f"{_g('ok', style)} supervisor running    "
            f"pid={pid}  socket={socket}"
        )
        if supervisor_workspace:
            ws_line = f"  workspace={supervisor_workspace}"
            if requested_workspace and requested_workspace != supervisor_workspace:
                ws_line += f"  requested={requested_workspace}  match={bool(workspace_match)}"
            c.print(ws_line)
        if log_p:
            c.print(f"  log={log_p}")
    workers = data.get("workers") or []
    if not workers:
        return
    t = _table("Workers" if get_verbosity() != Verbosity.QUIET else None, style)
    t.add_column("worker", justify="right")
    t.add_column("gpu", justify="right")
    t.add_column("pid", justify="right")
    t.add_column("state")
    t.add_column("free", justify="right")
    t.add_column("total", justify="right")
    t.add_column("loaded")
    for w in workers:
        worker_id = w.get("worker_id", w.get("gpu_id", "?"))
        gpu_id = w.get("gpu_id", "?")
        loaded = w.get("loaded_adapters") or []
        loaded_parts = []
        for a in loaded:
            health = ""
            if a.get("healthy") is False:
                health = " unhealthy"
            elif a.get("warning"):
                health = " warning"
            loaded_parts.append(f"{a.get('name')}@{a.get('version')}{health}")
        loaded_s = ", ".join(loaded_parts) or "-"
        free = w.get("free_vram_mib") or 0
        total = w.get("total_vram_mib") or 0
        busy = "busy" if w.get("busy") else ("alive" if w.get("alive") else "dead")
        t.add_row(
            str(worker_id),
            str(gpu_id),
            str(w.get("pid", "-")),
            busy,
            f"{free} MiB",
            f"{total} MiB" if total else "-",
            loaded_s,
        )
    c.print(t)


def render_daemon_start(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    pid = data.get("pid")
    socket = data.get("socket")
    workers = data.get("workers") or []
    workspace = data.get("workspace")
    if should_emit("info"):
        c.print(f"{_g('ok', style)} supervisor running    pid={pid}  socket={socket}")
        c.print(f"  workspace={workspace}  workers={len(workers)}")
    if get_verbosity() == Verbosity.QUIET or not workers:
        return
    for w in workers:
        worker_id = w.get("worker_id", w.get("gpu_id", "?"))
        gpu_id = w.get("gpu_id", "?")
        pid_w = w.get("pid", "-")
        total = w.get("total_vram_mib") or 0
        total_s = f"{total // 1024} GiB" if total else "cpu"
        device_s = f"gpu=#{gpu_id}" if isinstance(gpu_id, int) and gpu_id >= 0 else "cpu"
        c.print(
            f"{_g('ok', style)} worker {worker_id}  pid={pid_w}  {device_s}  vram={total_s}"
        )


def render_daemon_stop(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    pid = data.get("pid")
    c.print(f"{_g('ok', style)} supervisor stopped  pid={pid}")


def render_warm_result(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    tool = data.get("tool", "?")
    warmed = data.get("warmed") or []
    ok_n = sum(1 for w in warmed if w.get("success"))
    total = len(warmed)
    status_glyph = _g("ok", style) if ok_n == total else _g("warn", style)
    headline = f"{status_glyph} {tool}  loaded on {ok_n}/{total} workers"
    vrams = [
        w.get("vram_used_mib")
        for w in warmed
        if w.get("success") and w.get("vram_used_mib") is not None
    ]
    if vrams:
        median = sorted(vrams)[len(vrams) // 2]
        headline += f"   (vram p50={median} MiB)"
    # Codex round-3 Re2: surface "we waited on a busy worker" so the
    # operator can tell a slow cold-load from a queued backlog.
    busy_waits = [
        int(w.get("waited_for_busy_worker_ms") or 0)
        for w in warmed
        if w.get("success") and w.get("worker_state_before") == "busy"
    ]
    if busy_waits:
        worst = max(busy_waits)
        headline += f"   (waited {worst / 1000.0:.1f}s on busy worker)"
    if should_emit("info"):
        c.print(headline)
    if get_verbosity() == Verbosity.QUIET:
        return
    failures = [w for w in warmed if not w.get("success")]
    if failures:
        for f in failures:
            c.print(
                f"  {_g('err', style)} worker {f.get('gpu_id','?')}  "
                f"{f.get('error_code') or 'E_INTERNAL'}: {f.get('error','')}"
            )
    # Pi1 (Phase 20): surface healthcheck warnings for adapters whose load
    # succeeded but whose gate (license/env/weights) is not satisfied. The
    # slot is pinned — but the operator needs to know `run` will fail until
    # the gate is met.
    unhealthy = [
        w for w in warmed
        if w.get("success") and w.get("healthy") is False and w.get("warning")
    ]
    if unhealthy:
        seen: set[str] = set()
        for w in unhealthy:
            msg = str(w.get("warning") or "")
            if msg in seen:
                continue
            seen.add(msg)
            c.print(f"  {_g('warn', style)} {msg}")


# ---------------------------------------------------------------- workflows


def render_workflow_list(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    rows = data.get("workflows") or []
    if not rows:
        c.print("No workflows.")
        return
    t = _table("Workflows" if get_verbosity() != Verbosity.QUIET else None, style)
    t.add_column("name")
    t.add_column("version")
    t.add_column("nodes", justify="right")
    t.add_column("path", overflow="fold")
    for r in rows:
        t.add_row(
            str(r.get("name", "?")),
            str(r.get("version", "?")),
            str(r.get("nodes", "-")),
            str(r.get("path", "-")),
        )
    c.print(t)


def render_workflow_status(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    """Final summary panel + per-node table after ``workflow run`` / ``workflow status``."""
    style = style or get_style()
    c = _console_for(style, console)
    wf_id = data.get("workflow_id", "?")
    name = data.get("name", "?")
    ver = data.get("version", "?")
    state = data.get("state", "?")
    nodes = data.get("nodes") or []
    summary = _node_summary(nodes)
    head = (
        f"{_state_glyph(state, style)} {name} v{ver}   workflow_id={_short(wf_id, 16)}   "
        f"state={state}    {summary['cached']} cached / {summary['completed']} ran / "
        f"{summary['failed']} failed / {summary['skipped']} skipped"
    )
    if _is_pretty(style) and should_emit("info"):
        c.print(Panel.fit(head, border_style="cyan"))
    elif should_emit("info"):
        c.print(head)
        c.print("-" * min(120, len(head)))
    if get_verbosity() == Verbosity.QUIET:
        return
    t = _table(None, style)
    t.add_column("node")
    t.add_column("tool")
    t.add_column("state")
    t.add_column("job", overflow="fold")
    t.add_column("error", overflow="fold")
    for n in nodes:
        err = ""
        if n.get("error_code"):
            err = f"{n.get('error_code')}: {n.get('error_message') or ''}"
            if n.get("error_details"):
                details = json.dumps(n["error_details"], sort_keys=True)
                err = f"{err} details={details}"
        t.add_row(
            str(n.get("node_id", "?")),
            f"{n.get('tool','?')} v{n.get('tool_version','?')}",
            str(n.get("state", "?")),
            _short(n.get("job_id"), 24),
            err,
        )
    c.print(t)


def _node_summary(nodes: Iterable[dict[str, Any]]) -> dict[str, int]:
    out = {"completed": 0, "cached": 0, "failed": 0, "skipped": 0, "running": 0}
    for n in nodes:
        s = (n.get("state") or "").lower()
        if s in out:
            out[s] += 1
    return out


# ---------------------------------------------------------------- doctor


def render_doctor(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    head = "Environment check"
    if _is_pretty(style) and should_emit("info"):
        c.print(Panel.fit(head, border_style="cyan"))
    elif should_emit("info"):
        c.print(head)
        c.print("=" * len(head))
    rows = [
        ("python", data.get("python_version")),
        ("executable", data.get("executable")),
        ("torch", data.get("torch_version")),
        ("cuda", data.get("cuda_version")),
        ("cuda available", data.get("cuda_available")),
        ("num gpus", data.get("num_gpus")),
        ("active workspace", data.get("active_workspace")),
        ("workspace root", data.get("workspace_root")),
        ("model cache", data.get("model_cache_root")),
        ("daemon running", data.get("daemon_running")),
        ("adapters loaded", data.get("total_loaded_adapters_across_workers")),
        ("registry tools", ", ".join(data.get("registry_tools") or [])),
    ]
    t = _table(None, style)
    t.add_column("field", style="bold" if _is_pretty(style) else None)
    t.add_column("value", overflow="fold")
    for k, v in rows:
        t.add_row(str(k), "-" if v is None or v == "" else str(v))
    c.print(t)
    gpus = data.get("gpus") or []
    if gpus and get_verbosity() != Verbosity.QUIET:
        gt = _table("GPUs", style)
        gt.add_column("idx", justify="right")
        gt.add_column("name")
        gt.add_column("cc")
        gt.add_column("vram", justify="right")
        for g in gpus:
            gt.add_row(
                str(g.get("index", "?")),
                str(g.get("name", "-")),
                str(g.get("compute_capability", "-")),
                f"{g.get('total_memory_gib','-')} GiB",
            )
        c.print(gt)
    readiness = data.get("install_readiness") or {}
    tools = readiness.get("tools") or []
    if tools and get_verbosity() != Verbosity.QUIET:
        rt = _table("Tool readiness", style)
        rt.add_column("tool")
        rt.add_column("runtime")
        rt.add_column("ready")
        rt.add_column("reason", overflow="fold")
        for item in tools:
            rt.add_row(
                str(item.get("name", "?")),
                str(item.get("runtime") or "-"),
                str(bool(item.get("ready"))),
                str(item.get("reason") or "-"),
            )
        c.print(rt)


def render_install_result(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    status = data.get("status") or ("dry-run" if data.get("dry_run") else "plan")
    profile = data.get("profile") or "-"
    selected = data.get("selected_tools") or []
    head = f"VPA install {status}   profile={profile}   tools={len(selected)}"
    if _is_pretty(style) and should_emit("info"):
        c.print(Panel.fit(head, border_style="cyan"))
    elif should_emit("info"):
        c.print(head)
        c.print("=" * len(head))

    rows = [
        ("status", data.get("status")),
        ("cuda profile", data.get("cuda_profile")),
        ("cuda version", data.get("cuda_version")),
        ("install root", data.get("install_root")),
        ("shared env", data.get("shared_env")),
        ("sidecars", data.get("sidecars_root")),
        ("cache", data.get("cache_dir")),
        ("state", data.get("state_db")),
        ("disk estimate", _human_size(data.get("disk_estimate_bytes"))),
        ("next", data.get("next_command")),
    ]
    t = _table(None, style)
    t.add_column("field", style="bold" if _is_pretty(style) else None)
    t.add_column("value", overflow="fold")
    for key, value in rows:
        if value is not None:
            t.add_row(key, str(value))
    c.print(t)

    steps = data.get("steps") or data.get("results") or []
    if steps and get_verbosity() != Verbosity.QUIET:
        st = _table("Install units", style)
        st.add_column("unit")
        st.add_column("status/action", overflow="fold")
        for item in steps:
            status = item.get("status") or item.get("action") or "-"
            st.add_row(str(item.get("unit", "?")), str(status))
        c.print(st)

    capabilities = data.get("capabilities") or {}
    if isinstance(capabilities, dict) and capabilities and get_verbosity() != Verbosity.QUIET:
        cap = _table("Capability summary", style)
        cap.add_column("state")
        cap.add_column("count", justify="right")
        cap.add_column("tools", overflow="fold")
        for state in (
            "ready",
            "needs_credential",
            "needs_acceptance",
            "needs_local_artifact",
            "unavailable",
            "failed",
        ):
            items = capabilities.get(state) or []
            if not items:
                continue
            names = []
            for item in items:
                if isinstance(item, dict):
                    name = item.get("tool") or item.get("unit") or "?"
                    reason = item.get("reason")
                    names.append(f"{name} ({reason})" if reason else str(name))
                else:
                    names.append(str(item))
            cap.add_row(state.replace("_", " "), str(len(items)), ", ".join(names))
        c.print(cap)

    license_notices = data.get("license_notices") or []
    if license_notices and get_verbosity() != Verbosity.QUIET:
        lt = _table("Tool license notices", style)
        lt.add_column("tool")
        lt.add_column("notice", overflow="fold")
        lt.add_column("source", overflow="fold")
        for item in license_notices:
            lt.add_row(
                str(item.get("tool", "?")),
                str(item.get("notice", "")),
                str(item.get("url") or "-"),
            )
        c.print(lt)

    checks = data.get("checks") or []
    if checks and get_verbosity() != Verbosity.QUIET:
        ct = _table("CUDA preflight", style)
        ct.add_column("check")
        ct.add_column("status")
        ct.add_column("message", overflow="fold")
        for item in checks:
            ct.add_row(
                str(item.get("name", "?")),
                str(item.get("status", "?")),
                str(item.get("message") or "-"),
            )
        c.print(ct)

    readiness_tools = data.get("tools") or []
    if readiness_tools and get_verbosity() != Verbosity.QUIET:
        rt = _table("Tool readiness", style)
        rt.add_column("tool")
        rt.add_column("ready")
        rt.add_column("reason", overflow="fold")
        for item in readiness_tools:
            rt.add_row(
                str(item.get("name", "?")),
                str(bool(item.get("ready"))),
                str(item.get("reason") or "-"),
            )
        c.print(rt)


# ---------------------------------------------------------------- version / generic


def render_version(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    style = style or get_style()
    c = _console_for(style, console)
    version = data.get("version", "?")
    py = data.get("python", "?")
    torch_v = data.get("torch")
    cuda_v = data.get("cuda")
    if get_verbosity() == Verbosity.QUIET:
        c.print(version)
        return
    c.print(f"vpa {version}")
    c.print(f"  python {py}")
    if torch_v:
        c.print(f"  torch  {torch_v}" + (f"  (cuda {cuda_v})" if cuda_v else ""))


def render_generic(
    data: dict[str, Any],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    """Fallback for payloads without a dedicated renderer."""
    style = style or get_style()
    c = _console_for(style, console)
    if not data:
        c.print(f"{_g('ok', style)} ok")
        return
    # Key=value compact form.
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            c.print(f"  {k} = {v}")
        else:
            c.print(f"  {k} = {v!r}")


def render_error(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    """Render an error envelope to stderr."""
    style = style or get_style()
    err_console = Console(stderr=True, no_color=(not _is_pretty(style)),
                          highlight=False, force_terminal=_is_pretty(style))
    err_console.print(f"{_g('err', style)} [red bold]{code}[/]: {message}")
    if details:
        for k, v in details.items():
            err_console.print(f"  {k}={v}")


# ---------------------------------------------------------------- progress


def render_run_progress(
    events: Iterable[dict[str, Any]],
    style: LogStyle | None = None,
    console: Console | None = None,
) -> None:
    """Consume an NDJSON event stream and draw progress.

    Pretty mode uses a Rich Progress bar; plain mode emits one short line per
    state change. JSON callers do not invoke this.
    """
    style = style or get_style()
    c = _console_for(style, console)
    if not _is_pretty(style):
        for evt in events:
            t = evt.get("type", "?")
            payload = evt.get("payload") or {}
            if t == "progress":
                frac = payload.get("fraction")
                if frac is not None:
                    c.print(f"  progress {int(float(frac) * 100)}%")
            else:
                c.print(f"  {t}  job={_short(evt.get('job_id'), 12)}")
        return
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
    )

    progress = Progress(
        TextColumn("[bold]{task.fields[job_id]}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.fields[note]}"),
        transient=False,
        console=c,
    )
    tasks: dict[str, int] = {}
    with progress:
        for evt in events:
            jid = evt.get("job_id", "")
            short = _short(jid, 12)
            if jid not in tasks:
                tasks[jid] = progress.add_task("", total=1.0, job_id=short, note="")
            t = evt.get("type", "?")
            payload = evt.get("payload") or {}
            if t == "progress":
                frac = payload.get("fraction")
                if frac is not None:
                    progress.update(tasks[jid], completed=float(frac))
            elif t in ("completed", "failed", "cancelled"):
                note = f" {t}"
                progress.update(tasks[jid], completed=1.0, note=note)


__all__ = [
    "render_daemon_start",
    "render_daemon_status",
    "render_daemon_stop",
    "render_doctor",
    "render_error",
    "render_generic",
    "render_job",
    "render_job_list",
    "render_run_progress",
    "render_tool_info",
    "render_tool_list",
    "render_version",
    "render_warm_result",
    "render_workflow_list",
    "render_workflow_status",
    "render_workspace_list",
    "render_workspace_status",
]


# Side-effect quieting: rich autopicks terminal width from $COLUMNS or shutil.
_ = shutil  # silence unused warning if rich drops the dep later
_ = sys
_ = StringIO
