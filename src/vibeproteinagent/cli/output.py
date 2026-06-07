from __future__ import annotations

import contextlib
import json
import sys
import traceback
from collections.abc import Callable
from typing import Any, NoReturn

import typer

# Plain consoles by default; the Rich render path picks its own coloured console.
from rich.console import Console

from vibeproteinagent.core.errors import (
    ErrorCode,
    VPAError,
    exit_code_for,
)
from vibeproteinagent.core.log import LogStyle, get_style
from vibeproteinagent.core.types import err, ok

console = Console()
err_console = Console(stderr=True)

_TRACEBACK_MAX_BYTES = 8192


@contextlib.contextmanager
def protect_json_stdout(json_mode: bool):
    """Keep stdout reserved for the JSON envelope while preparing a command.

    Several scientific libraries print import-time notices to stdout. In normal
    human-readable mode that is tolerable; in ``--json`` mode it corrupts the
    CLI contract because callers expect stdout to contain exactly one JSON
    object. Redirect only the preparation phase so the final envelope still
    lands on stdout.
    """
    if json_mode:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    else:
        yield


# ---------------------------------------------------------------- envelopes


def print_envelope_ok(
    data: dict[str, Any],
    *,
    json_mode: bool,
    render: Callable[[dict[str, Any], LogStyle], None] | None = None,
) -> None:
    """Emit a success envelope.

    JSON mode: writes the envelope to stdout, bit-exact to the §10 contract.
    Non-JSON mode: delegates to ``render`` if provided; otherwise dispatches by
    payload shape so we keep coverage even for commands that haven't been
    explicitly migrated. Style (pretty | plain) is read from the CLI state.
    """
    if json_mode:
        payload = ok(data)
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    style = get_style()
    if style == LogStyle.JSON:
        # Defensive: caller asked for non-JSON but global state still says JSON.
        payload = ok(data)
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    if render is not None:
        render(data, style)
        return
    _dispatch_pretty(data, style)


def print_envelope_err(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    *,
    json_mode: bool,
    exit_code: int | None = None,
) -> NoReturn:
    """Emit an error envelope and raise typer.Exit with the right exit code."""
    payload = err(code, message, details or {})
    if json_mode:
        sys.stdout.write(json.dumps(payload))
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        from vibeproteinagent.cli.render import render_error

        render_error(code, message, details, style=get_style())
    raise typer.Exit(exit_code if exit_code is not None else exit_code_for(code))


def fail_from_exception(
    exc: BaseException,
    *,
    json_mode: bool,
    fallback_message: str = "internal error",
) -> NoReturn:
    """Convert any exception into the right error envelope + exit code.

    typer.Exit / SystemExit propagate unchanged so a command can call
    print_envelope_err inside a try/except without emitting a second envelope.
    """
    if isinstance(exc, (typer.Exit, SystemExit, KeyboardInterrupt)):
        raise exc
    if isinstance(exc, VPAError):
        print_envelope_err(exc.code, exc.message, exc.details, json_mode=json_mode)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if len(tb.encode("utf-8")) > _TRACEBACK_MAX_BYTES:
        tb = tb.encode("utf-8")[:_TRACEBACK_MAX_BYTES].decode("utf-8", errors="replace") + "...\n"
    print_envelope_err(
        ErrorCode.INTERNAL,
        f"{fallback_message}: {exc}",
        {"traceback": tb, "exception_type": type(exc).__name__},
        json_mode=json_mode,
    )


# ---------------------------------------------------------------- streaming


def print_ndjson_event(event: dict[str, Any]) -> None:
    """Emit one NDJSON event to stdout."""
    sys.stdout.write(json.dumps(event))
    sys.stdout.write("\n")
    sys.stdout.flush()


# ---------------------------------------------------------------- pretty dispatch


def _dispatch_pretty(data: dict[str, Any], style: LogStyle) -> None:
    """Best-effort fallback render based on payload shape."""
    from vibeproteinagent.cli import render

    if "tools" in data and isinstance(data["tools"], list):
        render.render_tool_list(data, style=style)
        return
    if "tool" in data and isinstance(data["tool"], dict):
        render.render_tool_info(data, style=style)
        return
    if "workspaces" in data and isinstance(data["workspaces"], list):
        render.render_workspace_list(data, style=style)
        return
    if "workflows" in data and isinstance(data["workflows"], list):
        render.render_workflow_list(data, style=style)
        return
    if data.get("job") and isinstance(data["job"], dict):
        render.render_job(data, style=style)
        return
    if data.get("jobs") and isinstance(data["jobs"], list):
        render.render_job_list(data, style=style)
        return
    if "lines" in data and isinstance(data["lines"], list):
        for line in data["lines"]:
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
        return
    if "version" in data and "python" in data:
        render.render_version(data, style=style)
        return
    if "workers" in data and "running" in data:
        render.render_daemon_status(data, style=style)
        return
    if "warmed" in data:
        render.render_warm_result(data, style=style)
        return
    if "nodes" in data and "workflow_id" in data:
        render.render_workflow_status(data, style=style)
        return
    if "torch_version" in data or "cuda_available" in data:
        render.render_doctor(data, style=style)
        return
    render.render_generic(data, style=style)


__all__ = [
    "console",
    "err_console",
    "fail_from_exception",
    "print_envelope_err",
    "print_envelope_ok",
    "print_ndjson_event",
    "protect_json_stdout",
]
