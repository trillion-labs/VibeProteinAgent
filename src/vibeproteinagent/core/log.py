"""Central logging facility for the CLI and worker daemon.

Three independent dimensions:

- :class:`LogStyle` picks the *user-facing* rendering mode (``json``, ``pretty``,
  ``plain``). Auto-detection lives in :func:`detect_style`.
- :class:`Verbosity` controls how *much* gets emitted in non-JSON mode
  (``quiet``, ``normal``, ``verbose``). JSON mode is unaffected.
- :class:`Logger` writes structured key=value lines to a file (the daemon and
  workers' forensic log). Format: ``[YYYY-MM-DD HH:MM:SS.mmm] [<comp>] [<LVL>] msg k=v``.

JSON envelope bit-exactness is preserved: render/log code never touches stdout
when ``style == LogStyle.JSON``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TextIO

# ---------------------------------------------------------------- style + verbosity


class LogStyle(StrEnum):
    """User-facing output mode."""

    JSON = "json"
    PRETTY = "pretty"
    PLAIN = "plain"


class Verbosity(StrEnum):
    """How much non-JSON output to produce."""

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"


def detect_style(
    *,
    json_mode: bool,
    isatty: bool,
    no_color: bool,
) -> LogStyle:
    """Pick the output style for one CLI invocation.

    Precedence: explicit ``--json`` wins; otherwise TTY + no-color decides.
    ``NO_COLOR`` env (https://no-color.org) is honoured by callers that
    forward it via ``no_color``.
    """
    if json_mode:
        return LogStyle.JSON
    if not isatty or no_color:
        return LogStyle.PLAIN
    return LogStyle.PRETTY


def style_from_env(*, json_mode: bool = False) -> LogStyle:
    """Convenience wrapper that reads ``NO_COLOR`` and stdout isatty itself."""
    no_color = bool(os.environ.get("NO_COLOR"))
    try:
        isatty = sys.stdout.isatty()
    except (AttributeError, ValueError):
        isatty = False
    return detect_style(json_mode=json_mode, isatty=isatty, no_color=no_color)


# ---------------------------------------------------------------- shared CLI state


class _CliState:
    """Process-wide CLI state set by the root Typer callback.

    Held in module scope (not a context) so adapter/render code can pick it up
    without threading ``ctx`` through every call. Initialised in :mod:`cli.main`.
    """

    __slots__ = ("style", "verbosity")

    def __init__(self) -> None:
        self.style: LogStyle = LogStyle.PLAIN
        self.verbosity: Verbosity = Verbosity.NORMAL


_state = _CliState()


def set_cli_state(style: LogStyle, verbosity: Verbosity) -> None:
    _state.style = style
    _state.verbosity = verbosity


def get_style() -> LogStyle:
    return _state.style


def get_verbosity() -> Verbosity:
    return _state.verbosity


def should_emit(level: str) -> bool:
    """Verbosity gate for non-JSON text. Errors always pass."""
    v = _state.verbosity
    if level == "error":
        return True
    if v == Verbosity.QUIET:
        return False
    if v == Verbosity.VERBOSE:
        return True
    # NORMAL: debug suppressed, info/warn shown.
    return level != "debug"


# ---------------------------------------------------------------- file logger


_LEVEL_W = 5  # fixed-width for INFO / WARN / ERROR / DEBUG


def _format_value(v: Any) -> str:
    """Coerce a kwarg value to a flat key=value token (no spaces, no newlines)."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    s = str(v)
    # Quote if contains whitespace, equals, or quotes.
    if any(c in s for c in (" ", "\t", "=", '"')):
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s


def _format_kv(kv: dict[str, Any]) -> str:
    return " ".join(f"{k}={_format_value(v)}" for k, v in kv.items())


def _timestamp(now: float | None = None) -> str:
    t = now if now is not None else time.time()
    dt = datetime.fromtimestamp(t)
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def format_record(
    component: str,
    level: str,
    message: str,
    kv: dict[str, Any] | None = None,
    *,
    now: float | None = None,
    comp_width: int = 14,
) -> str:
    """Format one log record. Used by :class:`Logger` and the tests."""
    lvl = level.upper().ljust(_LEVEL_W)
    comp = f"[{component}]".ljust(comp_width + 2)
    line = f"[{_timestamp(now)}] {comp} [{lvl}] {message}"
    if kv:
        line += " " + _format_kv(kv)
    return line


class Logger:
    """Write structured records to a file (and optionally stderr).

    Thread-safe; one ``Logger`` per ``(component, log_path)`` is fine.
    A failure to write is swallowed — logging must never crash the caller.
    """

    def __init__(
        self,
        component: str,
        log_path: Path | None = None,
        *,
        stderr_mirror: bool = False,
        comp_width: int = 14,
    ) -> None:
        self.component = component
        self.log_path = log_path
        self.stderr_mirror = stderr_mirror
        self.comp_width = comp_width
        self._lock = threading.Lock()
        self._stream: TextIO | None = None
        if log_path is not None:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = log_path.open("a", buffering=1, encoding="utf-8")
            except OSError:
                self._stream = None

    def _emit(self, level: str, message: str, kv: dict[str, Any] | None) -> None:
        line = format_record(
            self.component, level, message, kv,
            comp_width=self.comp_width,
        )
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.write(line + "\n")
                except OSError:
                    pass
            if self.stderr_mirror:
                try:
                    sys.stderr.write(line + "\n")
                    sys.stderr.flush()
                except OSError:
                    pass

    def info(self, message: str, **kv: Any) -> None:
        self._emit("info", message, kv or None)

    def warn(self, message: str, **kv: Any) -> None:
        self._emit("warn", message, kv or None)

    def warning(self, message: str, **kv: Any) -> None:  # alias
        self._emit("warn", message, kv or None)

    def error(self, message: str, **kv: Any) -> None:
        self._emit("error", message, kv or None)

    def debug(self, message: str, **kv: Any) -> None:
        self._emit("debug", message, kv or None)

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except OSError:
                    pass
                self._stream = None


def get_file_logger(component: str, log_path: Path) -> Logger:
    """Factory: a Logger that writes to ``log_path`` only."""
    return Logger(component, log_path)


__all__ = [
    "LogStyle",
    "Logger",
    "Verbosity",
    "detect_style",
    "format_record",
    "get_file_logger",
    "get_style",
    "get_verbosity",
    "set_cli_state",
    "should_emit",
    "style_from_env",
]
