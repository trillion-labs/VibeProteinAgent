from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _noop_log(_line: str) -> None:
    return None


def _noop_progress(_pct: float) -> None:
    return None


def _never_cancelled() -> bool:
    return False


@dataclass(slots=True, kw_only=True)
class AdapterRunContext:
    """Runtime sibling of `core.types.RunContext` — carries live callables.

    Adapters never write to SQLite; they only read these callables and write files
    under `job_dir/outputs/`. The supervisor wires `is_cancelled` to a real
    `JobStore.is_cancel_requested(job_id)` check, and `log` / `emit_progress` to
    the worker's IPC stream.
    """

    job_id: str
    workspace_root: Path
    job_dir: Path
    is_cancelled: Callable[[], bool]
    log: Callable[[str], None]
    emit_progress: Callable[[float], None]
    fusion_inputs: dict[str, Any] | None = field(default=None)


def null_context(job_dir: Path) -> AdapterRunContext:
    """Test/benchmark helper — all callables are no-ops, `is_cancelled` returns False."""
    job_dir = Path(job_dir)
    return AdapterRunContext(
        job_id="job_" + "0" * 32,
        workspace_root=job_dir.parent.parent if job_dir.parent.name == "jobs" else job_dir,
        job_dir=job_dir,
        is_cancelled=_never_cancelled,
        log=_noop_log,
        emit_progress=_noop_progress,
        fusion_inputs=None,
    )
