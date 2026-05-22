from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPolicy:
    submit: bool = True
    execute: bool = True
    warm: bool = True
    warm_required: bool = False
    max_wait_s: int = 3600
    wait_job_timeout_ms: int = 30_000
    priority: str = "normal"
    max_retries: int = 0
