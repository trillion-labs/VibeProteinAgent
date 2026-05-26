from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised by Python 3.10 sidecars
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042
        pass


class JobState(StrEnum):
    """Lifecycle states from Architecture v1 §5."""

    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(StrEnum):
    """Scheduling priority class."""

    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}
)


LEGAL_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.SUBMITTED: frozenset({JobState.QUEUED}),
    # QUEUED → FAILED covers scheduler-rejected jobs (e.g. multi-GPU jobs
    # asking for more workers than the host has — Codex round-2 RC4).
    JobState.QUEUED: frozenset({JobState.RUNNING, JobState.CANCELLED, JobState.FAILED}),
    JobState.RUNNING: frozenset({JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED}),
    JobState.COMPLETED: frozenset(),
    JobState.FAILED: frozenset({JobState.QUEUED}),  # explicit retry only
    JobState.CANCELLED: frozenset(),
}


class JobRecord(BaseModel):
    """One row of the ``jobs`` table."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    call_id: str | None = None
    state: JobState
    tool_name: str
    tool_version: str
    params: dict[str, Any]
    params_hash: str
    output: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None
    log_path: str
    workspace: str
    workflow_id: str | None = None
    workflow_node_id: str | None = None
    worker_id: int | None = None
    gpu_ids: list[int] = Field(default_factory=list)
    priority: Priority = Priority.NORMAL
    retry_count: int = 0
    max_retries: int = 0
    submitted_at: int
    queued_at: int | None = None
    started_at: int | None = None
    finished_at: int | None = None
    cache_hit: bool = False


class ArtifactRecord(BaseModel):
    """One row of the ``artifacts`` table."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    sha256: str
    kind: str
    mime: str | None = None
    size_bytes: int
    pool_path: str
    producer_job: str | None = None
    created_at: int
    last_used_at: int
    pinned: bool = False


class RunRecord(BaseModel):
    """One row of the ``runs`` table."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    job_id: str
    attempt: int
    worker_id: int
    gpu_ids: list[int]
    started_at: int
    finished_at: int | None = None
    exit_state: str | None = None
    vram_peak_mib: int | None = None
    duration_ms: int | None = None


class WorkflowRecord(BaseModel):
    """One row of the ``workflows`` table."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    name: str
    version: str
    descriptor: str
    params: dict[str, Any]
    state: str
    submitted_at: int
    finished_at: int | None = None


class WorkflowNodeRecord(BaseModel):
    """One row of the ``workflow_nodes`` table."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    node_id: str
    tool_name: str
    tool_version: str
    upstream: list[str]
    params_hash: str
    input_hash: str
    cache_key: str
    state: str
    job_id: str | None = None
    output: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None
