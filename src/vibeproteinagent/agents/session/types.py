from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RequestMode = Literal[
    "direct_tool",
    "direct_tool_set",
    "explicit_workflow",
    "agent_plan",
    "analysis_only",
]


class ModeClassification(BaseModel):
    """Persisted run classification.

    New ``AgentController.run`` requests are always ``agent_plan``. The broader
    mode enum remains for historical agent run records and report rendering.
    """

    model_config = ConfigDict(extra="forbid")

    mode: RequestMode
    reason: str
    requires_llm_planning: bool = False
    requested_tools: list[str] = Field(default_factory=list)
    workflow_ref: str | None = None
    job_refs: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    force_new: bool = False
    action: str = "submit"
    execution: dict[str, Any] = Field(default_factory=dict)
    direct_jobs: list[dict[str, Any]] = Field(default_factory=list)
    deliverable_contract: dict[str, Any] = Field(default_factory=dict)


class AgentRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    workspace: str
    user_request: str
    mode: RequestMode | None = None
    state: str
    provider: str | None = None
    model: str | None = None
    provider_snapshot: dict[str, Any] | None = None
    system_prompt_version: str
    tool_schema_version: str
    workflow_ids: list[str] = Field(default_factory=list)
    job_ids: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    final_report: dict[str, Any] | None = None
    created_at: int
    updated_at: int


class AgentMessageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_id: str
    run_id: str
    role: str
    content: str
    model: str | None = None
    token_count: int | None = None
    created_at: int


class AgentToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    run_id: str
    tool_name: str
    args: dict[str, Any]
    result: Any | None = None
    ok: bool
    created_at: int
