from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    provider: str
    model: str
    content: str
    finish_reason: str | None = None
    usage: ProviderUsage = Field(default_factory=ProviderUsage)
    raw: dict[str, Any] = Field(default_factory=dict)


class ChatProvider(Protocol):
    provider_name: str

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ProviderResponse: ...
