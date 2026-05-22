from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vibeproteinagent.agents.session.store import AgentSessionStore

from .types import ChatMessage, ChatProvider, ProviderResponse


class SessionLoggingProvider:
    """Persist provider boundary messages and response metadata for one agent run."""

    def __init__(
        self,
        inner: ChatProvider,
        session_store: AgentSessionStore,
        run_id: str,
    ) -> None:
        self.inner = inner
        self.session_store = session_store
        self.run_id = run_id
        self.provider_name = inner.provider_name

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ProviderResponse:
        for message in messages:
            self.session_store.add_message(
                self.run_id,
                message.role,
                message.content,
                model=model,
            )
        response = await self.inner.chat(
            messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            idempotency_key=idempotency_key,
        )
        usage = response.usage
        self.session_store.add_message(
            self.run_id,
            "assistant",
            response.content,
            model=response.model,
            token_count=usage.total_tokens,
        )
        self.session_store.update_run(
            self.run_id,
            provider=response.provider,
            model=response.model,
            provider_snapshot={
                "provider": response.provider,
                "model": response.model,
                "finish_reason": response.finish_reason,
                "usage": usage.model_dump(exclude_none=True),
            },
        )
        return response
