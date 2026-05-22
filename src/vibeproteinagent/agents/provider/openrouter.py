from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.core.errors import DependencyError, TimeoutError_

from .types import ChatMessage, ProviderResponse, ProviderUsage


class OpenRouterProvider:
    """Thin async OpenRouter client using the OpenAI-compatible chat API."""

    provider_name = "openrouter"

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ProviderResponse:
        return await asyncio.to_thread(
            self._chat_sync,
            messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
            idempotency_key=idempotency_key,
        )

    def _chat_sync(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None,
        temperature: float | None,
        response_format: dict[str, Any] | None,
        idempotency_key: str | None,
    ) -> ProviderResponse:
        if not self.config.api_key:
            raise DependencyError(
                "OPENROUTER_API_KEY (or VPA_OPENROUTER_CREDENTIAL) is required "
                "for OpenRouter-backed agent planning",
                details={"provider": self.provider_name},
            )

        body: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [
                m.model_dump(exclude_none=True)
                for m in _messages_for_model(messages, str(model or self.config.model))
            ],
            "temperature": self.config.temperature if temperature is None else temperature,
        }
        if response_format is not None:
            body["response_format"] = response_format

        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/VibeProteinAgent/VibeProteinAgent",
            "X-Title": "VibeProteinAgent",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        req = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )

        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                choice = (raw.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                usage = raw.get("usage") or {}
                return ProviderResponse(
                    provider=self.provider_name,
                    model=str(raw.get("model") or body["model"]),
                    content=str(message.get("content") or ""),
                    finish_reason=choice.get("finish_reason"),
                    usage=ProviderUsage(
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        cost_usd=usage.get("cost"),
                    ),
                    raw=raw,
                )
            except TimeoutError as exc:
                last_error = exc
                if attempt == 2:
                    raise TimeoutError_(
                        "OpenRouter request timed out",
                        details={
                            "provider": self.provider_name,
                            "timeout_s": self.config.timeout_s,
                        },
                    ) from exc
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code < 500 or attempt == 2:
                    details = {"provider": self.provider_name, "status": exc.code}
                    try:
                        details["body"] = exc.read().decode("utf-8")[:2000]
                    except Exception:
                        pass
                    raise DependencyError("OpenRouter request failed", details=details) from exc
            except urllib.error.URLError as exc:
                last_error = exc
                if attempt == 2:
                    raise DependencyError(
                        "OpenRouter request failed",
                        details={"provider": self.provider_name, "error": str(exc)},
                    ) from exc
            time.sleep(0.5 * (2 ** attempt))

        raise DependencyError(
            "OpenRouter request failed",
            details={"provider": self.provider_name, "error": str(last_error)},
        )


def _messages_for_model(
    messages: Sequence[ChatMessage],
    model: str,
) -> list[ChatMessage]:
    """Apply provider-specific prompt nudges without changing public contracts."""

    if "qwen3" not in model.lower():
        return list(messages)
    patched = list(messages)
    for index in range(len(patched) - 1, -1, -1):
        message = patched[index]
        if message.role != "user":
            continue
        patched[index] = ChatMessage(role=message.role, content=f"/no_think\n{message.content}")
        break
    return patched
