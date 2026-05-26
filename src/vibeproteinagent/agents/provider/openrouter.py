from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import queue
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
        if self.config.max_tokens is not None:
            body["max_tokens"] = self.config.max_tokens
        if self.config.disable_reasoning:
            body["reasoning"] = {"enabled": False}

        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/VibeProteinAgent/VibeProteinAgent",
            "X-Title": "VibeProteinAgent",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                raw = _post_json_with_total_timeout(
                    url,
                    data=data,
                    headers=headers,
                    timeout_s=self.config.timeout_s,
                )
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
            except TimeoutError_ as exc:
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


def _post_json_with_total_timeout(
    url: str,
    *,
    data: bytes,
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    ctx = _mp_context()
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_post_json_worker,
        args=(url, data, headers, timeout_s, result_queue),
    )
    proc.daemon = True
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2.0)
        raise TimeoutError_(
            "OpenRouter request timed out",
            details={"provider": OpenRouterProvider.provider_name, "timeout_s": timeout_s},
        )

    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise DependencyError(
            "OpenRouter request failed",
            details={
                "provider": OpenRouterProvider.provider_name,
                "error": f"worker exited without a response (exitcode={proc.exitcode})",
            },
        ) from exc

    kind = result.get("kind")
    if kind == "ok":
        raw = result.get("raw")
        if isinstance(raw, dict):
            return raw
        raise DependencyError(
            "OpenRouter request failed",
            details={"provider": OpenRouterProvider.provider_name, "error": "invalid JSON"},
        )
    if kind == "timeout":
        raise TimeoutError_(
            "OpenRouter request timed out",
            details={"provider": OpenRouterProvider.provider_name, "timeout_s": timeout_s},
        )
    if kind == "http_error":
        raise urllib.error.HTTPError(
            url,
            int(result.get("status") or 0),
            str(result.get("reason") or "OpenRouter request failed"),
            hdrs=None,
            fp=_ErrorBody(result.get("body") or ""),
        )
    if kind == "url_error":
        raise urllib.error.URLError(str(result.get("error") or "OpenRouter request failed"))
    raise DependencyError(
        "OpenRouter request failed",
        details={
            "provider": OpenRouterProvider.provider_name,
            "error": str(result.get("error") or "unknown worker error"),
            "error_type": result.get("error_type"),
        },
    )


def _post_json_worker(
    url: str,
    data: bytes,
    headers: dict[str, str],
    timeout_s: float,
    result_queue: mp.Queue,
) -> None:
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        result_queue.put({"kind": "ok", "raw": raw})
    except TimeoutError as exc:
        result_queue.put({"kind": "timeout", "error": str(exc)})
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")[:2000]
        except Exception:
            body = ""
        result_queue.put(
            {
                "kind": "http_error",
                "status": exc.code,
                "reason": exc.reason,
                "body": body,
            }
        )
    except urllib.error.URLError as exc:
        result_queue.put({"kind": "url_error", "error": str(exc)})
    except Exception as exc:
        result_queue.put(
            {
                "kind": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )


class _ErrorBody:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self, *_args: object, **_kwargs: object) -> bytes:
        return self._body


def _mp_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    if "spawn" in methods:
        return mp.get_context("spawn")
    return mp.get_context()


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
