from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Runtime knobs for the optional LLM orchestration layer."""

    provider: str = "codex"
    model: str = "gpt-5.2"
    planner_model: str | None = None
    summarizer_model: str | None = None
    temperature: float = 0.2
    timeout_s: float = 300.0
    max_rounds: int = 8
    max_entries: int = 3
    stream: bool = False
    agent_profile: str = "planner"
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        agent_profile: str | None = None,
        max_rounds: int | None = None,
        max_entries: int | None = None,
    ) -> AgentConfig:
        resolved_provider = provider or os.environ.get("VPA_AGENT_PROVIDER", "codex")
        resolved_model = model or os.environ.get("VPA_AGENT_MODEL", "gpt-5.2")
        parsed = parse_model_ref(resolved_model)
        if parsed is not None:
            parsed_provider, parsed_model = parsed
            if provider is None:
                resolved_provider = parsed_provider
                resolved_model = parsed_model
            elif normalize_provider_id(provider) == parsed_provider:
                resolved_model = parsed_model
        return cls(
            provider=normalize_provider_id(resolved_provider),
            model=resolved_model,
            planner_model=os.environ.get("VPA_AGENT_PLANNER_MODEL") or None,
            summarizer_model=os.environ.get("VPA_AGENT_SUMMARIZER_MODEL") or None,
            temperature=float(os.environ.get("VPA_AGENT_TEMPERATURE", "0.2")),
            timeout_s=float(os.environ.get("VPA_AGENT_TIMEOUT_S", "300")),
            max_rounds=max_rounds or int(os.environ.get("VPA_AGENT_MAX_ROUNDS", "8")),
            max_entries=max_entries or int(os.environ.get("VPA_AGENT_MAX_ENTRIES", "3")),
            agent_profile=agent_profile or os.environ.get("VPA_AGENT_PROFILE", "planner"),
            api_key=(
                api_key
                or os.environ.get("OPENROUTER_API_KEY")
                or os.environ.get("VPA_OPENROUTER_CREDENTIAL")
            ),
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )

    def model_for_role(self, role: str) -> str:
        if role == "planner" and self.planner_model:
            return self.planner_model
        if role in {"analyst", "summarizer"} and self.summarizer_model:
            return self.summarizer_model
        return self.model


def normalize_provider_id(provider: str) -> str:
    key = provider.strip().lower().replace("_", "-")
    aliases = {
        "claude": "claude-code",
        "claude_code": "claude-code",
        "claudecode": "claude-code",
        "anthropic-code": "claude-code",
        "openai-codex": "codex",
        "codex-cli": "codex",
    }
    return aliases.get(key, key)


def parse_model_ref(model_ref: str) -> tuple[str, str] | None:
    """Parse opencode-style provider/model refs for VPA-supported providers.

    OpenRouter model IDs themselves often contain slashes, so only known VPA
    provider prefixes are treated as provider selectors.
    """
    if "/" not in model_ref:
        return None
    provider, _, model = model_ref.partition("/")
    normalized = normalize_provider_id(provider)
    if normalized not in {"openrouter", "claude-code", "codex"}:
        return None
    return normalized, model
