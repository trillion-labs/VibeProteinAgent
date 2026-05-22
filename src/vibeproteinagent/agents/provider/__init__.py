from __future__ import annotations

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.core.errors import InvalidParamsError

from .cli_agent import CliAgentProvider, claude_code_provider, codex_provider
from .logging import SessionLoggingProvider
from .openrouter import OpenRouterProvider
from .types import ChatMessage, ChatProvider, ProviderResponse, ProviderUsage


def provider_from_config(config: AgentConfig) -> OpenRouterProvider | CliAgentProvider:
    if config.provider == "openrouter":
        return OpenRouterProvider(config)
    if config.provider == "claude-code":
        return claude_code_provider(config)
    if config.provider == "codex":
        return codex_provider(config)
    raise InvalidParamsError(
        f"unsupported agent provider {config.provider!r}",
        details={
            "provider": config.provider,
            "supported": ["openrouter", "claude-code", "codex"],
        },
    )


__all__ = [
    "ChatMessage",
    "ChatProvider",
    "CliAgentProvider",
    "OpenRouterProvider",
    "ProviderResponse",
    "ProviderUsage",
    "SessionLoggingProvider",
    "provider_from_config",
]
