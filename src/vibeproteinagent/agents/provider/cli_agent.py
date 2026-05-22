from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.core.errors import DependencyError, TimeoutError_

from .types import ChatMessage, ProviderResponse


@dataclass(frozen=True)
class CliAgentSpec:
    provider_name: str
    command_env: str
    args_env: str
    default_command: str
    default_args: str
    model_arg_template: str


class CliAgentProvider:
    """Chat-provider adapter for local coding-agent CLIs.

    The prompt is sent over stdin. Arguments are configurable via environment
    variables so each installation can match its installed Claude Code or Codex
    CLI without VPA baking in project-specific shell wrappers.
    """

    def __init__(self, config: AgentConfig, spec: CliAgentSpec) -> None:
        self.config = config
        self.spec = spec
        self.provider_name = spec.provider_name

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> ProviderResponse:
        del temperature, idempotency_key
        return await asyncio.to_thread(
            self._chat_sync,
            messages,
            model=model,
            response_format=response_format,
        )

    def _chat_sync(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None,
        response_format: dict[str, Any] | None,
    ) -> ProviderResponse:
        command = os.environ.get(self.spec.command_env, self.spec.default_command)
        args_text = os.environ.get(self.spec.args_env, self.spec.default_args)
        executable = shutil.which(command)
        if executable is None and command in {"python", "python3"}:
            executable = sys.executable
        if executable is None:
            raise DependencyError(
                f"{self.provider_name} command {command!r} was not found",
                details={
                    "provider": self.provider_name,
                    "command_env": self.spec.command_env,
                    "args_env": self.spec.args_env,
                },
            )

        chosen_model = model or self.config.model
        prompt = _messages_to_prompt(
            messages,
            provider=self.provider_name,
            model=chosen_model,
            agent_profile=self.config.agent_profile,
            wants_json=response_format == {"type": "json_object"},
        )
        cmd = [executable, *_render_args(args_text, chosen_model, self.spec.model_arg_template)]
        env = os.environ.copy()
        env["VPA_AGENT_PROVIDER"] = self.provider_name
        env["VPA_AGENT_MODEL"] = chosen_model
        env["VPA_AGENT_PROFILE"] = self.config.agent_profile
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError_(
                f"{self.provider_name} command timed out",
                details={"provider": self.provider_name, "timeout_s": self.config.timeout_s},
            ) from exc

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            raise DependencyError(
                f"{self.provider_name} command failed",
                details={
                    "provider": self.provider_name,
                    "returncode": proc.returncode,
                    "stdout": stdout[-4000:],
                    "stderr": stderr[-4000:],
                },
            )
        content = (
            _extract_json_object(stdout)
            if response_format == {"type": "json_object"}
            else stdout
        )
        return ProviderResponse(
            provider=self.provider_name,
            model=chosen_model,
            content=content,
            finish_reason="stop",
            raw={"stderr": stderr[-1000:] if stderr else None},
        )


def claude_code_provider(config: AgentConfig) -> CliAgentProvider:
    return CliAgentProvider(
        config,
        CliAgentSpec(
            provider_name="claude-code",
            command_env="VPA_CLAUDE_CODE_COMMAND",
            args_env="VPA_CLAUDE_CODE_ARGS",
            default_command="claude",
            default_args="-p --model {model} --output-format text",
            model_arg_template="--model {model}",
        ),
    )


def codex_provider(config: AgentConfig) -> CliAgentProvider:
    return CliAgentProvider(
        config,
        CliAgentSpec(
            provider_name="codex",
            command_env="VPA_CODEX_COMMAND",
            args_env="VPA_CODEX_ARGS",
            default_command="codex",
            default_args="exec --model {model} --skip-git-repo-check --sandbox read-only -",
            model_arg_template="--model {model}",
        ),
    )


def _render_args(args_text: str, model: str, model_arg_template: str) -> list[str]:
    if "{model}" in args_text:
        return shlex.split(args_text.format(model=shlex.quote(model)))
    args = shlex.split(args_text)
    model_args = shlex.split(model_arg_template.format(model=shlex.quote(model)))
    if _has_model_flag(args):
        return args
    return [*args, *model_args]


def _has_model_flag(args: list[str]) -> bool:
    return any(arg == "--model" or arg.startswith("--model=") or arg in {"-m"} for arg in args)


def _messages_to_prompt(
    messages: Sequence[ChatMessage],
    *,
    provider: str,
    model: str,
    agent_profile: str,
    wants_json: bool,
) -> str:
    lines = [
        f"VPA provider: {provider}",
        f"VPA model: {model}",
        f"VPA agent profile: {agent_profile}",
        "",
    ]
    for message in messages:
        lines.append(f"{message.role.upper()}:")
        lines.append(message.content)
        lines.append("")
    if wants_json:
        lines.append("Return only one valid JSON object. Do not wrap it in Markdown.")
    return "\n".join(lines)


def _extract_json_object(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return text[start : start + end]
    return text
