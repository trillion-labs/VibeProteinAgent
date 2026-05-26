from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.execution import controller as controller_mod
from vibeproteinagent.agents.execution.controller import AgentController
from vibeproteinagent.agents.planning.fixup import StructuredFixup
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.workspace import Workspace


class _FixupProvider:
    provider_name = "fake"

    def __init__(self, choices: list[dict[str, Any]]) -> None:
        self.choices = choices
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        self.messages.append(list(messages))
        choice = self.choices[min(len(self.messages) - 1, len(self.choices) - 1)]
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(choice))


def _ir_needing_fixup() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Fix an unsupported canonical consumer.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_fetch",
                "tool": "go_sequences",
                "intent": "Retrieve candidates.",
                "inputs": {"go_id": "GO:0003824"},
                "upstream": [],
            },
            {
                "id": "analyze",
                "tool": "noop",
                "intent": "Consume canonical candidates.",
                "inputs": {},
                "upstream": [
                    {"node": "go_fetch", "relationship": "candidate_source"}
                ],
            },
        ],
    }


def test_controller_compile_fixup_transport_retries_with_choice(
    tmp_workspace: Workspace,
) -> None:
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    provider = _FixupProvider([{"choice_id": "swap_tool", "tool": "rank_designs"}])

    plan = asyncio.run(
        controller._compile_with_fixup(_ir_needing_fixup(), provider=provider)
    )

    assert plan["workflow_plan"]["nodes"][1]["tool"] == "rank_designs"
    assert plan["workflow_plan"]["nodes"][1]["inputs"]["candidates"] == (
        "${outputs.go_fetch.candidates}"
    )
    assert len(provider.messages) == 1
    user_payload = json.loads(provider.messages[0][1].content)
    assert user_payload["structured_fixup"]["error_kind"] == (
        "missing_canonical_input_slot"
    )
    assert user_payload["planner_ir"]["nodes"][1]["tool"] == "noop"


def test_controller_compile_fixup_respects_retry_cap(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    provider = _FixupProvider([{"choice_id": "remove_node"}])

    def no_op_apply(llm_ir, fixup_payload, choice):
        del fixup_payload, choice
        return llm_ir

    monkeypatch.setattr(controller_mod, "apply_fixup_choice", no_op_apply)

    with pytest.raises(StructuredFixup):
        asyncio.run(
            controller._compile_with_fixup(
                _ir_needing_fixup(),
                provider=provider,
                max_retries=2,
            )
        )

    assert len(provider.messages) == 2


def test_late_planner_ir_compile_shim_is_removed(tmp_workspace: Workspace) -> None:
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    assert not hasattr(controller, "_maybe_compile_planner_ir")
