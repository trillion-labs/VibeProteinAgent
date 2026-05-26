from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
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


def _structured_fixup_message(provider: _FixupProvider, index: int) -> dict[str, Any]:
    user_payload = json.loads(provider.messages[index][1].content)
    return user_payload["structured_fixup"]


def test_malformed_swap_tool_choice_reprompts_and_then_compiles(
    tmp_workspace: Workspace,
) -> None:
    provider = _FixupProvider(
        [
            {"choice_id": "swap_tool"},
            {"choice_id": "swap_tool", "tool": "rank_designs"},
        ]
    )
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))

    plan = asyncio.run(
        controller._compile_with_fixup(_ir_needing_fixup(), provider=provider)
    )

    assert plan["workflow_plan"]["nodes"][1]["tool"] == "rank_designs"
    assert plan["workflow_plan"]["nodes"][1]["inputs"]["candidates"] == (
        "${outputs.go_fetch.candidates}"
    )
    assert len(provider.messages) == 2

    original_fixup = _structured_fixup_message(provider, 0)
    malformed_fixup = _structured_fixup_message(provider, 1)
    assert original_fixup["error_kind"] == "missing_canonical_input_slot"
    assert malformed_fixup["error_kind"] == "malformed_fixup_response"
    assert malformed_fixup["original_error_kind"] == "missing_canonical_input_slot"
    assert malformed_fixup["original_options"] == original_fixup["options"]
    assert malformed_fixup["options"] == original_fixup["options"]
    assert malformed_fixup["received_choice"] == {"choice_id": "swap_tool"}
    assert malformed_fixup["parse_error"] == (
        "swap_tool fixup choice requires a non-empty tool"
    )


def test_persistent_malformed_responses_raise_structured_fixup(
    tmp_workspace: Workspace,
) -> None:
    provider = _FixupProvider([{"choice_id": "swap_tool"}])
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))

    with pytest.raises(StructuredFixup) as exc_info:
        asyncio.run(
            controller._compile_with_fixup(
                _ir_needing_fixup(),
                provider=provider,
                max_retries=2,
            )
        )

    assert exc_info.value.payload["error_kind"] == "malformed_fixup_response"
    assert exc_info.value.payload["parse_error"] == (
        "swap_tool fixup choice requires a non-empty tool"
    )
    assert len(provider.messages) == 2


def test_invalid_choice_id_is_reprompted_and_surfaces_as_structured_fixup(
    tmp_workspace: Workspace,
) -> None:
    provider = _FixupProvider([{"choice_id": "nonsense"}])
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))

    with pytest.raises(StructuredFixup) as exc_info:
        asyncio.run(
            controller._compile_with_fixup(
                _ir_needing_fixup(),
                provider=provider,
                max_retries=1,
            )
        )

    assert exc_info.value.payload["error_kind"] == "malformed_fixup_response"
    assert exc_info.value.payload["received_choice"] == {"choice_id": "nonsense"}
    assert exc_info.value.payload["parse_error"] == (
        "invalid fixup choice_id: 'nonsense'"
    )


def test_other_apply_value_errors_are_handled_as_malformed_fixup_responses(
    tmp_workspace: Workspace,
) -> None:
    provider = _FixupProvider(
        [
            {
                "choice_id": "swap_tool",
                "tool": "rank_designs",
                "target_node_id": "missing_node",
            }
        ]
    )
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))

    with pytest.raises(StructuredFixup) as exc_info:
        asyncio.run(
            controller._compile_with_fixup(
                _ir_needing_fixup(),
                provider=provider,
                max_retries=1,
            )
        )

    assert exc_info.value.payload["error_kind"] == "malformed_fixup_response"
    assert exc_info.value.payload["parse_error"] == (
        "fixup target node 'missing_node' was not found"
    )
