from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.planning import workflow_planner as planner_mod
from vibeproteinagent.agents.planning.prompts import PLANNER_IR_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse


class _LegacyPlannerProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        self.messages.append(list(messages))
        prompt = json.loads(messages[-1].content)
        if prompt["output_contract"]["mode"] == "tool_selection":
            content = {
                "mode": "tool_selection",
                "candidate_tools": ["go_sequences"],
                "reason": "GO sequence retrieval is enough for this fixture.",
                "missing_inputs": [],
            }
        else:
            content = {
                "mode": "agent_plan",
                "goal": "Collect one GO-associated candidate sequence.",
                "assumptions": [],
                "missing_inputs": [],
                "required_tools": ["go_sequences"],
                "params": {},
                "workflow_plan": {
                    "name": "legacy_go_plan",
                    "version": "0.1.0",
                    "nodes": [
                        {
                            "id": "go_candidates",
                            "tool": "go_sequences",
                            "inputs": {"go_id": "GO:0003824", "max_sequences": 1},
                        }
                    ],
                },
                "success_criteria": {"workflow_completes": True},
                "warm_tools": [],
                "estimated_llm_calls": 2,
            }
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(content))


def test_legacy_draft_plan_runs_when_planner_ir_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    monkeypatch.setattr(
        planner_mod,
        "_descriptor_availability",
        lambda descriptor, adapter_cls: {"available": True},
    )
    monkeypatch.setattr(
        planner_mod,
        "_install_readiness",
        lambda tool_name: {"install_ready": True},
    )
    provider = _LegacyPlannerProvider()
    registry = ToolRegistry()
    registry.discover()

    plan = asyncio.run(
        WorkflowPlanner(registry, provider).draft_plan(
            "Find one representative GO sequence.",
            available_tools=[
                {
                    "name": "go_sequences",
                    "description": "Retrieve protein sequences associated with a GO term.",
                }
            ],
            workflows=[],
            model="fake",
        )
    )

    assert plan["workflow_plan"]["name"] == "legacy_go_plan"
    assert plan["workflow_plan"]["nodes"][0]["tool"] == "go_sequences"
    assert provider.messages[1][0].content == PLANNER_SYSTEM_PROMPT
    assert provider.messages[1][0].content != PLANNER_IR_SYSTEM_PROMPT


def test_deleted_legacy_heuristics_are_not_importable() -> None:
    assert not hasattr(planner_mod, "_repair_scalar_inputs_from_output_arrays")
    assert not hasattr(planner_mod, "_fallback_workflow_plan_from_selected_tools")
    assert not hasattr(planner_mod, "_CANDIDATE_OUTPUT_FIELDS")
