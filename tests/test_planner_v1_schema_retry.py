from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.planning.workflow_planner import (
    PlannerIRSchemaViolation,
    WorkflowPlanner,
)
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse


class _PlannerProvider:
    provider_name = "fake"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        self.messages.append(list(messages))
        payload = self.payloads[min(len(self.messages) - 1, len(self.payloads) - 1)]
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(payload))


def _valid_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Collect and rank candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_candidates",
                "tool": "go_sequences",
                "intent": "Retrieve candidates.",
                "inputs": {"go_id": "GO:0003824"},
                "upstream": [],
            },
            {
                "id": "rank_candidates",
                "tool": "rank_designs",
                "intent": "Rank candidates.",
                "inputs": {"top_k": 1},
                "upstream": [
                    {"node": "go_candidates", "relationship": "candidate_source"}
                ],
            },
        ],
    }


def _selection() -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "selected_tools": [
            {"name": "go_sequences", "reason": "Use retrieval followed by ranking."},
            {"name": "rank_designs", "reason": "Use retrieval followed by ranking."},
        ],
        "missing_inputs": [],
    }


def _invalid_relationship_ir() -> dict[str, Any]:
    ir = _valid_ir()
    ir["nodes"][1]["upstream"][0]["relationship"] = "evidence"
    return ir


def test_planner_v1_retries_with_structured_schema_error() -> None:
    provider = _PlannerProvider([_selection(), _invalid_relationship_ir(), _valid_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "find GO candidates",
            available_tools=[
                {"name": "go_sequences", "description": "Retrieve GO sequences."},
                {"name": "rank_designs", "description": "Rank candidates."},
            ],
            max_schema_retries=2,
        )
    )

    assert len(provider.messages) == 3
    retry_prompt = json.loads(provider.messages[2][1].content)
    structured_error = retry_prompt["previous_planner_ir_error"]["structured_error"]
    assert structured_error["error_kind"] == "planner_ir_schema_violation"
    assert structured_error["violations"][0]["issue"] == (
        "relationship_not_allowed_in_v1"
    )
    assert plan["workflow_plan"]["nodes"][1]["inputs"]["candidates"] == (
        "${outputs.go_candidates.candidates}"
    )


def test_planner_v1_raises_after_schema_retry_exhaustion() -> None:
    provider = _PlannerProvider([_selection(), _invalid_relationship_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    with pytest.raises(PlannerIRSchemaViolation) as excinfo:
        asyncio.run(
            planner._plan_v1(
                "find GO candidates",
                available_tools=[
                    {"name": "go_sequences", "description": "Retrieve GO sequences."},
                    {"name": "rank_designs", "description": "Rank candidates."},
                ],
                max_schema_retries=1,
            )
        )

    assert len(provider.messages) == 3
    assert excinfo.value.structured_error["violations"][0]["issue"] == (
        "relationship_not_allowed_in_v1"
    )
