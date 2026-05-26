from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.execution.controller import AgentController
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.workspace import Workspace

FIXTURE = Path(__file__).parent / "fixtures" / "planner_v1" / "go_rank_request.json"


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


def _fixture_ir() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())["planner_ir"]


def _tool_catalog() -> list[dict[str, str]]:
    return [
        {"name": "go_sequences", "description": "Retrieve GO sequences."},
        {"name": "sequence_analysis", "description": "Analyze protein sequences."},
        {"name": "rank_designs", "description": "Rank candidates."},
    ]


def test_plan_v1_compiles_single_producer_consumer() -> None:
    provider = _PlannerProvider([_fixture_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates and rank them.",
            available_tools=_tool_catalog(),
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == ["go_candidates", "rank_candidates"]
    assert nodes[1]["inputs"]["candidates"] == "${outputs.go_candidates.candidates}"


def test_plan_v1_compiles_multi_upstream_merge() -> None:
    ir = _fixture_ir()
    second = copy.deepcopy(ir["nodes"][0])
    second["id"] = "go_more_candidates"
    second["inputs"] = {"go_id": "GO:0008150", "max_sequences": 2}
    ir["nodes"].insert(1, second)
    ir["nodes"][2]["upstream"].append(
        {"node": "go_more_candidates", "relationship": "candidate_source"}
    )
    provider = _PlannerProvider([ir])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1("Find and rank GO candidates.", available_tools=_tool_catalog())
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == [
        "go_candidates",
        "go_more_candidates",
        "merge_rank_candidates_candidates",
        "rank_candidates",
    ]
    assert nodes[2]["tool"] == "merge_candidates"
    assert nodes[2]["inputs"]["candidate_sets"] == [
        "${outputs.go_candidates.candidates}",
        "${outputs.go_more_candidates.candidates}",
    ]
    assert nodes[3]["inputs"]["candidates"] == (
        "${outputs.merge_rank_candidates_candidates.candidates}"
    )


def test_plan_v1_reuses_controller_fixup_transport(tmp_workspace: Workspace) -> None:
    ir = _fixture_ir()
    ir["nodes"][1]["tool"] = "go_sequences"
    provider = _PlannerProvider(
        [ir, {"choice_id": "swap_tool", "tool": "rank_designs"}]
    )
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    planner = WorkflowPlanner(ToolRegistry(), provider)

    async def compile_with_fixup(planner_ir: dict[str, Any]) -> dict[str, Any]:
        return await controller._compile_with_fixup(planner_ir, provider=provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates and rank them.",
            available_tools=_tool_catalog(),
            compile_with_fixup=compile_with_fixup,
        )
    )

    assert len(provider.messages) == 2
    assert plan["workflow_plan"]["nodes"][1]["tool"] == "rank_designs"
    assert plan["workflow_plan"]["nodes"][1]["inputs"]["candidates"] == (
        "${outputs.go_candidates.candidates}"
    )
