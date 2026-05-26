from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.planning import workflow_planner as planner_mod
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse

PLANNER_IR_FIXTURE = Path(__file__).parent / "fixtures" / "planner_ir" / "two_go_to_rank.json"


class _Provider:
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


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.discover()
    return registry


def _spy_legacy_validators(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {
        "_validate_static_output_refs": 0,
        "_validate_static_output_ref_types": 0,
        "_validate_static_job_refs": 0,
    }

    def _spy(name: str):
        def _inner(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            calls[name] += 1

        return _inner

    for name in calls:
        monkeypatch.setattr(planner_mod, name, _spy(name))
    return calls


def _allow_tool_environment(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_plan_v1_compiled_output_skips_legacy_static_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    calls = _spy_legacy_validators(monkeypatch)
    planner_ir = json.loads(PLANNER_IR_FIXTURE.read_text())
    provider = _Provider([planner_ir])
    planner = WorkflowPlanner(_registry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates from two sources and rank them.",
            available_tools=[
                {"name": "go_sequences", "description": "Retrieve GO sequences."},
                {"name": "rank_designs", "description": "Rank candidate designs."},
            ],
        )
    )
    assert "merge_rank_candidates" in {
        node["id"] for node in plan["workflow_plan"]["nodes"]
    }

    planner.validate_invocation(plan, {}, allow_internal_tools=True)

    assert calls == {
        "_validate_static_output_refs": 0,
        "_validate_static_output_ref_types": 0,
        "_validate_static_job_refs": 0,
    }


def test_legacy_draft_plan_still_runs_legacy_static_validators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    _allow_tool_environment(monkeypatch)
    calls = _spy_legacy_validators(monkeypatch)
    provider = _Provider(
        [
            {
                "mode": "tool_selection",
                "candidate_tools": ["go_sequences"],
                "reason": "GO sequence retrieval is enough for this fixture.",
                "missing_inputs": [],
            },
            {
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
            },
        ]
    )
    planner = WorkflowPlanner(_registry(), provider)

    plan = asyncio.run(
        planner.draft_plan(
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
    assert calls == {
        "_validate_static_output_refs": 1,
        "_validate_static_output_ref_types": 1,
        "_validate_static_job_refs": 1,
    }
