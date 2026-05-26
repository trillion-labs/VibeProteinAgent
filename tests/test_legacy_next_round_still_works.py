from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.evaluation import decision as decision_mod
from vibeproteinagent.agents.execution import ExecutionPolicy
from vibeproteinagent.agents.execution import controller as controller_mod
from vibeproteinagent.agents.execution.controller import AgentController, WorkflowExecutionOutcome
from vibeproteinagent.agents.planning import workflow_planner as planner_mod
from vibeproteinagent.agents.planning.prompts import NEXT_ROUND_SYSTEM_PROMPT
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.workspace import Workspace


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
        if messages[0].content == NEXT_ROUND_SYSTEM_PROMPT and len(self.payloads) >= 2:
            payload = self.payloads[1]
        else:
            payload = self.payloads[0] if self.payloads else {}
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(payload))


def _plan(name: str = "legacy_plan") -> dict[str, Any]:
    return {
        "mode": "agent_plan",
        "goal": "Collect GO candidates.",
        "workflow_plan": {
            "name": name,
            "version": "0.1.0",
            "nodes": [
                {
                    "id": "go_candidates",
                    "tool": "go_sequences",
                    "inputs": {"go_id": "GO:0003824", "max_sequences": 1},
                }
            ],
        },
        "success_criteria": {},
    }


def _selection() -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "candidate_tools": ["go_sequences"],
        "reason": "GO sequence retrieval remains useful.",
        "missing_inputs": [],
    }


def _classification() -> ModeClassification:
    return ModeClassification(
        mode="agent_plan",
        reason="test",
        requires_llm_planning=True,
    )


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


def _run(
    controller: AgentController,
    workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    executed_plans: list[dict[str, Any]] = []

    def fake_execute(self, run_id, broker, plan, params, policy, prior_round_context=None):
        del self, run_id, broker, params, policy, prior_round_context
        executed_plans.append(plan)
        if len(executed_plans) == 1:
            return WorkflowExecutionOutcome(
                "wf1",
                "completed",
                True,
                {
                    "workflow_id": "wf1",
                    "state": "completed",
                    "job_ids": ["job_go"],
                    "nodes": [],
                    "metrics_used": [],
                    "top_designs": [],
                },
            )
        return WorkflowExecutionOutcome(
            "wf2",
            "completed",
            True,
            {
                "workflow_id": "wf2",
                "state": "completed",
                "job_ids": ["job_go_2"],
                "nodes": [],
                "final_submission": {"submission_id": "sub_final"},
                "metrics_used": [],
            },
        )

    monkeypatch.setattr(controller_mod.AgentController, "_execute_validated_plan", fake_execute)
    run = controller.session_store.create_run("legacy next round test")
    broker = AgentToolBroker(workspace, controller.registry, controller.session_store, run.run_id)
    asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "Find GO candidates.",
            _classification(),
            ExecutionPolicy(submit=True),
        )
    )
    return executed_plans


def test_legacy_flag_off_automatic_next_round_uses_legacy_prompt(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    _allow_tool_environment(monkeypatch)
    provider = _Provider([])
    monkeypatch.setattr(controller_mod, "provider_from_config", lambda _config: provider)
    monkeypatch.setattr(planner_mod.WorkflowPlanner, "draft_plan", _fake_initial_plan)
    legacy_calls = {"draft_next_round": 0}

    async def fake_draft_next_round(self, *args, **kwargs):
        del self, args, kwargs
        legacy_calls["draft_next_round"] += 1
        return _plan("legacy_next_round")

    monkeypatch.setattr(
        planner_mod.WorkflowPlanner,
        "draft_next_round_plan",
        fake_draft_next_round,
    )

    async def fake_decide(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "next_round",
            "reasoning": "Try another legacy round.",
            "next_round": {"reason": "Try another round.", "recommended": True},
            "metrics_used": [],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", fake_decide)
    plan_v1_calls = {"count": 0}

    async def fail_plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        plan_v1_calls["count"] += 1
        raise AssertionError("legacy next-round path must not call _plan_v1")

    monkeypatch.setattr(planner_mod.WorkflowPlanner, "_plan_v1", fail_plan_v1)
    controller = AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3))

    executed_plans = _run(controller, tmp_workspace, monkeypatch)

    assert executed_plans[1]["workflow_plan"]["name"] == "legacy_next_round"
    assert plan_v1_calls["count"] == 0
    assert legacy_calls["draft_next_round"] == 1
    assert NEXT_ROUND_SYSTEM_PROMPT


def test_legacy_flag_off_decision_plan_escape_hatch_still_works(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "compat")
    _allow_tool_environment(monkeypatch)
    provider = _Provider([])
    monkeypatch.setattr(controller_mod, "provider_from_config", lambda _config: provider)
    monkeypatch.setattr(planner_mod.WorkflowPlanner, "draft_plan", _fake_initial_plan)

    supplied_next_plan = _plan("decision_supplied_next_round")

    async def fake_decide(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "next_round",
            "reasoning": "Use the legacy escape hatch.",
            "next_round": {
                "reason": "Use supplied executor plan.",
                "recommended": True,
                "plan": supplied_next_plan,
            },
            "metrics_used": [],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", fake_decide)

    async def fail_draft_next_round(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("decision-supplied legacy plan should bypass planner")

    monkeypatch.setattr(
        planner_mod.WorkflowPlanner,
        "draft_next_round_plan",
        fail_draft_next_round,
    )
    controller = AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3))

    executed_plans = _run(controller, tmp_workspace, monkeypatch)

    assert executed_plans[1]["workflow_plan"]["name"] == "decision_supplied_next_round"
    assert provider.messages == []


async def _fake_initial_plan(self, *args, **kwargs):
    del self, args, kwargs
    return _plan("legacy_initial")
