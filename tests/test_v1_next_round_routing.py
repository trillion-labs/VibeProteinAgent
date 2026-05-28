from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.evaluation import decision as decision_mod
from vibeproteinagent.agents.execution import ExecutionPolicy
from vibeproteinagent.agents.execution import controller as controller_mod
from vibeproteinagent.agents.execution.controller import AgentController, WorkflowExecutionOutcome
from vibeproteinagent.agents.execution.round_state import PriorRoundContext
from vibeproteinagent.agents.planning import workflow_planner as planner_mod
from vibeproteinagent.agents.planning.prompts import (
    PLANNER_IR_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.workspace import Workspace


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
        payload = self.payloads[len(self.messages) - 1]
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(payload))


def _candidate(candidate_id: str = "cand_prior") -> dict[str, Any]:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {"tool": "go_sequences", "local_id": "seq_0000", "index": 0},
        "sequence": "ACDE",
        "scores": {"go_sequences": {"confidence": 0.8}},
        "artifacts": {},
    }


def _round1_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Collect GO candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_candidates",
                "tool": "go_sequences",
                "intent": "Retrieve GO candidates.",
                "inputs": {"go_id": "GO:0003824", "max_sequences": 1},
                "upstream": [],
            }
        ],
        "finalization": {"success_criteria": {"needs_ranking": True}},
    }


def _round2_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Rank prior-round survivors.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "rank_prior",
                "tool": "rank_designs",
                "intent": "Rank surviving prior-round candidates.",
                "inputs": {"top_k": 1},
                "upstream": [
                    {"node": "prior_round", "relationship": "candidate_source"}
                ],
            }
        ],
        "finalization": {"success_criteria": {"ranked_candidates": True}},
    }


def _selection(*tools: str) -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "selected_tools": [
            {"name": tool, "reason": "Selected tools for the current planning round."}
            for tool in tools
        ],
        "missing_inputs": [],
    }


def _classification() -> ModeClassification:
    return ModeClassification(
        mode="agent_plan",
        reason="test",
        requires_llm_planning=True,
    )


def _run_controller(controller: AgentController, workspace: Workspace) -> dict[str, Any]:
    run = controller.session_store.create_run("next round v1 test")
    broker = AgentToolBroker(workspace, controller.registry, controller.session_store, run.run_id)
    return asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "Find and rank GO candidates.",
            _classification(),
            ExecutionPolicy(submit=True),
        )
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


def test_v1_next_round_uses_planner_ir_with_prior_round_scope(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    provider = _PlannerProvider(
        [
            _selection("go_sequences"),
            _round1_ir(),
            _selection("rank_designs"),
            _round2_ir(),
        ]
    )
    monkeypatch.setattr(controller_mod, "provider_from_config", lambda _config: provider)
    calls = {"plan_v1": 0}
    original_plan_v1 = planner_mod.WorkflowPlanner._plan_v1

    async def spy_plan_v1(self, *args, **kwargs):
        calls["plan_v1"] += 1
        return await original_plan_v1(self, *args, **kwargs)

    monkeypatch.setattr(planner_mod.WorkflowPlanner, "_plan_v1", spy_plan_v1)

    async def fake_decide(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "next_round",
            "reasoning": "Rank survivors in a second round.",
            "next_round": {
                "reason": "Survivors need ranking before final submission.",
                "recommended": True,
            },
            "metrics_used": ["go_sequences.confidence"],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", fake_decide)
    captured: dict[str, Any] = {"execute_calls": []}

    def fake_execute(
        self,
        run_id,
        broker,
        plan,
        params,
        policy,
        prior_round_context=None,
    ):
        del self, run_id, broker, params, policy
        scope = controller._build_executor_scope(prior_round=prior_round_context)
        captured["execute_calls"].append(
            {
                "plan": plan,
                "prior_round_context": prior_round_context,
                "scope": scope,
            }
        )
        if len(captured["execute_calls"]) == 1:
            return WorkflowExecutionOutcome(
                "wf_round1",
                "completed",
                True,
                {
                    "workflow_id": "wf_round1",
                    "state": "completed",
                    "job_ids": ["job_go"],
                    "nodes": [
                        {
                            "node_id": "go_candidates",
                            "tool": "go_sequences",
                            "state": "completed",
                            "job_id": "job_go",
                            "output": {"candidates": [_candidate()]},
                        }
                    ],
                    "metrics_used": ["go_sequences.confidence"],
                    "top_designs": [],
                },
            )
        return WorkflowExecutionOutcome(
            "wf_round2",
            "completed",
            True,
            {
                "workflow_id": "wf_round2",
                "state": "completed",
                "job_ids": ["job_rank"],
                "nodes": [],
                "final_submission": {"submission_id": "sub_final"},
                "metrics_used": ["rank_designs.score"],
            },
        )

    monkeypatch.setattr(controller_mod.AgentController, "_execute_validated_plan", fake_execute)
    controller = AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3))

    report = _run_controller(controller, tmp_workspace)

    assert report["state"] == "completed"
    assert calls == {"plan_v1": 2}
    assert not hasattr(controller, "_maybe_compile_planner_ir")
    round2_prior_context = captured["execute_calls"][1]["prior_round_context"]
    assert isinstance(round2_prior_context, PriorRoundContext)
    assert round2_prior_context.round_number == 2
    assert round2_prior_context.surviving_candidates == [_candidate()]
    assert round2_prior_context.prior_decision_reason == (
        "Survivors need ranking before final submission."
    )
    assert controller._round_state is None
    assert [message[0].content for message in provider.messages] == [
        TOOL_SELECTION_SYSTEM_PROMPT,
        PLANNER_IR_SYSTEM_PROMPT,
        TOOL_SELECTION_SYSTEM_PROMPT,
        PLANNER_IR_SYSTEM_PROMPT,
    ]
    round2_user_message = json.loads(provider.messages[3][1].content)
    assert "prior_round_context" in round2_user_message
    assert "surviving_candidates" not in round2_user_message["prior_round_context"]
    assert (
        round2_user_message["prior_round_context"]["digest"]["latest_round"][
            "survivor_pool_summary"
        ]["count"]
        == 1
    )
    round2_execute = captured["execute_calls"][1]
    assert round2_execute["plan"]["workflow_plan"]["nodes"][0]["inputs"]["candidates"] == (
        "${outputs.prior_round.candidates}"
    )
    assert round2_execute["scope"]["outputs"]["prior_round"]["candidates"] == [_candidate()]
