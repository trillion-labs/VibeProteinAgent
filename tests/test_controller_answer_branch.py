from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.evaluation import decision as decision_mod
from vibeproteinagent.agents.execution.controller import (
    AgentController,
    WorkflowExecutionOutcome,
)
from vibeproteinagent.agents.execution.policies import ExecutionPolicy
from vibeproteinagent.agents.execution.run_result import AgentRunResult
from vibeproteinagent.agents.planning import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.workspace import Workspace


def _plan(name: str = "analysis") -> dict[str, Any]:
    return {
        "mode": "agent_plan",
        "source_kind": "workflow_yaml",
        "goal": "run local analysis",
        "assumptions": [],
        "missing_inputs": [],
        "required_tools": ["noop"],
        "workflow_plan": {
            "name": name,
            "version": "0.1.0",
            "nodes": [
                {
                    "id": "analyze",
                    "tool": "noop",
                    "inputs": {"payload": "analysis"},
                }
            ],
        },
        "success_criteria": {"answer_question": True},
        "warm_tools": [],
        "estimated_llm_calls": 1,
    }


def _classification() -> ModeClassification:
    return ModeClassification(
        mode="agent_plan",
        reason="test",
        requires_llm_planning=True,
    )


def _run_controller(controller: AgentController, workspace: Workspace) -> dict[str, Any]:
    run = controller.session_store.create_run("answer branch test")
    broker = AgentToolBroker(workspace, controller.registry, controller.session_store, run.run_id)
    return asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "What is the pI of P44612?",
            _classification(),
            ExecutionPolicy(submit=True),
        )
    )


def _outcome(workflow_id: str, job_id: str) -> WorkflowExecutionOutcome:
    return WorkflowExecutionOutcome(
        workflow_id=workflow_id,
        state="completed",
        executed=True,
        summary={
            "state": "completed",
            "workflow_id": workflow_id,
            "job_ids": [job_id],
            "metrics_used": ["pI"],
            "nodes": [
                {
                    "node_id": "analyze",
                    "tool": "noop",
                    "state": "completed",
                    "job_id": job_id,
                    "output": {"representative_sequences": [{"pI": 6.74}]},
                }
            ],
        },
    )


class _DecisionProvider:
    provider_name = "fake"

    def __init__(self, decisions: Sequence[dict[str, Any]]) -> None:
        self.decisions = list(decisions)
        self.index = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        if "result critic" not in messages[0].content:
            raise AssertionError("planner should be monkeypatched")
        payload = self.decisions[self.index]
        self.index += 1
        return ProviderResponse(
            provider="fake",
            model="fake",
            content=json.dumps(payload),
        )


def _answer_decision() -> dict[str, Any]:
    return {
        "decision": "answer",
        "reasoning": "User asked for pI value; computed via sequence_analysis.",
        "answer": {
            "text": "The pI of UniProt P44612 is 6.74.",
            "evidence": [
                {
                    "node_id": "analyze",
                    "field": "representative_sequences[0].pI",
                    "summary": "Computed pI from sequence_analysis",
                }
            ],
        },
        "metrics_used": ["pI"],
    }


def test_answer_decision_returns_answer_result_without_submission_or_next_round(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    provider = _DecisionProvider([_answer_decision()])
    execute_calls = {"count": 0}

    async def draft_plan(self, *args, **kwargs):
        del self, args, kwargs
        return _plan()

    async def fail_next_round(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("answer decision must not start another round")

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        execute_calls["count"] += 1
        return _outcome("wf_answer", "job_analysis")

    def fail_submit(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("answer decision must not call submit_candidate")

    monkeypatch.setattr(WorkflowPlanner, "draft_plan", draft_plan)
    monkeypatch.setattr(WorkflowPlanner, "draft_next_round_plan", fail_next_round)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(AgentController, "_execute_submit_candidate_decision", fail_submit)

    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake", max_rounds=2),
    )
    report = _run_controller(controller, tmp_workspace)

    assert execute_calls["count"] == 1
    assert report["state"] == "completed"
    assert report["result"] == {
        "kind": "answer",
        "reasoning": "User asked for pI value; computed via sequence_analysis.",
        "text": "The pI of UniProt P44612 is 6.74.",
        "evidence": [
            {
                "node_id": "analyze",
                "field": "representative_sequences[0].pI",
                "summary": "Computed pI from sequence_analysis",
            }
        ],
    }
    assert report["summary"]["latest"]["answer"] == report["result"]


def test_submit_candidate_decision_behavior_is_unchanged(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    provider = _DecisionProvider(
        [
            {
                "decision": "submit_candidate",
                "reasoning": "ranking is sufficient",
                "submit_candidate": {
                    "candidates": [
                        {
                            "job_id": "job_rank",
                            "candidate_id": "design_1",
                            "justification": "top ranked",
                        }
                    ]
                },
                "metrics_used": ["score"],
            }
        ]
    )
    submit_calls = {"count": 0}

    async def draft_plan(self, *args, **kwargs):
        del self, args, kwargs
        return _plan()

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        return _outcome("wf_submit", "job_rank")

    def submit(self, run_id, broker, decision, *, round_index, policy):
        del self, run_id, broker, round_index, policy
        submit_calls["count"] += 1
        assert decision["decision"] == "submit_candidate"
        return (
            {
                "submission_id": "sub_1",
                "outcome": "accepted",
                "terminal": True,
                "candidates_resolved": [{"candidate_id": "design_1"}],
                "reasoning": "ranking is sufficient",
            },
            "job_submit",
        )

    monkeypatch.setattr(WorkflowPlanner, "draft_plan", draft_plan)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(AgentController, "_execute_submit_candidate_decision", submit)

    report = _run_controller(
        AgentController(tmp_workspace, config=AgentConfig(max_rounds=1, max_entries=3)),
        tmp_workspace,
    )

    assert submit_calls["count"] == 1
    assert report["result"] == {"workflow_ids": ["wf_submit"], "state": "completed"}
    assert report["decision"]["submit_job_id"] == "job_submit"


def test_next_round_decision_behavior_is_unchanged(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    provider = _DecisionProvider(
        [
            {
                "decision": "next_round",
                "reasoning": "More local analysis is useful.",
                "next_round": {"reason": "Run one more pass.", "recommended": True},
                "metrics_used": ["pI"],
            },
            {
                "decision": "stop",
                "reasoning": "Second round completed.",
                "metrics_used": ["pI"],
            },
        ]
    )
    executed: list[str] = []

    async def draft_plan(self, *args, **kwargs):
        del self, args, kwargs
        return _plan("round1")

    async def draft_next_round(self, *args, **kwargs):
        del self, args, kwargs
        return _plan("round2")

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        index = len(executed) + 1
        executed.append(f"wf_round{index}")
        return _outcome(f"wf_round{index}", f"job_round{index}")

    monkeypatch.setattr(WorkflowPlanner, "draft_plan", draft_plan)
    monkeypatch.setattr(WorkflowPlanner, "draft_next_round_plan", draft_next_round)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)

    report = _run_controller(
        AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3)),
        tmp_workspace,
    )

    assert executed == ["wf_round1", "wf_round2"]
    assert [item["decision"] for item in report["decisions"]] == ["next_round", "stop"]
    assert report["result"] == {
        "workflow_ids": ["wf_round1", "wf_round2"],
        "state": "completed",
    }


def test_multi_round_agent_can_end_with_answer(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_PLANNER_IR", "off")
    provider = _DecisionProvider(
        [
            {
                "decision": "next_round",
                "reasoning": "Need a second local calculation.",
                "next_round": {"reason": "Compute the requested property.", "recommended": True},
                "metrics_used": ["pI"],
            },
            _answer_decision(),
        ]
    )

    async def draft_plan(self, *args, **kwargs):
        del self, args, kwargs
        return _plan("round1")

    async def draft_next_round(self, *args, **kwargs):
        del self, args, kwargs
        return _plan("round2")

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        index = provider.index + 1
        return _outcome(f"wf_round{index}", f"job_round{index}")

    def fail_submit(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("answer decision must not call submit_candidate")

    monkeypatch.setattr(WorkflowPlanner, "draft_plan", draft_plan)
    monkeypatch.setattr(WorkflowPlanner, "draft_next_round_plan", draft_next_round)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(AgentController, "_execute_submit_candidate_decision", fail_submit)

    report = _run_controller(
        AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3)),
        tmp_workspace,
    )

    assert [item["decision"] for item in report["decisions"]] == ["next_round", "answer"]
    assert report["result"]["kind"] == "answer"
    assert report["result"]["text"] == "The pI of UniProt P44612 is 6.74."


def test_answer_result_requires_text() -> None:
    with pytest.raises(ValueError):
        AgentRunResult(kind="answer", text=None)
