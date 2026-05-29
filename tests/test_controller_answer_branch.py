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
    _candidate_pool_from_summary,
    _first_submit_candidate_ref,
    _summary_has_candidate_pool,
)
from vibeproteinagent.agents.execution.policies import ExecutionPolicy
from vibeproteinagent.agents.execution.run_result import AgentRunResult
from vibeproteinagent.agents.planning import WorkflowPlanner
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.core.errors import InvalidParamsError
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


def _candidate_outcome(workflow_id: str, job_id: str) -> WorkflowExecutionOutcome:
    out = _outcome(workflow_id, job_id)
    out.summary["nodes"][0]["output"] = {
        "representative_sequences": [
            {
                "candidate_id": "design_1",
                "sequence": "ACDEFGHIK",
                "scores": {"sequence_analysis": {"pI": 6.74}},
                "capabilities": {"has_sequence": True},
            }
        ]
    }
    return out


def test_metric_only_representative_sequences_are_not_candidate_pool() -> None:
    summary = {
        "nodes": [
            {
                "node_id": "analyze",
                "tool": "sequence_analysis",
                "state": "completed",
                "job_id": "job_metrics",
                "output": {
                    "representative_sequences": [
                        {"pI": 6.74, "scores": {"sequence_analysis": {"pI": 6.74}}}
                    ]
                },
            }
        ]
    }

    assert _summary_has_candidate_pool(summary) is False
    assert _first_submit_candidate_ref(summary) is None


def test_sequence_representative_sequences_are_candidate_pool() -> None:
    summary = {
        "nodes": [
            {
                "node_id": "analyze",
                "tool": "sequence_analysis",
                "state": "completed",
                "job_id": "job_seq",
                "output": {
                    "representative_sequences": [
                        {
                            "candidate_id": "seq_1",
                            "sequence": "ACDEFGHIK",
                            "scores": {"sequence_analysis": {"pI": 6.74}},
                        }
                    ]
                },
            }
        ]
    }

    assert _summary_has_candidate_pool(summary) is True


def test_legacy_candidate_materialization_drops_non_schema_metadata() -> None:
    summary = {
        "nodes": [
            {
                "node_id": "analyze",
                "tool": "sequence_analysis",
                "state": "completed",
                "job_id": "job_seq",
                "output": {
                    "representative_sequences": [
                        {
                            "candidate_id": "seq_1",
                            "sequence": "ACDEFGHIK",
                            "metadata": {"raw_adapter": "kept out of candidate_v1"},
                            "scores": {"sequence_analysis": {"pI": 6.74}},
                        }
                    ]
                },
            }
        ]
    }

    pool = _candidate_pool_from_summary(summary)

    assert len(pool) == 1
    assert "metadata" not in pool[0]
    validate_candidate_v1(pool[0])


def test_structure_required_contract_does_not_submit_sequence_only_candidate() -> None:
    summary = {
        "nodes": [
            {
                "node_id": "analyze",
                "tool": "sequence_analysis",
                "state": "completed",
                "job_id": "job_seq",
                "output": {
                    "representative_sequences": [
                        {
                            "candidate_id": "seq_1",
                            "sequence": "ACDEFGHIK",
                            "scores": {"sequence_analysis": {"pI": 6.74}},
                        }
                    ]
                },
            }
        ]
    }

    assert (
        _first_submit_candidate_ref(
            summary,
            deliverable_contract={
                "required_candidate_capabilities": ["has_full_structure"]
            },
        )
        is None
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


def _no_acceptable_intent() -> dict[str, Any]:
    return {
        "decision": "next_round_with_no_acceptable_candidate",
        "reasoning": "No acceptable candidate is available.",
        "next_round": {"reason": "No acceptable candidate is available."},
        "metrics_used": ["pI"],
    }


def test_no_acceptable_intent_without_candidates_closes_no_available(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DecisionProvider([_no_acceptable_intent()])
    execute_calls = {"count": 0}

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return _plan()

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        execute_calls["count"] += 1
        return _outcome("wf_answer", "job_analysis")

    def fail_submit(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("non-submit decision must not call submit_candidate")

    def no_acceptable_submit(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("no_available_candidate must not call submit_candidate")

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(AgentController, "_execute_submit_candidate_decision", fail_submit)
    monkeypatch.setattr(
        AgentController,
        "_execute_no_acceptable_candidate_submission",
        no_acceptable_submit,
    )

    controller = AgentController(
        tmp_workspace,
        config=AgentConfig.from_env(provider="codex", model="fake", max_rounds=1),
    )
    report = _run_controller(controller, tmp_workspace)

    assert execute_calls["count"] == 1
    assert report["state"] == "completed"
    assert report["decision"]["decision"] == "no_available_candidate"
    assert report["decision"]["resolved_branch"]["action"] == "stop"


def test_submit_candidate_decision_behavior_is_unchanged(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return _plan()

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        return _candidate_outcome("wf_submit", "job_rank")

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

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
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


def test_max_round_semantic_sequence_only_survivor_can_submit(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DecisionProvider(
        [
            {
                "decision": "next_round_with_surviving_candidate",
                "reasoning": "Representative sequence is sufficient for the semantic task.",
                "next_round": {
                    "reason": "More diversity would be useful if another round existed.",
                    "recommended": True,
                },
                "metrics_used": ["pI"],
            }
        ]
    )
    submit_calls = {"count": 0}

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return _plan()

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        return _candidate_outcome("wf_semantic", "job_sequence_summary")

    def submit(self, run_id, broker, decision, *, round_index, policy):
        del self, run_id, broker, round_index, policy
        submit_calls["count"] += 1
        candidate_ref = decision["submit_candidate"]["candidates"][0]
        assert candidate_ref["job_id"] == "job_sequence_summary"
        assert candidate_ref["candidate_id"] == "noop:job_sequence_summary:design_1"
        assert candidate_ref["index"] == 0
        return (
            {
                "submission_id": "sub_sequence",
                "outcome": "accepted",
                "terminal": True,
                "candidates_resolved": [{"sequence": "ACDEFGHIK", "index": 0}],
                "reasoning": "Representative sequence is sufficient for the semantic task.",
            },
            "job_submit_sequence",
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
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
    assert report["decision"]["decision"] == "submit_candidate"
    assert report["decision"]["submit_job_id"] == "job_submit_sequence"
    assert report["decision"]["submission_policy"]["version"] == "terminal_submission_policy_v1"
    assert "has_sequence" in report["decision"]["submission_policy"]["matched_candidate_capabilities"]
    assert report["result"] == {"workflow_ids": ["wf_semantic"], "state": "completed"}


def test_max_round_structure_contract_does_not_submit_sequence_only_candidate(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DecisionProvider(
        [
            {
                "decision": "next_round_with_surviving_candidate",
                "reasoning": "The user asked for a structural design candidate.",
                "next_round": {"reason": "Need structural evidence.", "recommended": True},
                "metrics_used": ["pI"],
            }
        ]
    )

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return _plan()

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        return _candidate_outcome("wf_structure", "job_sequence_summary")

    def fail_submit(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("sequence-only candidate must not be submitted")

    def no_acceptable_submit(self, run_id, broker, decision, *, round_index, policy):
        del self, run_id, broker, round_index, policy
        assert decision["decision"] == "next_round_with_surviving_candidate"
        return (
            {
                "submission_id": "sub_no_structure",
                "outcome": "no_acceptable_candidate",
                "terminal": True,
                "candidates_resolved": [],
                "reasoning": "No candidate satisfied the requested deliverable capabilities.",
            },
            "job_no_structure",
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(AgentController, "_execute_submit_candidate_decision", fail_submit)
    monkeypatch.setattr(
        AgentController,
        "_execute_no_acceptable_candidate_submission",
        no_acceptable_submit,
    )

    classification = ModeClassification(
        mode="agent_plan",
        reason="test",
        requires_llm_planning=True,
        deliverable_contract={
            "version": "deliverable_contract_v1",
            "deliverable_kind": "candidate_submission",
            "required_candidate_capabilities": ["has_sequence", "has_full_structure"],
            "terminal_submission_allowed": True,
        },
    )
    controller = AgentController(
        tmp_workspace,
        config=AgentConfig(max_rounds=1, max_entries=3),
    )
    run = controller.session_store.create_run("structure contract test")
    broker = AgentToolBroker(tmp_workspace, controller.registry, controller.session_store, run.run_id)
    report = asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "Design a full structure candidate.",
            classification,
            ExecutionPolicy(submit=True),
        )
    )

    assert report["state"] == "completed"
    assert report["decision"]["decision"] == "no_acceptable_candidate"
    assert report["summary"]["latest"]["final_submission"]["outcome"] == "no_acceptable_candidate"


def test_next_round_decision_behavior_is_unchanged(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DecisionProvider(
        [
                {
                    "decision": "next_round_with_surviving_candidate",
                    "reasoning": "More local analysis is useful.",
                    "next_round": {"reason": "Run one more pass.", "recommended": True},
                    "metrics_used": ["pI"],
                },
                {
                    "decision": "next_round_with_no_acceptable_candidate",
                    "reasoning": "Second round completed.",
                    "next_round": {"reason": "Second round completed."},
                    "metrics_used": ["pI"],
                },
        ]
    )
    executed: list[str] = []

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return _plan(f"round{len(executed) + 1}")

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        index = len(executed) + 1
        executed.append(f"wf_round{index}")
        return _candidate_outcome(f"wf_round{index}", f"job_round{index}")

    def no_acceptable_submit(self, run_id, broker, decision, *, round_index, policy):
        del self, run_id, broker, decision, round_index, policy
        return (
            {
                "submission_id": "sub_no_acceptable",
                "outcome": "no_acceptable_candidate",
                "terminal": True,
                "candidates_resolved": [],
                "reasoning": "Second round completed.",
            },
            "job_no_acceptable",
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(
        AgentController,
        "_execute_no_acceptable_candidate_submission",
        no_acceptable_submit,
    )

    report = _run_controller(
        AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3)),
        tmp_workspace,
    )

    assert executed == ["wf_round1", "wf_round2"]
    assert [item["decision"] for item in report["decisions"]] == [
        "next_round_with_surviving_candidate",
        "no_acceptable_candidate",
    ]
    assert report["result"]["status"] == "no_acceptable_candidate"
    assert report["result"]["technical_failure"] is False


def test_next_round_planner_failure_after_completed_workflow_closes_no_acceptable(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DecisionProvider(
        [
            {
                "decision": "next_round_with_surviving_candidate",
                "reasoning": "Existing candidates need another round but are not acceptable yet.",
                "next_round": {"reason": "Improve candidate quality.", "recommended": True},
                "metrics_used": ["score"],
            }
        ]
    )
    plan_calls = {"count": 0}

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        plan_calls["count"] += 1
        if plan_calls["count"] == 1:
            return _plan("round1")
        raise InvalidParamsError("next round planner failed", details={"phase": "next"})

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        return _candidate_outcome("wf_round1", "job_round1")

    def no_acceptable_submit(self, run_id, broker, decision, *, round_index, policy):
        del self, run_id, broker, round_index, policy
        assert decision["decision"] == "no_acceptable_candidate"
        return (
            {
                "submission_id": "sub_no_acceptable",
                "outcome": "no_acceptable_candidate",
                "terminal": True,
                "candidates_resolved": [],
                "reasoning": decision["reasoning"],
            },
            "job_no_acceptable",
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(
        AgentController,
        "_execute_no_acceptable_candidate_submission",
        no_acceptable_submit,
    )

    report = _run_controller(
        AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3)),
        tmp_workspace,
    )

    assert plan_calls["count"] >= 2
    assert report["state"] == "completed"
    assert report["decision"]["decision"] == "no_acceptable_candidate"
    assert report["decision"]["resolved_branch"]["failure"] is False
    assert len(report["decisions"]) == 1
    assert report["result"]["technical_failure"] is False


def test_multi_round_agent_can_end_with_non_submit_stop(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _DecisionProvider(
        [
                {
                    "decision": "next_round_with_surviving_candidate",
                    "reasoning": "Need a second local calculation.",
                    "next_round": {"reason": "Compute the requested property.", "recommended": True},
                    "metrics_used": ["pI"],
                },
            _no_acceptable_intent(),
        ]
    )

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return _plan(f"round{provider.index + 1}")

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        index = provider.index + 1
        return _candidate_outcome(f"wf_round{index}", f"job_round{index}")

    def fail_submit(self, *args, **kwargs):
        del self, args, kwargs
        raise AssertionError("non-submit stop must not call submit_candidate")

    def no_acceptable_submit(self, run_id, broker, decision, *, round_index, policy):
        del self, run_id, broker, decision, round_index, policy
        return (
            {
                "submission_id": "sub_no_acceptable",
                "outcome": "no_acceptable_candidate",
                "terminal": True,
                "candidates_resolved": [],
                "reasoning": "No acceptable candidate is available.",
            },
            "job_no_acceptable",
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)
    monkeypatch.setattr(AgentController, "_execute_submit_candidate_decision", fail_submit)
    monkeypatch.setattr(
        AgentController,
        "_execute_no_acceptable_candidate_submission",
        no_acceptable_submit,
    )

    report = _run_controller(
        AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3)),
        tmp_workspace,
    )

    assert [item["decision"] for item in report["decisions"]] == [
        "next_round_with_surviving_candidate",
        "no_acceptable_candidate",
    ]
    assert report["decision"]["resolved_branch"]["action"] == "stop"


def test_answer_result_requires_text() -> None:
    with pytest.raises(ValueError):
        AgentRunResult(kind="answer", text=None)
