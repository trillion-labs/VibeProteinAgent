from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.execution.controller import AgentController, WorkflowExecutionOutcome
from vibeproteinagent.agents.execution.policies import ExecutionPolicy
from vibeproteinagent.agents.planning import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.workspace import Workspace


class _NoDecisionProvider:
    provider_name = "fake"

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        if "result critic" in messages[0].content:
            raise AssertionError("decision LLM must not be called for no_available_candidate")
        return ProviderResponse(provider="fake", model="fake", content="{}")


class _NextRoundDecisionProvider:
    provider_name = "fake"

    def __init__(self) -> None:
        self.decision_calls = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        if "result critic" in messages[0].content:
            self.decision_calls += 1
            return ProviderResponse(
                provider="fake",
                model="fake",
                content=(
                    '{"decision":"next_round_with_surviving_candidate",'
                    '"reasoning":"partial empty branch was recorded, but current '
                    'round candidates remain viable",'
                    '"next_round":{"reason":"continue with viable current candidates",'
                    '"recommended":true,"changes":[]},'
                    '"metrics_used":["candidate_count"]}'
                ),
            )
        return ProviderResponse(provider="fake", model="fake", content="{}")


def _classification() -> ModeClassification:
    return ModeClassification(mode="agent_plan", reason="test", requires_llm_planning=True)


def _candidate(candidate_id: str) -> dict[str, Any]:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {"tool": "go_sequences", "local_id": candidate_id},
        "capabilities": {"has_sequence": True},
        "sequence": "ACDE",
    }


@pytest.fixture(autouse=True)
def _allow_mocked_workflow_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        WorkflowPlanner,
        "validate_invocation",
        lambda self, plan, *args, **kwargs: None,
    )


def _completed_outcome(workflow_id: str, summary: dict[str, Any]) -> WorkflowExecutionOutcome:
    return WorkflowExecutionOutcome(
        workflow_id=workflow_id,
        state="completed",
        executed=True,
        summary=summary,
        raw_workflow_summary=summary,
    )


def test_max_round_empty_candidate_without_survivors_is_deterministic(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "mode": "agent_plan",
            "source_kind": "workflow_yaml",
            "goal": "produce candidates",
            "assumptions": [],
            "missing_inputs": [],
            "required_tools": ["go_sequences", "sequence_analysis"],
            "workflow_plan": {
                "name": "empty",
                "version": "0.1.0",
                "nodes": [
                    {
                        "id": "producer",
                        "tool": "go_sequences",
                        "inputs": {"go_id": "GO:0000001"},
                    }
                ],
            },
            "success_criteria": {},
        }

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        return _completed_outcome(
            "wf_empty",
            {
                "state": "completed",
                "workflow_id": "wf_empty",
                "job_ids": [],
                "nodes": [
                    {
                        "node_id": "producer",
                        "tool": "go_sequences",
                        "state": "completed",
                        "output": {
                            "candidates": [],
                            "empty_candidate_event": {
                                "event_type": "empty_candidate_event",
                                "node_id": "producer",
                                "tool": "go_sequences",
                            },
                        },
                    },
                    {
                        "node_id": "consumer",
                        "tool": "sequence_analysis",
                        "state": "skipped",
                        "output": {
                            "skip_kind": "skipped_no_candidates",
                            "skip_reason": "upstream candidate stream is empty",
                        },
                    },
                ],
            },
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: _NoDecisionProvider(),
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)

    controller = AgentController(
        tmp_workspace,
        config=AgentConfig(max_rounds=1, max_entries=3),
    )
    run = controller.session_store.create_run("empty candidate")
    broker = AgentToolBroker(
        tmp_workspace,
        controller.registry,
        controller.session_store,
        run.run_id,
    )
    report = asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "find candidates",
            _classification(),
            ExecutionPolicy(submit=True),
        )
    )

    assert report["state"] == "completed"
    assert report["decision"]["decision"] == "no_available_candidate"
    assert report["result"]["status"] == "no_available_candidate"
    assert report["result"]["technical_failure"] is False


def test_max_round_no_survivors_without_empty_event_is_deterministic(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "mode": "agent_plan",
            "source_kind": "workflow_yaml",
            "goal": "produce candidates",
            "assumptions": [],
            "missing_inputs": [],
            "required_tools": ["literature_search"],
            "workflow_plan": {
                "name": "no_candidates",
                "version": "0.1.0",
                "nodes": [{"id": "context", "tool": "literature_search", "inputs": {"query": "x"}}],
            },
            "success_criteria": {},
        }

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        return _completed_outcome(
            "wf_no_candidates",
            {
                "state": "completed",
                "workflow_id": "wf_no_candidates",
                "job_ids": [],
                "nodes": [
                    {
                        "node_id": "context",
                        "tool": "literature_search",
                        "state": "completed",
                        "output": {"status": "no_hits", "records": []},
                    }
                ],
            },
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: _NoDecisionProvider(),
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)

    controller = AgentController(tmp_workspace, config=AgentConfig(max_rounds=1, max_entries=3))
    run = controller.session_store.create_run("no candidates")
    broker = AgentToolBroker(
        tmp_workspace,
        controller.registry,
        controller.session_store,
        run.run_id,
    )
    report = asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "find candidates",
            _classification(),
            ExecutionPolicy(submit=True),
        )
    )

    assert report["state"] == "completed"
    assert report["decision"]["decision"] == "no_available_candidate"
    assert report["result"]["technical_failure"] is False


def test_nonmax_empty_candidate_branches_next_round_without_decision_llm(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    plan_calls = {"count": 0}
    execute_calls: list[dict[str, Any]] = []

    async def plan_v1(self, *args, **kwargs):
        del self, args
        plan_calls["count"] += 1
        if plan_calls["count"] == 2:
            prior = kwargs.get("prior_round_context")
            assert prior is not None
            assert prior.survivor_snapshot_id is not None
            assert prior.prompt_digest["latest_round"]["branch_decision"]["action"] == "next_round"
        return {
            "mode": "agent_plan",
            "source_kind": "workflow_yaml",
            "goal": "produce candidates",
            "assumptions": [],
            "missing_inputs": [],
            "required_tools": ["go_sequences", "sequence_analysis"],
            "workflow_plan": {
                "name": f"round_{plan_calls['count']}",
                "version": "0.1.0",
                "nodes": [
                    {
                        "id": "producer",
                        "tool": "go_sequences",
                        "inputs": {"go_id": "GO:0000001"},
                    }
                ],
            },
            "success_criteria": {},
        }

    def execute(self, *args, **kwargs):
        del self, args
        execute_calls.append(kwargs)
        if len(execute_calls) == 1:
            return _completed_outcome(
                "wf_empty",
                {
                    "state": "completed",
                    "workflow_id": "wf_empty",
                    "job_ids": [],
                    "nodes": [
                        {
                            "node_id": "producer",
                            "tool": "go_sequences",
                            "state": "completed",
                            "output": {
                                "candidates": [],
                                "empty_candidate_event": {
                                    "event_type": "empty_candidate_event",
                                    "node_id": "producer",
                                    "tool": "go_sequences",
                                },
                            },
                        }
                    ],
                },
            )
        return _completed_outcome(
            "wf_done",
            {
                "state": "completed",
                "workflow_id": "wf_done",
                "job_ids": [],
                "nodes": [],
                "final_submission": {"submission_id": "sub_final"},
            },
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: _NoDecisionProvider(),
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)

    controller = AgentController(
        tmp_workspace,
        config=AgentConfig(max_rounds=2, max_entries=3),
    )
    run = controller.session_store.create_run("empty candidate")
    broker = AgentToolBroker(
        tmp_workspace,
        controller.registry,
        controller.session_store,
        run.run_id,
    )
    report = asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "find candidates",
            _classification(),
            ExecutionPolicy(submit=True),
        )
    )

    assert report["state"] == "completed"
    assert plan_calls["count"] == 2
    assert report["decisions"][0]["decision"] == "next_round_with_no_acceptable_candidate"
    assert report["decisions"][0]["resolved_branch"]["action"] == "next_round"


def test_partial_empty_event_with_current_candidates_uses_decision_llm(
    tmp_workspace: Workspace,
    monkeypatch,
) -> None:
    provider = _NextRoundDecisionProvider()
    plan_calls = {"count": 0}

    async def plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        plan_calls["count"] += 1
        return {
            "mode": "agent_plan",
            "source_kind": "workflow_yaml",
            "goal": "produce candidates",
            "assumptions": [],
            "missing_inputs": [],
            "required_tools": ["go_sequences", "sequence_analysis"],
            "workflow_plan": {
                "name": f"round_{plan_calls['count']}",
                "version": "0.1.0",
                "nodes": [
                    {
                        "id": "producer",
                        "tool": "go_sequences",
                        "inputs": {"go_id": "GO:0000001"},
                    }
                ],
            },
            "success_criteria": {},
        }

    def execute(self, *args, **kwargs):
        del self, args, kwargs
        if plan_calls["count"] == 1:
            return _completed_outcome(
                "wf_partial_empty",
                {
                    "state": "completed",
                    "workflow_id": "wf_partial_empty",
                    "job_ids": [],
                    "nodes": [
                        {
                            "node_id": "empty_side_branch",
                            "tool": "interpro_sequences",
                            "state": "completed",
                            "output": {
                                "candidates": [],
                                "empty_candidate_event": {
                                    "event_type": "empty_candidate_event",
                                    "node_id": "empty_side_branch",
                                    "tool": "interpro_sequences",
                                },
                            },
                        },
                        {
                            "node_id": "valid_branch",
                            "tool": "go_sequences",
                            "state": "completed",
                            "output": {"candidates": [_candidate("cand_ok")]},
                        },
                    ],
                },
            )
        return _completed_outcome(
            "wf_done",
            {
                "state": "completed",
                "workflow_id": "wf_done",
                "job_ids": [],
                "nodes": [],
                "final_submission": {"submission_id": "sub_final"},
            },
        )

    monkeypatch.setattr(WorkflowPlanner, "_plan_v1", plan_v1)
    monkeypatch.setattr(
        "vibeproteinagent.agents.execution.controller.provider_from_config",
        lambda _config: provider,
    )
    monkeypatch.setattr(AgentController, "_execute_validated_plan", execute)

    controller = AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3))
    run = controller.session_store.create_run("partial empty with candidate")
    broker = AgentToolBroker(
        tmp_workspace,
        controller.registry,
        controller.session_store,
        run.run_id,
    )

    report = asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "find candidates",
            _classification(),
            ExecutionPolicy(submit=True),
        )
    )

    assert report["state"] == "completed"
    assert provider.decision_calls == 1
    assert report["decisions"][0]["reasoning"].startswith("partial empty branch")
    assert report["summary"]["rounds"][0]["current_round_candidate_summary"]["count"] == 1
