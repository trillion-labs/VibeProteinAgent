from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.evaluation import decision as decision_mod
from vibeproteinagent.agents.execution import ExecutionPolicy
from vibeproteinagent.agents.execution import controller as controller_mod
from vibeproteinagent.agents.execution.controller import AgentController, WorkflowExecutionOutcome
from vibeproteinagent.agents.execution.round_state import PriorRoundContext
from vibeproteinagent.agents.planning import workflow_planner as planner_mod
from vibeproteinagent.agents.provider import ProviderResponse
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.workspace import Workspace


class _Provider:
    provider_name = "fake"

    async def chat(self, messages, **kwargs) -> ProviderResponse:
        del messages, kwargs
        return ProviderResponse(provider="fake", model="fake", content="{}")


def _classification() -> ModeClassification:
    return ModeClassification(
        mode="agent_plan",
        reason="test",
        requires_llm_planning=True,
    )


def _compiled_plan(node_id: str = "go_candidates") -> dict[str, Any]:
    return {
        "mode": "agent_plan",
        "source_kind": "planner_ir_compiled",
        "goal": "Test plan.",
        "assumptions": [],
        "missing_inputs": [],
        "required_tools": ["go_sequences"],
        "workflow_plan": {
            "name": "test_plan",
            "version": "0.1.0",
            "nodes": [
                {
                    "id": node_id,
                    "tool": "go_sequences",
                    "inputs": {"go_id": "GO:0003824", "max_sequences": 1},
                }
            ],
        },
        "success_criteria": {},
    }


def _candidate() -> dict[str, Any]:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_1",
        "source": {"tool": "go_sequences", "local_id": "seq_0000", "index": 0},
        "sequence": "ACDE",
        "scores": {"go_sequences": {"confidence": 0.8}},
        "artifacts": {},
    }


def _completed_outcome(workflow_id: str = "wf_1") -> WorkflowExecutionOutcome:
    return WorkflowExecutionOutcome(
        workflow_id,
        "completed",
        True,
        {
            "workflow_id": workflow_id,
            "state": "completed",
            "job_ids": [f"job_{workflow_id}"],
            "nodes": [
                {
                    "node_id": "go_candidates",
                    "tool": "go_sequences",
                    "state": "completed",
                    "job_id": f"job_{workflow_id}",
                    "output": {"candidates": [_candidate()]},
                }
            ],
            "metrics_used": ["go_sequences.confidence"],
            "top_designs": [],
        },
    )


def _final_submission_outcome() -> WorkflowExecutionOutcome:
    outcome = _completed_outcome("wf_final")
    outcome.summary["final_submission"] = {"submission_id": "sub_final"}
    return outcome


def _controller(tmp_workspace: Workspace, *, max_rounds: int = 1) -> AgentController:
    return AgentController(
        tmp_workspace,
        config=AgentConfig(max_rounds=max_rounds, max_entries=4),
    )


def _run_agent_plan(controller: AgentController, workspace: Workspace) -> dict[str, Any]:
    run = controller.session_store.create_run("round state cleanup")
    broker = AgentToolBroker(workspace, controller.registry, controller.session_store, run.run_id)
    return asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "Find candidates.",
            _classification(),
            ExecutionPolicy(submit=True),
        )
    )


def _allow_tool_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(controller_mod, "provider_from_config", lambda _config: _Provider())
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


def _patch_plan_v1(monkeypatch: pytest.MonkeyPatch, plan: dict[str, Any] | None = None) -> None:
    async def fake_plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        return plan or _compiled_plan()

    monkeypatch.setattr(planner_mod.WorkflowPlanner, "_plan_v1", fake_plan_v1)


def test_round_state_cleanup_after_normal_submission(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    _patch_plan_v1(monkeypatch)
    controller = _controller(tmp_workspace)
    controller._round_state = PriorRoundContext(99, [], "stale")

    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_validated_plan",
        lambda self, *args, **kwargs: _completed_outcome(),
    )

    async def submit_decision(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "submit_candidate",
            "reasoning": "Submit the best candidate.",
            "submit_candidate": {"candidate_ref": {"candidate_id": "cand_1"}},
            "metrics_used": [],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", submit_decision)
    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_submit_candidate_decision",
        lambda self, *args, **kwargs: ({"submission_id": "sub_1"}, "job_submit"),
    )

    _run_agent_plan(controller, tmp_workspace)

    assert controller._round_state is None


def test_round_state_cleanup_after_exception(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    _patch_plan_v1(monkeypatch)
    controller = _controller(tmp_workspace)
    controller._round_state = PriorRoundContext(99, [], "stale")

    def fail_execute(self, *args, **kwargs):
        del self, args, kwargs
        raise RuntimeError("planner/executor failure")

    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_validated_plan",
        fail_execute,
    )

    with pytest.raises(RuntimeError, match="planner/executor failure"):
        _run_agent_plan(controller, tmp_workspace)

    assert controller._round_state is None


def test_round_state_cleanup_after_stop_decision(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    _patch_plan_v1(monkeypatch)
    controller = _controller(tmp_workspace)

    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_validated_plan",
        lambda self, *args, **kwargs: _completed_outcome(),
    )

    async def stop_decision(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "stop",
            "reasoning": "Stop without another round.",
            "metrics_used": [],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", stop_decision)

    _run_agent_plan(controller, tmp_workspace)

    assert controller._round_state is None


def test_round_state_cleanup_after_no_acceptable_candidate(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    _patch_plan_v1(monkeypatch)
    controller = _controller(tmp_workspace)

    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_validated_plan",
        lambda self, *args, **kwargs: _completed_outcome(),
    )

    async def no_candidate_decision(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "no_acceptable_candidate",
            "reasoning": "No candidate meets the bar.",
            "no_acceptable_candidate": {"reason": "No candidate meets the bar."},
            "metrics_used": [],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", no_candidate_decision)
    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_submit_candidate_decision",
        lambda self, *args, **kwargs: ({"submission_id": "sub_none"}, "job_submit"),
    )

    _run_agent_plan(controller, tmp_workspace)

    assert controller._round_state is None


def test_controller_reuse_does_not_leak_round_state(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    _patch_plan_v1(monkeypatch)
    controller = _controller(tmp_workspace)
    observed_prior_contexts: list[PriorRoundContext | None] = []

    def fake_execute(self, run_id, broker, plan, params, policy, prior_round_context=None):
        del self, run_id, broker, plan, params, policy
        observed_prior_contexts.append(prior_round_context)
        return _final_submission_outcome()

    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_validated_plan",
        fake_execute,
    )

    _run_agent_plan(controller, tmp_workspace)
    controller._round_state = PriorRoundContext(99, [{"candidate_id": "stale"}], "stale")
    _run_agent_plan(controller, tmp_workspace)

    assert observed_prior_contexts == [None, None]
    assert controller._round_state is None


def test_round_state_is_available_between_rounds(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    controller = _controller(tmp_workspace, max_rounds=2)
    probes: dict[str, Any] = {"plan_contexts": []}

    async def fake_plan_v1(self, *args, **kwargs):
        del self, args
        probes["plan_contexts"].append(kwargs.get("prior_round_context"))
        if len(probes["plan_contexts"]) == 2:
            assert controller._round_state is not None
            assert kwargs["prior_round_context"] is controller._round_state
        return _compiled_plan(f"go_candidates_{len(probes['plan_contexts'])}")

    monkeypatch.setattr(planner_mod.WorkflowPlanner, "_plan_v1", fake_plan_v1)

    execute_calls = {"count": 0}

    def fake_execute(self, run_id, broker, plan, params, policy, prior_round_context=None):
        del self, run_id, broker, plan, params, policy
        execute_calls["count"] += 1
        if execute_calls["count"] == 1:
            assert prior_round_context is None
            return _completed_outcome("wf_1")
        assert prior_round_context is not None
        return _final_submission_outcome()

    monkeypatch.setattr(
        controller_mod.AgentController,
        "_execute_validated_plan",
        fake_execute,
    )

    async def next_round_decision(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "next_round",
            "reasoning": "Carry survivors forward.",
            "next_round": {
                "recommended": True,
                "reason": "Rank survivors in another round.",
            },
            "metrics_used": [],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", next_round_decision)

    _run_agent_plan(controller, tmp_workspace)

    assert probes["plan_contexts"][0] is None
    assert isinstance(probes["plan_contexts"][1], PriorRoundContext)
    assert controller._round_state is None
