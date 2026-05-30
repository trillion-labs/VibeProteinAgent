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
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.workspace import Workspace


class _Provider:
    provider_name = "fake"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.calls = 0

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del messages, kwargs
        payload = self.payloads[self.calls]
        self.calls += 1
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(payload))


def _candidate() -> dict[str, Any]:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_prior",
        "source": {"tool": "go_sequences", "local_id": "seq_0000", "index": 0},
        "sequence": "ACDE",
        "scores": {},
        "artifacts": {},
    }


def _workflow_summary(workflow_id: str, job_id: str) -> dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "state": "completed",
        "job_ids": [job_id],
        "nodes": [
            {
                "node_id": "go_candidates",
                "tool": "go_sequences",
                "state": "completed",
                "job_id": job_id,
                "output": {"candidates": [_candidate()]},
            }
        ],
        "metrics_used": [],
        "top_designs": [],
    }


def _selection(*tools: str) -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "selected_tools": [
            {"name": tool, "reason": "selected for test"}
            for tool in tools
        ],
        "missing_inputs": [],
    }


def _initial_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Collect candidates.",
        "nodes": [
            {
                "id": "go_candidates",
                "tool": "go_sequences",
                "intent": "Collect candidates.",
                "inputs": {"go_id": "GO:0003824"},
                "upstream": [],
            }
        ],
    }


def _next_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Rank carried candidates.",
        "nodes": [
            {
                "id": "rank_prior",
                "tool": "rank_designs",
                "intent": "Rank prior candidates.",
                "inputs": {},
                "upstream": [
                    {"node": "prior_round", "relationship": "candidate_source"}
                ],
            }
        ],
    }


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


def test_next_round_v1_plan_preserves_source_kind_and_skips_legacy_validators(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_tool_environment(monkeypatch)
    provider = _Provider(
        [
            _selection("go_sequences"),
            _initial_ir(),
            _selection("rank_designs"),
            _next_ir(),
        ]
    )
    monkeypatch.setattr(controller_mod, "provider_from_config", lambda _config: provider)
    validator_calls = {
        "_validate_static_output_refs": 0,
        "_validate_static_output_ref_types": 0,
        "_validate_static_job_refs": 0,
    }

    def _spy(name: str):
        def inner(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            validator_calls[name] += 1

        return inner

    for name in validator_calls:
        monkeypatch.setattr(planner_mod, name, _spy(name))

    async def fake_decide(self, *args, **kwargs):
        del self, args, kwargs
        return {
            "decision": "next_round",
            "reasoning": "Rank the survivors.",
            "next_round": {"reason": "Rank survivors.", "recommended": True},
            "metrics_used": [],
        }

    monkeypatch.setattr(decision_mod.DecisionEngine, "decide", fake_decide)
    executed_plans: list[dict[str, Any]] = []

    def fake_execute(self, run_id, broker, plan, params, policy, prior_round_context=None):
        del self, run_id, broker, params, policy, prior_round_context
        executed_plans.append(plan)
        if len(executed_plans) == 1:
            raw_summary = _workflow_summary("wf1", "job_go")
            return WorkflowExecutionOutcome(
                "wf1",
                "completed",
                True,
                raw_summary,
                raw_workflow_summary=raw_summary,
            )
        raw_summary = {
            "workflow_id": "wf2",
            "state": "completed",
            "job_ids": ["job_rank"],
            "nodes": [],
            "final_submission": {"submission_id": "sub_final"},
            "metrics_used": [],
        }
        return WorkflowExecutionOutcome(
            "wf2",
            "completed",
            True,
            raw_summary,
            raw_workflow_summary=raw_summary,
        )

    monkeypatch.setattr(controller_mod.AgentController, "_execute_validated_plan", fake_execute)
    controller = AgentController(tmp_workspace, config=AgentConfig(max_rounds=2, max_entries=3))
    run = controller.session_store.create_run("source kind test")
    broker = AgentToolBroker(tmp_workspace, controller.registry, controller.session_store, run.run_id)

    asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "Collect and rank candidates.",
            ModeClassification(
                mode="agent_plan",
                reason="test",
                requires_llm_planning=True,
            ),
            ExecutionPolicy(submit=True),
        )
    )

    assert executed_plans[1]["source_kind"] == "planner_compiled"
    assert validator_calls == {
        "_validate_static_output_refs": 0,
        "_validate_static_output_ref_types": 0,
        "_validate_static_job_refs": 0,
    }
