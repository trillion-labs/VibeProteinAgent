from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.execution import ExecutionPolicy
from vibeproteinagent.agents.execution import controller as controller_mod
from vibeproteinagent.agents.execution.controller import AgentController
from vibeproteinagent.agents.planning import workflow_planner as planner_mod
from vibeproteinagent.agents.session import ModeClassification
from vibeproteinagent.agents.tools import AgentToolBroker
from vibeproteinagent.workspace import Workspace


class _NeverUsedProvider:
    provider_name = "fake"


def _plan() -> dict[str, Any]:
    return {
        "mode": "agent_plan",
        "goal": "Collect GO candidates.",
        "workflow_plan": {
            "name": "route_test",
            "version": "0.1.0",
            "nodes": [
                {
                    "id": "go_candidates",
                    "tool": "go_sequences",
                    "inputs": {"go_id": "GO:0003824"},
                }
            ],
        },
        "success_criteria": {},
    }


def _run_plan_only(controller: AgentController, tmp_workspace: Workspace) -> dict[str, Any]:
    run = controller.session_store.create_run("route test")
    broker = AgentToolBroker(
        tmp_workspace,
        controller.registry,
        controller.session_store,
        run.run_id,
    )
    classification = ModeClassification(
        mode="agent_plan",
        reason="test",
        requires_llm_planning=True,
    )
    return asyncio.run(
        controller._run_agent_plan(
            run.run_id,
            broker,
            "Find GO candidates.",
            classification,
            ExecutionPolicy(submit=False),
        )
    )


def test_controller_calls_planner_ir_v1_path(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        controller_mod,
        "provider_from_config",
        lambda _config: _NeverUsedProvider(),
    )
    calls = {"legacy": 0, "v1": 0}

    async def fake_draft_plan(self, *args, **kwargs):
        del self, args, kwargs
        calls["legacy"] += 1
        return _plan()

    async def fake_plan_v1(self, *args, **kwargs):
        del self, args, kwargs
        calls["v1"] += 1
        return _plan()

    monkeypatch.setattr(planner_mod.WorkflowPlanner, "draft_plan", fake_draft_plan)
    monkeypatch.setattr(planner_mod.WorkflowPlanner, "_plan_v1", fake_plan_v1)
    monkeypatch.setattr(
        planner_mod.WorkflowPlanner,
        "validate_invocation",
        lambda _self, plan, *_args, **_kwargs: plan,
    )
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=1))

    report = _run_plan_only(controller, tmp_workspace)

    assert report["state"] == "planned"
    assert calls == {"legacy": 0, "v1": 1}
