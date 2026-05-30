from __future__ import annotations

import logging

import pytest

from vibeproteinagent.agents.config import resolve_decision_normalization_mode
from vibeproteinagent.agents.evaluation import DecisionSchemaViolation
from vibeproteinagent.agents.execution import controller as controller_mod
from vibeproteinagent.agents.execution.controller import (
    AgentController,
    _json_safe_decision_details,
    _submit_candidate_params_from_decision,
)
from vibeproteinagent.agents.execution.policies import ExecutionPolicy
from vibeproteinagent.workspace.store import JobStore
from vibeproteinagent.workflows.store import WorkflowStore
from vibeproteinagent.workspace import Workspace


def _misnested_decision() -> dict:
    return {
        "decision": "submit_candidate",
        "submit_candidate": {
            "outcome": "accepted",
            "candidates": [
                {
                    "job_id": "job_rank",
                    "candidate_id": "design_1",
                    "justification": "top ranked",
                    "reasoning": "The candidate satisfies the criteria.",
                    "rank": 1,
                }
            ],
        },
        "next_round": {"recommended": False, "reason": "ready"},
        "metrics_used": ["score"],
        "notes": [],
    }


def _valid_decision() -> dict:
    return {
        "decision": "submit_candidate",
        "reasoning": "The candidate satisfies the criteria.",
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


def test_default_mode_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VPA_NORMALIZE_DECISION_REASONING", raising=False)
    assert resolve_decision_normalization_mode() == "strict"


def test_compat_mode_lifts_misnested_reasoning_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "compat")
    caplog.set_level(logging.WARNING, logger="vibeproteinagent.audit")

    params = _submit_candidate_params_from_decision(_misnested_decision())

    assert params["reasoning"] == "The candidate satisfies the criteria."
    assert "reasoning" not in params["candidates"][0]
    assert "rank" not in params["candidates"][0]
    assert any(
        getattr(record, "pattern_key", None) == "decision.reasoning.lifted_from_candidate"
        for record in caplog.records
    )


def test_strict_mode_rejects_misnested_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")
    with pytest.raises(DecisionSchemaViolation):
        _submit_candidate_params_from_decision(_misnested_decision())


def test_strict_mode_valid_payload_uses_no_legacy_lift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")

    def _fail_legacy(decision):
        del decision
        raise AssertionError("legacy lift should not run in strict mode")

    monkeypatch.setattr(controller_mod, "_legacy_lift_submit_candidate_params", _fail_legacy)
    params = _submit_candidate_params_from_decision(_valid_decision())

    assert params["outcome"] == "accepted"
    assert params["reasoning"] == "The candidate satisfies the criteria."


def test_off_mode_rejects_misnested_reasoning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "off")
    with pytest.raises(DecisionSchemaViolation):
        _submit_candidate_params_from_decision(_misnested_decision())


def test_off_mode_valid_payload_uses_no_legacy_lift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "off")

    def _fail_legacy(decision):
        del decision
        raise AssertionError("legacy lift should not run in off mode")

    monkeypatch.setattr(controller_mod, "_legacy_lift_submit_candidate_params", _fail_legacy)
    params = _submit_candidate_params_from_decision(_valid_decision())

    assert params["outcome"] == "accepted"
    assert params["reasoning"] == "The candidate satisfies the criteria."


def test_invalid_normalization_env_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "maybe")
    with pytest.raises(ValueError, match="Invalid VPA_NORMALIZE_DECISION_REASONING"):
        resolve_decision_normalization_mode()


class _SubmitBroker:
    def __init__(self) -> None:
        self.params = None

    def submit_job(self, tool_name, params, **kwargs):
        del kwargs
        assert tool_name == "submit_candidate"
        self.params = params
        return {"ok": True, "job_id": "job_submit"}

    def wait_job(self, job_id, **kwargs):
        del kwargs
        assert job_id == "job_submit"
        return {
            "state": "completed",
            "output": {
                "terminal": True,
                "outcome": "accepted",
                "submission_id": "sub_1",
            },
        }


def test_submit_execution_validates_public_decision_payload_only(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")
    controller = AgentController(tmp_workspace)
    broker = _SubmitBroker()
    decision = {
        **_valid_decision(),
        "resolved_branch": {
            "action": "submit_candidate",
            "failure": False,
            "reason": "internal controller branch",
        },
        "notes": [{"level": "debug", "message": "internal note"}],
    }

    submission, job_id = controller._execute_submit_candidate_decision(
        "run_1",
        broker,
        decision,
        round_index=0,
        policy=ExecutionPolicy(submit=True),
    )

    assert job_id == "job_submit"
    assert submission["terminal"] is True
    assert broker.params["reasoning"] == "The candidate satisfies the criteria."


def test_submit_execution_resolves_workflow_node_alias_to_job_id(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")
    controller = AgentController(tmp_workspace)
    run = controller.session_store.create_run("submit alias")
    workflow_store = WorkflowStore(tmp_workspace)
    workflow_id = workflow_store.create(
        name="alias workflow",
        version="1",
        descriptor_yaml="name: alias workflow\n",
        params={},
    )
    workflow_store.add_node(
        workflow_id=workflow_id,
        node_id="binder_design",
        tool_name="pxdesign",
        tool_version="1.0.0",
        upstream=[],
        params_hash="params",
        input_hash="input",
        cache_key="cache",
    )
    job = JobStore(tmp_workspace).submit(
        "pxdesign",
        "1.0.0",
        {"target_pdb": "/tmp/target.pdb"},
        workflow_id=workflow_id,
        workflow_node_id="binder_design",
    )
    workflow_store.update_node_state(
        workflow_id,
        "binder_design",
        "completed",
        job_id=job.job_id,
        output={"candidates": [{"candidate_id": "cand_a"}]},
    )
    controller.session_store.update_run(run.run_id, workflow_ids=[workflow_id])
    broker = _SubmitBroker()
    decision = _valid_decision()
    decision["submit_candidate"]["candidates"][0]["job_id"] = "binder_design"

    controller._execute_submit_candidate_decision(
        run.run_id,
        broker,
        decision,
        round_index=0,
        policy=ExecutionPolicy(submit=True),
    )

    assert broker.params["candidates"][0]["job_id"] == job.job_id


def test_submit_execution_leaves_unknown_job_alias_for_schema_validation(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")
    controller = AgentController(tmp_workspace)
    run = controller.session_store.create_run("submit unknown alias")
    broker = _SubmitBroker()
    decision = _valid_decision()
    decision["submit_candidate"]["candidates"][0]["job_id"] = "not_a_known_node"

    controller._execute_submit_candidate_decision(
        run.run_id,
        broker,
        decision,
        round_index=0,
        policy=ExecutionPolicy(submit=True),
    )

    assert broker.params["candidates"][0]["job_id"] == "not_a_known_node"


def test_decision_error_details_are_json_safe_without_payload_cycle() -> None:
    decision = _valid_decision()
    details = {"payload": decision}
    decision["notes"] = [{"details": details}]

    safe = _json_safe_decision_details(details)

    assert safe["payload"]["decision"] == "submit_candidate"
    assert safe["payload"]["candidate_ref_count"] == 1
