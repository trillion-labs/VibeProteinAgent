from __future__ import annotations

import logging

import pytest

from vibeproteinagent.agents.config import resolve_decision_normalization_mode
from vibeproteinagent.agents.evaluation import DecisionSchemaViolation
from vibeproteinagent.agents.execution import controller as controller_mod
from vibeproteinagent.agents.execution.controller import _submit_candidate_params_from_decision


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
