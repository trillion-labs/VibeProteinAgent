from __future__ import annotations

import pytest

from vibeproteinagent.agents.evaluation import DecisionSchemaViolation, validate_decision_v1


def _answer_decision() -> dict:
    return {
        "decision": "answer",
        "reasoning": "sequence_analysis produced the requested property.",
        "answer": {"text": "The pI is 6.74."},
        "metrics_used": ["pI"],
    }


def test_answer_is_not_a_decision_intent() -> None:
    with pytest.raises(DecisionSchemaViolation) as excinfo:
        validate_decision_v1(_answer_decision())

    assert any(
        item.path in {"/decision", "/answer"} for item in excinfo.value.violations
    )


def test_answer_payload_on_submit_decision_fails() -> None:
    payload = {
        "decision": "submit_candidate",
        "reasoning": "The ranked candidate satisfies the criteria.",
        "submit_candidate": {
            "candidates": [
                {
                    "job_id": "job_rank",
                    "candidate_id": "design_1",
                    "justification": "top ranked",
                }
            ]
        },
        "answer": {"text": "This must be absent."},
        "metrics_used": ["score"],
    }

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)
