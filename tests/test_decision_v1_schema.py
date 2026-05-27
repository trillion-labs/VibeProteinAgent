from __future__ import annotations

import pytest

from vibeproteinagent.agents.evaluation import (
    DecisionSchemaViolation,
    validate_decision_v1,
)


def _valid_submit_decision() -> dict:
    return {
        "decision": "submit_candidate",
        "reasoning": "The ranked candidate satisfies the criteria.",
        "submit_candidate": {
            "candidates": [
                {
                    "job_id": "job_rank",
                    "candidate_id": "design_1",
                    "justification": "top ranked",
                    "scores_to_highlight": ["score"],
                }
            ]
        },
        "metrics_used": ["score"],
        "notes": "ready",
    }


def test_valid_submit_candidate_decision_passes() -> None:
    validate_decision_v1(_valid_submit_decision())


def test_valid_next_round_decision_passes() -> None:
    validate_decision_v1(
        {
            "decision": "next_round_with_surviving_candidate",
            "reasoning": "More exploration is needed.",
            "next_round": {"reason": "Improve diversity.", "recommended": True},
            "metrics_used": ["diversity"],
        }
    )


def test_valid_no_acceptable_next_round_intent_passes() -> None:
    validate_decision_v1(
        {
            "decision": "next_round_with_no_acceptable_candidate",
            "reasoning": "All candidates fall below the threshold.",
            "next_round": {"reason": "All candidates are weak."},
            "metrics_used": ["score"],
        }
    )


def test_reasoning_inside_candidate_ref_fails_with_clear_path() -> None:
    payload = _valid_submit_decision()
    payload["submit_candidate"]["candidates"][0]["reasoning"] = "misnested"

    with pytest.raises(DecisionSchemaViolation) as err:
        validate_decision_v1(payload)

    violation = err.value.violations[0]
    assert violation.path == "/submit_candidate/candidates/0/reasoning"
    assert violation.issue == "reasoning_inside_candidate_forbidden"


def test_missing_top_level_reasoning_fails_for_submit_candidate() -> None:
    payload = _valid_submit_decision()
    payload.pop("reasoning")

    with pytest.raises(DecisionSchemaViolation) as err:
        validate_decision_v1(payload)

    assert any(
        item.path == "/reasoning" and item.issue == "reasoning_missing_at_top_level"
        for item in err.value.violations
    )


def test_unknown_top_level_field_fails() -> None:
    payload = _valid_submit_decision()
    payload["extra"] = True
    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


def test_unknown_candidate_ref_field_fails() -> None:
    payload = _valid_submit_decision()
    payload["submit_candidate"]["candidates"][0]["rank"] = 1
    with pytest.raises(DecisionSchemaViolation) as err:
        validate_decision_v1(payload)
    assert any(item.issue == "additional_property_not_allowed" for item in err.value.violations)


def test_empty_submit_candidate_candidates_fails() -> None:
    payload = _valid_submit_decision()
    payload["submit_candidate"]["candidates"] = []
    with pytest.raises(DecisionSchemaViolation) as err:
        validate_decision_v1(payload)
    assert any(item.issue == "array_too_short" for item in err.value.violations)


def test_invalid_decision_enum_fails() -> None:
    payload = _valid_submit_decision()
    payload["decision"] = "submit_final"
    with pytest.raises(DecisionSchemaViolation) as err:
        validate_decision_v1(payload)
    assert any(item.issue == "invalid_decision_enum" for item in err.value.violations)
