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


def _submit_decision() -> dict:
    return {
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
        "metrics_used": ["score"],
    }


def test_valid_answer_decision_with_minimal_text_passes() -> None:
    validate_decision_v1(_answer_decision())


def test_valid_answer_decision_with_evidence_passes() -> None:
    payload = _answer_decision()
    payload["answer"]["evidence"] = [
        {
            "node_id": "analyze",
            "field": "representative_sequences[0].pI",
            "summary": "Computed pI from sequence_analysis",
        }
    ]

    validate_decision_v1(payload)


def test_valid_answer_decision_with_empty_evidence_passes() -> None:
    payload = _answer_decision()
    payload["answer"]["evidence"] = []

    validate_decision_v1(payload)


def test_answer_decision_missing_answer_payload_fails() -> None:
    payload = _answer_decision()
    payload.pop("answer")

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


def test_answer_decision_missing_text_fails() -> None:
    payload = _answer_decision()
    payload["answer"].pop("text")

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


def test_answer_decision_empty_text_fails() -> None:
    payload = _answer_decision()
    payload["answer"]["text"] = ""

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


def test_answer_payload_on_non_answer_decision_fails() -> None:
    payload = _submit_decision()
    payload["answer"] = {"text": "This must be absent."}

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


def test_submit_candidate_payload_on_answer_decision_fails() -> None:
    payload = _answer_decision()
    payload["submit_candidate"] = _submit_decision()["submit_candidate"]

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


def test_extra_answer_payload_field_fails() -> None:
    payload = _answer_decision()
    payload["answer"]["confidence"] = 0.8

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


def test_extra_evidence_item_field_fails() -> None:
    payload = _answer_decision()
    payload["answer"]["evidence"] = [
        {
            "node_id": "analyze",
            "field": "representative_sequences[0].pI",
            "summary": "Computed pI from sequence_analysis",
            "value": 6.74,
        }
    ]

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)


@pytest.mark.parametrize("decision_value", ["reply", "explain"])
def test_invalid_answer_like_decision_enum_fails(decision_value: str) -> None:
    payload = _answer_decision()
    payload["decision"] = decision_value

    with pytest.raises(DecisionSchemaViolation):
        validate_decision_v1(payload)
