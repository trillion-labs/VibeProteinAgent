from __future__ import annotations

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.routing.mode_router import ModeRouter


def test_sequence_request_derives_sequence_deliverable_contract() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Find and rank representative enzyme sequences for this protein family."
    )

    contract = classification.deliverable_contract
    assert contract["version"] == "deliverable_contract_v1"
    assert contract["deliverable_contract_source"] == "fallback"
    assert contract["answer_type"] == "candidate_submission"
    assert contract["deliverable_kind"] == "candidate_submission"
    assert contract["required_candidate_capabilities"] == ["has_sequence"]
    assert contract["terminal_submission_allowed"] is True
    assert contract["rationale"]


def test_structure_design_request_derives_full_structure_contract() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Design a binder and return a full structure candidate."
    )

    contract = classification.deliverable_contract
    assert contract["answer_type"] == "candidate_submission"
    assert contract["deliverable_kind"] == "candidate_submission"
    assert contract["required_candidate_capabilities"] == [
        "has_sequence",
        "has_full_structure",
    ]
    assert "structure" in contract["rationale"]


def test_structure_param_design_request_derives_full_structure_contract() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Design a binder.",
        params={"target_pdb": "/tmp/target.pdb"},
    )

    contract = classification.deliverable_contract
    assert contract["answer_type"] == "candidate_submission"
    assert contract["required_candidate_capabilities"] == [
        "has_sequence",
        "has_full_structure",
    ]
    assert contract["terminal_submission_allowed"] is True
    assert "structure target" in contract["rationale"]


def test_structure_param_explanation_request_stays_plain_language() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Explain this protein target.",
        params={"target_pdb": "/tmp/target.pdb"},
    )

    contract = classification.deliverable_contract
    assert contract["answer_type"] == "plain_language_answer"
    assert contract["required_candidate_capabilities"] == []
    assert contract["terminal_submission_allowed"] is False


def test_plain_language_question_does_not_require_candidate_submission() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Explain what NADPHX repair enzymes do."
    )

    contract = classification.deliverable_contract
    assert contract["answer_type"] == "plain_language_answer"
    assert contract["deliverable_kind"] == "plain_language_answer"
    assert contract["required_candidate_capabilities"] == []
    assert contract["terminal_submission_allowed"] is False
    assert contract["rationale"]


def test_binder_sequence_evidence_request_does_not_force_structure_contract() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Find and rank binder sequence candidates for this target family."
    )

    contract = classification.deliverable_contract
    assert contract["answer_type"] == "candidate_submission"
    assert contract["required_candidate_capabilities"] == ["has_sequence"]


def test_explicit_candidate_count_is_added_to_deliverable_contract() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Design 5 candidates for this target.",
    )

    assert classification.deliverable_contract["requested_candidate_count"] == 5


def test_requested_candidate_count_param_priority() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Design 5 binder candidates for this target.",
        params={
            "requested_candidate_count": 4,
            "num_candidates": 3,
            "candidate_count": 2,
            "top_k": 1,
        },
    )

    assert classification.deliverable_contract["requested_candidate_count"] == 4


def test_zero_candidate_count_is_not_added_to_deliverable_contract() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Design binder candidates.",
        params={"requested_candidate_count": 0},
    )

    assert "requested_candidate_count" not in classification.deliverable_contract


def test_budget_like_count_params_are_not_requested_candidate_count() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Design binder candidates.",
        params={"count": 9, "num_designs": 8, "top_k_designs": 7},
    )

    assert "requested_candidate_count" not in classification.deliverable_contract


def test_structure_count_text_is_not_requested_candidate_count() -> None:
    classification = ModeRouter(ToolRegistry()).classify(
        "Design 3 structures for this target."
    )

    assert "requested_candidate_count" not in classification.deliverable_contract
