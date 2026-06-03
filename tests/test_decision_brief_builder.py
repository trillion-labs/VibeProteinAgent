from __future__ import annotations

import json

from vibeproteinagent.agents.evaluation import DecisionEngine
from vibeproteinagent.agents.execution.controller import (
    _candidate_pool_items_from_summary,
    _first_submit_candidate_ref,
)
from vibeproteinagent.agents.execution.candidate_pool import candidate_pool_items_from_summary
from vibeproteinagent.agents.execution.decision_brief import DecisionBriefBuilder


def _raw_workflow_summary() -> dict:
    return {
        "workflow_id": "wf_brief",
        "state": "completed",
        "nodes": [
            {
                "node_id": "rank",
                "tool": "rank_designs",
                "job_id": "job_rank_real",
                "state": "completed",
                "output": {
                    "ranked_candidates": [
                        {
                            "schema_version": "candidate_v1",
                            "candidate_id": "cand_full",
                            "source": {
                                "tool": "rank_designs",
                                "local_id": "cand_full",
                                "job_id": "job_not_submit_source",
                            },
                            "sequence": "ACDEFGHIKLMNPQRSTVWY",
                            "full_structure_path": "/tmp/full_structure.cif",
                            "scores": {
                                "rank_designs": {
                                    "ranking_score": 0.91,
                                    "raw_metric": 12.3,
                                }
                            },
                            "artifacts": {"rank": {"path": "/tmp/rank.json"}},
                            "capabilities": {
                                "has_sequence": True,
                                "has_full_structure": True,
                                "has_candidate_scores": True,
                            },
                        },
                        {
                            "schema_version": "candidate_v1",
                            "candidate_id": "cand_seq_only",
                            "source": {"tool": "rank_designs", "local_id": "cand_seq_only"},
                            "sequence": "MMMMMMMM",
                            "capabilities": {
                                "has_sequence": True,
                                "has_full_structure": False,
                            },
                        },
                    ]
                },
            }
        ],
    }


def test_decision_brief_recommended_candidates_use_raw_node_job_id_safely() -> None:
    workflow_summary = _raw_workflow_summary()
    round_record = {
        "round_index": 0,
        "workflow_id": "wf_brief",
        "state": "completed",
        "workflow_summary_raw": workflow_summary,
        "empty_candidate_events": [],
        "skipped_nodes": [],
    }
    prompt_context = {
        "latest_round": {
            "candidate_summary": {
                "count": 2,
                "items": [
                    {
                        "candidate_id": "cand_full",
                        "source": {"job_id": "job_from_prompt_context"},
                    }
                ],
                "truncated": False,
            },
            "survivor_summary": {"count": 2, "items": [], "truncated": False},
            "candidate_capability_filter_events": [],
            "failure_cards": [],
        }
    }
    contract = {
        "answer_type": "candidate_submission",
        "required_candidate_capabilities": ["has_sequence", "has_full_structure"],
        "optional_candidate_capabilities": ["has_candidate_scores"],
        "terminal_submission_allowed": True,
    }
    local = DecisionEngine(None).local_decision(
        {"state": "completed"},
        success_criteria={"return_final": True},
    )

    brief = DecisionBriefBuilder().build(
        round_record=round_record,
        prompt_context=prompt_context,
        outcome_summary={"state": "completed", "workflow_id": "wf_brief"},
        success_criteria={"return_final": True},
        deliverable_contract=contract,
        local_decision=local,
        round_index=0,
        max_rounds=4,
    )

    top = brief["recommended_candidates"][0]
    assert top["candidate_id"] == "cand_full"
    assert top["job_id"] == "job_rank_real"
    assert top["index"] == 0
    assert top["rank"] == 1
    assert top["source_tool"] == "rank_designs"
    assert top["container_key"] == "ranked_candidates"
    assert top["capabilities_satisfied"] is True
    assert top["missing_capabilities"] == []
    assert "ranked" in top["evidence"]
    assert "full_structure_available" in top["evidence"]
    assert "candidate_scores_available" in top["evidence"]
    assert brief["candidate_pool"]["current_round_count"] == 2
    assert brief["candidate_pool"]["survivor_count"] == 2
    assert brief["candidate_pool"]["submit_ready_count"] == 1
    assert brief["candidate_pool"]["missing_capability_count"] == 1

    serialized = json.dumps(brief, sort_keys=True)
    assert "job_from_prompt_context" not in serialized
    assert "job_not_submit_source" not in serialized
    assert "ACDEFGHIKLMNPQRSTVWY" not in serialized
    assert "MMMMMMMM" not in serialized
    assert "/tmp/full_structure.cif" not in serialized
    assert "/tmp/rank.json" not in serialized
    assert "ranking_score" not in serialized
    assert "raw_metric" not in serialized
    assert "0.91" not in serialized
    assert "artifacts" not in serialized


def test_decision_brief_top_ref_matches_submit_fallback_contract() -> None:
    workflow_summary = _raw_workflow_summary()
    contract = {
        "required_candidate_capabilities": ["has_sequence", "has_full_structure"],
        "terminal_submission_allowed": True,
    }
    brief = DecisionBriefBuilder().build(
        round_record={
            "round_index": 0,
            "workflow_id": "wf_brief",
            "state": "completed",
            "workflow_summary_raw": workflow_summary,
        },
        prompt_context={
            "latest_round": {
                "candidate_summary": {"count": 2, "items": [], "truncated": False},
                "survivor_summary": {"count": 2, "items": [], "truncated": False},
            }
        },
        outcome_summary={"state": "completed"},
        success_criteria={},
        deliverable_contract=contract,
        local_decision={},
        round_index=0,
        max_rounds=1,
    )
    fallback_ref = _first_submit_candidate_ref(
        workflow_summary,
        deliverable_contract=contract,
    )

    assert fallback_ref is not None
    top = brief["recommended_candidates"][0]
    assert top["job_id"] == fallback_ref["job_id"]
    assert top["candidate_id"] == fallback_ref["candidate_id"]
    assert top["index"] == fallback_ref["index"]


def test_controller_candidate_pool_alias_matches_shared_helper() -> None:
    workflow_summary = _raw_workflow_summary()

    controller_items = _candidate_pool_items_from_summary(workflow_summary)
    shared_items = candidate_pool_items_from_summary(workflow_summary)

    assert controller_items == shared_items


def test_decision_brief_malformed_inputs_degrade_to_empty_safe_shape() -> None:
    brief = DecisionBriefBuilder().build(
        round_record={"workflow_summary_raw": {"nodes": "not-a-list"}},
        prompt_context=None,
        outcome_summary=None,
        success_criteria=None,
        deliverable_contract=None,
        local_decision=None,
        round_index=2,
        max_rounds=4,
    )

    assert brief["version"] == "decision_brief_v1"
    assert brief["round"]["index"] == 2
    assert brief["round"]["remaining_rounds"] == 1
    assert brief["recommended_candidates"] == []
    assert brief["candidate_pool"]["current_round_count"] == 0


def test_decision_brief_does_not_use_legacy_workflow_summary_for_refs() -> None:
    brief = DecisionBriefBuilder().build(
        round_record={
            "workflow_summary": _raw_workflow_summary(),
            "workflow_id": "wf_legacy_only",
            "state": "completed",
        },
        prompt_context={
            "latest_round": {
                "candidate_summary": {"count": 1, "items": [], "truncated": False},
                "survivor_summary": {"count": 1, "items": [], "truncated": False},
            }
        },
        outcome_summary={"state": "completed"},
        success_criteria={},
        deliverable_contract={
            "required_candidate_capabilities": ["has_sequence", "has_full_structure"],
            "terminal_submission_allowed": True,
        },
        local_decision={},
        round_index=0,
        max_rounds=1,
    )

    assert brief["recommended_candidates"] == []
    assert brief["candidate_pool"]["submit_ready_count"] == 0
    assert brief["candidate_pool"]["missing_capability_count"] == 0


def test_decision_brief_pool_counts_use_full_raw_pool_not_card_limit() -> None:
    candidates = []
    for index in range(12):
        candidate = {
            "schema_version": "candidate_v1",
            "candidate_id": f"cand_{index}",
            "source": {"tool": "rank_designs", "local_id": f"cand_{index}"},
            "sequence": f"ACDE{index}",
            "capabilities": {"has_sequence": True},
        }
        if index < 11:
            candidate["full_structure_path"] = f"/tmp/cand_{index}.cif"
            candidate["capabilities"]["has_full_structure"] = True
        else:
            candidate["capabilities"]["has_full_structure"] = False
        candidates.append(candidate)
    workflow_summary = {
        "workflow_id": "wf_many",
        "state": "completed",
        "nodes": [
            {
                "node_id": "rank_many",
                "tool": "rank_designs",
                "job_id": "job_rank_many",
                "state": "completed",
                "output": {"ranked_candidates": candidates},
            }
        ],
    }

    brief = DecisionBriefBuilder().build(
        round_record={
            "workflow_summary_raw": workflow_summary,
            "workflow_id": "wf_many",
            "state": "completed",
        },
        prompt_context={
            "latest_round": {
                "candidate_summary": {"count": 12, "items": [], "truncated": True},
                "survivor_summary": {"count": 12, "items": [], "truncated": True},
            }
        },
        outcome_summary={"state": "completed"},
        success_criteria={},
        deliverable_contract={
            "required_candidate_capabilities": ["has_sequence", "has_full_structure"],
            "terminal_submission_allowed": True,
        },
        local_decision={},
        round_index=0,
        max_rounds=1,
    )

    assert len(brief["recommended_candidates"]) == 10
    assert brief["candidate_pool"]["submit_ready_count"] == 11
    assert brief["candidate_pool"]["missing_capability_count"] == 1
    assert brief["candidate_pool"]["truncated"] is True


def test_decision_brief_includes_previous_only_survivor_refs_safely() -> None:
    brief = DecisionBriefBuilder().build(
        round_record={
            "workflow_summary_raw": _raw_workflow_summary(),
            "workflow_id": "wf_brief",
            "state": "completed",
        },
        prompt_context={
            "latest_round": {
                "candidate_summary": {"count": 2, "items": [], "truncated": False},
                "survivor_summary": {
                    "count": 3,
                    "truncated": False,
                    "items": [
                        {
                            "candidate_id": "cand_full",
                            "source": {"tool": "rank_designs", "job_id": "job_duplicate"},
                            "capabilities": {
                                "has_sequence": True,
                                "has_full_structure": True,
                            },
                            "sequence": "SHOULD_NOT_LEAK",
                            "score_keys": ["rank_designs"],
                        },
                        {
                            "candidate_id": "cand_survivor",
                            "source": {"tool": "go_sequences", "job_id": "job_survivor"},
                            "capabilities": {
                                "has_sequence": True,
                                "has_full_structure": True,
                            },
                            "structure_path": "/tmp/survivor.pdb",
                            "score_keys": ["go_sequences"],
                        },
                    ],
                },
            }
        },
        outcome_summary={"state": "completed", "workflow_id": "wf_brief"},
        success_criteria={},
        deliverable_contract={
            "required_candidate_capabilities": ["has_sequence", "has_full_structure"],
            "terminal_submission_allowed": True,
        },
        local_decision={},
        round_index=0,
        max_rounds=2,
    )

    ids = [card["candidate_id"] for card in brief["recommended_candidates"]]
    assert ids == ["cand_full", "cand_seq_only", "cand_survivor"]
    survivor = brief["recommended_candidates"][2]
    assert survivor["job_id"] == "job_survivor"
    assert survivor["source_surface"] == "survivor_summary"
    assert survivor["capabilities_satisfied"] is True
    assert brief["candidate_pool"]["submit_ready_count"] == 2
    assert brief["candidate_pool"]["missing_capability_count"] == 1

    serialized = json.dumps(brief, sort_keys=True)
    assert "SHOULD_NOT_LEAK" not in serialized
    assert "/tmp/survivor.pdb" not in serialized
    assert "job_duplicate" not in serialized


def test_decision_brief_exposes_only_positive_requested_candidate_count() -> None:
    kwargs = {
        "round_record": {"workflow_summary_raw": _raw_workflow_summary()},
        "prompt_context": {
            "latest_round": {
                "candidate_summary": {"count": 2, "items": [], "truncated": False},
                "survivor_summary": {"count": 2, "items": [], "truncated": False},
            }
        },
        "outcome_summary": {"state": "completed"},
        "success_criteria": {},
        "local_decision": {},
        "round_index": 0,
        "max_rounds": 1,
    }

    positive = DecisionBriefBuilder().build(
        **kwargs,
        deliverable_contract={"requested_candidate_count": 3},
    )
    zero = DecisionBriefBuilder().build(
        **kwargs,
        deliverable_contract={"requested_candidate_count": 0},
    )

    assert positive["user_goal"]["requested_candidate_count"] == 3
    assert zero["user_goal"]["requested_candidate_count"] is None


def test_decision_brief_requested_candidate_count_priority_matches_router() -> None:
    kwargs = {
        "round_record": {"workflow_summary_raw": _raw_workflow_summary()},
        "prompt_context": {
            "latest_round": {
                "candidate_summary": {"count": 2, "items": [], "truncated": False},
                "survivor_summary": {"count": 2, "items": [], "truncated": False},
            }
        },
        "outcome_summary": {"state": "completed"},
        "success_criteria": {},
        "local_decision": {},
        "round_index": 0,
        "max_rounds": 1,
    }

    normalized = DecisionBriefBuilder().build(
        **kwargs,
        deliverable_contract={
            "requested_candidate_count": 4,
            "num_candidates": 3,
            "candidate_count": 2,
            "top_k": 1,
        },
    )
    legacy_overlap = DecisionBriefBuilder().build(
        **kwargs,
        deliverable_contract={
            "num_candidates": 3,
            "candidate_count": 2,
            "top_k": 1,
        },
    )

    assert normalized["user_goal"]["requested_candidate_count"] == 4
    assert legacy_overlap["user_goal"]["requested_candidate_count"] == 3


def test_decision_brief_uses_full_previous_survivor_stats_when_summary_truncated() -> None:
    survivor_items = []
    for index in range(12):
        survivor_items.append(
            {
                "candidate_id": f"prev_{index}",
                "source": {"tool": "rank_designs", "job_id": f"job_prev_{index}"},
                "capabilities": {
                    "has_sequence": True,
                    "has_full_structure": index < 11,
                },
                "has_sequence": True,
                "has_structure": index < 11,
                "sequence": "SHOULD_NOT_LEAK",
                "structure_path": f"/tmp/prev_{index}.pdb",
                "scores": {"raw_score": index},
                "artifacts": {"unsafe": f"/tmp/prev_{index}.json"},
            }
        )

    brief = DecisionBriefBuilder().build(
        round_record={
            "workflow_summary_raw": {"workflow_id": "wf_empty", "nodes": []},
            "previous_survivor_pool_stats": {
                "submit_ready_count": 11,
                "missing_capability_count": 1,
            },
        },
        prompt_context={
            "latest_round": {
                "candidate_summary": {"count": 0, "items": [], "truncated": False},
                "survivor_summary": {
                    "count": 12,
                    "items": survivor_items,
                    "truncated": True,
                },
            }
        },
        outcome_summary={"state": "completed"},
        success_criteria={},
        deliverable_contract={
            "required_candidate_capabilities": ["has_sequence", "has_full_structure"],
            "terminal_submission_allowed": True,
        },
        local_decision={},
        round_index=1,
        max_rounds=3,
    )

    assert len(brief["recommended_candidates"]) == 10
    assert brief["candidate_pool"]["submit_ready_count"] == 11
    assert brief["candidate_pool"]["missing_capability_count"] == 1
    assert brief["candidate_pool"]["truncated"] is True

    serialized = json.dumps(brief, sort_keys=True)
    assert "SHOULD_NOT_LEAK" not in serialized
    assert "/tmp/prev_0.pdb" not in serialized
    assert "raw_score" not in serialized
    assert "unsafe" not in serialized
