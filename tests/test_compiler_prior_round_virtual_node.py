from __future__ import annotations

import pytest

from vibeproteinagent.agents.execution.round_state import PriorRoundContext
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup


def _ir(nodes: list[dict]) -> dict:
    return {
        "version": "planner_ir_v1",
        "goal": "Continue from prior round candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": nodes,
        "finalization": {"success_criteria": {}},
    }


def _prior_context() -> PriorRoundContext:
    return PriorRoundContext(
        round_number=2,
        surviving_candidates=[],
        prior_decision_reason="Continue from the prior round.",
    )


def test_prior_round_single_upstream_wires_directly_without_real_node() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "rank",
                    "tool": "rank_designs",
                    "intent": "Rank surviving prior-round candidates.",
                    "inputs": {"top_k": 3},
                    "upstream": [
                        {"node": "prior_round", "relationship": "candidate_source"}
                    ],
                }
            ]
        ),
        prior_round_context=_prior_context(),
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert nodes == [
        {
            "id": "rank",
            "tool": "rank_designs",
            "inputs": {
                "top_k": 3,
                "candidates": "${outputs.prior_round.candidates}",
            },
        }
    ]
    assert "prior_round" not in {node["id"] for node in nodes}


def test_prior_round_and_real_upstream_merge_in_declared_order() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "go_fresh",
                    "tool": "go_sequences",
                    "intent": "Retrieve fresh candidates.",
                    "inputs": {"go_id": "GO:0003824"},
                    "upstream": [],
                },
                {
                    "id": "rank",
                    "tool": "rank_designs",
                    "intent": "Rank old and new candidates together.",
                    "inputs": {},
                    "upstream": [
                        {"node": "prior_round", "relationship": "candidate_source"},
                        {"node": "go_fresh", "relationship": "candidate_source"},
                    ],
                },
            ]
        ),
        prior_round_context=_prior_context(),
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["id"] for node in nodes] == [
        "go_fresh",
        "merge_rank_candidates",
        "rank",
    ]
    assert nodes[1]["inputs"]["candidate_sets"] == [
        "${outputs.prior_round.candidates}",
        "${outputs.go_fresh.candidates}",
    ]
    assert nodes[2]["inputs"]["candidates"] == "${outputs.merge_rank_candidates.candidates}"


def test_prior_round_into_non_candidate_consumer_raises_fixup() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "go_fresh",
                        "tool": "go_sequences",
                        "intent": "This producer cannot consume candidates.",
                        "inputs": {"go_id": "GO:0003824"},
                        "upstream": [
                            {"node": "prior_round", "relationship": "candidate_source"}
                        ],
                    }
                ]
            ),
            prior_round_context=_prior_context(),
        )

    assert exc_info.value.payload["error_kind"] == "missing_canonical_input_slot"
