from __future__ import annotations

import copy

import pytest
from jsonschema import ValidationError

from vibeproteinagent.agents.planning.ir import validate_planner_ir_v1


def _valid_ir() -> dict:
    return {
        "version": "planner_ir_v1",
        "goal": "Design and rank candidate protein sequences.",
        "assumptions": ["Use local tools only."],
        "missing_inputs": [
            {
                "name": "target_structure",
                "reason": "Needed for structure-conditioned binding.",
                "required_for_execution": False,
            }
        ],
        "nodes": [
            {
                "id": "generate",
                "tool": "go_sequences",
                "intent": "Find candidate source sequences.",
                "inputs": {"go_term": "GO:0005524", "limit": 3},
                "upstream": [],
            },
            {
                "id": "rank",
                "tool": "rank_designs",
                "intent": "Rank generated candidates.",
                "inputs": {"top_k": 1},
                "upstream": [
                    {"node": "generate", "relationship": "candidate_source"}
                ],
            },
        ],
        "finalization": {
            "ranking_preference": "Prefer ATP-binding candidates.",
            "success_criteria": {"requires_sequence": True},
        },
    }


def _invalid(ir: dict) -> None:
    with pytest.raises(ValidationError):
        validate_planner_ir_v1(ir)


def test_valid_planner_ir_passes() -> None:
    validate_planner_ir_v1(_valid_ir())


def test_empty_nodes_fails() -> None:
    ir = _valid_ir()
    ir["nodes"] = []
    _invalid(ir)


@pytest.mark.parametrize("bad_id", ["1node", "bad-node"])
def test_invalid_node_id_fails(bad_id: str) -> None:
    ir = _valid_ir()
    ir["nodes"][0]["id"] = bad_id
    _invalid(ir)


def test_duplicate_node_ids_fail() -> None:
    ir = _valid_ir()
    ir["nodes"][1]["id"] = "generate"
    _invalid(ir)


def test_upstream_reference_to_later_node_fails() -> None:
    ir = _valid_ir()
    ir["nodes"][0]["upstream"] = [
        {"node": "rank", "relationship": "candidate_source"}
    ]
    _invalid(ir)


def test_upstream_reference_to_unknown_node_fails() -> None:
    ir = _valid_ir()
    ir["nodes"][1]["upstream"] = [
        {"node": "missing", "relationship": "candidate_source"}
    ]
    _invalid(ir)


def test_relationship_other_than_candidate_source_fails() -> None:
    ir = _valid_ir()
    ir["nodes"][1]["upstream"][0]["relationship"] = "evidence"
    _invalid(ir)


@pytest.mark.parametrize(
    "bad_ref",
    ["${outputs.generate.sequence}", "${jobs.rank}", "${item.sequence}"],
)
def test_executor_refs_anywhere_in_strings_fail(bad_ref: str) -> None:
    ir = _valid_ir()
    ir["nodes"][1]["intent"] = f"Use {bad_ref}"
    _invalid(ir)


@pytest.mark.parametrize("bad_token", ["workflow_ref", "submit_final"])
def test_workflow_ref_and_submit_final_anywhere_fail(bad_token: str) -> None:
    ir = _valid_ir()
    ir["nodes"][0]["inputs"][bad_token] = True
    _invalid(ir)


def test_submit_candidate_tool_fails() -> None:
    ir = _valid_ir()
    ir["nodes"][1]["tool"] = "submit_candidate"
    _invalid(ir)


def test_submit_candidate_node_id_fails() -> None:
    ir = _valid_ir()
    ir["nodes"][1]["id"] = "submit_candidate"
    _invalid(ir)


def test_finalization_is_optional() -> None:
    ir = _valid_ir()
    ir.pop("finalization")
    validate_planner_ir_v1(ir)


def test_finalization_expected_decision_is_rejected() -> None:
    ir = copy.deepcopy(_valid_ir())
    ir["finalization"]["expected_decision"] = "submit_candidate"
    _invalid(ir)
