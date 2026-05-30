from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.parameter_resolution import normalize_tool_parameters

FIXTURES = Path(__file__).parent / "fixtures" / "planner_ir"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_single_node_ir_compiles_with_inputs_passed_through() -> None:
    plan = compile_planner_ir(_fixture("single_go.json"))
    nodes = plan["workflow_plan"]["nodes"]

    assert plan["source_kind"] == "planner_compiled"
    assert nodes == [
        {
            "id": "go_fetch",
            "tool": "go_sequences",
            "inputs": {"go_id": "GO:0003824", "max_sequences": 2},
            "input_provenance": {
                "go_id": "planner_literal",
                "max_sequences": "planner_literal",
            },
        }
    ]


def test_single_upstream_candidate_source_wires_directly() -> None:
    plan = compile_planner_ir(_fixture("go_to_rank.json"))
    nodes = plan["workflow_plan"]["nodes"]

    assert [node["id"] for node in nodes] == ["go_fetch", "rank"]
    assert nodes[0]["inputs"] == {"go_id": "GO:0003824"}
    assert nodes[1]["inputs"] == {
        "top_k": 3,
        "candidates": "${outputs.go_fetch.candidates}",
    }
    assert nodes[1]["input_provenance"] == {
        "top_k": "planner_literal",
        "candidates": "compiled_wiring",
    }
    assert plan["required_tools"] == ["go_sequences", "rank_designs"]


def _rank_with_aux_ir(
    aux_inputs: dict,
    *,
    input_provenance: dict | None = None,
) -> dict:
    node = {
        "id": "rank",
        "tool": "rank_designs",
        "intent": "Rank retrieved candidate sequences.",
        "inputs": {"top_k": 3, **aux_inputs},
        "upstream": [{"node": "go_fetch", "relationship": "candidate_source"}],
    }
    if input_provenance is not None:
        node["input_provenance"] = input_provenance
    return {
        "version": "planner_ir_v1",
        "goal": "Retrieve and rank candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_fetch",
                "tool": "go_sequences",
                "intent": "Retrieve candidate sequences.",
                "inputs": {"go_id": "GO:0003824"},
                "upstream": [],
            },
            node,
        ],
        "finalization": {"success_criteria": {}},
    }


def test_rank_designs_drops_empty_planner_literal_aux_candidate_streams() -> None:
    plan = compile_planner_ir(
        _rank_with_aux_ir(
            {
                "predictions": [],
                "analyses": [],
                "qc_results": [],
            }
        )
    )

    rank = plan["workflow_plan"]["nodes"][1]

    assert rank["inputs"] == {
        "top_k": 3,
        "candidates": "${outputs.go_fetch.candidates}",
    }
    assert rank["input_provenance"] == {
        "top_k": "planner_literal",
        "candidates": "compiled_wiring",
    }


@pytest.mark.parametrize(
    "source",
    [
        "compiled_wiring",
        "runtime_derived",
        "upstream_candidate",
        "user_explicit",
        "known_input",
        "adapter_default",
    ],
)
def test_rank_designs_preserves_non_planner_empty_aux_candidate_streams(
    source: str,
) -> None:
    plan = compile_planner_ir(
        _rank_with_aux_ir(
            {"qc_results": []},
            input_provenance={"qc_results": source, "top_k": "planner_literal"},
        )
    )

    rank = plan["workflow_plan"]["nodes"][1]

    assert rank["inputs"]["qc_results"] == []
    assert rank["input_provenance"]["qc_results"] == source
    assert rank["inputs"]["candidates"] == "${outputs.go_fetch.candidates}"
    assert rank["input_provenance"]["candidates"] == "compiled_wiring"


def test_rank_designs_non_empty_planner_literal_aux_stream_still_rejected() -> None:
    plan = compile_planner_ir(
        _rank_with_aux_ir({"qc_results": [{"id": "qc_1", "flags": []}]})
    )
    rank = plan["workflow_plan"]["nodes"][1]

    assert rank["inputs"]["qc_results"] == [{"id": "qc_1", "flags": []}]
    assert rank["input_provenance"]["qc_results"] == "planner_literal"

    with pytest.raises(InvalidParamsError) as excinfo:
        normalize_tool_parameters(
            tool="rank_designs",
            inputs={
                "qc_results": [{"id": "qc_1", "flags": []}],
                "candidates": [],
                "top_k": 3,
            },
            provenance=rank["input_provenance"],
            original_inputs=rank["inputs"],
        )

    event = excinfo.value.details["parameter_conflict_event"]
    assert event["field"] == "qc_results"
    assert event["ownership"] == "candidate_stream"
    assert event["status"] == "failed_planner_literal_not_allowed"
