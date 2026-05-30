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


def _downstream_protenix_ir(
    value: str,
    *,
    source: str = "planner_literal",
) -> dict:
    return {
        "version": "planner_ir_v1",
        "goal": "Generate, sequence, predict, and rank one candidate.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "backbone_gen",
                "tool": "rfdiffusion",
                "intent": "Generate a backbone.",
                "inputs": {"mode": "unconditional", "num_designs": 1},
                "upstream": [],
            },
            {
                "id": "sequence_design",
                "tool": "proteinmpnn",
                "intent": "Design a sequence.",
                "inputs": {"num_designs": 1},
                "upstream": [
                    {"node": "backbone_gen", "relationship": "candidate_source"}
                ],
            },
            {
                "id": "structure_pred",
                "tool": "protenix_v2",
                "intent": "Predict structure from the sequence candidate.",
                "inputs": {"input_json": value},
                "input_provenance": {"input_json": source},
                "upstream": [
                    {"node": "sequence_design", "relationship": "candidate_source"}
                ],
            },
        ],
        "finalization": {"success_criteria": {}},
    }


@pytest.mark.parametrize(
    "value",
    [
        "/mnt/vast/trillion/hyunjin/VibeProteinAgent/tests/fixtures/6MRR.pdb",
        "sequence_design",
    ],
)
def test_downstream_protenix_drops_planner_literal_input_json(value: str) -> None:
    plan = compile_planner_ir(_downstream_protenix_ir(value))
    structure = plan["workflow_plan"]["nodes"][2]

    assert structure["tool"] == "protenix_v2"
    assert structure["inputs"] == {
        "candidates": "${outputs.sequence_design.candidates}",
    }
    assert structure["input_provenance"] == {
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
def test_downstream_protenix_preserves_non_planner_input_json(source: str) -> None:
    plan = compile_planner_ir(
        _downstream_protenix_ir("/tmp/protenix_input.json", source=source)
    )
    structure = plan["workflow_plan"]["nodes"][2]

    assert structure["inputs"]["input_json"] == "/tmp/protenix_input.json"
    assert structure["input_provenance"]["input_json"] == source
    assert structure["inputs"]["candidates"] == "${outputs.sequence_design.candidates}"
    assert structure["input_provenance"]["candidates"] == "compiled_wiring"


def test_root_protenix_preserves_planner_literal_input_json() -> None:
    plan = compile_planner_ir(
        {
            "version": "planner_ir_v1",
            "goal": "Predict from a prepared Protenix input JSON.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "structure_pred",
                    "tool": "protenix_v2",
                    "intent": "Predict a prepared molecular system.",
                    "inputs": {"input_json": "/tmp/protenix_input.json"},
                    "upstream": [],
                }
            ],
            "finalization": {"success_criteria": {}},
        }
    )
    structure = plan["workflow_plan"]["nodes"][0]

    assert structure["inputs"] == {"input_json": "/tmp/protenix_input.json"}
    assert structure["input_provenance"] == {"input_json": "planner_literal"}


def test_downstream_proteinmpnn_drops_planner_literal_pdb_path() -> None:
    plan = compile_planner_ir(
        {
            "version": "planner_ir_v1",
            "goal": "Generate and sequence one backbone.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "backbone_gen",
                    "tool": "rfdiffusion",
                    "intent": "Generate a backbone.",
                    "inputs": {"mode": "unconditional", "num_designs": 1},
                    "upstream": [],
                },
                {
                    "id": "sequence_design",
                    "tool": "proteinmpnn",
                    "intent": "Design a sequence.",
                    "inputs": {"pdb_path": "backbone_gen", "num_designs": 1},
                    "upstream": [
                        {"node": "backbone_gen", "relationship": "candidate_source"}
                    ],
                },
            ],
            "finalization": {"success_criteria": {}},
        }
    )
    sequence = plan["workflow_plan"]["nodes"][1]

    assert sequence["inputs"] == {
        "num_designs": 1,
        "candidates": "${outputs.backbone_gen.candidates}",
    }
    assert sequence["input_provenance"] == {
        "num_designs": "planner_literal",
        "candidates": "compiled_wiring",
    }
