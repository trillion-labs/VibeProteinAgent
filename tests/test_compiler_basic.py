from __future__ import annotations

import json
from pathlib import Path

from vibeproteinagent.agents.planning.compiler import compile_planner_ir

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
    assert plan["required_tools"] == ["go_sequences", "rank_designs"]
