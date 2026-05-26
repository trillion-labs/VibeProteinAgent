from __future__ import annotations

import copy
import json
from pathlib import Path

from vibeproteinagent.agents.planning.compiler import compile_planner_ir

FIXTURES = Path(__file__).parent / "fixtures" / "planner_ir"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_two_candidate_upstreams_insert_merge_node_in_order() -> None:
    plan = compile_planner_ir(_fixture("two_go_to_rank.json"))
    nodes = plan["workflow_plan"]["nodes"]

    assert [node["id"] for node in nodes] == [
        "go_a",
        "go_b",
        "merge_rank_candidates",
        "rank",
    ]
    merge = nodes[2]
    assert merge["tool"] == "merge_candidates"
    assert merge["inputs"] == {
        "candidate_sets": [
            "${outputs.go_a.candidates}",
            "${outputs.go_b.candidates}",
        ],
        "dedup_policy": "keep_all",
    }
    assert nodes[3]["inputs"]["candidates"] == "${outputs.merge_rank_candidates.candidates}"


def test_three_candidate_upstreams_preserve_declared_order() -> None:
    ir = _fixture("two_go_to_rank.json")
    go_c = copy.deepcopy(ir["nodes"][1])
    go_c["id"] = "go_c"
    go_c["inputs"] = {"go_id": "GO:0009987"}
    ir["nodes"].insert(2, go_c)
    ir["nodes"][3]["upstream"].append(
        {"node": "go_c", "relationship": "candidate_source"}
    )

    plan = compile_planner_ir(ir)
    merge = plan["workflow_plan"]["nodes"][3]

    assert merge["id"] == "merge_rank_candidates"
    assert merge["inputs"]["candidate_sets"] == [
        "${outputs.go_a.candidates}",
        "${outputs.go_b.candidates}",
        "${outputs.go_c.candidates}",
    ]
