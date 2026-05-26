from __future__ import annotations

from vibeproteinagent.agents.planning import canonical_tools
from vibeproteinagent.agents.planning.compiler import compile_planner_ir


def test_stub_scalar_consumer_emits_foreach_over_canonical_candidates() -> None:
    original = dict(canonical_tools.CANONICAL_SCALAR_CONSUMERS)
    canonical_tools.CANONICAL_SCALAR_CONSUMERS.clear()
    canonical_tools.CANONICAL_SCALAR_CONSUMERS["stub_tool"] = {"sequence": "sequence"}
    try:
        ir = {
            "version": "planner_ir_v1",
            "goal": "Test scalar fanout.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "go_fetch",
                    "tool": "go_sequences",
                    "intent": "Retrieve candidates.",
                    "inputs": {"go_id": "GO:0003824"},
                    "upstream": [],
                },
                {
                    "id": "per_candidate",
                    "tool": "stub_tool",
                    "intent": "Consume each candidate sequence.",
                    "inputs": {},
                    "upstream": [
                        {"node": "go_fetch", "relationship": "candidate_source"}
                    ],
                },
            ],
        }

        plan = compile_planner_ir(ir)
    finally:
        canonical_tools.CANONICAL_SCALAR_CONSUMERS.clear()
        canonical_tools.CANONICAL_SCALAR_CONSUMERS.update(original)

    consumer = plan["workflow_plan"]["nodes"][1]
    assert consumer["foreach"] == "${outputs.go_fetch.candidates}"
    assert consumer["inputs"]["sequence"] == "${item.sequence}"
