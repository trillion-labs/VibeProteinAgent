from __future__ import annotations

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.adapters.visibility import is_agent_visible_tool
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.canonical_tools import (
    CANONICAL_PRODUCERS,
    candidate_contract,
)


def test_search_sequence_is_canonical_sequence_score_producer() -> None:
    assert "search_sequence" in CANONICAL_PRODUCERS
    contract = candidate_contract("search_sequence")
    assert {"has_sequence", "has_candidate_scores"} <= set(contract.produces)


def test_search_sequence_is_valid_first_tool_for_sequence_workflow() -> None:
    plan = compile_planner_ir(
        {
            "version": "planner_ir_v1",
            "goal": "Retrieve and analyze representative sequences.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "retrieve_sequences",
                    "tool": "search_sequence",
                    "intent": "Retrieve sequence candidates.",
                    "inputs": {"context": "NAD(P)HX repair proteins", "max_downstream_candidates": 10},
                    "upstream": [],
                },
                {
                    "id": "analyze_sequences",
                    "tool": "sequence_analysis",
                    "intent": "Analyze retrieved sequences.",
                    "inputs": {},
                    "upstream": [
                        {"node": "retrieve_sequences", "relationship": "candidate_source"}
                    ],
                },
            ],
            "finalization": {"ranking_preference": None, "success_criteria": {}},
        }
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["tool"] for node in nodes] == ["search_sequence", "sequence_analysis"]
    assert nodes[1]["inputs"]["candidates"] == "${outputs.retrieve_sequences.candidates}"


def test_agent_visible_catalog_prefers_search_sequence_over_legacy_streams() -> None:
    registry = ToolRegistry()
    registry.discover()
    descriptors = {
        descriptor.name: descriptor
        for descriptor in registry.list()
        if is_agent_visible_tool(descriptor)
    }

    assert "search_sequence" in descriptors
    assert "go_sequences" in descriptors
    assert "interpro_sequences" in descriptors
    assert "legacy/internal" in descriptors["go_sequences"].description.lower()
    assert "legacy/internal" in descriptors["interpro_sequences"].description.lower()
    assert "function_term_resolver" not in descriptors
    assert registry.get("go_sequences")[0].name == "go_sequences"
