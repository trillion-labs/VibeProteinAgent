from __future__ import annotations

import pytest

from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup
from vibeproteinagent.agents.planning.ir import validate_planner_ir_v1
from vibeproteinagent.agents.planning.workflow_planner import (
    _validate_planner_ir_against_selected_tools,
)
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.parameter_resolution import normalize_tool_parameters


def _ir(nodes: list[dict]) -> dict:
    return {
        "version": "planner_ir_v1",
        "goal": "Build and use family profile evidence.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": nodes,
        "finalization": {"success_criteria": {}},
    }


def test_profile_source_passes_planner_ir_schema_validation() -> None:
    payload = _ir(
        [
            {
                "id": "family_ref",
                "tool": "family_profile",
                "intent": "Build a family reference profile.",
                "inputs": {"fasta_path": "/tmp/family.fasta"},
                "upstream": [],
            },
            {
                "id": "score_family",
                "tool": "family_profile",
                "intent": "Score generated candidates against the reference profile.",
                "inputs": {},
                "upstream": [
                    {"node": "family_ref", "relationship": "profile_source"}
                ],
            },
        ]
    )

    validate_planner_ir_v1(payload)


def test_candidate_and_profile_sources_compile_to_separate_inputs() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "natural_hits",
                    "tool": "search_sequence",
                    "intent": "Retrieve natural family exemplars.",
                    "inputs": {"query": "TIM barrel hydrolase", "max_downstream_candidates": 10},
                    "upstream": [],
                },
                {
                    "id": "family_ref",
                    "tool": "family_profile",
                    "intent": "Build profile evidence from natural exemplars.",
                    "inputs": {"family_label": "TIM barrel hydrolase"},
                    "upstream": [
                        {"node": "natural_hits", "relationship": "candidate_source"}
                    ],
                },
                {
                    "id": "generated",
                    "tool": "esm2",
                    "intent": "Generate candidate variants.",
                    "inputs": {"mode": "sequence_generation"},
                    "upstream": [
                        {"node": "natural_hits", "relationship": "candidate_source"}
                    ],
                },
                {
                    "id": "score_family",
                    "tool": "family_profile",
                    "intent": "Annotate generated candidates with profile evidence.",
                    "inputs": {},
                    "upstream": [
                        {"node": "generated", "relationship": "candidate_source"},
                        {"node": "family_ref", "relationship": "profile_source"},
                    ],
                },
                {
                    "id": "rank",
                    "tool": "rank_designs",
                    "intent": "Rank profile-annotated generated candidates.",
                    "inputs": {"top_k": 3},
                    "upstream": [
                        {"node": "score_family", "relationship": "candidate_source"}
                    ],
                },
            ]
        )
    )

    nodes = {node["id"]: node for node in plan["workflow_plan"]["nodes"]}
    assert nodes["family_ref"]["inputs"]["candidates"] == "${outputs.natural_hits.candidates}"
    assert nodes["family_ref"]["input_provenance"]["candidates"] == "compiled_wiring"
    assert nodes["score_family"]["inputs"]["candidates"] == "${outputs.generated.candidates}"
    assert nodes["score_family"]["inputs"]["profile"] == "${outputs.family_ref.profile}"
    assert nodes["score_family"]["input_provenance"]["candidates"] == "compiled_wiring"
    assert nodes["score_family"]["input_provenance"]["profile"] == "compiled_wiring"
    assert nodes["rank"]["inputs"]["candidates"] == "${outputs.score_family.candidates}"


def test_family_profile_build_output_cannot_feed_candidate_consumer() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "natural_hits",
                        "tool": "search_sequence",
                        "intent": "Retrieve natural family exemplars.",
                        "inputs": {"query": "TIM barrel hydrolase"},
                        "upstream": [],
                    },
                    {
                        "id": "family_ref",
                        "tool": "family_profile",
                        "intent": "Build profile evidence from natural exemplars.",
                        "inputs": {},
                        "upstream": [
                            {"node": "natural_hits", "relationship": "candidate_source"}
                        ],
                    },
                    {
                        "id": "rank",
                        "tool": "rank_designs",
                        "intent": "Incorrectly rank profile-only build output.",
                        "inputs": {"top_k": 3},
                        "upstream": [
                            {"node": "family_ref", "relationship": "candidate_source"}
                        ],
                    },
                ]
            )
        )

    payload = exc_info.value.payload
    assert payload["error_kind"] == "upstream_does_not_produce_candidates"
    assert payload["upstream_node"] == "family_ref"


def test_profile_source_from_non_profile_producer_fails_structured() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "natural_hits",
                        "tool": "search_sequence",
                        "intent": "Retrieve natural family exemplars.",
                        "inputs": {"query": "hydrolase"},
                        "upstream": [],
                    },
                    {
                        "id": "score_family",
                        "tool": "family_profile",
                        "intent": "Incorrectly consume candidates as a profile.",
                        "inputs": {},
                        "upstream": [
                            {"node": "natural_hits", "relationship": "profile_source"}
                        ],
                    },
                ]
            )
        )

    payload = exc_info.value.payload
    assert payload["error_kind"] == "upstream_does_not_produce_profile"
    assert payload["field"] == "profile"
    assert payload["upstream_node"] == "natural_hits"


def test_profile_source_into_non_profile_consumer_fails_structured() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "family_ref",
                        "tool": "family_profile",
                        "intent": "Build a profile.",
                        "inputs": {"fasta_path": "/tmp/family.fasta"},
                        "upstream": [],
                    },
                    {
                        "id": "rank",
                        "tool": "rank_designs",
                        "intent": "Incorrectly consume profile evidence directly.",
                        "inputs": {"top_k": 3},
                        "upstream": [
                            {"node": "family_ref", "relationship": "profile_source"}
                        ],
                    },
                ]
            )
        )

    payload = exc_info.value.payload
    assert payload["error_kind"] == "missing_profile_input_slot"
    assert payload["field"] == "profile"
    assert payload["node_id"] == "rank"


@pytest.mark.parametrize("tool", ["esm2", "esm3", "rfdiffusion", "proteinmpnn"])
def test_profile_source_into_generator_compiles_to_profile_guidance(tool: str) -> None:
    inputs_by_tool = {
        "esm2": {"sequence": "ACDE"},
        "esm3": {"tracks": ["embedding"]},
        "rfdiffusion": {"mode": "unconditional", "num_designs": 1},
        "proteinmpnn": {"pdb_path": "/tmp/backbone.pdb"},
    }
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "family_ref",
                    "tool": "family_profile",
                    "intent": "Build a family profile.",
                    "inputs": {"fasta_path": "/tmp/family.fasta"},
                    "upstream": [],
                },
                {
                    "id": "guided",
                    "tool": tool,
                    "intent": "Use profile guidance softly.",
                    "inputs": inputs_by_tool[tool],
                    "upstream": [
                        {"node": "family_ref", "relationship": "profile_source"}
                    ],
                },
            ]
        )
    )

    guided = {node["id"]: node for node in plan["workflow_plan"]["nodes"]}["guided"]
    assert guided["inputs"]["profile_guidance"] == "${outputs.family_ref.profile}"
    assert guided["input_provenance"]["profile_guidance"] == "compiled_wiring"
    if tool == "esm2":
        assert guided["inputs"]["mode"] == "sequence_generation"
        assert guided["input_provenance"]["mode"] == "compiled_wiring"


def test_planner_validation_allows_esm2_profile_guided_mode_omission() -> None:
    payload = _ir(
        [
            {
                "id": "family_ref",
                "tool": "family_profile",
                "intent": "Build profile evidence.",
                "inputs": {"fasta_path": "/tmp/family.fasta"},
                "upstream": [],
            },
            {
                "id": "guided",
                "tool": "esm2",
                "intent": "Use profile guidance softly.",
                "inputs": {"sequence": "ACDE"},
                "upstream": [
                    {"node": "family_ref", "relationship": "profile_source"}
                ],
            },
        ]
    )
    selected_tools = [
        {
            "name": "family_profile",
            "input_schema": {
                "type": "object",
                "properties": {"fasta_path": {"type": "string"}},
            },
        },
        {
            "name": "esm2",
            "selected_mode": "sequence_generation",
            "input_schema": {
                "type": "object",
                "anyOf": [{"required": ["sequence"]}],
                "properties": {
                    "sequence": {"type": "string"},
                    "mode": {"type": "string"},
                    "profile_guidance": {"type": "object"},
                },
            },
        },
    ]

    _validate_planner_ir_against_selected_tools(payload, selected_tools)

    guided = payload["nodes"][1]
    assert guided["inputs"]["mode"] == "sequence_generation"


def test_profile_guided_generation_must_score_before_ranking() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "family_ref",
                    "tool": "family_profile",
                    "intent": "Build profile evidence.",
                    "inputs": {"fasta_path": "/tmp/family.fasta"},
                    "upstream": [],
                },
                {
                    "id": "guided",
                    "tool": "esm2",
                    "intent": "Generate with soft profile guidance.",
                    "inputs": {"sequence": "ACDE", "mode": "sequence_generation"},
                    "upstream": [
                        {"node": "family_ref", "relationship": "profile_source"}
                    ],
                },
                {
                    "id": "score_family",
                    "tool": "family_profile",
                    "intent": "Annotate generated candidates with profile evidence.",
                    "inputs": {},
                    "upstream": [
                        {"node": "guided", "relationship": "candidate_source"},
                        {"node": "family_ref", "relationship": "profile_source"},
                    ],
                },
                {
                    "id": "rank",
                    "tool": "rank_designs",
                    "intent": "Rank profile-annotated candidates.",
                    "inputs": {"top_k": 3},
                    "upstream": [
                        {"node": "score_family", "relationship": "candidate_source"}
                    ],
                },
            ]
        )
    )

    nodes = {node["id"]: node for node in plan["workflow_plan"]["nodes"]}
    assert nodes["guided"]["inputs"]["profile_guidance"] == (
        "${outputs.family_ref.profile}"
    )
    assert nodes["score_family"]["inputs"]["candidates"] == (
        "${outputs.guided.candidates}"
    )
    assert nodes["score_family"]["inputs"]["profile"] == "${outputs.family_ref.profile}"
    assert nodes["rank"]["inputs"]["candidates"] == "${outputs.score_family.candidates}"


def test_planner_literal_profile_guidance_is_rejected_by_compiler() -> None:
    with pytest.raises(StructuredFixup) as fixup_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "guided",
                        "tool": "esm2",
                        "intent": "Incorrectly use planner-authored profile guidance.",
                        "inputs": {
                            "sequence": "ACDE",
                            "profile_guidance": {"schema_version": "profile_guidance_v1"},
                        },
                        "upstream": [],
                    }
                ]
            )
        )

    payload = fixup_info.value.payload
    assert payload["error_kind"] == "planner_literal_profile_reference"
    assert payload["field"] == "profile_guidance"


def test_planner_literal_profile_is_rejected_by_ownership_validation() -> None:
    with pytest.raises(StructuredFixup) as fixup_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "score_family",
                        "tool": "family_profile",
                        "intent": "Incorrectly score using a planner-authored profile.",
                        "inputs": {"profile": {"schema_version": "family_profile_v1"}},
                        "upstream": [],
                    }
                ]
            )
        )

    payload = fixup_info.value.payload
    assert payload["error_kind"] == "planner_literal_profile_reference"
    assert payload["field"] == "profile"


def test_planner_literal_profile_is_still_rejected_by_ownership_validation() -> None:
    with pytest.raises(InvalidParamsError) as exc_info:
        normalize_tool_parameters(
            tool="family_profile",
            inputs={"profile": {"schema_version": "family_profile_v1"}},
            provenance={"profile": "planner_literal"},
            original_inputs={"profile": {"schema_version": "family_profile_v1"}},
        )

    event = exc_info.value.details["parameter_conflict_event"]
    assert event["field"] == "profile"
    assert event["ownership"] == "profile_evidence"
    assert event["status"] == "failed_planner_literal_not_allowed"


def test_planner_literal_profile_guidance_is_still_rejected_by_runtime_guard() -> None:
    with pytest.raises(InvalidParamsError) as exc_info:
        normalize_tool_parameters(
            tool="esm2",
            inputs={
                "sequence": "ACDE",
                "profile_guidance": {"schema_version": "profile_guidance_v1"},
            },
            provenance={"profile_guidance": "planner_literal"},
            original_inputs={"profile_guidance": {"schema_version": "profile_guidance_v1"}},
        )

    event = exc_info.value.details["parameter_conflict_event"]
    assert event["field"] == "profile_guidance"
    assert event["ownership"] == "profile_evidence"
    assert event["status"] == "failed_planner_literal_not_allowed"
