from __future__ import annotations

import pytest

from vibeproteinagent.agents.execution.round_state import PriorRoundContext
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup, apply_enum_fixup_choice


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
        prior_decision_reason="Continue from the prior round.",
        survivor_snapshot_id="csnap_survivor",
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
            "input_provenance": {
                "top_k": "planner_literal",
                "candidates": "compiled_wiring",
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


def test_multi_candidate_item_requirement_is_not_static_upstream_count_gate() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "backbone_gen",
                    "tool": "rfdiffusion",
                    "intent": "Generate structure candidates.",
                    "inputs": {"mode": "unconditional"},
                    "upstream": [],
                },
                {
                    "id": "align",
                    "tool": "tmalign",
                    "intent": "Compare generated structures if enough candidates exist.",
                    "inputs": {},
                    "upstream": [
                        {"node": "backbone_gen", "relationship": "candidate_source"}
                    ],
                },
            ]
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert nodes[1]["tool"] == "tmalign"
    assert nodes[1]["inputs"]["candidates"] == "${outputs.backbone_gen.candidates}"


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


def test_prior_round_tool_node_raises_structured_fixup() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "prior_context",
                        "tool": "prior_round",
                        "intent": "Incorrectly treat prior round as a tool.",
                        "inputs": {},
                        "upstream": [],
                    }
                ]
            ),
            prior_round_context=_prior_context(),
        )

    assert exc_info.value.payload["error_kind"] == "prior_round_used_as_tool"


def test_node_id_used_as_literal_path_raises_structured_fixup() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "backbone_gen",
                        "tool": "rfdiffusion3",
                        "intent": "Generate a backbone.",
                        "inputs": {"mode": "unconditional"},
                        "upstream": [],
                    },
                    {
                        "id": "seq_design",
                        "tool": "proteinmpnn",
                        "intent": "Design sequence from backbone.",
                        "inputs": {"pdb_path": "backbone_gen"},
                        "upstream": [
                            {"node": "backbone_gen", "relationship": "candidate_source"}
                        ],
                    },
                ]
            )
        )

    payload = exc_info.value.payload
    assert payload["error_kind"] == "node_reference_used_as_literal_path"
    assert payload["field"] == "pdb_path"
    assert payload["literal_value"] == "backbone_gen"


def test_nested_node_id_path_fixup_removes_nested_input() -> None:
    llm_ir = _ir(
        [
            {
                "id": "backbone_gen",
                "tool": "rfdiffusion",
                "intent": "Generate a backbone.",
                "inputs": {"mode": "binder_design", "target_pdb": "/tmp/target.pdb"},
                "upstream": [],
            },
            {
                "id": "seq_design",
                "tool": "proteinmpnn",
                "intent": "Design sequence from backbone.",
                "inputs": {"nested": {"pdb_path": "backbone_gen"}},
                "upstream": [
                    {"node": "backbone_gen", "relationship": "candidate_source"}
                ],
            },
        ]
    )
    payload = {
        "error_kind": "node_reference_used_as_literal_path",
        "node_id": "seq_design",
        "field": "nested.pdb_path",
        "literal_value": "backbone_gen",
        "upstream_node": "backbone_gen",
        "options": [{"choice_id": "remove_input", "field": "nested.pdb_path"}],
    }

    repaired = apply_enum_fixup_choice(
        llm_ir,
        payload,
        {"choice_id": "remove_input", "field": "nested.pdb_path"},
    )

    assert repaired["nodes"][1]["inputs"] == {"nested": {}}


def test_node_id_used_as_literal_path_inside_list_raises_structured_fixup() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "backbone_gen",
                        "tool": "rfdiffusion",
                        "intent": "Generate a backbone.",
                        "inputs": {"mode": "unconditional"},
                        "upstream": [],
                    },
                    {
                        "id": "seq_design",
                        "tool": "proteinmpnn",
                        "intent": "Design sequence from backbone.",
                        "inputs": {"structures": [{"pdb_path": "backbone_gen"}]},
                        "upstream": [
                            {"node": "backbone_gen", "relationship": "candidate_source"}
                        ],
                    },
                ]
            )
        )

    assert exc_info.value.payload["error_kind"] == "node_reference_used_as_literal_path"
    assert exc_info.value.payload["field"] == "structures[0].pdb_path"


def test_virtual_prior_round_used_as_literal_path_raises_structured_fixup() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "seq_design",
                        "tool": "proteinmpnn",
                        "intent": "Design sequence from prior survivor.",
                        "inputs": {"pdb_path": "prior_round"},
                        "upstream": [
                            {"node": "prior_round", "relationship": "candidate_source"}
                        ],
                    },
                ]
            ),
            prior_round_context=_prior_context(),
        )

    payload = exc_info.value.payload
    assert payload["error_kind"] == "node_reference_used_as_literal_path"
    assert payload["field"] == "pdb_path"
    assert payload["literal_value"] == "prior_round"


def test_list_index_node_id_path_fixup_removes_nested_input() -> None:
    llm_ir = _ir(
        [
            {
                "id": "backbone_gen",
                "tool": "rfdiffusion",
                "intent": "Generate a backbone.",
                "inputs": {"mode": "unconditional"},
                "upstream": [],
            },
            {
                "id": "seq_design",
                "tool": "proteinmpnn",
                "intent": "Design sequence from backbone.",
                "inputs": {"structures": [{"pdb_path": "backbone_gen", "chain": "A"}]},
                "upstream": [
                    {"node": "backbone_gen", "relationship": "candidate_source"}
                ],
            },
        ]
    )
    repaired = apply_enum_fixup_choice(
        llm_ir,
        {
            "error_kind": "node_reference_used_as_literal_path",
            "node_id": "seq_design",
            "field": "structures[0].pdb_path",
            "literal_value": "backbone_gen",
            "upstream_node": "backbone_gen",
            "options": [{"choice_id": "remove_input", "field": "structures[0].pdb_path"}],
        },
        {"choice_id": "remove_input", "field": "structures[0].pdb_path"},
    )

    assert repaired["nodes"][1]["inputs"] == {"structures": [{"chain": "A"}]}


def test_function_resolver_wires_multi_intent_term_lists_to_sequence_tools() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "resolve_terms",
                    "tool": "function_term_resolver",
                    "intent": "Resolve independent facets.",
                    "inputs": {"query": "facet A and facet B"},
                    "upstream": [],
                },
                {
                    "id": "go_fetch",
                    "tool": "go_sequences",
                    "intent": "Retrieve GO sequences for all resolved facets.",
                    "inputs": {},
                    "upstream": [
                        {"node": "resolve_terms", "relationship": "term_source"}
                    ],
                },
                {
                    "id": "ipr_fetch",
                    "tool": "interpro_sequences",
                    "intent": "Retrieve InterPro sequences for all resolved facets.",
                    "inputs": {},
                    "upstream": [
                        {"node": "resolve_terms", "relationship": "term_source"}
                    ],
                },
            ]
        )
    )

    nodes = {node["id"]: node for node in plan["workflow_plan"]["nodes"]}
    assert nodes["go_fetch"]["inputs"]["go_id"] is None
    assert nodes["go_fetch"]["inputs"]["go_terms"] == "${outputs.resolve_terms.selected_go_terms}"
    assert nodes["ipr_fetch"]["inputs"]["interpro_id"] is None
    assert (
        nodes["ipr_fetch"]["inputs"]["interpro_terms"]
        == "${outputs.resolve_terms.selected_interpro_terms}"
    )
