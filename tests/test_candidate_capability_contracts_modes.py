from __future__ import annotations

import pytest

from vibeproteinagent.agents.planning.canonical_tools import (
    annotate_candidate_capabilities,
    candidate_capabilities_from_record,
    candidate_capability_flags,
    candidate_contract_for_node,
    candidate_contract_for_inputs,
)
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup


def _ir(nodes: list[dict]) -> dict:
    return {
        "version": "planner_ir_v1",
        "goal": "capability wiring test",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": nodes,
    }


def _node(
    node_id: str,
    tool: str,
    inputs: dict | None = None,
    upstream: list[dict] | None = None,
) -> dict:
    return {
        "id": node_id,
        "tool": tool,
        "intent": f"run {tool}",
        "inputs": inputs or {},
        "upstream": upstream or [],
    }


def _candidate_source(node_id: str) -> dict:
    return {"node": node_id, "relationship": "candidate_source"}


def test_explicit_capabilities_prevent_backbone_only_full_structure_promotion() -> None:
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_backbone",
        "source": {"tool": "rfdiffusion", "local_id": "design_0000"},
        "structure_path": "/tmp/backbone.pdb",
        "scores": {"rfdiffusion": {"score": None}},
        "capabilities": candidate_capability_flags(
            has_backbone_structure=True,
            has_full_structure=False,
            has_candidate_scores=True,
        ),
    }

    capabilities = candidate_capabilities_from_record(candidate)

    assert "has_backbone_structure" in capabilities
    assert "has_candidate_scores" in capabilities
    assert "has_full_structure" not in capabilities
    assert annotate_candidate_capabilities(candidate)["capabilities"]["has_full_structure"] is False


def test_mode_sensitive_generation_contracts() -> None:
    assert candidate_contract_for_inputs("rfdiffusion", {"mode": "unconditional"}).consumes == frozenset()
    assert candidate_contract_for_inputs("rfdiffusion", {"mode": "partial_diffusion"}).consumes == frozenset(
        {"has_backbone_structure"}
    )
    assert candidate_contract_for_inputs("rfpeptides", {}).consumes == frozenset()
    assert candidate_contract_for_inputs("rfpeptides", {"target_pdb": "/tmp/t.pdb"}).consumes == frozenset(
        {"has_backbone_structure"}
    )


def test_family_profile_candidate_contract_depends_on_profile_source() -> None:
    build = candidate_contract_for_inputs("family_profile", {})
    score_literal = candidate_contract_for_inputs(
        "family_profile",
        {"profile": {"schema_version": "family_profile_v1"}},
    )
    score_wired = candidate_contract_for_node(
        "family_profile",
        {},
        [{"node": "family_ref", "relationship": "profile_source"}],
    )

    assert build.consumes == frozenset({"has_sequence"})
    assert build.produces == frozenset()
    assert build.preserves_upstream is False
    assert "has_candidate_scores" in score_literal.produces
    assert "has_candidate_scores" in score_wired.produces


def test_esm3_tracks_control_produced_structure_capabilities() -> None:
    embedding_only = candidate_contract_for_inputs("esm3", {"tracks": ["embedding"]})
    with_structure = candidate_contract_for_inputs("esm3", {"tracks": ["sequence", "structure"]})

    assert "has_full_structure" not in embedding_only.produces
    assert "has_candidate_scores" in embedding_only.produces
    assert {"has_sequence", "has_backbone_structure", "has_full_structure", "has_candidate_scores"} <= set(
        with_structure.produces
    )


def test_pxdesign_produces_sequence_and_backbone_not_full_structure() -> None:
    contract = candidate_contract_for_inputs("pxdesign", {"target_pdb": "/tmp/target.cif"})

    assert {"has_sequence", "has_backbone_structure", "has_candidate_scores"} <= set(
        contract.produces
    )
    assert "has_full_structure" not in contract.produces
    assert "has_full_structure" in contract.consumes


def test_pxdesign_compiler_adds_full_structure_tail_before_downstream_consumers() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                _node("pxd", "pxdesign", {"target_pdb": "/tmp/target.cif", "top_k_designs": 3}),
                _node("analyze", "structure_analysis", {}, [_candidate_source("pxd")]),
            ]
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["tool"] for node in nodes] == [
        "pxdesign",
        "protenix_v2",
        "protein_qc",
        "rank_designs",
        "structure_analysis",
    ]
    assert nodes[1]["inputs"]["candidates"] == "${outputs.pxd.candidates}"
    assert nodes[2]["inputs"]["candidates"] == "${outputs.pxd_protenix.candidates}"
    assert nodes[3]["inputs"]["candidates"] == "${outputs.pxd_qc.candidates}"
    assert nodes[3]["inputs"]["top_k"] == 3
    assert nodes[4]["inputs"]["candidates"] == "${outputs.pxd_qc.candidates}"


def test_backbone_only_generator_can_feed_proteinmpnn_but_not_target_structure_binder() -> None:
    good = compile_planner_ir(
        _ir(
            [
                _node("bb", "rfdiffusion", {"mode": "unconditional"}),
                _node("seq", "proteinmpnn", {}, [_candidate_source("bb")]),
            ]
        )
    )
    assert good["workflow_plan"]["nodes"][1]["inputs"]["candidates"] == "${outputs.bb.candidates}"

    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    _node("bb", "rfdiffusion", {"mode": "unconditional"}),
                    _node(
                        "binder",
                        "bindcraft",
                        {"target_chain": "A", "binder_length": 60},
                        [_candidate_source("bb")],
                    ),
                ]
            )
        )

    payload = exc_info.value.payload
    assert payload["error_kind"] == "candidate_capability_mismatch"
    assert "has_full_structure" in payload["missing_capabilities"]


def test_pairwise_structure_tools_accept_single_stream_and_check_item_count_at_runtime() -> None:
    single_stream = compile_planner_ir(
        _ir(
            [
                _node("a", "rfdiffusion", {"mode": "unconditional"}),
                _node("align", "tmalign", {}, [_candidate_source("a")]),
            ]
        )
    )

    assert (
        single_stream["workflow_plan"]["nodes"][-1]["inputs"]["candidates"]
        == "${outputs.a.candidates}"
    )

    plan = compile_planner_ir(
        _ir(
            [
                _node("a", "rfdiffusion", {"mode": "unconditional"}),
                _node("b", "rfdiffusion", {"mode": "unconditional"}),
                _node("align", "tmalign", {}, [_candidate_source("a"), _candidate_source("b")]),
            ]
        )
    )
    node_ids = [node["id"] for node in plan["workflow_plan"]["nodes"]]
    assert "merge_align_candidates" in node_ids
    assert plan["workflow_plan"]["nodes"][-1]["inputs"]["candidates"] == "${outputs.merge_align_candidates.candidates}"
