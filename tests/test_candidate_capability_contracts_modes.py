from __future__ import annotations

import pytest

from vibeproteinagent.agents.planning.canonical_tools import (
    annotate_candidate_capabilities,
    candidate_capabilities_from_record,
    candidate_capability_flags,
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


def test_esm3_tracks_control_produced_structure_capabilities() -> None:
    embedding_only = candidate_contract_for_inputs("esm3", {"tracks": ["embedding"]})
    with_structure = candidate_contract_for_inputs("esm3", {"tracks": ["sequence", "structure"]})

    assert "has_full_structure" not in embedding_only.produces
    assert "has_candidate_scores" in embedding_only.produces
    assert {"has_sequence", "has_backbone_structure", "has_full_structure", "has_candidate_scores"} <= set(
        with_structure.produces
    )


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


def test_pairwise_structure_tools_require_two_candidate_items_when_wired_from_nodes() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(
            _ir(
                [
                    _node("a", "rfdiffusion", {"mode": "unconditional"}),
                    _node("align", "tmalign", {}, [_candidate_source("a")]),
                ]
            )
        )

    assert "at_least_2_candidate_items" in exc_info.value.payload["missing_capabilities"]

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
