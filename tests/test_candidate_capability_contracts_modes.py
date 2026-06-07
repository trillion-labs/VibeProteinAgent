from __future__ import annotations

from vibeproteinagent.candidates.capabilities import (
    annotate_candidate_capabilities,
    candidate_capabilities_from_record,
    candidate_capability_flags,
    candidate_contract_for_node,
    candidate_contract_for_inputs,
)


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
