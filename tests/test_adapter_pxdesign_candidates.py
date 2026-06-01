from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.pxdesign.adapter import PXDesignDesign, PXDesignInput, PXDesignOutput, _pxdesign_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_target", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/target.pdb", "scores": {}, "artifacts": {}}


def _ir(inputs: dict | None = None) -> dict:
    return {"version": "planner_ir_v1", "goal": "x", "assumptions": [], "missing_inputs": [], "nodes": [{"id": "n", "tool": "pxdesign", "intent": "x", "inputs": inputs or {}, "upstream": []}]}


def test_pxdesign_emits_canonical_candidates() -> None:
    params = PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60)
    design = PXDesignDesign(candidate_id="legacy", cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4, score=0.7)
    candidate = _pxdesign_candidates(params, [design])[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "pxdesign"
    assert candidate["source"]["native_id"] == "legacy"
    assert candidate["structure_path"] == "/tmp/d.cif"
    assert candidate["backbone_path"] == "/tmp/d.cif"
    assert "full_structure_path" not in candidate
    assert candidate["capabilities"]["has_sequence"] is True
    assert candidate["capabilities"]["has_backbone_structure"] is True
    assert candidate["capabilities"]["has_full_structure"] is False
    assert candidate["scores"]["pxdesign"]["pipeline"] == "diffusion_af2_initial_guess_proteinmpnn"
    assert (
        candidate["scores"]["pxdesign"]["structure_detail"]
        == "binder_backbone_with_target_context"
    )


def test_pxdesign_candidate_ids_are_deterministic() -> None:
    params = PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60)
    design = PXDesignDesign(cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4)
    assert _pxdesign_candidates(params, [design])[0]["candidate_id"] == _pxdesign_candidates(params, [design])[0]["candidate_id"]


def test_pxdesign_output_preserves_legacy_fields() -> None:
    out = PXDesignOutput(designs=[PXDesignDesign(cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4)], runtime_ms=1, num_samples=1)
    assert out.model_dump()["num_samples"] == 1


def test_pxdesign_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert PXDesignInput(candidates=[_candidate()], target_chain="A", binder_length=60).target_pdb == "/tmp/target.pdb"
    full = {**_candidate(), "structure_path": None, "full_structure_path": "/tmp/full.cif"}
    full.pop("structure_path")
    assert PXDesignInput(candidates=[full], target_chain="A", binder_length=60).target_pdb == "/tmp/full.cif"
    backbone = {**_candidate(), "structure_path": None, "backbone_path": "/tmp/backbone.pdb"}
    backbone.pop("structure_path")
    assert PXDesignInput(candidates=[backbone], target_chain="A", binder_length=60).target_pdb == "/tmp/backbone.pdb"
    assert PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60).target_pdb == "/tmp/target.pdb"
    no_structure = {**_candidate()}
    no_structure.pop("structure_path")
    with pytest.raises(ValidationError, match="structure_path, full_structure_path, or backbone_path"):
        PXDesignInput(candidates=[no_structure], target_chain="A", binder_length=60)
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        PXDesignInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}], target_chain="A", binder_length=60)
