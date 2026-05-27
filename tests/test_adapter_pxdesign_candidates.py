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
    assert candidate["scores"]["pxdesign"]["pipeline"] == "diffusion_af2_initial_guess_proteinmpnn"


def test_pxdesign_candidate_ids_are_deterministic() -> None:
    params = PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60)
    design = PXDesignDesign(cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4)
    assert _pxdesign_candidates(params, [design])[0]["candidate_id"] == _pxdesign_candidates(params, [design])[0]["candidate_id"]


def test_pxdesign_output_preserves_legacy_fields() -> None:
    out = PXDesignOutput(designs=[PXDesignDesign(cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4)], runtime_ms=1, num_samples=1)
    assert out.model_dump()["num_samples"] == 1


def test_pxdesign_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert PXDesignInput(candidates=[_candidate()], target_chain="A", binder_length=60).target_pdb == "/tmp/target.pdb"
    assert PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60).target_pdb == "/tmp/target.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        PXDesignInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}], target_chain="A", binder_length=60)
