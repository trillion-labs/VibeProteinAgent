from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.rfdiffusion.adapter import (
    RFdiffusionDesign,
    RFdiffusionInput,
    RFdiffusionOutput,
    _canonical_candidate_from_design,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(structure_path: str = "/tmp/backbone.pdb") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_backbone",
        "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "design_0000", "native_id": None, "index": 0},
        "structure_path": structure_path,
        "scores": {"upstream": {"score": 1.0}},
        "artifacts": {},
    }


def test_rfdiffusion_designs_emit_canonical_candidates() -> None:
    params = RFdiffusionInput()
    design = {"pdb_path": "/tmp/design_0.pdb", "trb_path": "/tmp/design_0.trb", "sequence": None, "length": 100, "score": 0.7}
    candidate = _canonical_candidate_from_design(design, 0, params=params)
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "rfdiffusion"
    assert candidate["source"]["local_id"] == "design_0000"
    assert candidate["structure_path"] == "/tmp/design_0.pdb"
    assert candidate["scores"]["rfdiffusion"]["length"] == 100


def test_rfdiffusion_candidate_ids_are_deterministic() -> None:
    params = RFdiffusionInput()
    design = {"pdb_path": "/tmp/design_0.pdb", "length": 100}
    assert _canonical_candidate_from_design(design, 0, params=params)["candidate_id"] == _canonical_candidate_from_design(design, 0, params=params)["candidate_id"]


def test_rfdiffusion_output_preserves_legacy_fields() -> None:
    out = RFdiffusionOutput(mode="unconditional", contigs=["100-100"], designs=[RFdiffusionDesign(pdb_path="/tmp/d.pdb", length=100)], runtime_ms=1, num_designs_returned=1)
    dumped = out.model_dump()
    assert dumped["designs"][0]["pdb_path"] == "/tmp/d.pdb"
    assert dumped["top_candidates"] == []


def test_rfdiffusion_consumes_canonical_and_rejects_mixed() -> None:
    params = RFdiffusionInput(candidates=[_candidate()])
    assert params.target_pdb == "/tmp/backbone.pdb"
    assert params.mode == "partial_diffusion"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        RFdiffusionInput(candidates=[_candidate(), {"structure_path": "/tmp/legacy.pdb"}])
