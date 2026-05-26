from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.dssp.adapter import DSSPInput, DSSPOutput, _dssp_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_dssp", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/a.pdb", "scores": {}, "artifacts": {}}


def test_dssp_emits_canonical_candidates() -> None:
    params = DSSPInput(pdb_path="/tmp/a.pdb")
    candidates = _dssp_candidates(params, "/tmp/a.pdb", {"helix_fraction": 0.5, "sheet_fraction": 0.25, "loop_fraction": 0.25}, "HH--", "/tmp/a.dssp")
    candidate = candidates[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "dssp"
    assert candidate["source"]["local_id"] == "dssp_0000"
    assert candidate["scores"]["dssp"]["helix_fraction"] == 0.5


def test_dssp_candidate_ids_are_deterministic() -> None:
    params = DSSPInput(pdb_path="/tmp/a.pdb")
    first = _dssp_candidates(params, "/tmp/a.pdb", {}, "", "/tmp/a.dssp")[0]
    second = _dssp_candidates(params, "/tmp/a.pdb", {}, "", "/tmp/a.dssp")[0]
    assert first["candidate_id"] == second["candidate_id"]


def test_dssp_output_preserves_legacy_fields() -> None:
    out = DSSPOutput(dssp_output_path="/tmp/a.dssp", secondary_structure_string="HH", helix_fraction=1.0, sheet_fraction=0.0, coil_fraction=0.0, num_residues=2, runtime_ms=1)
    assert out.model_dump()["secondary_structure_string"] == "HH"


def test_dssp_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert DSSPInput(candidates=[_candidate()]).pdb_path == "/tmp/a.pdb"
    assert DSSPInput(pdb_path="/tmp/a.pdb").pdb_path == "/tmp/a.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        DSSPInput(candidates=[_candidate(), {"pdb_path": "/tmp/legacy.pdb"}])
