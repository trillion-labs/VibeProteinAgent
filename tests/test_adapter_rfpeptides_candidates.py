from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.rfpeptides.adapter import RFPeptidesDesign, RFPeptidesInput, RFPeptidesOutput, _rfpeptides_candidates
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_target", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/target.pdb", "scores": {}, "artifacts": {}}


def test_rfpeptides_emits_canonical_candidates() -> None:
    params = RFPeptidesInput(target_pdb="/tmp/target.pdb")
    design = RFPeptidesDesign(pdb_path="/tmp/design.pdb", trb_path="/tmp/design.trb", length=10)
    candidate = _rfpeptides_candidates(params, [design])[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "rfpeptides"
    assert candidate["source"]["local_id"] == "design_0000"
    assert candidate["scores"]["rfpeptides"]["length"] == 10


def test_rfpeptides_candidate_ids_are_deterministic() -> None:
    params = RFPeptidesInput(target_pdb="/tmp/target.pdb")
    design = RFPeptidesDesign(pdb_path="/tmp/design.pdb", length=10)
    assert _rfpeptides_candidates(params, [design])[0]["candidate_id"] == _rfpeptides_candidates(params, [design])[0]["candidate_id"]


def test_rfpeptides_output_preserves_legacy_fields() -> None:
    out = RFPeptidesOutput(designs=[RFPeptidesDesign(pdb_path="/tmp/d.pdb", length=10)], runtime_ms=1, num_designs_returned=1)
    assert out.model_dump()["designs"][0]["length"] == 10


def test_rfpeptides_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert RFPeptidesInput(candidates=[_candidate()]).target_pdb == "/tmp/target.pdb"
    assert RFPeptidesInput(peptide_length=12).target_pdb is None
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        RFPeptidesInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}])
