from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.rfdiffusion3.adapter import RFdiffusion3Design, RFdiffusion3Input, RFdiffusion3Output, _rfdiffusion3_candidates
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_target", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/target.pdb", "scores": {}, "artifacts": {}}


def test_rfdiffusion3_emits_canonical_candidates() -> None:
    params = RFdiffusion3Input(target_pdb="/tmp/target.pdb")
    design = RFdiffusion3Design(pdb_path="/tmp/design.pdb", sequence="ACDE", length=4, score=0.2)
    candidate = _rfdiffusion3_candidates(params, [design])[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "rfdiffusion3"
    assert candidate["source"]["local_id"] == "design_0000"
    assert candidate["sequence"] == "ACDE"
    assert candidate["scores"]["rfdiffusion3"]["score"] == 0.2


def test_rfdiffusion3_candidate_ids_are_deterministic() -> None:
    params = RFdiffusion3Input(target_pdb="/tmp/target.pdb")
    design = RFdiffusion3Design(pdb_path="/tmp/design.pdb", length=4)
    assert _rfdiffusion3_candidates(params, [design])[0]["candidate_id"] == _rfdiffusion3_candidates(params, [design])[0]["candidate_id"]


def test_rfdiffusion3_output_preserves_legacy_fields() -> None:
    out = RFdiffusion3Output(designs=[RFdiffusion3Design(pdb_path="/tmp/d.pdb", length=4)], runtime_ms=1, num_designs_returned=1)
    assert out.model_dump()["designs"][0]["pdb_path"] == "/tmp/d.pdb"


def test_rfdiffusion3_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert RFdiffusion3Input(candidates=[_candidate()]).target_pdb == "/tmp/target.pdb"
    assert RFdiffusion3Input(target_pdb="/tmp/target.pdb").target_pdb == "/tmp/target.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        RFdiffusion3Input(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}])
