from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.boltzgen.adapter import BoltzGenDesign, BoltzGenInput, BoltzGenOutput, _boltzgen_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_target", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/target.pdb", "scores": {}, "artifacts": {}}


def test_boltzgen_emits_canonical_candidates() -> None:
    params = BoltzGenInput(target_pdb="/tmp/target.pdb")
    design = BoltzGenDesign(candidate_id="legacy", cif_path="/tmp/d.cif", sequence="ACDE", length=4, score=0.4)
    candidate = _boltzgen_candidates(params, [design])[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "boltzgen"
    assert candidate["source"]["native_id"] == "legacy"
    assert candidate["scores"]["boltzgen"]["score"] == 0.4


def test_boltzgen_candidate_ids_are_deterministic() -> None:
    params = BoltzGenInput(target_pdb="/tmp/target.pdb")
    design = BoltzGenDesign(cif_path="/tmp/d.cif", length=4)
    assert _boltzgen_candidates(params, [design])[0]["candidate_id"] == _boltzgen_candidates(params, [design])[0]["candidate_id"]


def test_boltzgen_output_preserves_legacy_fields() -> None:
    out = BoltzGenOutput(designs=[BoltzGenDesign(cif_path="/tmp/d.cif", length=4)], runtime_ms=1, num_designs_returned=1)
    assert out.model_dump()["num_designs_returned"] == 1


def test_boltzgen_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert BoltzGenInput(candidates=[_candidate()]).target_pdb == "/tmp/target.pdb"
    assert BoltzGenInput(target_pdb="/tmp/target.pdb").target_pdb == "/tmp/target.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        BoltzGenInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}])
