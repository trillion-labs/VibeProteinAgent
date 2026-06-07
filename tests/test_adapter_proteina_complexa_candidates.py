from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.proteina_complexa.adapter import ProteinaComplexaDesign, ProteinaComplexaInput, ProteinaComplexaOutput, _proteina_complexa_candidates
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_struct", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/a.pdb", "sequence": "ACDE", "scores": {}, "artifacts": {}}


def test_proteina_complexa_emits_canonical_candidates() -> None:
    params = ProteinaComplexaInput(mode="protein_binder", target_pdb="/tmp/a.pdb")
    design = ProteinaComplexaDesign(pdb_path="/tmp/d.pdb", sequence="ACDE", length=4, score=0.8, iptm=0.6)
    candidate = _proteina_complexa_candidates(params, [design])[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "proteina_complexa"
    assert candidate["source"]["local_id"] == "design_0000"
    assert candidate["scores"]["proteina_complexa"]["mode"] == "protein_binder"


def test_proteina_complexa_candidate_ids_are_deterministic() -> None:
    params = ProteinaComplexaInput(mode="protein_binder", target_pdb="/tmp/a.pdb")
    design = ProteinaComplexaDesign(pdb_path="/tmp/d.pdb", length=4)
    assert _proteina_complexa_candidates(params, [design])[0]["candidate_id"] == _proteina_complexa_candidates(params, [design])[0]["candidate_id"]


def test_proteina_complexa_output_preserves_legacy_fields() -> None:
    out = ProteinaComplexaOutput(mode="protein_binder", designs=[ProteinaComplexaDesign(pdb_path="/tmp/d.pdb", length=4)], runtime_ms=1, num_designs_returned=1, seeds_used=[])
    assert out.model_dump()["mode"] == "protein_binder"


def test_proteina_complexa_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert ProteinaComplexaInput(mode="protein_binder", candidates=[_candidate()]).target_pdb == "/tmp/a.pdb"
    assert ProteinaComplexaInput(mode="motif", candidates=[_candidate()], motif_residues=[1], total_length=60).motif_pdb == "/tmp/a.pdb"
    assert ProteinaComplexaInput(mode="protein_binder", target_pdb="/tmp/a.pdb").target_pdb == "/tmp/a.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ProteinaComplexaInput(mode="protein_binder", candidates=[_candidate(), {"target_pdb": "/tmp/b.pdb"}])
