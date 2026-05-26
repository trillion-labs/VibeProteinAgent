from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.bindcraft.adapter import BindCraftDesign, BindCraftInput, BindCraftOutput, _canonical_candidate_from_design
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_target", "source": {"tool": "target", "node_id": None, "job_id": None, "local_id": "target_0000", "native_id": None, "index": 0}, "sequence": "AAAA", "structure_path": "/tmp/target.pdb", "scores": {}, "artifacts": {}}


def test_bindcraft_designs_emit_canonical_candidates() -> None:
    candidate = _canonical_candidate_from_design({"candidate_id": "legacy", "pdb_path": "/tmp/binder.pdb", "sequence": "MKKL", "binder_length": 4, "iptm": 0.8}, 0)
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "bindcraft"
    assert candidate["source"]["local_id"] == "binder_0000"
    assert candidate["sequence"] == "MKKL"
    assert candidate["scores"]["bindcraft"]["iptm"] == 0.8


def test_bindcraft_candidate_ids_are_deterministic() -> None:
    row = {"pdb_path": "/tmp/binder.pdb", "sequence": "MKKL", "binder_length": 4}
    assert _canonical_candidate_from_design(row, 0)["candidate_id"] == _canonical_candidate_from_design(row, 0)["candidate_id"]


def test_bindcraft_output_preserves_legacy_fields() -> None:
    out = BindCraftOutput(designs=[BindCraftDesign(pdb_path="/tmp/b.pdb", sequence="MKKL", binder_length=4)], runtime_ms=1, num_designs_returned=1, seeds_used=[])
    assert out.model_dump()["designs"][0]["pdb_path"] == "/tmp/b.pdb"


def test_bindcraft_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert BindCraftInput(candidates=[_candidate()]).target_pdb == "/tmp/target.pdb"
    assert BindCraftInput(target_pdb="/tmp/target.pdb").target_pdb == "/tmp/target.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        BindCraftInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}])
