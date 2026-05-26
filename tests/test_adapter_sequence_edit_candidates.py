from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.sequence_edit.adapter import SequenceEditInput, SequenceEditOutput, _sequence_edit_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup


def _candidate(sequence: str = "ACDE") -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": f"cand_{sequence}", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "seq_0000", "native_id": None, "index": 0}, "sequence": sequence, "scores": {}, "artifacts": {}}


def _ir(inputs: dict | None = None) -> dict:
    return {"version": "planner_ir_v1", "goal": "x", "assumptions": [], "missing_inputs": [], "nodes": [{"id": "n", "tool": "sequence_edit", "intent": "x", "inputs": inputs or {}, "upstream": []}]}


def test_sequence_edit_emits_canonical_candidates() -> None:
    params = SequenceEditInput(operation="mutate_residues", sequence="ACDE", mutation_notation=["A1C"])
    candidate = _sequence_edit_candidates(params, "CCDE", "/tmp/m.fa", {"mismatch_count": 1})[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "sequence_edit"
    assert candidate["sequence"] == "CCDE"
    assert candidate["scores"]["sequence_edit"]["mismatch_count"] == 1


def test_sequence_edit_candidate_ids_are_deterministic() -> None:
    params = SequenceEditInput(operation="mutate_residues", sequence="ACDE", mutation_notation=["A1C"])
    assert _sequence_edit_candidates(params, "CCDE", None, {})[0]["candidate_id"] == _sequence_edit_candidates(params, "CCDE", None, {})[0]["candidate_id"]


def test_sequence_edit_output_preserves_legacy_fields() -> None:
    out = SequenceEditOutput(operation="mutate_residues", wild_type_length=4, mutant_length=4, mutations=["A1C"], changed_positions=[1], mutant_sequence="CCDE", mismatch_count=1, runtime_ms=1)
    assert out.model_dump()["mutations"] == ["A1C"]


def test_sequence_edit_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert SequenceEditInput(operation="mutate_residues", candidates=[_candidate()], mutation_notation=["A1C"]).sequence == "ACDE"
    assert SequenceEditInput(operation="mutate_residues", sequence="ACDE", mutation_notation=["A1C"]).sequence == "ACDE"
    assert SequenceEditInput(operation="check_wt_consistency", candidates=[_candidate("ACDE"), _candidate("CCDE")]).mutant_sequence == "CCDE"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        SequenceEditInput(operation="mutate_residues", candidates=[_candidate(), {"sequence": "AAAA"}], mutation_notation=["A1C"])


def test_sequence_edit_explicit_mode_enforcement() -> None:
    with pytest.raises(StructuredFixup):
        compile_planner_ir(_ir())
    assert compile_planner_ir(_ir({"mode": "mutate_residues"}))["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "mutate_residues"
    with pytest.raises(StructuredFixup):
        compile_planner_ir(_ir({"mode": "bad"}))
