from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.pyrosetta.adapter import PyRosettaInput, PyRosettaOutput, _pyrosetta_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_struct", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/a.pdb", "scores": {}, "artifacts": {}}


def _ir(inputs: dict | None = None) -> dict:
    return {"version": "planner_ir_v1", "goal": "x", "assumptions": [], "missing_inputs": [], "nodes": [{"id": "n", "tool": "pyrosetta", "intent": "x", "inputs": inputs or {}, "upstream": []}]}


def test_pyrosetta_emits_canonical_candidates() -> None:
    params = PyRosettaInput(pdb_path="/tmp/a.pdb")
    candidate = _pyrosetta_candidates(params, -12.5, None, "/tmp/a.pdb")[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "pyrosetta"
    assert candidate["source"]["local_id"] == "score_0000"
    assert candidate["scores"]["pyrosetta"]["total_score"] == -12.5


def test_pyrosetta_candidate_ids_are_deterministic() -> None:
    params = PyRosettaInput(pdb_path="/tmp/a.pdb")
    assert _pyrosetta_candidates(params, -1.0, None, "/tmp/a.pdb")[0]["candidate_id"] == _pyrosetta_candidates(params, -1.0, None, "/tmp/a.pdb")[0]["candidate_id"]


def test_pyrosetta_output_preserves_legacy_fields() -> None:
    assert PyRosettaOutput(mode="score", total_score=-1.0, runtime_ms=1).model_dump()["total_score"] == -1.0


def test_pyrosetta_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert PyRosettaInput(candidates=[_candidate()]).pdb_path == "/tmp/a.pdb"
    assert PyRosettaInput(pdb_path="/tmp/a.pdb").pdb_path == "/tmp/a.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        PyRosettaInput(candidates=[_candidate(), {"pdb_path": "/tmp/b.pdb"}])


def test_pyrosetta_explicit_mode_enforcement() -> None:
    with pytest.raises(StructuredFixup):
        compile_planner_ir(_ir())
    assert compile_planner_ir(_ir({"mode": "score"}))["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "score"
    with pytest.raises(StructuredFixup):
        compile_planner_ir(_ir({"mode": "bad"}))
