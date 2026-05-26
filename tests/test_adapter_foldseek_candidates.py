from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.foldseek.adapter import FoldseekInput, FoldseekOutput, _foldseek_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_struct", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/a.pdb", "scores": {}, "artifacts": {}}


def _ir(inputs: dict | None = None) -> dict:
    return {"version": "planner_ir_v1", "goal": "x", "assumptions": [], "missing_inputs": [], "nodes": [{"id": "n", "tool": "foldseek", "intent": "x", "inputs": inputs or {}, "upstream": []}]}


def test_foldseek_emits_canonical_candidates(tmp_path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text("q1\tt1\t0.9\t100\t1e-10\t55\n")
    candidates = _foldseek_candidates(FoldseekInput(query_pdb="/tmp/q.pdb", target_db="/tmp/db"), hits)
    assert len(candidates) == 1
    validate_candidate_v1(candidates[0])
    assert candidates[0]["source"]["tool"] == "foldseek"
    assert candidates[0]["source"]["local_id"] == "hit_0000"
    assert candidates[0]["scores"]["foldseek"]["evalue"] == 1e-10


def test_foldseek_candidate_ids_are_deterministic(tmp_path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text("q1\tt1\t0.9\t100\t1e-10\t55\n")
    params = FoldseekInput(query_pdb="/tmp/q.pdb", target_db="/tmp/db")
    assert _foldseek_candidates(params, hits)[0]["candidate_id"] == _foldseek_candidates(params, hits)[0]["candidate_id"]


def test_foldseek_output_preserves_legacy_fields() -> None:
    assert FoldseekOutput(hits_tsv="/tmp/h.tsv", num_hits=0, runtime_ms=1).model_dump()["num_hits"] == 0


def test_foldseek_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert FoldseekInput(candidates=[_candidate()], target_db="/tmp/db").query_pdb == "/tmp/a.pdb"
    assert FoldseekInput(query_pdb="/tmp/a.pdb", target_db="/tmp/db").query_pdb == "/tmp/a.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        FoldseekInput(candidates=[_candidate(), {"query_pdb": "/tmp/b.pdb"}], target_db="/tmp/db")


def test_foldseek_explicit_mode_enforcement() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(_ir())
    assert exc_info.value.payload["error_kind"] == "missing_explicit_mode"
    assert exc_info.value.payload["allowed_modes"] == ["easy-search", "easy-cluster"]
    assert compile_planner_ir(_ir({"mode": "easy-search"}))["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "easy-search"
    with pytest.raises(StructuredFixup):
        compile_planner_ir(_ir({"mode": "bad"}))
