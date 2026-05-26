from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.structure_analysis.adapter import StructureAnalysisAdapter, StructureAnalysisInput
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _pdb(tmp_path):
    path = tmp_path / "mini.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  CYS A   2       4.000   0.000   0.000  1.00 20.00           C\n"
    )
    return path


def _candidate(path: str) -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_struct", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": path, "scores": {"upstream": {"score": 1.0}}, "artifacts": {}}


def test_structure_analysis_emits_canonical_candidates(tmp_path) -> None:
    pdb = _pdb(tmp_path)
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(StructureAnalysisInput(structure_path=str(pdb), run_dssp=False), null_context(tmp_path / "job"))
    assert len(out.candidates) == 1
    validate_candidate_v1(out.candidates[0])
    assert out.candidates[0]["source"]["tool"] == "structure_analysis"
    assert "structure_analysis" in out.candidates[0]["scores"]
    assert out.structure_path == str(pdb)


def test_structure_analysis_candidate_ids_are_deterministic(tmp_path) -> None:
    pdb = _pdb(tmp_path)
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    first = adapter.run(StructureAnalysisInput(structure_path=str(pdb), run_dssp=False), null_context(tmp_path / "a"))
    second = adapter.run(StructureAnalysisInput(structure_path=str(pdb), run_dssp=False), null_context(tmp_path / "b"))
    assert first.candidates[0]["candidate_id"] == second.candidates[0]["candidate_id"]


def test_structure_analysis_consumes_canonical_legacy_and_rejects_mixed(tmp_path) -> None:
    pdb = str(_pdb(tmp_path))
    assert StructureAnalysisInput(candidates=[_candidate(pdb)]).structure_path == pdb
    assert StructureAnalysisInput(structure_path=pdb).structure_path == pdb
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        StructureAnalysisInput(candidates=[_candidate(pdb), {"structure_path": pdb}])
