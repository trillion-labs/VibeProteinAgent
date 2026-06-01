from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.structure_analysis.adapter import (
    StructureAnalysisAdapter,
    StructureAnalysisInput,
)
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _pdb(tmp_path):
    path = tmp_path / "mini.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  CYS A   2       4.000   0.000   0.000  1.00 20.00           C\n"
    )
    return path


def _pdb_with_residue_count(tmp_path, name: str, count: int):
    path = tmp_path / name
    path.write_text(
        "".join(
            f"ATOM  {index:5d}  CA  ALA A{index:4d}"
            f"    {float(index * 4):8.3f}{0.0:8.3f}{0.0:8.3f}"
            "  1.00 20.00           C\n"
            for index in range(1, count + 1)
        )
    )
    return path


def _candidate(path: str, candidate_id: str = "cand_struct", index: int = 0) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": f"struct_{index:04d}",
            "native_id": None,
            "index": index,
        },
        "structure_path": path,
        "scores": {"upstream": {"score": 1.0}},
        "artifacts": {},
    }


def test_structure_analysis_emits_canonical_candidates(tmp_path) -> None:
    pdb = _pdb(tmp_path)
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        StructureAnalysisInput(structure_path=str(pdb), run_dssp=False),
        null_context(tmp_path / "job"),
    )
    assert len(out.candidates) == 1
    validate_candidate_v1(out.candidates[0])
    assert out.candidates[0]["source"]["tool"] == "structure_analysis"
    assert "structure_analysis" in out.candidates[0]["scores"]
    assert out.structure_path == str(pdb)


def test_structure_analysis_candidate_ids_are_deterministic(tmp_path) -> None:
    pdb = _pdb(tmp_path)
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    first = adapter.run(
        StructureAnalysisInput(structure_path=str(pdb), run_dssp=False),
        null_context(tmp_path / "a"),
    )
    second = adapter.run(
        StructureAnalysisInput(structure_path=str(pdb), run_dssp=False),
        null_context(tmp_path / "b"),
    )
    assert first.candidates[0]["candidate_id"] == second.candidates[0]["candidate_id"]


def test_structure_analysis_consumes_canonical_legacy_and_rejects_mixed(tmp_path) -> None:
    pdb = str(_pdb(tmp_path))
    assert StructureAnalysisInput(candidates=[_candidate(pdb)]).structure_path == pdb
    assert StructureAnalysisInput(structure_path=pdb).structure_path == pdb
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        StructureAnalysisInput(candidates=[_candidate(pdb), {"structure_path": pdb}])


def test_structure_analysis_scores_each_candidate_structure(tmp_path) -> None:
    first_pdb = _pdb_with_residue_count(tmp_path, "first.pdb", 2)
    second_pdb = _pdb_with_residue_count(tmp_path, "second.pdb", 4)
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        StructureAnalysisInput(
            candidates=[
                _candidate(str(first_pdb), "cand_first", 0),
                _candidate(str(second_pdb), "cand_second", 1),
            ],
            run_dssp=False,
        ),
        null_context(tmp_path / "job"),
    )

    assert len(out.candidates) == 2
    first_score = out.candidates[0]["scores"]["structure_analysis"]
    second_score = out.candidates[1]["scores"]["structure_analysis"]
    assert sum(first_score["secondary_structure"][key] for key in ("helix", "sheet", "loop")) == 2
    assert sum(second_score["secondary_structure"][key] for key in ("helix", "sheet", "loop")) == 4
    assert first_score != second_score
    assert (
        out.candidates[0]["artifacts"]["structure_analysis"]["summary_json"]
        != out.candidates[1]["artifacts"]["structure_analysis"]["summary_json"]
    )
