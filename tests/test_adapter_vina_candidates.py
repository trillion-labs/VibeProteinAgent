from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.vina.adapter import VinaInput, VinaMode, VinaOutput, _vina_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate(path: str, index: int) -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": f"cand_{index}", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": f"struct_{index:04d}", "native_id": None, "index": index}, "structure_path": path, "scores": {}, "artifacts": {}}


def test_vina_emits_canonical_candidates() -> None:
    params = VinaInput(receptor_pdbqt="/tmp/r.pdbqt", ligand_pdbqt="/tmp/l.pdbqt", center=[0, 0, 0], box_size=[1, 1, 1])
    candidate = _vina_candidates(params, [VinaMode(rank=1, affinity_kcal_mol=-8.0, rmsd_lb=0, rmsd_ub=0)], "/tmp/docked.pdbqt")[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "vina"
    assert candidate["source"]["local_id"] == "mode_0000"
    assert candidate["scores"]["vina"]["affinity_kcal_mol"] == -8.0


def test_vina_candidate_ids_are_deterministic() -> None:
    params = VinaInput(receptor_pdbqt="/tmp/r.pdbqt", ligand_pdbqt="/tmp/l.pdbqt", center=[0, 0, 0], box_size=[1, 1, 1])
    mode = VinaMode(rank=1, affinity_kcal_mol=-8.0, rmsd_lb=0, rmsd_ub=0)
    assert _vina_candidates(params, [mode], "/tmp/docked.pdbqt")[0]["candidate_id"] == _vina_candidates(params, [mode], "/tmp/docked.pdbqt")[0]["candidate_id"]


def test_vina_output_preserves_legacy_fields() -> None:
    out = VinaOutput(docked_pdbqt="/tmp/d.pdbqt", best_affinity_kcal_mol=-8, modes=[VinaMode(rank=1, affinity_kcal_mol=-8, rmsd_lb=0, rmsd_ub=0)], runtime_ms=1)
    assert out.model_dump()["best_affinity_kcal_mol"] == -8


def test_vina_consumes_canonical_legacy_and_rejects_mixed() -> None:
    params = VinaInput(candidates=[_candidate("/tmp/r.pdbqt", 0), _candidate("/tmp/l.pdbqt", 1)], center=[0, 0, 0], box_size=[1, 1, 1])
    assert params.receptor_pdbqt == "/tmp/r.pdbqt"
    assert VinaInput(receptor_pdbqt="/tmp/r.pdbqt", ligand_pdbqt="/tmp/l.pdbqt", center=[0, 0, 0], box_size=[1, 1, 1]).ligand_pdbqt == "/tmp/l.pdbqt"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        VinaInput(candidates=[_candidate("/tmp/r.pdbqt", 0), {"ligand_pdbqt": "/tmp/l.pdbqt"}], center=[0, 0, 0], box_size=[1, 1, 1])
