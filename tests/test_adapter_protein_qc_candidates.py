from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.protein_qc.adapter import ProteinQCAdapter, ProteinQCInput
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(
    sequence: str = "MKKK",
    candidate_id: str = "cand_qc",
    index: int = 0,
    *,
    structure_path: str | None = None,
    scores: dict | None = None,
) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": f"seq_{index:04d}",
            "native_id": None,
            "index": index,
        },
        "sequence": sequence,
        **({"structure_path": structure_path} if structure_path else {}),
        "scores": scores if scores is not None else {"upstream": {"score": 1.0}},
        "artifacts": {},
    }


def test_protein_qc_emits_canonical_candidates(tmp_path) -> None:
    adapter = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(ProteinQCInput(sequence="MKKK"), null_context(tmp_path / "job"))

    assert out.metrics["polybasic_hits"] == 1.0
    assert len(out.candidates) == 1
    candidate = out.candidates[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "protein_qc"
    assert candidate["source"]["local_id"] == "qc_0000"
    assert candidate["scores"]["protein_qc"]["passes_filters"] is False


def test_protein_qc_candidate_ids_are_deterministic(tmp_path) -> None:
    adapter = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    params = ProteinQCInput(sequence="MKKK")
    first = adapter.run(params, null_context(tmp_path / "a"))
    second = adapter.run(params, null_context(tmp_path / "b"))
    assert first.candidates[0]["candidate_id"] == second.candidates[0]["candidate_id"]


def test_protein_qc_consumes_canonical_and_legacy_paths(tmp_path) -> None:
    adapter = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    canonical = adapter.run(ProteinQCInput(candidates=[_candidate()]), null_context(tmp_path / "c"))
    legacy = adapter.run(ProteinQCInput(sequence="MKKK"), null_context(tmp_path / "l"))
    assert canonical.candidates[0]["candidate_id"] == "cand_qc"
    assert canonical.candidates[0]["scores"]["protein_qc"]["polybasic_hits"] == 1.0
    assert legacy.metrics == canonical.metrics


def test_protein_qc_preserves_candidate_identity_and_appends_global_plddt(
    tmp_path,
) -> None:
    pdb_path = tmp_path / "design.pdb"
    pdb_path.write_text(
        "ATOM      1  CA  ALA A   1      11.104  13.207   2.100  1.00 83.36           C\n"
        "ATOM      2  CB  ALA A   1      11.904  14.207   2.100  1.00 75.66           C\n"
        "TER\n"
        "END\n"
    )
    source = _candidate(
        "A",
        candidate_id="v40",
        structure_path=str(pdb_path),
        scores={
            "esm2": {"score": 0.4},
            "protenix_v2": {"plddt": 79.512, "ptm": 0.745},
        },
    )
    adapter = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        ProteinQCInput(candidates=[source], metrics=["plddt"]),
        null_context(tmp_path / "global_plddt"),
    )

    candidate = out.candidates[0]
    assert candidate["candidate_id"] == "v40"
    assert candidate["scores"]["esm2"]["score"] == 0.4
    assert candidate["scores"]["protenix_v2"]["plddt"] == pytest.approx(79.512)
    assert candidate["scores"]["protein_qc"]["plddt"] == pytest.approx(79.512)
    assert candidate["scores"]["protein_qc"]["ca_bfactor_plddt"] == pytest.approx(83.36)
    assert candidate["scores"]["protein_qc"]["plddt_source"] == "scores.protenix_v2.plddt"


def test_protein_qc_scores_each_candidate_sequence_independently(tmp_path) -> None:
    adapter = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        ProteinQCInput(
            candidates=[
                _candidate("AAAA", candidate_id="cand_a", index=0),
                _candidate("RRRR", candidate_id="cand_r", index=1),
            ],
            metrics=["gravy", "pi", "instability_index", "polybasic"],
        ),
        null_context(tmp_path / "per_candidate"),
    )

    first = out.candidates[0]["scores"]["protein_qc"]
    second = out.candidates[1]["scores"]["protein_qc"]
    assert first["gravy"] != second["gravy"]
    assert first["pi"] != second["pi"]
    assert first["passes_filters"] is False
    assert second["passes_filters"] is False
    assert "high_GRAVY" in first["flags"]
    assert "contains_polybasic_cluster" in second["flags"]
    assert out.metrics["gravy"] == first["gravy"]
    assert out.metrics["pi"] == first["pi"]


def test_protein_qc_rejects_mixed_candidates() -> None:
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ProteinQCInput(candidates=[_candidate(), {"sequence": "AAAA"}])
