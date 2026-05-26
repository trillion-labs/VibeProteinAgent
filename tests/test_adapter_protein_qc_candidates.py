from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.protein_qc.adapter import ProteinQCAdapter, ProteinQCInput
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate(sequence: str = "MKKK") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_qc",
        "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "seq_0000", "native_id": None, "index": 0},
        "sequence": sequence,
        "scores": {"upstream": {"score": 1.0}},
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


def test_protein_qc_rejects_mixed_candidates() -> None:
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ProteinQCInput(candidates=[_candidate(), {"sequence": "AAAA"}])
