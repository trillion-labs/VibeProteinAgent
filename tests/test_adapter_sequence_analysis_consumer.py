from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.sequence_analysis.adapter import SequenceAnalysisAdapter, SequenceAnalysisInput
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(sequence: str, index: int = 0) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": f"cand_seq_{index}",
        "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": f"seq_{index:04d}", "native_id": None, "index": index},
        "sequence": sequence,
        "scores": {"upstream": {"score": index}},
        "artifacts": {},
    }


def test_sequence_analysis_consumes_and_reemits_canonical_candidates(tmp_path) -> None:
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        SequenceAnalysisInput(candidates=[_candidate("MKKLL", 0), _candidate("ACDE", 1)]),
        null_context(tmp_path / "job"),
    )

    assert len(out.candidates) == len(out.representative_sequences) == 2
    for candidate in out.candidates:
        validate_candidate_v1(candidate)
        assert candidate["candidate_id"].startswith("cand_seq_")
        assert "sequence_analysis" in candidate["scores"]


def test_sequence_analysis_reemits_all_canonical_candidates_not_only_representatives(tmp_path) -> None:
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidates = [_candidate("ACDEFGHIK" + "A" * index, index) for index in range(7)]

    out = adapter.run(
        SequenceAnalysisInput(candidates=candidates),
        null_context(tmp_path / "job_all"),
    )

    assert len(out.representative_sequences) == 5
    assert [candidate["candidate_id"] for candidate in out.candidates] == [
        candidate["candidate_id"] for candidate in candidates
    ]
    assert all("sequence_analysis" in candidate["scores"] for candidate in out.candidates)


def test_sequence_analysis_legacy_path_unchanged(tmp_path) -> None:
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(SequenceAnalysisInput(sequence="MKKLL"), null_context(tmp_path / "job"))
    assert out.representative_sequences[0]["sequence"] == "MKKLL"
    assert out.candidates[0]["source"]["tool"] == "sequence_analysis"


def test_sequence_analysis_rejects_mixed_candidates() -> None:
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        SequenceAnalysisInput(candidates=[_candidate("MKKLL"), {"sequence": "AAAA"}])
