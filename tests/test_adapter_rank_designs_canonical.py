from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.rank_designs.adapter import RankDesignsAdapter, RankDesignsInput


def _canonical_candidate(
    candidate_id: str,
    *,
    local_id: str,
    sequence: str,
    score: float,
) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "go_sequences",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": None,
            "index": int(local_id.rsplit("_", 1)[1]),
        },
        "sequence": sequence,
        "scores": {"go_sequences": {"score": score, "plddt": 90.0}},
        "artifacts": {},
    }


def test_rank_designs_accepts_all_canonical_candidates(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            candidates=[
                _canonical_candidate("cand_weak", local_id="seq_0000", sequence="AAAA", score=0.1),
                _canonical_candidate(
                    "cand_strong",
                    local_id="seq_0001",
                    sequence="CCCC",
                    score=0.9,
                ),
            ],
            top_k=1,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.ranked_candidates[0]["candidate_id"] == "cand_strong"
    assert out.ranked_candidates[0]["sequence"] == "CCCC"
    assert out.ranked_candidates[0]["candidate_score"] == 0.9


def test_rank_designs_legacy_candidate_path_still_works(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            candidates=[
                {"candidate_id": "weak", "sequence": "AAAA", "score": 0.1},
                {"candidate_id": "strong", "sequence": "CCCC", "score": 0.9},
            ],
            top_k=1,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.ranked_candidates[0]["candidate_id"] == "strong"
    assert out.ranked_candidates[0]["sequence"] == "CCCC"


def test_rank_designs_rejects_mixed_canonical_and_legacy_candidates() -> None:
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        RankDesignsInput(
            candidates=[
                _canonical_candidate("cand_a", local_id="seq_0000", sequence="AAAA", score=0.1),
                {"candidate_id": "legacy", "sequence": "CCCC", "score": 0.9},
            ]
        )


def test_rank_designs_canonical_detection_only_applies_to_candidates_slot(tmp_path) -> None:
    # PR #3 intentionally routes canonical Candidate only in the candidates slot.
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            predictions=[
                _canonical_candidate(
                    "cand_prediction",
                    local_id="seq_0000",
                    sequence="AAAA",
                    score=0.9,
                )
            ],
            top_k=1,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.ranked_candidates[0]["candidate_id"] == "cand_prediction"
    assert out.ranked_candidates[0]["sequence"] == "AAAA"
