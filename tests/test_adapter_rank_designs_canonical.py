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


def test_rank_designs_preserves_canonical_score_namespaces(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = _canonical_candidate(
        "cand_qc",
        local_id="seq_0000",
        sequence="AAAA",
        score=0.5,
    )
    candidate["source"]["tool"] = "protenix_v2"
    candidate["scores"] = {
        "protenix_v2": {"ranking_score": 0.8, "plddt": 91.0},
        "protein_qc": {"flags": ["high_instability_index"], "plddt": 88.0},
        "sequence_analysis": {"hydrophobic_fraction": 0.2},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["ranking_score"] == 0.8
    assert row["qc_flags"] == ["high_instability_index"]
    assert row["qc_flag_count"] == 1
    assert row["scores"]["protein_qc"]["flags"] == ["high_instability_index"]
    assert row["scores"]["sequence_analysis"]["hydrophobic_fraction"] == 0.2


def test_rank_designs_normalizes_esm3_plddt_and_reads_ptm_namespace(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_esm3",
        "source": {
            "tool": "esm3",
            "node_id": None,
            "job_id": None,
            "local_id": "esm3_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": "ACDE",
        "scores": {
            "esm3": {"mean_plddt": 0.981, "ptm": 0.864},
            "protein_qc": {"flags": [], "plddt": 98.1},
        },
        "artifacts": {},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["plddt"] == pytest.approx(98.1)
    assert row["ptm"] == pytest.approx(0.864)
    assert row["score"] == pytest.approx(0.981 + 0.25 * 0.864)
    assert row["score"] > 0


def test_rank_designs_applies_qc_penalty_without_double_dividing_plddt(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_esm3_flags",
        "source": {
            "tool": "esm3",
            "node_id": None,
            "job_id": None,
            "local_id": "esm3_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": "ACDE",
        "scores": {
            "esm3": {"mean_plddt": 0.981, "ptm": 0.864},
            "protein_qc": {
                "flags": [
                    "low_plddt",
                    "contains_polybasic_cluster",
                    "high_instability_index",
                ],
                "plddt": 98.1,
            },
        },
        "artifacts": {},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["qc_flag_count"] == 3
    assert row["score"] == pytest.approx(0.981 + 0.25 * 0.864 - 0.30)


def test_rank_designs_keeps_protenix_plddt_percent_scale_semantics(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = _canonical_candidate(
        "cand_protenix",
        local_id="seq_0000",
        sequence="AAAA",
        score=0.5,
    )
    candidate["source"]["tool"] = "protenix_v2"
    candidate["scores"] = {
        "protenix_v2": {
            "ranking_score": 0.8,
            "plddt": 91.0,
            "iptm": 0.6,
            "ptm": 0.7,
        },
        "protein_qc": {"flags": ["high_instability_index"], "plddt": 88.0},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["plddt"] == pytest.approx(91.0)
    assert row["score"] == pytest.approx(0.8 + 0.91 + 0.5 * 0.6 + 0.25 * 0.7 - 0.10)


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
