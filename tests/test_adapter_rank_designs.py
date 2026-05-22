from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.rank_designs.adapter import (
    RankDesignsAdapter,
    RankDesignsInput,
    RankDesignsOutput,
)


def test_rank_designs_combines_prediction_analysis_and_qc(tmp_path):
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            predictions=[
                {"name": "weak", "metrics": {"ranking_score": 0.3, "plddt": 70}},
                {"name": "strong", "metrics": {"ranking_score": 0.8, "plddt": 90}},
            ],
            analyses=[
                {"liabilities": [{"type": "surface", "positions": [1], "reason": "x"}]},
                {"liabilities": []},
            ],
            qc_results=[{"flags": ["low_plddt"]}, {"flags": []}],
            top_k=2,
        ),
        null_context(tmp_path),
    )

    assert out.candidate_count == 2
    assert out.ranked_candidates[0]["candidate_id"] == "strong"
    assert Path(out.ranked_json).exists()
    assert Path(out.ranked_tsv).exists()


def test_rank_designs_requires_input(tmp_path):
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="requires at least one"):
        adapter.run(RankDesignsInput(), null_context(tmp_path))


def test_rank_designs_output_round_trip():
    out = RankDesignsOutput(
        candidate_count=1,
        ranked_candidates=[{"rank": 1, "candidate_id": "a", "score": 1.0}],
        ranked_json="/tmp/r.json",
        ranked_tsv="/tmp/r.tsv",
        runtime_ms=1,
    )
    assert RankDesignsOutput.model_validate(out.model_dump()) == out
