from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.base import torch
from vibeproteinagent.adapters.family_profile.adapter import (
    FamilyProfileAdapter,
    FamilyProfileInput,
)


def _candidate(candidate_id: str, sequence: str, *, score: float = 0.5) -> dict:
    index = int(candidate_id.rsplit("_", 1)[1])
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "search_sequence",
            "node_id": None,
            "job_id": None,
            "local_id": f"seq_{index:04d}",
            "native_id": None,
            "index": index,
        },
        "sequence": sequence,
        "scores": {
            "search_sequence": {"score": score},
            "mmseqs2_gpu": {"best_hit_bitscore": 123.0},
        },
        "artifacts": {"template_select": {"template_id": "1ABC_A"}},
    }


def _adapter(tmp_path: Path) -> FamilyProfileAdapter:
    return FamilyProfileAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))


def test_family_profile_builds_from_fasta_without_emitting_candidates(tmp_path) -> None:
    fasta = tmp_path / "family.fasta"
    fasta.write_text(">a\nACDEFG\n>b\nACDEYG\n>c\nACDFFG\n")

    out = _adapter(tmp_path).run(
        FamilyProfileInput(
            fasta_path=str(fasta),
            family_id="IPR000001",
            family_label="example family",
            min_conservation=0.67,
            output_prefix="family_ref",
        ),
        null_context(tmp_path / "job"),
    )

    assert out.mode == "build"
    assert out.candidates == []
    assert out.profile["sequence_count"] == 3
    assert out.profile["length_stats"]["median"] == 6.0
    assert out.profile["aa_frequency"]["A"] == pytest.approx(3 / 18, abs=1e-6)
    assert out.profile["aa_frequency"]["G"] == pytest.approx(3 / 18, abs=1e-6)
    assert out.profile["consensus_sequence"] == "ACDEFG"
    assert out.profile["conserved_positions"][0]["position"] == 1
    assert Path(out.artifacts["profile_json"]).is_file()
    assert Path(out.artifacts["summary_json"]).is_file()
    assert Path(out.artifacts["sequences_fasta"]).is_file()
    assert Path(out.artifacts["position_frequencies_json"]).is_file()
    assert Path(out.artifacts["motif_windows_json"]).is_file()


def test_family_profile_aligned_input_computes_column_conservation(tmp_path) -> None:
    alignment = tmp_path / "family.aln.fasta"
    alignment.write_text(">a\nACD-E\n>b\nACD-E\n>c\nACDFE\n")

    out = _adapter(tmp_path).run(
        FamilyProfileInput(
            alignment_path=str(alignment),
            min_conservation=1.0,
            output_prefix="aligned_family",
        ),
        null_context(tmp_path / "job"),
    )

    assert out.profile["aligned"] is True
    positions = out.profile["position_frequencies"]
    assert positions[0]["consensus"] == "A"
    assert positions[0]["conservation"] == 1.0
    assert positions[3]["observed"] == 1
    assert positions[3]["consensus"] == "F"


def test_family_profile_builds_from_canonical_candidates_and_summarizes_evidence(tmp_path) -> None:
    out = _adapter(tmp_path).run(
        FamilyProfileInput(
            candidates=[
                _candidate("cand_0", "AAAA"),
                _candidate("cand_1", "AAAT"),
            ],
            family_id="GO:0003824",
            output_prefix="candidate_family",
        ),
        null_context(tmp_path / "job"),
    )

    assert out.mode == "build"
    assert out.candidates == []
    evidence = out.profile["evidence_summary"]
    assert evidence["source_tools"]["search_sequence"] == 2
    assert evidence["score_namespaces"]["mmseqs2_gpu"] == 2
    assert evidence["artifact_namespaces"]["template_select"] == 2


def test_family_profile_score_preserves_and_annotates_candidates(tmp_path) -> None:
    profile = {
        "schema_version": "family_profile_v1",
        "family_id": "family-a",
        "family_label": "Family A",
        "sequence_count": 3,
        "consensus_sequence": "ACDE",
        "length_stats": {"mean": 4.0, "stdev": 0.0},
        "conserved_positions": [
            {"position": 1, "consensus": "A", "conservation": 1.0},
            {"position": 2, "consensus": "C", "conservation": 1.0},
        ],
        "motif_windows": [{"start": 1, "end": 2, "length": 2, "consensus": "AC"}],
        "position_frequencies": [],
        "evidence_summary": {"source_tools": {"search_sequence": 3}},
    }
    original = _candidate("cand_0", "ACDE", score=0.2)

    out = _adapter(tmp_path).run(
        FamilyProfileInput(
            candidates=[original],
            profile=profile,
            output_prefix="scored_family",
        ),
        null_context(tmp_path / "job"),
    )

    assert out.mode == "score"
    assert len(out.candidates) == 1
    scored = out.candidates[0]
    assert scored["candidate_id"] == original["candidate_id"]
    assert scored["source"] == original["source"]
    assert scored["scores"]["search_sequence"]["score"] == 0.2
    family_scores = scored["scores"]["family_profile"]
    assert family_scores["ranking_score"] == pytest.approx(1.0)
    assert family_scores["family_similarity_score"] == pytest.approx(1.0)
    assert scored["artifacts"]["family_profile"]["family_id"] == "family-a"
    assert scored["capabilities"]["has_candidate_scores"] is True
    summary = json.loads(Path(out.artifacts["summary_json"]).read_text())
    assert summary["mode"] == "score"
    assert summary["reference_sequence_count"] == 3
    assert summary["annotated_candidate_count"] == 1
    assert Path(out.artifacts["sequences_fasta"]).read_text() == ""


def test_family_profile_rejects_mixed_candidate_input() -> None:
    with pytest.raises(ValidationError, match="all canonical"):
        FamilyProfileInput(
            candidates=[
                _candidate("cand_0", "ACDE"),
                {"candidate_id": "legacy", "sequence": "ACDE"},
            ]
        )
