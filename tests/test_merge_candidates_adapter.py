from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context, validate_output
from vibeproteinagent.adapters.merge_candidates.adapter import (
    MergeCandidatesAdapter,
    MergeCandidatesInput,
)


def _candidate(
    candidate_id: str,
    *,
    sequence: str,
    index: int,
) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "go_sequences",
            "node_id": None,
            "job_id": None,
            "local_id": f"seq_{index:04d}",
            "native_id": None,
            "index": index,
        },
        "sequence": sequence,
        "scores": {"go_sequences": {"length": len(sequence)}},
        "artifacts": {},
    }


def _run(tmp_path, params: MergeCandidatesInput):
    adapter = MergeCandidatesAdapter(
        config={},
        weights_dir=tmp_path,
        device=torch.device("cpu"),
    )
    return adapter.run(params, null_context(tmp_path / "job"))


def test_keep_all_concatenates_in_upstream_order(tmp_path) -> None:
    a = _candidate("cand_a", sequence="MA", index=0)
    b = _candidate("cand_b", sequence="MB", index=1)
    c = _candidate("cand_c", sequence="MC", index=2)
    d = _candidate("cand_d", sequence="MD", index=3)

    out = _run(tmp_path, MergeCandidatesInput(candidate_sets=[[a, b], [c, d]]))

    assert out.candidates == [a, b, c, d]
    assert out.merge_audit == {
        "input_counts": [2, 2],
        "output_count": 4,
        "dedup_policy": "keep_all",
        "duplicates_removed": 0,
        "duplicate_groups": [],
        "conflicts": [],
    }


def test_duplicate_enrichment_is_merged_with_audit(tmp_path) -> None:
    source = _candidate("cand_a", sequence="MA", index=0)
    enriched = _candidate("cand_a", sequence="MA", index=1)
    enriched["scores"] = {"sequence_analysis": {"gravy": -0.2}}
    enriched["source"]["tool"] = "sequence_analysis"
    enriched["source"]["native_id"] = "analysis-native"

    out = _run(tmp_path, MergeCandidatesInput(candidate_sets=[[source], [enriched]]))

    assert len(out.candidates) == 1
    assert out.candidates[0]["candidate_id"] == "cand_a"
    assert set(out.candidates[0]["scores"]) == {"go_sequences", "sequence_analysis"}
    assert out.candidates[0]["source"]["tool"] == "sequence_analysis"
    assert out.merge_audit["duplicates_removed"] == 1
    assert out.merge_audit["duplicate_groups"][0]["retained_candidate_id"] == "cand_a"
    assert out.merge_audit["duplicate_groups"][0]["versions"][1]["source"]["native_id"] == (
        "analysis-native"
    )


def test_empty_input_set_rejected() -> None:
    with pytest.raises(ValidationError):
        MergeCandidatesInput(candidate_sets=[])


def test_adapter_schema_rejects_compact_scores_string() -> None:
    candidate = _candidate("cand_a", sequence="MA", index=0)
    candidate["scores"] = "<dict>"

    with pytest.raises(ValidationError):
        MergeCandidatesInput(candidate_sets=[[candidate]])


def test_adapter_output_never_exposes_private_merge_fields(tmp_path) -> None:
    source = _candidate("cand_a", sequence="MA", index=0)
    enriched = _candidate("cand_a", sequence="MA", index=1)
    enriched["scores"] = {"sequence_analysis": {"gravy": -0.2}}

    out = _run(tmp_path, MergeCandidatesInput(candidate_sets=[[source], [enriched]]))

    assert "__candidate_merge_positions" not in out.candidates[0]
    assert "__candidate_merge_representative_position" not in out.candidates[0]


def test_output_validates_against_descriptor_schema(tmp_path) -> None:
    out = _run(
        tmp_path,
        MergeCandidatesInput(candidate_sets=[[_candidate("cand_a", sequence="MA", index=0)]]),
    )

    validate_output(MergeCandidatesAdapter.descriptor, out.model_dump(mode="json"))
