from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.tmalign.adapter import (
    TMalignInput,
    TMalignOutput,
    _tmalign_candidates,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(candidate_id: str, path: str, index: int) -> dict:
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
        "scores": {},
        "artifacts": {},
    }


def test_tmalign_emits_canonical_candidates() -> None:
    params = TMalignInput(pdb_a="/tmp/ref.pdb", pdb_b="/tmp/query.pdb")
    candidate = _tmalign_candidates(
        params,
        "/tmp/query.pdb",
        {"tm_score": 0.8, "rmsd": 1.2},
        None,
    )[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "tmalign"
    assert candidate["scores"]["tmalign"]["tm_score"] == 0.8


def test_tmalign_candidate_ids_are_deterministic() -> None:
    params = TMalignInput(pdb_a="/tmp/ref.pdb", pdb_b="/tmp/query.pdb")
    first = _tmalign_candidates(params, "/tmp/query.pdb", {}, None)
    second = _tmalign_candidates(params, "/tmp/query.pdb", {}, None)
    assert first[0]["candidate_id"] == second[0]["candidate_id"]


def test_tmalign_output_preserves_legacy_fields() -> None:
    out = TMalignOutput(
        tm_score_normalized_by_a=0.9,
        tm_score_normalized_by_b=0.8,
        tm_score_normalized=0.8,
        rmsd=1.0,
        aligned_length=10,
        length_a=10,
        length_b=12,
        seq_identity=0.5,
        runtime_ms=1,
    )
    assert out.model_dump()["rmsd"] == 1.0


def test_tmalign_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert (
        TMalignInput(
            candidates=[
                _candidate("ref", "/tmp/ref.pdb", 0),
                _candidate("query", "/tmp/query.pdb", 1),
            ]
        ).pdb_b
        == "/tmp/query.pdb"
    )
    assert TMalignInput(pdb_a="/tmp/ref.pdb", pdb_b="/tmp/query.pdb").pdb_a == "/tmp/ref.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        TMalignInput(
            candidates=[
                _candidate("ref", "/tmp/ref.pdb", 0),
                {"pdb_b": "/tmp/query.pdb"},
            ]
        )


def test_tmalign_candidate_input_appends_scores_to_design_not_reference() -> None:
    params = TMalignInput(
        candidates=[
            _candidate("reference", "/tmp/ref.pdb", 0),
            _candidate("v40", "/tmp/v40.pdb", 1),
        ]
    )

    candidates = _tmalign_candidates(
        params,
        "/tmp/v40.pdb",
        {"tm_score_normalized": 0.88058, "rmsd": 1.1},
        "/tmp/v40_sup.pdb",
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["candidate_id"] == "v40"
    assert candidate["structure_path"] == "/tmp/v40.pdb"
    assert candidate["scores"]["tmalign"]["tm_score_normalized"] == pytest.approx(0.88058)
    assert candidate["artifacts"]["tmalign"]["superposition_pdb"] == "/tmp/v40_sup.pdb"
