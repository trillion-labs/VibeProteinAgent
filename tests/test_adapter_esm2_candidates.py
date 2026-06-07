from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.esm2.adapter import (
    ESM2Input,
    ESM2Output,
    _DEFAULT_MODEL,
    _esm2_candidates,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(sequence: str = "ACDE") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_esm2_input",
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": "seq_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": sequence,
        "scores": {},
        "artifacts": {},
    }


def test_esm2_embedding_emits_canonical_candidate() -> None:
    params = ESM2Input(sequence="ACDE", mode="embedding")
    out = ESM2Output(
        per_residue_embeddings_npz="/tmp/esm2.npz",
        pooled_embedding=[0.1, 0.2, 0.3],
        num_residues=4,
        model_name=_DEFAULT_MODEL,
        mode="embedding",
        runtime_ms=1,
    )
    out.candidates = _esm2_candidates(params, out)

    assert len(out.candidates) == 1
    candidate = out.candidates[0]
    validate_candidate_v1(candidate)
    assert candidate["sequence"] == "ACDE"
    assert candidate["scores"]["esm2"]["embedding_dim"] == 3
    assert candidate["artifacts"]["esm2"]["per_residue_embeddings_npz"] == "/tmp/esm2.npz"


def test_esm2_sequence_generation_emits_canonical_candidates() -> None:
    params = ESM2Input(sequence="ACX", mode="sequence_generation")
    out = ESM2Output(
        generated_sequences=["ACA"],
        filled_positions=[3],
        generation_strategy="esm2_mlm_argmax",
        num_residues=3,
        model_name=_DEFAULT_MODEL,
        mode="sequence_generation",
        runtime_ms=1,
    )
    out.candidates = _esm2_candidates(params, out)

    assert len(out.candidates) == len(out.generated_sequences)
    validate_candidate_v1(out.candidates[0])
    assert out.candidates[0]["sequence"] == "ACA"
    assert out.candidates[0]["scores"]["esm2"]["filled_positions"] == [3]


def test_esm2_profile_guidance_forces_generation_and_records_metadata() -> None:
    profile = {
        "schema_version": "family_profile_v1",
        "consensus_sequence": "ACDE",
        "conserved_positions": [
            {"position": 2, "consensus": "C", "conservation": 1.0},
            {"position": 4, "consensus": "E", "conservation": 1.0},
        ],
    }
    params = ESM2Input(sequence="AAAA", profile_guidance=profile, mask_positions=[3])
    out = ESM2Output(
        generated_sequences=["ACDE"],
        filled_positions=[2, 3, 4],
        generation_strategy="esm2_mlm_argmax",
        num_residues=4,
        model_name=_DEFAULT_MODEL,
        mode="sequence_generation",
        runtime_ms=1,
    )
    candidate = _esm2_candidates(params, out)[0]

    assert params.mode == "sequence_generation"
    assert candidate["scores"]["esm2"]["profile_guidance_applied"] is True
    assert candidate["artifacts"]["profile_guidance"]["consensus_length"] == 4


def test_esm2_candidate_ids_are_deterministic() -> None:
    params = ESM2Input(sequence="ACDE", mode="embedding")
    out = ESM2Output(
        pooled_embedding=[0.1, 0.2],
        num_residues=4,
        model_name=_DEFAULT_MODEL,
        mode="embedding",
        runtime_ms=1,
    )
    assert _esm2_candidates(params, out)[0]["candidate_id"] == _esm2_candidates(params, out)[0]["candidate_id"]


def test_esm2_candidate_source_fields() -> None:
    params = ESM2Input(sequence="ACDE", mode="embedding")
    out = ESM2Output(
        pooled_embedding=[0.1],
        num_residues=4,
        model_name=_DEFAULT_MODEL,
        mode="embedding",
        runtime_ms=1,
    )
    source = _esm2_candidates(params, out)[0]["source"]
    assert source["tool"] == "esm2"
    assert source["local_id"] == "esm2_0000"
    assert source["index"] == 0
    assert source["native_id"] is None


def test_esm2_output_preserves_legacy_fields() -> None:
    out = ESM2Output(
        per_residue_embeddings_npz="/tmp/esm2.npz",
        pooled_embedding=[0.1, 0.2],
        generated_sequences=[],
        filled_positions=[],
        fallback_used=False,
        num_residues=4,
        model_name=_DEFAULT_MODEL,
        mode="embedding",
        runtime_ms=1,
    )
    dumped = out.model_dump()
    assert dumped["per_residue_embeddings_npz"] == "/tmp/esm2.npz"
    assert dumped["pooled_embedding"] == [0.1, 0.2]
    assert dumped["generated_sequences"] == []
    assert dumped["filled_positions"] == []
    assert dumped["fallback_used"] is False


def test_esm2_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert ESM2Input(candidates=[_candidate("MKKL")], mode="embedding").sequence == "MKKL"
    assert ESM2Input(sequence="MKKL", mode="embedding").sequence == "MKKL"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ESM2Input(candidates=[_candidate("MKKL"), {"sequence": "ACDE"}], mode="embedding")
