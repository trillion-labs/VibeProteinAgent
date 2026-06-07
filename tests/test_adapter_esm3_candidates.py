from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.esm3.adapter import (
    ESM3Input,
    ESM3Output,
    _esm3_candidates,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(sequence: str = "ACDE") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_esm3_input",
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


def test_esm3_emits_canonical_candidate() -> None:
    params = ESM3Input(sequence="ACDE", tracks=["sequence", "structure", "embedding"])
    out = ESM3Output(
        per_residue_embeddings_npz="/tmp/esm3.npz",
        predicted_pdb="/tmp/esm3.pdb",
        confidence_scores={"mean_plddt": 82.0, "ptm": 0.7},
        num_residues=4,
        model_name="esm3_sm_open_v1",
        runtime_ms=1,
    )
    out.candidates = _esm3_candidates(params, out)

    assert len(out.candidates) == 1
    candidate = out.candidates[0]
    validate_candidate_v1(candidate)
    assert candidate["sequence"] == "ACDE"
    assert candidate["structure_path"] == "/tmp/esm3.pdb"
    assert candidate["scores"]["esm3"]["mean_plddt"] == 82.0
    assert candidate["artifacts"]["esm3"]["per_residue_embeddings_npz"] == "/tmp/esm3.npz"


def test_esm3_profile_guidance_can_supply_consensus_sequence() -> None:
    params = ESM3Input(
        profile_guidance={
            "schema_version": "family_profile_v1",
            "consensus_sequence": "ACDE",
            "sequence_count": 3,
        },
        tracks=["embedding"],
    )
    out = ESM3Output(
        per_residue_embeddings_npz="/tmp/esm3.npz",
        predicted_pdb=None,
        confidence_scores=None,
        num_residues=4,
        model_name="esm3_sm_open_v1",
        runtime_ms=1,
    )
    candidate = _esm3_candidates(params, out)[0]

    assert params.sequence == "ACDE"
    assert candidate["scores"]["esm3"]["profile_guidance_applied"] is True
    assert candidate["artifacts"]["profile_guidance"]["sequence_count"] == 3


def test_esm3_profile_guidance_keeps_explicit_sequence() -> None:
    params = ESM3Input(
        sequence="YYYY",
        profile_guidance={"schema_version": "family_profile_v1", "consensus_sequence": "ACDE"},
        tracks=["embedding"],
    )
    out = ESM3Output(
        per_residue_embeddings_npz=None,
        predicted_pdb=None,
        confidence_scores=None,
        num_residues=4,
        model_name="esm3_sm_open_v1",
        runtime_ms=1,
    )
    candidate = _esm3_candidates(params, out)[0]

    assert candidate["sequence"] == "YYYY"
    assert candidate["scores"]["esm3"]["profile_guidance_applied"] is False
    assert candidate["artifacts"]["profile_guidance"]["consensus_length"] == 4


def test_esm3_candidate_ids_are_deterministic() -> None:
    params = ESM3Input(sequence="ACDE", tracks=["embedding"])
    out = ESM3Output(
        per_residue_embeddings_npz="/tmp/esm3.npz",
        predicted_pdb=None,
        confidence_scores=None,
        num_residues=4,
        model_name="esm3_sm_open_v1",
        runtime_ms=1,
    )
    assert _esm3_candidates(params, out)[0]["candidate_id"] == _esm3_candidates(params, out)[0]["candidate_id"]


def test_esm3_candidate_source_fields() -> None:
    params = ESM3Input(sequence="ACDE", tracks=["structure"])
    out = ESM3Output(
        per_residue_embeddings_npz=None,
        predicted_pdb="/tmp/esm3.pdb",
        confidence_scores={"mean_plddt": 82.0},
        num_residues=4,
        model_name="esm3_sm_open_v1",
        runtime_ms=1,
    )
    source = _esm3_candidates(params, out)[0]["source"]
    assert source["tool"] == "esm3"
    assert source["local_id"] == "esm3_0000"
    assert source["index"] == 0


def test_esm3_output_preserves_legacy_fields() -> None:
    out = ESM3Output(
        per_residue_embeddings_npz="/tmp/esm3.npz",
        predicted_pdb="/tmp/esm3.pdb",
        confidence_scores={"mean_plddt": 82.0},
        num_residues=4,
        model_name="esm3_sm_open_v1",
        runtime_ms=1,
    )
    dumped = out.model_dump()
    assert dumped["per_residue_embeddings_npz"] == "/tmp/esm3.npz"
    assert dumped["predicted_pdb"] == "/tmp/esm3.pdb"
    assert dumped["confidence_scores"] == {"mean_plddt": 82.0}
    assert dumped["num_residues"] == 4


def test_esm3_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert ESM3Input(candidates=[_candidate("MKKL")]).sequence == "MKKL"
    assert ESM3Input(sequence="MKKL").sequence == "MKKL"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ESM3Input(candidates=[_candidate("MKKL"), {"sequence": "ACDE"}])
