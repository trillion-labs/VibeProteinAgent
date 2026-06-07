from __future__ import annotations

import pytest
from jsonschema import ValidationError

from vibeproteinagent.candidates.schema import validate_candidate_v1


def _valid_candidate(*, native_id: str | None = "P44612") -> dict:
    source = {
        "tool": "go_sequences",
        "node_id": "generate",
        "job_id": "job_abc123",
        "local_id": "seq_0000",
        "native_id": native_id,
        "index": 0,
    }
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_abc123",
        "source": source,
        "sequence": "ACDEFGHIK",
        "scores": {"go_sequences": {"confidence": 0.8}},
        "artifacts": {"go_sequences": {"record_path": "seqs.json"}},
    }


def _invalid(candidate: dict) -> None:
    with pytest.raises(ValidationError):
        validate_candidate_v1(candidate)


def test_valid_candidate_passes_with_native_id() -> None:
    validate_candidate_v1(_valid_candidate(native_id="P44612"))


def test_valid_candidate_passes_without_native_id() -> None:
    candidate = _valid_candidate(native_id=None)
    candidate["source"].pop("native_id")
    validate_candidate_v1(candidate)


def test_missing_candidate_id_fails() -> None:
    candidate = _valid_candidate()
    candidate.pop("candidate_id")
    _invalid(candidate)


def test_missing_source_tool_fails() -> None:
    candidate = _valid_candidate()
    candidate["source"].pop("tool")
    _invalid(candidate)


def test_missing_source_local_id_fails() -> None:
    candidate = _valid_candidate()
    candidate["source"].pop("local_id")
    _invalid(candidate)


def test_negative_source_index_fails() -> None:
    candidate = _valid_candidate()
    candidate["source"]["index"] = -1
    _invalid(candidate)


def test_scores_top_level_scalar_fails() -> None:
    candidate = _valid_candidate()
    candidate["scores"] = {"plddt": 0.8}
    _invalid(candidate)


def test_scores_top_level_array_fails() -> None:
    candidate = _valid_candidate()
    candidate["scores"] = {"rank_designs": [0.8]}
    _invalid(candidate)


def test_artifacts_top_level_scalar_fails() -> None:
    candidate = _valid_candidate()
    candidate["artifacts"] = {"structure_path": "design.pdb"}
    _invalid(candidate)


def test_artifacts_top_level_array_fails() -> None:
    candidate = _valid_candidate()
    candidate["artifacts"] = {"rank_designs": ["design.pdb"]}
    _invalid(candidate)


@pytest.mark.parametrize("field", ["kind", "rank", "metadata"])
def test_deferred_fields_are_rejected(field: str) -> None:
    candidate = _valid_candidate()
    candidate[field] = "deferred"
    _invalid(candidate)
