from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from vibeproteinagent.adapters.merge_candidates.adapter import MergeCandidatesInput
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


REPO_ROOT = Path(__file__).resolve().parents[1]


def _candidate() -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_capabilities",
        "source": {"tool": "boltzgen", "local_id": "0"},
        "sequence": "MKTAYIAK",
        "structure_path": "/tmp/design.pdb",
        "scores": {"boltzgen": {"confidence": 0.82}},
        "artifacts": {},
        "capabilities": {
            "has_sequence": True,
            "has_backbone_structure": True,
            "has_full_structure": True,
            "has_candidate_scores": True,
        },
    }


def _schema(path: str) -> dict:
    return json.loads((REPO_ROOT / path).read_text())


def test_candidate_v1_accepts_known_capabilities() -> None:
    validate_candidate_v1(_candidate())


def test_candidate_v1_rejects_unknown_capability_keys() -> None:
    candidate = _candidate()
    candidate["capabilities"]["has_magic"] = True

    with pytest.raises(ValidationError):
        validate_candidate_v1(candidate)


def test_merge_candidates_input_accepts_candidate_capabilities() -> None:
    parsed = MergeCandidatesInput(candidate_sets=[[_candidate()]])

    assert parsed.candidate_sets[0][0]["capabilities"]["has_full_structure"] is True


@pytest.mark.parametrize(
    "schema_path,payload",
    [
        (
            "src/vibeproteinagent/adapters/merge_candidates/schemas/merge_candidates.input.json",
            {"candidate_sets": [[_candidate()]]},
        ),
        (
            "src/vibeproteinagent/adapters/merge_candidates/schemas/merge_candidates.output.json",
            {
                "candidates": [_candidate()],
                "merge_audit": {
                    "input_counts": [1],
                    "output_count": 1,
                    "dedup_policy": "keep_all",
                    "duplicates_removed": 0,
                },
            },
        ),
    ],
)
def test_merge_candidates_json_schemas_accept_candidate_capabilities(
    schema_path: str,
    payload: dict,
) -> None:
    Draft202012Validator(_schema(schema_path)).validate(payload)
