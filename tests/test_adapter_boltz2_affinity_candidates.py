from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from vibeproteinagent.adapters.boltz2_affinity.adapter import (
    Boltz2AffinityInput,
    Boltz2AffinityOutput,
    _boltz2_affinity_candidates,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(sequence: str, index: int) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": f"cand_boltz_{index}",
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": f"seq_{index:04d}",
            "native_id": None,
            "index": index,
        },
        "sequence": sequence,
        "scores": {"upstream": {"score": float(index)}},
        "artifacts": {"upstream": {"path": f"/tmp/upstream_{index}.cif"}},
    }


def test_boltz2_affinity_emits_canonical_candidates() -> None:
    params = Boltz2AffinityInput(
        sequences=[
            {"type": "protein", "id": "A", "sequence": "AAAA"},
            {"type": "ligand", "id": "L", "smiles": "C"},
        ],
        binder_chain="L",
    )
    candidate = _boltz2_affinity_candidates(
        params,
        "/tmp/pred.cif",
        {"predicted_affinity": 1.2},
    )[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "boltz2_affinity"
    assert candidate["structure_path"] == "/tmp/pred.cif"
    assert candidate["scores"]["boltz2_affinity"]["predicted_affinity"] == 1.2


def test_boltz2_affinity_candidate_ids_are_deterministic() -> None:
    params = Boltz2AffinityInput(
        sequences=[{"type": "protein", "id": "A", "sequence": "AAAA"}],
        binder_chain="A",
    )
    first = _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})[0]
    second = _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})[0]
    assert first["candidate_id"] == second["candidate_id"]


def test_boltz2_affinity_output_preserves_legacy_fields() -> None:
    out = Boltz2AffinityOutput(
        predicted_cif="/tmp/pred.cif",
        affinity_pred_value=1.2,
        affinity_probability_binary=0.7,
        confidence_score=0.9,
        runtime_ms=1,
        seeds_used=[42],
    )
    assert out.model_dump()["affinity_pred_value"] == 1.2


def test_boltz2_affinity_consumes_canonical_legacy_and_rejects_mixed() -> None:
    params = Boltz2AffinityInput(candidates=[_candidate("AAAA", 0), _candidate("CCCC", 1)])
    assert params.sequences[0]["sequence"] == "AAAA"
    assert params.binder_chain == "B"
    assert (
        Boltz2AffinityInput(
            sequences=[{"type": "protein", "id": "A", "sequence": "AAAA"}],
            binder_chain="A",
        ).binder_chain
        == "A"
    )
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        Boltz2AffinityInput(candidates=[_candidate("AAAA", 0), {"sequence": "CCCC"}])


def test_boltz2_affinity_preserves_single_upstream_candidate_identity() -> None:
    upstream = _candidate("AAAA", 0)
    params = Boltz2AffinityInput(
        candidates=[upstream],
        sequences=[
            {"type": "protein", "id": "A", "sequence": "AAAA"},
            {"type": "ligand", "id": "L", "smiles": "C"},
        ],
        binder_chain="L",
    )

    candidate = _boltz2_affinity_candidates(
        params,
        "/tmp/pred.cif",
        {"affinity_pred_value": -1.0},
    )[0]

    assert candidate["candidate_id"] == upstream["candidate_id"]
    assert candidate["sequence"] == "AAAA"
    assert candidate["structure_path"] == "/tmp/pred.cif"
    assert candidate["scores"]["upstream"] == {"score": 0.0}
    assert candidate["scores"]["boltz2_affinity"]["affinity_pred_value"] == -1.0
    assert candidate["artifacts"]["upstream"] == {"path": "/tmp/upstream_0.cif"}
    assert candidate["artifacts"]["boltz2_affinity"]["predicted_cif"] == "/tmp/pred.cif"


def test_boltz2_affinity_selects_unique_candidate_by_sequence() -> None:
    params = Boltz2AffinityInput(
        candidates=[_candidate("AAAA", 0), _candidate("CCCC", 1)],
        sequences=[
            {"type": "protein", "id": "A", "sequence": "CCCC"},
            {"type": "ligand", "id": "L", "smiles": "C"},
        ],
        binder_chain="L",
    )

    candidate = _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})[0]

    assert candidate["candidate_id"] == "cand_boltz_1"


def test_boltz2_affinity_selects_unique_candidate_from_input_yaml(
    tmp_path,
) -> None:
    input_yaml = tmp_path / "input.yaml"
    input_yaml.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "CCCC"}},
                    {"ligand": {"id": "L", "smiles": "C"}},
                ],
                "properties": [{"affinity": {"binder": "L"}}],
            },
            sort_keys=False,
        )
    )
    params = Boltz2AffinityInput(
        candidates=[_candidate("AAAA", 0), _candidate("CCCC", 1)],
        input_yaml=str(input_yaml),
    )

    candidate = _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})[0]

    assert candidate["candidate_id"] == "cand_boltz_1"


def test_boltz2_affinity_rejects_ambiguous_multiple_candidates() -> None:
    params = Boltz2AffinityInput(
        candidates=[_candidate("AAAA", 0), _candidate("AAAA", 1)],
        sequences=[
            {"type": "protein", "id": "A", "sequence": "AAAA"},
            {"type": "ligand", "id": "L", "smiles": "C"},
        ],
        binder_chain="L",
    )

    with pytest.raises(ValueError, match="multiple canonical candidates match"):
        _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})


def test_boltz2_affinity_rejects_ambiguous_input_yaml_sequence_match(
    tmp_path,
) -> None:
    input_yaml = tmp_path / "input.yaml"
    input_yaml.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", "sequence": "AAAA"}},
                    {"ligand": {"id": "L", "smiles": "C"}},
                ],
                "properties": [{"affinity": {"binder": "L"}}],
            },
            sort_keys=False,
        )
    )
    params = Boltz2AffinityInput(
        candidates=[_candidate("AAAA", 0), _candidate("AAAA", 1)],
        input_yaml=str(input_yaml),
    )

    with pytest.raises(ValueError, match="sequences/input_yaml"):
        _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})
