from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.boltz2_affinity.adapter import (
    Boltz2AffinityInput,
    Boltz2AffinityOutput,
    _boltz2_affinity_candidates,
)
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate(sequence: str, index: int) -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": f"cand_boltz_{index}", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": f"seq_{index:04d}", "native_id": None, "index": index}, "sequence": sequence, "scores": {}, "artifacts": {}}


def test_boltz2_affinity_emits_canonical_candidates() -> None:
    params = Boltz2AffinityInput(sequences=[{"type": "protein", "id": "A", "sequence": "AAAA"}, {"type": "ligand", "id": "L", "smiles": "C"}], binder_chain="L")
    candidate = _boltz2_affinity_candidates(params, "/tmp/pred.cif", {"predicted_affinity": 1.2})[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "boltz2_affinity"
    assert candidate["structure_path"] == "/tmp/pred.cif"
    assert candidate["scores"]["boltz2_affinity"]["predicted_affinity"] == 1.2


def test_boltz2_affinity_candidate_ids_are_deterministic() -> None:
    params = Boltz2AffinityInput(sequences=[{"type": "protein", "id": "A", "sequence": "AAAA"}], binder_chain="A")
    assert _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})[0]["candidate_id"] == _boltz2_affinity_candidates(params, "/tmp/pred.cif", {})[0]["candidate_id"]


def test_boltz2_affinity_output_preserves_legacy_fields() -> None:
    out = Boltz2AffinityOutput(predicted_cif="/tmp/pred.cif", affinity_pred_value=1.2, affinity_probability_binary=0.7, confidence_score=0.9, runtime_ms=1, seeds_used=[42])
    assert out.model_dump()["affinity_pred_value"] == 1.2


def test_boltz2_affinity_consumes_canonical_legacy_and_rejects_mixed() -> None:
    params = Boltz2AffinityInput(candidates=[_candidate("AAAA", 0), _candidate("CCCC", 1)])
    assert params.sequences[0]["sequence"] == "AAAA"
    assert params.binder_chain == "B"
    assert Boltz2AffinityInput(sequences=[{"type": "protein", "id": "A", "sequence": "AAAA"}], binder_chain="A").binder_chain == "A"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        Boltz2AffinityInput(candidates=[_candidate("AAAA", 0), {"sequence": "CCCC"}])
