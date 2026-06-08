from __future__ import annotations

import pytest
from pydantic import ValidationError

pytest.importorskip("freesasa")

from vibeproteinagent.adapters.prodigy.adapter import (
    ProdigyInput,
    ProdigyOutput,
    _prodigy_candidates,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_prod",
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": "struct_0000",
            "native_id": None,
            "index": 0,
        },
        "structure_path": "/tmp/a.pdb",
        "scores": {},
        "artifacts": {},
    }


def test_prodigy_emits_canonical_candidates() -> None:
    params = ProdigyInput(pdb_path="/tmp/a.pdb", chain_a="A", chain_b="B")
    candidate = _prodigy_candidates(
        params,
        "/tmp/a.pdb",
        {"delta_g_kcal_mol": -9.0, "kd_nm": 10.0},
    )[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "prodigy"
    assert candidate["scores"]["prodigy"]["delta_g_kcal_mol"] == -9.0


def test_prodigy_candidate_ids_are_deterministic() -> None:
    params = ProdigyInput(pdb_path="/tmp/a.pdb", chain_a="A", chain_b="B")
    first = _prodigy_candidates(params, "/tmp/a.pdb", {})
    second = _prodigy_candidates(params, "/tmp/a.pdb", {})
    assert first[0]["candidate_id"] == second[0]["candidate_id"]


def test_prodigy_output_preserves_legacy_fields() -> None:
    out = ProdigyOutput(delta_g_kcal_mol=-9.0, kd_nm=10.0, interface_contacts=5, runtime_ms=1)
    assert out.model_dump()["kd_nm"] == 10.0


def test_prodigy_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert (
        ProdigyInput(candidates=[_candidate()], chain_a="A", chain_b="B").pdb_path
        == "/tmp/a.pdb"
    )
    assert (
        ProdigyInput(pdb_path="/tmp/a.pdb", chain_a="A", chain_b="B").pdb_path
        == "/tmp/a.pdb"
    )
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ProdigyInput(
            candidates=[_candidate(), {"pdb_path": "/tmp/legacy.pdb"}],
            chain_a="A",
            chain_b="B",
        )


def test_prodigy_preserves_canonical_identity_and_appends_scores() -> None:
    source = _candidate()
    source["scores"] = {"bindcraft": {"iptm": 0.78}}

    candidate = _prodigy_candidates(
        ProdigyInput(candidates=[source], chain_a="A", chain_b="B"),
        "/tmp/a.pdb",
        {"delta_g_kcal_mol": -10.2, "kd_nm": 36.0},
    )[0]

    assert candidate["candidate_id"] == "cand_prod"
    assert candidate["scores"]["bindcraft"]["iptm"] == 0.78
    assert candidate["scores"]["prodigy"]["kd_nm"] == 36.0
