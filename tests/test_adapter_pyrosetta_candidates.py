from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.pyrosetta.adapter import (
    PyRosettaInput,
    PyRosettaOutput,
    _pyrosetta_candidates,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(
    structure_path: str = "/tmp/a.pdb",
    artifacts: dict | None = None,
) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_struct",
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": "struct_0000",
            "native_id": None,
            "index": 0,
        },
        "structure_path": structure_path,
        "scores": {},
        "artifacts": artifacts or {},
    }


def test_pyrosetta_emits_canonical_candidates() -> None:
    params = PyRosettaInput(pdb_path="/tmp/a.pdb")
    candidate = _pyrosetta_candidates(params, -12.5, None, "/tmp/a.pdb")[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "pyrosetta"
    assert candidate["source"]["local_id"] == "score_0000"
    assert candidate["scores"]["pyrosetta"]["total_score"] == -12.5


def test_pyrosetta_candidate_ids_are_deterministic() -> None:
    params = PyRosettaInput(pdb_path="/tmp/a.pdb")
    assert (
        _pyrosetta_candidates(params, -1.0, None, "/tmp/a.pdb")[0]["candidate_id"]
        == _pyrosetta_candidates(params, -1.0, None, "/tmp/a.pdb")[0][
            "candidate_id"
        ]
    )


def test_pyrosetta_output_preserves_legacy_fields() -> None:
    assert (
        PyRosettaOutput(mode="score", total_score=-1.0, runtime_ms=1).model_dump()[
            "total_score"
        ]
        == -1.0
    )


def test_pyrosetta_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert PyRosettaInput(candidates=[_candidate()]).pdb_path == "/tmp/a.pdb"
    assert PyRosettaInput(
        candidates=[
            _candidate(
                "/tmp/a.cif",
                {"protenix_v2": {"predicted_pdb": "/tmp/a.pdb"}},
            )
        ]
    ).pdb_path == "/tmp/a.pdb"
    with pytest.raises(ValidationError, match="requires a PDB artifact"):
        PyRosettaInput(candidates=[_candidate("/tmp/a.cif")])
    assert PyRosettaInput(pdb_path="/tmp/a.pdb").pdb_path == "/tmp/a.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        PyRosettaInput(candidates=[_candidate(), {"pdb_path": "/tmp/b.pdb"}])


def test_pyrosetta_preserves_upstream_identity_and_records_interface_pair() -> None:
    upstream = _candidate(
        artifacts={
            "protenix_v2": {
                "complex_chains": [
                    {"role": "target", "chain_id": "A", "sequence_length": 100},
                    {"role": "binder", "chain_id": "B", "sequence_length": 20},
                ]
            }
        }
    )
    params = PyRosettaInput(
        candidates=[upstream],
        mode="interface",
        chains_a="A",
        chains_b="B",
    )

    candidate = _pyrosetta_candidates(
        params,
        -12.5,
        {"dG_separated": -2.0},
        "/tmp/relaxed.pdb",
    )[0]

    assert candidate["candidate_id"] == "cand_struct"
    assert candidate["structure_path"] == "/tmp/relaxed.pdb"
    assert candidate["artifacts"]["protenix_v2"]["complex_chains"][1]["role"] == "binder"
    assert candidate["artifacts"]["pyrosetta"]["interface_chain_pairs"] == [
        {"chain_a": "A", "chain_b": "B", "source": "pyrosetta"}
    ]
    assert candidate["scores"]["pyrosetta"]["interface_metrics"]["dG_separated"] == -2.0
