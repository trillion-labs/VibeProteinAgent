from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.bindcraft.adapter import (
    BindCraftDesign,
    BindCraftInput,
    BindCraftOutput,
    _canonical_candidate_from_design,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_target",
        "source": {
            "tool": "target",
            "node_id": None,
            "job_id": None,
            "local_id": "target_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": "AAAA",
        "structure_path": "/tmp/target.pdb",
        "scores": {},
        "artifacts": {},
    }


def test_bindcraft_designs_emit_canonical_candidates() -> None:
    candidate = _canonical_candidate_from_design(
        {
            "candidate_id": "legacy",
            "pdb_path": "/tmp/binder.pdb",
            "sequence": "MKKL",
            "binder_length": 4,
            "iptm": 0.8,
        },
        0,
    )
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "bindcraft"
    assert candidate["source"]["local_id"] == "binder_0000"
    assert candidate["sequence"] == "MKKL"
    assert candidate["scores"]["bindcraft"]["iptm"] == 0.8


def test_bindcraft_candidate_ids_are_deterministic() -> None:
    row = {"pdb_path": "/tmp/binder.pdb", "sequence": "MKKL", "binder_length": 4}
    assert (
        _canonical_candidate_from_design(row, 0)["candidate_id"]
        == _canonical_candidate_from_design(row, 0)["candidate_id"]
    )


def test_bindcraft_records_target_context_without_changing_binder_sequence(tmp_path) -> None:
    target = tmp_path / "target.pdb"
    output = tmp_path / "binder.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  CYS A   2       1.000   0.000   0.000  1.00 20.00           C\n"
    )
    output.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  CYS A   2       1.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  CA  MET B   1       2.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      4  CA  LYS B   2       3.000   0.000   0.000  1.00 20.00           C\n"
    )
    params = BindCraftInput(target_pdb=str(target), target_chain="A")
    candidate = _canonical_candidate_from_design(
        {
            "candidate_id": "legacy",
            "pdb_path": str(output),
            "sequence": "MK",
            "binder_length": 2,
        },
        0,
        params=params,
    )

    assert candidate["sequence"] == "MK"
    assert candidate["artifacts"]["bindcraft"]["complex_chains"] == [
        {
            "role": "target",
            "chain_id": "A",
            "input_order": 0,
            "sequence_length": 2,
            "observed_chain_id": "A",
        },
        {
            "role": "binder",
            "chain_id": "B",
            "input_order": 1,
            "sequence_length": 2,
            "observed_chain_id": "B",
        },
    ]


def test_bindcraft_infers_binder_chain_when_target_chain_is_b(tmp_path) -> None:
    target = tmp_path / "target.pdb"
    output = tmp_path / "binder_target_b.pdb"
    target.write_text(
        "ATOM      1  CA  ALA B   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  CYS B   2       1.000   0.000   0.000  1.00 20.00           C\n"
    )
    output.write_text(
        "ATOM      1  CA  MET A   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  LYS A   2       1.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      3  CA  ALA B   1       2.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      4  CA  CYS B   2       3.000   0.000   0.000  1.00 20.00           C\n"
    )
    params = BindCraftInput(target_pdb=str(target), target_chain="B")

    candidate = _canonical_candidate_from_design(
        {
            "candidate_id": "legacy",
            "pdb_path": str(output),
            "sequence": "MK",
            "binder_length": 2,
        },
        0,
        params=params,
    )

    rows = candidate["artifacts"]["bindcraft"]["complex_chains"]
    assert rows[0]["role"] == "target"
    assert rows[0]["chain_id"] == "B"
    assert rows[0]["observed_chain_id"] == "B"
    assert rows[1]["role"] == "binder"
    assert rows[1]["chain_id"] == "A"
    assert rows[1]["observed_chain_id"] == "A"


def test_bindcraft_does_not_assert_non_unique_binder_chain(tmp_path) -> None:
    target = tmp_path / "target.pdb"
    output = tmp_path / "duplicate_binder.pdb"
    target.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C\n"
    )
    output.write_text(
        "ATOM      1  CA  MET B   1       0.000   0.000   0.000  1.00 20.00           C\n"
        "ATOM      2  CA  MET C   1       1.000   0.000   0.000  1.00 20.00           C\n"
    )
    params = BindCraftInput(target_pdb=str(target), target_chain="A")

    candidate = _canonical_candidate_from_design(
        {
            "candidate_id": "legacy",
            "pdb_path": str(output),
            "sequence": "M",
            "binder_length": 1,
        },
        0,
        params=params,
    )

    binder = candidate["artifacts"]["bindcraft"]["complex_chains"][1]
    assert candidate["sequence"] == "M"
    assert binder["role"] == "binder"
    assert "chain_id" not in binder
    assert "observed_chain_id" not in binder


def test_bindcraft_output_preserves_legacy_fields() -> None:
    out = BindCraftOutput(
        designs=[
            BindCraftDesign(
                pdb_path="/tmp/b.pdb",
                sequence="MKKL",
                binder_length=4,
            )
        ],
        runtime_ms=1,
        num_designs_returned=1,
        seeds_used=[],
    )
    assert out.model_dump()["designs"][0]["pdb_path"] == "/tmp/b.pdb"


def test_bindcraft_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert BindCraftInput(candidates=[_candidate()]).target_pdb == "/tmp/target.pdb"
    assert BindCraftInput(target_pdb="/tmp/target.pdb").target_pdb == "/tmp/target.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        BindCraftInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}])
