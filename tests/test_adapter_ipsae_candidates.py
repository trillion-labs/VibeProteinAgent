from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.ipsae.adapter import IpSaeInput, IpSaeOutput, _ipsae_candidates
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_ipsae",
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


def test_ipsae_emits_canonical_candidates() -> None:
    params = IpSaeInput(pae_path="/tmp/a.json", pdb_path="/tmp/a.pdb", chain_a="A", chain_b="B")
    candidate = _ipsae_candidates(
        params,
        "/tmp/a.pdb",
        {"interface_pae": 0.6},
        "/tmp/s.txt",
        "/tmp/byres.txt",
    )[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "ipsae"
    assert candidate["scores"]["ipsae"]["interface_pae"] == 0.6


def test_ipsae_candidate_ids_are_deterministic() -> None:
    params = IpSaeInput(pae_path="/tmp/a.json", pdb_path="/tmp/a.pdb", chain_a="A", chain_b="B")
    first = _ipsae_candidates(params, "/tmp/a.pdb", {}, "/tmp/s.txt", "/tmp/byres.txt")
    second = _ipsae_candidates(params, "/tmp/a.pdb", {}, "/tmp/s.txt", "/tmp/byres.txt")
    assert first[0]["candidate_id"] == second[0]["candidate_id"]


def test_ipsae_output_preserves_legacy_fields() -> None:
    out = IpSaeOutput(
        ipsae_score=0.5,
        ipsae_asym_ab=0.4,
        ipsae_asym_ba=0.5,
        ipsae_max=0.5,
        interface_residues_a=[1],
        interface_residues_b=[2],
        scoreboard_path="/tmp/s.txt",
        byres_path="/tmp/b.txt",
        runtime_ms=1,
    )
    assert out.model_dump()["ipsae_score"] == 0.5


def test_ipsae_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert (
        IpSaeInput(
            pae_path="/tmp/a.json",
            candidates=[_candidate()],
            chain_a="A",
            chain_b="B",
        ).pdb_path
        == "/tmp/a.pdb"
    )
    assert (
        IpSaeInput(
            pae_path="/tmp/a.json",
            pdb_path="/tmp/a.pdb",
            chain_a="A",
            chain_b="B",
        ).pdb_path
        == "/tmp/a.pdb"
    )
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        IpSaeInput(
            pae_path="/tmp/a.json",
            candidates=[_candidate(), {"pdb_path": "/tmp/legacy.pdb"}],
            chain_a="A",
            chain_b="B",
        )


def test_ipsae_preserves_canonical_identity_and_appends_scores() -> None:
    source = _candidate()
    source["scores"] = {"protenix_v2": {"iptm": 0.31}}

    candidate = _ipsae_candidates(
        IpSaeInput(pae_path="/tmp/a.json", candidates=[source], chain_a="A", chain_b="B"),
        "/tmp/a.pdb",
        {"ipsae_score": 0.0},
        "/tmp/s.txt",
        "/tmp/byres.txt",
    )[0]

    assert candidate["candidate_id"] == "cand_ipsae"
    assert candidate["scores"]["protenix_v2"]["iptm"] == 0.31
    assert candidate["scores"]["ipsae"]["ipsae_score"] == 0.0
