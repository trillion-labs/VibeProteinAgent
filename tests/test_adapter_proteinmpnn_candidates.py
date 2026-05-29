from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.proteinmpnn.adapter import ProteinMPNNDesign, ProteinMPNNInput, ProteinMPNNOutput, _canonical_candidate_from_design
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_backbone", "source": {"tool": "rfdiffusion", "node_id": None, "job_id": None, "local_id": "design_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/backbone.pdb", "scores": {"rfdiffusion": {"length": 3}}, "artifacts": {}}


def test_proteinmpnn_designs_emit_canonical_candidates() -> None:
    params = ProteinMPNNInput(pdb_path="/tmp/backbone.pdb")
    candidate = _canonical_candidate_from_design({"candidate_id": "legacy", "sequence": "AAA", "score": 0.5, "seq_recovery": 0.3}, 0, params=params, structure_path="/tmp/backbone.pdb")
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "proteinmpnn"
    assert candidate["source"]["local_id"] == "seq_0000"
    assert candidate["sequence"] == "AAA"
    assert candidate["structure_path"] == "/tmp/backbone.pdb"
    assert candidate["scores"]["proteinmpnn"]["score"] == 0.5


def test_proteinmpnn_candidate_ids_are_deterministic() -> None:
    params = ProteinMPNNInput(pdb_path="/tmp/backbone.pdb")
    row = {"sequence": "AAA", "score": 0.5}
    assert _canonical_candidate_from_design(row, 0, params=params, structure_path="/tmp/backbone.pdb")["candidate_id"] == _canonical_candidate_from_design(row, 0, params=params, structure_path="/tmp/backbone.pdb")["candidate_id"]


def test_proteinmpnn_output_preserves_legacy_fields() -> None:
    out = ProteinMPNNOutput(designs=[ProteinMPNNDesign(sequence="AAA", score=0.5)], designs_fasta="/tmp/d.fa", log_prob_npz="/tmp/d.npz", num_residues=3, runtime_ms=1)
    assert out.model_dump()["designs"][0]["sequence"] == "AAA"


def test_proteinmpnn_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert ProteinMPNNInput(candidates=[_candidate()]).pdb_path == "/tmp/backbone.pdb"
    assert ProteinMPNNInput(pdb_path="/tmp/backbone.pdb").pdb_path == "/tmp/backbone.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ProteinMPNNInput(candidates=[_candidate(), {"pdb_path": "/tmp/legacy.pdb"}])


def test_proteinmpnn_infers_all_pdb_atom_chains(tmp_path) -> None:
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY B   1      1.000   0.000   0.000  1.00 10.00           C",
                "HETATM    3  C1  LIG C   1      2.000   0.000   0.000  1.00 10.00           C",
            ]
        )
        + "\n"
    )

    params = ProteinMPNNInput(pdb_path=str(pdb))

    assert params.chains == ["A", "B"]


def test_proteinmpnn_canonical_candidate_prefers_artifact_pdb_over_cif(tmp_path) -> None:
    pdb = tmp_path / "design.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA B   1      0.000   0.000   0.000  1.00 10.00           C\n"
    )
    candidate = _candidate()
    candidate["structure_path"] = str(tmp_path / "design.cif")
    candidate["artifacts"] = {"pxdesign": {"pdb_path": str(pdb)}}
    candidate["source"]["tool"] = "pxdesign"

    params = ProteinMPNNInput(candidates=[candidate])

    assert params.pdb_path == str(pdb)
    assert params.chains == ["B"]


def test_proteinmpnn_infers_all_cif_atom_chains(tmp_path) -> None:
    cif = tmp_path / "complex.cif"
    cif.write_text(
        "\n".join(
            [
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.label_asym_id",
                "_atom_site.auth_asym_id",
                "ATOM A X",
                "ATOM B Y",
                "HETATM C Z",
                "#",
            ]
        )
        + "\n"
    )

    params = ProteinMPNNInput(pdb_path=str(cif))

    assert params.chains == ["X", "Y"]
