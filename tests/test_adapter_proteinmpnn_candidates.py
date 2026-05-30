from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.proteinmpnn.adapter import (
    ProteinMPNNAdapter,
    ProteinMPNNDesign,
    ProteinMPNNInput,
    ProteinMPNNOutput,
    _canonical_candidate_from_design,
    _materialize_mpnn_pdb_input,
)
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate() -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_backbone",
        "source": {
            "tool": "rfdiffusion",
            "node_id": None,
            "job_id": None,
            "local_id": "design_0000",
            "native_id": None,
            "index": 0,
        },
        "structure_path": "/tmp/backbone.pdb",
        "scores": {"rfdiffusion": {"length": 3}},
        "artifacts": {},
    }


def test_proteinmpnn_designs_emit_canonical_candidates() -> None:
    params = ProteinMPNNInput(pdb_path="/tmp/backbone.pdb")
    candidate = _canonical_candidate_from_design(
        {
            "candidate_id": "legacy",
            "sequence": "AAA",
            "score": 0.5,
            "seq_recovery": 0.3,
        },
        0,
        params=params,
        structure_path="/tmp/backbone.pdb",
    )
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "proteinmpnn"
    assert candidate["source"]["local_id"] == "seq_0000"
    assert candidate["sequence"] == "AAA"
    assert candidate["structure_path"] == "/tmp/backbone.pdb"
    assert candidate["scores"]["proteinmpnn"]["score"] == 0.5


def test_proteinmpnn_candidate_ids_are_deterministic() -> None:
    params = ProteinMPNNInput(pdb_path="/tmp/backbone.pdb")
    row = {"sequence": "AAA", "score": 0.5}
    first = _canonical_candidate_from_design(
        row, 0, params=params, structure_path="/tmp/backbone.pdb"
    )
    second = _canonical_candidate_from_design(
        row, 0, params=params, structure_path="/tmp/backbone.pdb"
    )
    assert first["candidate_id"] == second["candidate_id"]


def test_proteinmpnn_output_preserves_legacy_fields() -> None:
    out = ProteinMPNNOutput(
        designs=[ProteinMPNNDesign(sequence="AAA", score=0.5)],
        designs_fasta="/tmp/d.fa",
        log_prob_npz="/tmp/d.npz",
        num_residues=3,
        runtime_ms=1,
    )
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


def test_proteinmpnn_materializes_multichar_cif_chains_for_mpnn(tmp_path) -> None:
    cif = tmp_path / "pxdesign_like.cif"
    cif.write_text(
        "\n".join(
            [
                "data_design",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.type_symbol",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.auth_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.auth_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "ATOM N N ALA A0 A0 1 1 100000.0 0.0 -200000.0",
                "ATOM C CA ALA A0 A0 1 1 100001.0 0.0 -199999.0",
                "ATOM C C ALA A0 A0 1 1 100002.0 0.0 -199998.0",
                "ATOM O O ALA A0 A0 1 1 100003.0 0.0 -199997.0",
                "ATOM N N GLY B0 B0 1 1 100000.0 1.0 -200000.0",
                "ATOM C CA GLY B0 B0 1 1 100001.0 1.0 -199999.0",
                "ATOM C C GLY B0 B0 1 1 100002.0 1.0 -199998.0",
                "ATOM O O GLY B0 B0 1 1 100003.0 1.0 -199997.0",
                "#",
            ]
        )
        + "\n"
    )

    pdb_path, chains, metadata = _materialize_mpnn_pdb_input(
        cif,
        requested_chains=["B0"],
        job_dir=tmp_path / "job",
    )

    assert chains == ["B"]
    assert metadata is not None
    assert metadata["raw_to_pdb_chain"] == {"A0": "A", "B0": "B"}
    text = pdb_path.read_text()
    assert " A   1" in text
    assert " B   1" in text
    assert "A0" not in text
    assert "B0" not in text
    for line in text.splitlines():
        if line.startswith("ATOM"):
            float(line[30:38])
            float(line[38:46])
            float(line[46:54])


def test_proteinmpnn_run_uses_materialized_chain_ids_for_mpnn() -> None:
    source = inspect.getsource(ProteinMPNNAdapter.run)

    assert "list(input_chains)" in source
    assert "list(params.chains)" not in source


def test_proteinmpnn_candidate_records_mpnn_input_pdb_artifact() -> None:
    params = ProteinMPNNInput(pdb_path="/tmp/backbone.cif")
    candidate = _canonical_candidate_from_design(
        {"candidate_id": "legacy", "sequence": "AAA", "score": 0.5},
        0,
        params=params,
        structure_path="/tmp/backbone.cif",
        mpnn_input_pdb_path="/tmp/backbone_proteinmpnn_input.pdb",
    )

    assert candidate["structure_path"] == "/tmp/backbone.cif"
    assert candidate["artifacts"]["proteinmpnn"]["mpnn_input_pdb_path"] == (
        "/tmp/backbone_proteinmpnn_input.pdb"
    )
