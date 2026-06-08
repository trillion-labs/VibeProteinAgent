from __future__ import annotations

from vibeproteinagent.candidates.chain_roles import (
    chain_sequence_from_structure,
    chain_sequences_from_structure,
    complex_chain_metadata,
    observed_chain_for_sequence,
)


def _pdb_line(
    serial: int,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
    )


def test_chain_role_helper_records_explicit_target_and_binder(tmp_path) -> None:
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        "\n".join(
            [
                _pdb_line(1, "ALA", "A", 1, 0.0),
                _pdb_line(2, "CYS", "A", 2, 1.0),
                _pdb_line(3, "MET", "B", 1, 2.0),
                _pdb_line(4, "LYS", "B", 2, 3.0),
            ]
        )
        + "\n"
    )

    assert chain_sequences_from_structure(pdb) == {"A": "AC", "B": "MK"}
    assert chain_sequence_from_structure(pdb, "B") == "MK"

    rows = complex_chain_metadata(
        target_sequence="AC",
        binder_sequence="MK",
        structure_path=pdb,
        target_chain_id="A",
        binder_chain_id="B",
    )

    assert rows == [
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


def test_chain_role_helper_does_not_invent_roles_for_raw_structure(tmp_path) -> None:
    pdb = tmp_path / "two_chain.pdb"
    pdb.write_text(
        "\n".join(
            [
                _pdb_line(1, "ALA", "A", 1, 0.0),
                _pdb_line(2, "MET", "B", 1, 1.0),
            ]
        )
        + "\n"
    )

    assert chain_sequences_from_structure(pdb) == {"A": "A", "B": "M"}
    assert complex_chain_metadata(structure_path=pdb) == []


def test_chain_role_helper_validates_explicit_chain_against_sequence(tmp_path) -> None:
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        "\n".join(
            [
                _pdb_line(1, "MET", "A", 1, 0.0),
                _pdb_line(2, "LYS", "A", 2, 1.0),
                _pdb_line(3, "ALA", "B", 1, 2.0),
                _pdb_line(4, "CYS", "B", 2, 3.0),
            ]
        )
        + "\n"
    )

    assert observed_chain_for_sequence(pdb, "MK", preferred_chain_id="B") == "A"
    rows = complex_chain_metadata(
        target_sequence="AC",
        binder_sequence="MK",
        structure_path=pdb,
        target_chain_id="B",
        binder_chain_id="B",
    )

    assert rows[0]["observed_chain_id"] == "B"
    assert rows[1]["observed_chain_id"] == "A"


def test_chain_role_helper_leaves_duplicate_sequence_ambiguous(tmp_path) -> None:
    pdb = tmp_path / "duplicate.pdb"
    pdb.write_text(
        "\n".join(
            [
                _pdb_line(1, "MET", "A", 1, 0.0),
                _pdb_line(2, "LYS", "A", 2, 1.0),
                _pdb_line(3, "MET", "B", 1, 2.0),
                _pdb_line(4, "LYS", "B", 2, 3.0),
            ]
        )
        + "\n"
    )

    assert observed_chain_for_sequence(pdb, "MK") is None
    row = complex_chain_metadata(
        binder_sequence="MK",
        structure_path=pdb,
        binder_chain_id="C",
    )[0]

    assert "observed_chain_id" not in row
