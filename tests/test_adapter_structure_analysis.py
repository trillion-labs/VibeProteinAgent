from __future__ import annotations

from pathlib import Path

import pytest

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.base import torch
from vibeproteinagent.adapters.structure_analysis import adapter as structure_adapter
from vibeproteinagent.adapters.structure_analysis.adapter import (
    StructureAnalysisAdapter,
    StructureAnalysisInput,
    _parse_structure,
)
from vibeproteinagent.core.binary import find_binary

_PDB = Path(__file__).resolve().parent / "fixtures" / "6MRR.pdb"
_CIF = Path(__file__).resolve().parent / "fixtures" / "5o45.cif"


def _ca_line(serial: int, resname: str, chain: str, resseq: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}"
        f"    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
    )


def _atom_line(
    serial: int,
    record: str,
    atom: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float = 0.0,
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:<4s} {resname:>3s} {chain}{resseq:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def _atom_line_with_icode(
    serial: int,
    record: str,
    atom: str,
    resname: str,
    chain: str,
    resseq: int,
    icode: str,
    x: float,
    y: float,
    z: float = 0.0,
    element: str = "C",
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:<4s} {resname:>3s} {chain}{resseq:4d}{icode:1s}"
        f"   {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def _dssp_available() -> bool:
    try:
        find_binary(("mkdssp", "dssp"), env_override="VPA_DSSP_BINARY")
    except FileNotFoundError:
        return False
    return True


def _hetatm_standard_aa_cif(tmp_path: Path) -> Path:
    path = tmp_path / "hetatm_standard_aa.cif"
    path.write_text(
        "\n".join(
            [
                "data_hetatm_standard_aa",
                "loop_",
                "_atom_site.group_PDB",
                "_atom_site.label_atom_id",
                "_atom_site.label_comp_id",
                "_atom_site.label_asym_id",
                "_atom_site.label_seq_id",
                "_atom_site.Cartn_x",
                "_atom_site.Cartn_y",
                "_atom_site.Cartn_z",
                "HETATM N ALA A 1 0.0 0.0 0.0",
                "HETATM CA ALA A 1 1.0 0.0 0.0",
                "HETATM C ALA A 1 2.0 0.0 0.0",
                "HETATM O ALA A 1 3.0 0.0 0.0",
                "HETATM N GLY A 2 4.0 0.0 0.0",
                "HETATM CA GLY A 2 5.0 0.0 0.0",
                "HETATM C GLY A 2 6.0 0.0 0.0",
                "HETATM O GLY A 2 7.0 0.0 0.0",
                "HETATM C1 LIG L 1 5.0 1.0 0.0",
                "#",
            ]
        )
        + "\n"
    )
    return path


def test_structure_analysis_pdb_fixture(tmp_path):
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        StructureAnalysisInput(structure_path=str(_PDB), max_annotations=5),
        null_context(tmp_path),
    )
    assert out.chains
    assert sum(out.burial.values()) > 0
    assert Path(out.artifacts["summary_json"]).exists()
    assert Path(out.artifacts["residue_annotations_tsv"]).exists()
    assert out.annotation_scope["summary_scope"] == "all_residues_after_chain_filter"


def test_structure_analysis_positions_do_not_narrow_global_summary(tmp_path):
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    full = adapter.run(
        StructureAnalysisInput(structure_path=str(_PDB), chain_id="A", max_annotations=5),
        null_context(tmp_path / "full"),
    )
    subset = adapter.run(
        StructureAnalysisInput(
            structure_path=str(_PDB),
            chain_id="A",
            positions=[1],
            max_annotations=5,
        ),
        null_context(tmp_path / "subset"),
    )
    assert subset.burial == full.burial
    assert subset.annotation_scope["annotation_scope"] == "requested_positions"
    assert subset.annotation_scope["summary_residue_count"] == sum(full.burial.values())
    assert subset.annotation_scope["annotated_residue_count"] <= 1


def test_structure_analysis_maps_requested_sequence_positions_to_structure_residues(tmp_path):
    pdb_path = tmp_path / "offset_numbering.pdb"
    pdb_path.write_text(
        "".join(
            [
                _ca_line(1, "ALA", "A", 5, 0.0),
                _ca_line(2, "CYS", "A", 6, 4.0),
                _ca_line(3, "ASP", "A", 7, 8.0),
            ]
        )
    )
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        StructureAnalysisInput(
            structure_path=str(pdb_path),
            sequence=">full\nMACD\n",
            chain_id="A",
            positions=[1, 2, 4],
            run_dssp=False,
        ),
        null_context(tmp_path),
    )

    scope = out.annotation_scope
    assert scope["position_index_basis"] == "sequence_alignment_substring"
    assert scope["annotated_residue_count"] == 2
    assert scope["missing_positions"] == [1]
    assert {
        item["requested_position"]: item["resseq"]
        for item in scope["mapped_positions"]
    } == {2: 5, 4: 7}
    assert any(
        "requested positions missing from parsed structure after sequence alignment: 1" in warning
        for warning in out.warnings
    )
    residue_tsv = Path(out.artifacts["residue_annotations_tsv"]).read_text()
    assert "\t5\t\tA\tALA\t" in residue_tsv
    assert "\t7\t\tD\tASP\t" in residue_tsv
    assert "\t6\t\tC\tCYS\t" not in residue_tsv


def test_structure_analysis_rejects_ambiguous_multichain_sequence_mapping(tmp_path):
    pdb_path = tmp_path / "two_chains.pdb"
    pdb_path.write_text(
        "".join(
            [
                _ca_line(1, "ALA", "A", 5, 0.0),
                _ca_line(2, "CYS", "B", 5, 4.0),
            ]
        )
    )
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    with pytest.raises(ValueError, match="requires chain_id"):
        adapter.run(
            StructureAnalysisInput(
                structure_path=str(pdb_path),
                sequence="AC",
                positions=[1],
                run_dssp=False,
            ),
            null_context(tmp_path),
        )


def test_structure_analysis_mmcif_fixture(tmp_path):
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        StructureAnalysisInput(structure_path=str(_CIF), chain_id="A", max_annotations=5),
        null_context(tmp_path),
    )
    assert out.chains == ["A"]
    assert out.residue_counts["A"] > 30
    assert Path(out.artifacts["dssp_tsv"]).exists()


def test_structure_analysis_mmcif_standard_aa_hetatm_is_residue_not_ligand(tmp_path):
    cif = _hetatm_standard_aa_cif(tmp_path)
    residues, ligands = _parse_structure(cif)

    assert [(res.chain_id, res.resseq, res.resname) for res in residues] == [
        ("A", 1, "ALA"),
        ("A", 2, "GLY"),
    ]
    assert [lig.resname for lig in ligands] == ["LIG"]


def test_structure_analysis_falls_back_after_mkdssp_failure_for_hetatm_standard_aa(
    monkeypatch,
    tmp_path,
):
    cif = _hetatm_standard_aa_cif(tmp_path)

    def fail_dssp(*_args, **_kwargs):
        return None, None, "DSSP execution failed: mocked exit 1"

    def fake_pydssp(residues):
        return {residue.key: "loop" for residue in residues}, None

    monkeypatch.setattr(structure_adapter, "_run_dssp", fail_dssp)
    monkeypatch.setattr(structure_adapter, "_run_pydssp", fake_pydssp)

    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        StructureAnalysisInput(structure_path=str(cif), run_dssp=True),
        null_context(tmp_path / "job"),
    )

    assert out.chains == ["A"]
    assert out.residue_counts == {"A": 2}
    assert out.ligand_contacts["contact_count"] == 2
    assert out.secondary_structure["loop"] == 2
    assert any("used pydssp" in warning for warning in out.warnings)


def test_structure_analysis_ligand_contacts_use_heavy_atoms_not_ca(tmp_path):
    pdb_path = tmp_path / "sidechain_ligand_contact.pdb"
    pdb_path.write_text(
        "".join(
            [
                _atom_line(1, "ATOM", "CA", "ALA", "A", 1, 20.0, 0.0),
                _atom_line(2, "ATOM", "CB", "ALA", "A", 1, 0.0, 0.0),
                _atom_line(3, "HETATM", "C1", "LIG", "L", 1, 1.0, 0.0),
                _atom_line(4, "HETATM", "O1", "LIG", "L", 1, 8.0, 0.0, element="O"),
                _atom_line(5, "HETATM", "H1", "LIG", "L", 1, 0.5, 0.0, element="H"),
            ]
        )
    )
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        StructureAnalysisInput(
            structure_path=str(pdb_path),
            ligand_resnames=["LIG"],
            contact_distance_angstrom=2.0,
            run_dssp=False,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.interface["ligand_contact_count"] == 1
    assert out.ligand_contacts["contact_count"] == 1
    assert out.ligand_contacts["contact_residue_count"] == 1
    assert out.ligand_contacts["ligand_heavy_atom_count"] == 2
    assert out.ligand_contacts["contact_ligand_atom_count"] == 1
    assert out.ligand_pocket["ligand_heavy_atom_count"] == 2
    assert out.ligand_pocket["ligand_buried_atom_count"] == 1
    assert out.ligand_pocket["ligand_buried_fraction"] == pytest.approx(0.5)
    ligand_tsv = Path(out.artifacts["ligand_contacts_tsv"]).read_text()
    assert "\tCB\tC1\t1.0\t" in ligand_tsv
    assert "H1" not in ligand_tsv


def test_structure_analysis_ligand_legacy_counts_are_contacting_residues(tmp_path):
    pdb_path = tmp_path / "multi_ligand_residue_contact.pdb"
    pdb_path.write_text(
        "".join(
            [
                _atom_line(1, "ATOM", "CA", "ALA", "A", 1, 20.0, 0.0),
                _atom_line(2, "ATOM", "CB", "ALA", "A", 1, 0.0, 0.0),
                _atom_line(3, "HETATM", "C1", "LIG", "L", 1, 1.0, 0.0),
                _atom_line(4, "HETATM", "C2", "LIG", "L", 2, 1.2, 0.0),
            ]
        )
    )
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        StructureAnalysisInput(
            structure_path=str(pdb_path),
            ligand_resnames=["LIG"],
            contact_distance_angstrom=2.0,
            run_dssp=False,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.interface["ligand_contact_count"] == 1
    assert out.ligand_contacts["contact_count"] == 1
    assert out.ligand_contacts["contact_residue_count"] == 1
    assert out.ligand_contacts["contact_atom_pair_count"] == 2
    assert out.ligand_contacts["contact_ligand_atom_count"] == 2
    assert out.ligand_pocket["pocket_residue_count"] == 1
    assert out.ligand_pocket["ligand_buried_atom_count"] == 2


def test_structure_analysis_ligand_counts_distinct_insertion_code_residues(tmp_path):
    pdb_path = tmp_path / "icode_ligand_contacts.pdb"
    pdb_path.write_text(
        "".join(
            [
                _atom_line_with_icode(1, "ATOM", "CB", "ALA", "A", 10, "A", 0.0, 0.0),
                _atom_line_with_icode(2, "ATOM", "CB", "GLY", "A", 10, "B", 0.0, 1.0),
                _atom_line(3, "HETATM", "C1", "LIG", "L", 1, 1.0, 0.0),
                _atom_line(4, "HETATM", "C2", "LIG", "L", 1, 1.0, 1.0),
            ]
        )
    )
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        StructureAnalysisInput(
            structure_path=str(pdb_path),
            ligand_resnames=["LIG"],
            contact_distance_angstrom=2.0,
            run_dssp=False,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.interface["ligand_contact_count"] == 2
    assert out.ligand_contacts["contact_count"] == 2
    assert out.ligand_contacts["contact_residue_count"] == 2
    assert out.ligand_pocket["pocket_residue_count"] == 2
    ligand_tsv = Path(out.artifacts["ligand_contacts_tsv"]).read_text()
    assert ligand_tsv.splitlines()[0].split("\t")[:3] == ["chain_id", "resseq", "icode"]
    assert "\t10\tA\tLIG\t" in ligand_tsv
    assert "\t10\tB\tLIG\t" in ligand_tsv


def test_structure_analysis_dssp_input_sanitizes_standard_aa_hetatm(tmp_path):
    cif = _hetatm_standard_aa_cif(tmp_path)
    residues, _ligands = _parse_structure(cif)

    sanitized = structure_adapter._write_dssp_input_pdb(residues, tmp_path, "sanitized")
    text = sanitized.read_text()

    assert "HETATM" not in text
    assert "ATOM" in text
    assert " ALA " in text
    assert " GLY " in text
    assert " LIG " not in text


@pytest.mark.slow
@pytest.mark.skipif(
    not _dssp_available(),
    reason="mkdssp/dssp binary unavailable",
)
def test_structure_analysis_uses_real_dssp_when_available(tmp_path):
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        StructureAnalysisInput(structure_path=str(_PDB), max_annotations=5, run_dssp=True),
        null_context(tmp_path),
    )
    assert "raw_dssp" in out.artifacts
    assert Path(out.artifacts["raw_dssp"]).exists()
    assert any("DSSP" in warning for warning in out.warnings)


@pytest.mark.slow
def test_structure_analysis_uses_pydssp_when_available(tmp_path):
    pytest.importorskip("pydssp")
    adapter = StructureAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        StructureAnalysisInput(structure_path=str(_PDB), max_annotations=5, run_dssp=True),
        null_context(tmp_path),
    )
    assert out.secondary_structure["helix"] > 0
    assert "dssp_tsv" in out.artifacts
