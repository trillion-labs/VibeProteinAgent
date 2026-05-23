from __future__ import annotations

from pathlib import Path

import pytest

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.base import torch
from vibeproteinagent.adapters.structure_analysis.adapter import (
    StructureAnalysisAdapter,
    StructureAnalysisInput,
)
from vibeproteinagent.core.binary import find_binary

_PDB = Path(__file__).resolve().parent / "fixtures" / "6MRR.pdb"
_CIF = Path(__file__).resolve().parent / "fixtures" / "5o45.cif"


def _ca_line(serial: int, resname: str, chain: str, resseq: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}"
        f"    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
    )


def _dssp_available() -> bool:
    try:
        find_binary(("mkdssp", "dssp"), env_override="VPA_DSSP_BINARY")
    except FileNotFoundError:
        return False
    return True


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
