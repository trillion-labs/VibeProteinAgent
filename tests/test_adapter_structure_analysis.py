from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.structure_analysis.adapter import (
    StructureAnalysisAdapter,
    StructureAnalysisInput,
)
from vibeproteinagent.core.binary import find_binary

_PDB = Path(__file__).resolve().parent / "fixtures" / "6MRR.pdb"
_CIF = Path(__file__).resolve().parent / "fixtures" / "5o45.cif"


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
