"""Tests for the Protein-QC first-party adapter."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.protein_qc.adapter import (
    ProteinQCAdapter,
    ProteinQCInput,
    ProteinQCOutput,
    _polybasic_hits,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PDB_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "6MRR.pdb"
_CIF_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "5o45.cif"


def test_descriptor_fields():
    d = ProteinQCAdapter.descriptor
    assert d.name == "protein_qc"
    assert d.category.value == "scoring"
    assert d.gpu_required is False
    assert d.pin_warm is True


def test_input_requires_at_least_one():
    with pytest.raises(ValidationError):
        ProteinQCInput()
    # sequence-only is fine.
    ProteinQCInput(sequence="MKKL")
    # pdb-only is fine.
    ProteinQCInput(pdb_path="/tmp/x.pdb")


def test_polybasic_helper():
    assert _polybasic_hits("MKKK") == [1]
    assert _polybasic_hits("MKR") == []
    assert _polybasic_hits("AAAKRKKMAAARRR") == [3, 11]


def test_sequence_only_smoke(tmp_path):
    a = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    seq = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFKKKR"
    out = a.run(ProteinQCInput(sequence=seq), null_context(tmp_path))
    assert "gravy" in out.metrics
    assert "polybasic_hits" in out.metrics
    assert out.metrics["polybasic_hits"] == 1.0
    assert "contains_polybasic_cluster" in out.flags
    assert out.metrics["plddt"] is None  # no PDB → pLDDT unset
    assert out.runtime_ms >= 0


def test_selected_metrics_only(tmp_path):
    a = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = a.run(
        ProteinQCInput(sequence="MKKL", metrics=["gravy", "pi"]),
        null_context(tmp_path),
    )
    assert set(out.metrics) >= {"gravy", "pi"}
    assert "instability_index" not in out.metrics
    assert "plddt" not in out.metrics


def test_plddt_from_unit_interval_bfactor_is_persisted_on_percent_scale(tmp_path):
    pdb_path = tmp_path / "unit_plddt.pdb"
    pdb_path.write_text(
        "ATOM      1  CA  ALA A   1      11.104  13.207   2.100  1.00  0.98           C\n"
        "TER\n"
        "END\n"
    )
    a = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = a.run(
        ProteinQCInput(pdb_path=str(pdb_path), metrics=["plddt"]),
        null_context(tmp_path),
    )

    assert out.metrics["plddt"] == pytest.approx(98.0)
    assert out.metrics["plddt_source"] == "ca_bfactor_plddt"
    assert out.metrics["ca_bfactor_plddt"] == pytest.approx(98.0)
    assert out.metrics["atom_bfactor_plddt"] == pytest.approx(98.0)
    assert "low_plddt" not in out.flags
    assert out.candidates[0]["scores"]["protein_qc"]["plddt"] == pytest.approx(98.0)


def test_canonical_protenix_global_plddt_wins_over_ca_bfactor(tmp_path):
    pdb_path = tmp_path / "protenix_like.pdb"
    pdb_path.write_text(
        "ATOM      1  CA  ALA A   1      11.104  13.207   2.100  1.00 83.36           C\n"
        "ATOM      2  CB  ALA A   1      11.904  14.207   2.100  1.00 75.66           C\n"
        "TER\n"
        "END\n"
    )
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": "v40",
        "source": {"tool": "esm2", "local_id": "v40", "index": 0},
        "sequence": "A",
        "structure_path": str(pdb_path),
        "scores": {"protenix_v2": {"plddt": 79.512}},
        "artifacts": {},
    }
    adapter = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        ProteinQCInput(candidates=[candidate], metrics=["plddt"]),
        null_context(tmp_path),
    )

    assert out.metrics["plddt"] == pytest.approx(79.512)
    assert out.metrics["plddt_source"] == "scores.protenix_v2.plddt"
    assert out.metrics["ca_bfactor_plddt"] == pytest.approx(83.36)
    assert out.metrics["atom_bfactor_plddt"] == pytest.approx(79.51)
    qc_scores = out.candidates[0]["scores"]["protein_qc"]
    assert qc_scores["plddt"] == pytest.approx(79.512)
    assert qc_scores["plddt_source"] == "scores.protenix_v2.plddt"
    assert qc_scores["ca_bfactor_plddt"] == pytest.approx(83.36)


@pytest.mark.skipif(not _PDB_FIXTURE.exists(), reason="6MRR fixture missing")
def test_pdb_path_extracts_plddt(tmp_path):
    a = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = a.run(
        ProteinQCInput(pdb_path=str(_PDB_FIXTURE), metrics=["plddt", "gravy"]),
        null_context(tmp_path),
    )
    assert out.metrics["plddt"] is not None
    assert out.metrics["gravy"] is not None


@pytest.mark.skipif(not _CIF_FIXTURE.exists(), reason="5o45 CIF fixture missing")
def test_mmcif_path_extracts_plddt(tmp_path):
    a = ProteinQCAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = a.run(
        ProteinQCInput(pdb_path=str(_CIF_FIXTURE), metrics=["plddt"]),
        null_context(tmp_path),
    )
    assert out.metrics["plddt"] is not None


def test_output_round_trip():
    out = ProteinQCOutput(
        metrics={"gravy": -0.5, "pi": 6.5, "instability_index": 30.0},
        flags=["high_GRAVY"],
        runtime_ms=12,
    )
    rebuilt = ProteinQCOutput.model_validate(out.model_dump())
    assert rebuilt == out
