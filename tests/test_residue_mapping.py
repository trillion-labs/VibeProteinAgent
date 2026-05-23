from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from pathlib import Path

from jsonschema import Draft202012Validator

from vibeproteinagent.reporting.collect import _warnings_from_job
from vibeproteinagent.adapters.residue_mapping import (
    parse_structure_residue_refs,
    resolve_position_range,
    resolve_positions,
)


def _ca_line(serial: int, resname: str, chain: str, resseq: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}"
        f"    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
    )


def _offset_pdb(tmp_path: Path) -> Path:
    pdb = tmp_path / "offset.pdb"
    pdb.write_text(
        "".join(
            [
                _ca_line(1, "ALA", "A", 5, 0.0),
                _ca_line(2, "CYS", "A", 6, 4.0),
                _ca_line(3, "ASP", "A", 7, 8.0),
            ]
        )
    )
    return pdb


def test_sequence_positions_map_to_structure_residue_ids(tmp_path):
    pdb = _offset_pdb(tmp_path)
    refs = parse_structure_residue_refs(pdb, "A")

    result = resolve_positions(
        residues=refs,
        positions=[1, 2, 4],
        chain_id="A",
        sequence="MACD",
        position_basis="auto",
        label="target_hotspots",
    )

    assert result.position_index_basis == "sequence_alignment_substring"
    assert result.residue_ids == ["A5", "A7"]
    assert result.missing_positions == [1]
    assert {
        item["requested_position"]: item["residue_id"]
        for item in result.mapped_positions
    } == {2: "A5", 4: "A7"}


def test_sequence_ranges_collapse_to_structure_ranges(tmp_path):
    pdb = _offset_pdb(tmp_path)
    refs = parse_structure_residue_refs(pdb, "A")

    span, result = resolve_position_range(
        residues=refs,
        range_text="2-4",
        chain_id="A",
        sequence="MACD",
        position_basis="auto",
        label="target_residues",
    )

    assert span == "A5-7"
    assert result.residue_ids == ["A5", "A6", "A7"]


def test_legacy_structure_basis_still_passes_structure_numbers(tmp_path):
    pdb = _offset_pdb(tmp_path)
    refs = parse_structure_residue_refs(pdb, "A")

    result = resolve_positions(
        residues=refs,
        positions=[5, 7],
        chain_id="A",
        position_basis="auto",
        label="target_hotspots",
    )

    assert result.position_index_basis == "structure_resseq"
    assert result.residue_ids == ["A5", "A7"]
    assert any("no sequence/fasta" in warning for warning in result.warnings)


def test_proteina_complexa_schema_requires_one_pocket_or_motif_selector():
    schema = Path(
        "src/vibeproteinagent/adapters/proteina_complexa/schemas/proteina_complexa.input.json"
    )
    validator = Draft202012Validator(json_schema := json.loads(schema.read_text()))
    Draft202012Validator.check_schema(json_schema)

    ligand_errors = list(
        validator.iter_errors(
            {
                "mode": "ligand_binder",
                "target_pdb": "target.pdb",
                "ligand_smiles": "CCO",
            }
        )
    )
    assert ligand_errors
    validator.validate(
        {
            "mode": "ligand_binder",
            "target_pdb": "target.pdb",
            "ligand_smiles": "CCO",
            "target_pocket_residue_ids": ["A10"],
        }
    )

    motif_errors = list(
        validator.iter_errors(
            {
                "mode": "motif",
                "motif_pdb": "motif.pdb",
                "total_length": 80,
            }
        )
    )
    assert motif_errors
    validator.validate(
        {
            "mode": "motif",
            "motif_pdb": "motif.pdb",
            "total_length": 80,
            "motif_residue_ids": ["A5"],
        }
    )


class _DummyTorch:
    @staticmethod
    def no_grad():
        def decorate(fn):
            return fn

        return decorate

    @staticmethod
    def device(name):
        return name


def _install_dummy_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _DummyTorch())
    for name in list(sys.modules):
        if name.startswith("vibeproteinagent.adapters.rfdiffusion") or name.startswith(
            "vibeproteinagent.adapters.rfpeptides"
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_explicit_hotspot_ids_drive_rfdiffusion_command(monkeypatch, tmp_path):
    _install_dummy_torch(monkeypatch)
    from vibeproteinagent.adapters.rfdiffusion.adapter import (
        RFdiffusionInput,
        _build_cmd,
        _build_contigs_with_mappings,
    )

    pdb = _offset_pdb(tmp_path)
    params = RFdiffusionInput(
        mode="binder_design",
        target_pdb=str(pdb),
        target_hotspot_residue_ids=["A5"],
    )

    contigs, mappings, warnings, hotspots, selectors = _build_contigs_with_mappings(params, pdb)
    cmd = _build_cmd(
        params=params,
        output_prefix=tmp_path / "out" / "design",
        contigs=contigs,
        target_pdb=pdb,
        target_ss=None,
        target_adj=None,
        checkpoint=tmp_path / "Base_ckpt.pt",
        resolved_hotspots=hotspots,
        resolved_selectors=selectors,
    )

    assert warnings == []
    assert hotspots == ["A5"]
    assert mappings["target_hotspots"]["residue_ids"] == ["A5"]
    assert "ppi.hotspot_res=[A5]" in cmd[2]


def test_explicit_hotspot_ids_drive_rfpeptides_command(monkeypatch, tmp_path):
    _install_dummy_torch(monkeypatch)
    from vibeproteinagent.adapters.rfpeptides.adapter import (
        RFPeptidesInput,
        _build_cmd,
        _build_contigs_with_mappings,
    )

    pdb = _offset_pdb(tmp_path)
    params = RFPeptidesInput(
        target_pdb=str(pdb),
        target_hotspot_residue_ids=["A5"],
    )

    contig, mappings, warnings, hotspots = _build_contigs_with_mappings(params, pdb)
    cmd = _build_cmd(
        params=params,
        output_prefix=tmp_path / "out" / "design",
        target_pdb=pdb,
        contigs=contig,
        weights_dir=tmp_path,
        resolved_hotspots=hotspots,
    )

    assert warnings == []
    assert hotspots == ["A5"]
    assert mappings["target_hotspots"]["residue_ids"] == ["A5"]
    assert "ppi.hotspot_res" in cmd[2]
    assert "A5" in cmd[2]


def test_report_warnings_include_adapter_output_warnings():
    job = SimpleNamespace(
        job_id="job_1",
        error_code=None,
        error_message=None,
        output={"warnings": ["target_hotspots missing from parsed structure after sequence alignment: 1"]},
    )

    warnings = _warnings_from_job(job)

    assert len(warnings) == 1
    assert warnings[0].level == "warning"
    assert warnings[0].source == "job_1"
    assert "target_hotspots missing" in warnings[0].message
