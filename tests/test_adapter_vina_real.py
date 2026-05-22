"""Tests for the Vina binary-wrapper adapter."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.vina.adapter import (
    VinaAdapter,
    VinaInput,
    VinaMode,
    VinaOutput,
    _find_binary,
    _parse_modes,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RECEPTOR = _PROJECT_ROOT / "tests" / "fixtures" / "vina_receptor.pdbqt"
_LIGAND = _PROJECT_ROOT / "tests" / "fixtures" / "vina_ligand.pdbqt"


def _binary_available() -> bool:
    try:
        _find_binary()
        return True
    except FileNotFoundError:
        return False


def test_descriptor_fields():
    d = VinaAdapter.descriptor
    assert d.name == "vina"
    assert d.category.value == "docking"
    assert d.backend.value == "binary"
    assert d.gpu_required is False
    assert d.pin_warm is True


def test_input_defaults():
    p = VinaInput(
        receptor_pdbqt="/tmp/r.pdbqt",
        ligand_pdbqt="/tmp/l.pdbqt",
        center=[0.0, 0.0, 0.0],
        box_size=[20.0, 20.0, 20.0],
    )
    assert p.exhaustiveness == 8
    assert p.num_modes == 9
    assert p.seed is None


def test_input_rejects_wrong_center_length():
    with pytest.raises(ValidationError):
        VinaInput(
            receptor_pdbqt="/tmp/r.pdbqt",
            ligand_pdbqt="/tmp/l.pdbqt",
            center=[0.0, 0.0],
            box_size=[20.0, 20.0, 20.0],
        )


def test_input_rejects_nonpositive_box_size():
    with pytest.raises(ValidationError):
        VinaInput(
            receptor_pdbqt="/tmp/r.pdbqt",
            ligand_pdbqt="/tmp/l.pdbqt",
            center=[0.0, 0.0, 0.0],
            box_size=[0.0, 20.0, 20.0],
        )


def test_parse_modes_table():
    stdout = (
        "Reading input ... done.\n"
        "Performing docking (random seed: 42) ...\n"
        "mode |   affinity | dist from best mode\n"
        "     | (kcal/mol) | rmsd l.b.| rmsd u.b.\n"
        "-----+------------+----------+----------\n"
        "   1       -8.123      0.000      0.000\n"
        "   2       -7.880      1.234      2.001\n"
        "   3       -7.500      2.100      3.300\n"
        "\n"
        "(trailing noise)\n"
    )
    modes = _parse_modes(stdout)
    assert len(modes) == 3
    assert modes[0].rank == 1
    assert modes[0].affinity_kcal_mol == -8.123
    assert modes[0].rmsd_lb == 0.0
    assert modes[2].affinity_kcal_mol == -7.5


def test_output_round_trip():
    out = VinaOutput(
        docked_pdbqt="/tmp/d.pdbqt",
        best_affinity_kcal_mol=-8.1,
        modes=[VinaMode(rank=1, affinity_kcal_mol=-8.1, rmsd_lb=0.0, rmsd_ub=0.0)],
        runtime_ms=1234,
    )
    assert VinaOutput.model_validate(out.model_dump()) == out


@pytest.mark.slow
@pytest.mark.skipif(not _binary_available(), reason="vina binary unavailable")
@pytest.mark.skipif(
    not (_RECEPTOR.exists() and _LIGAND.exists()),
    reason="vina pdbqt fixtures not bundled",
)
def test_run_basic_dock(tmp_path):
    a = VinaAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    # Box centred on the 1iep STI binding pocket — values from the bundled
    # AutoDock-Vina v1.2.5 example/basic_docking/solution/1iep_receptor_vina_box.txt.
    out = a.run(
        VinaInput(
            receptor_pdbqt=str(_RECEPTOR),
            ligand_pdbqt=str(_LIGAND),
            center=[15.190, 53.903, 16.917],
            box_size=[20.0, 20.0, 20.0],
            exhaustiveness=1,
            num_modes=3,
            seed=42,
        ),
        null_context(job_dir),
    )
    assert Path(out.docked_pdbqt).is_file()
    assert len(out.modes) >= 1
    assert out.best_affinity_kcal_mol <= out.modes[0].affinity_kcal_mol
