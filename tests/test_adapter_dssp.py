"""Tests for the DSSP binary-wrapper adapter (CPU-only)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.dssp.adapter import (
    DSSPAdapter,
    DSSPInput,
    DSSPOutput,
    _find_binary,
    _parse_dssp_ss,
    _run_dssp_binary,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "6MRR.pdb"


def _binary_available() -> bool:
    try:
        _find_binary()
        return True
    except FileNotFoundError:
        return False


def test_descriptor_fields():
    d = DSSPAdapter.descriptor
    assert d.name == "dssp"
    assert d.category.value == "misc"
    assert d.backend.value == "binary"
    assert d.gpu_required is False
    assert d.pin_warm is True


def test_input_defaults():
    p = DSSPInput(pdb_path="/tmp/a.pdb")
    assert p.format == "json"


def test_input_rejects_bad_format():
    with pytest.raises(ValidationError):
        DSSPInput(pdb_path="/tmp/a.pdb", format="weird")


def test_parse_dssp_ss_minimal():
    body = (
        "  #  RESIDUE AA STRUCTURE BP1 BP2  ACC     N-H-->O    O-->H-N    N-H-->O    O-->H-N    TCO  KAPPA ALPHA  PHI   PSI    X-CA   Y-CA   Z-CA\n"  # noqa: E501
        "    1    1 A G              0   0  124      0, 0.0     2,-0.4     0, 0.0    30,-0.2   0.000 360.0 360.0 360.0  37.9  -15.5    3.4   11.9\n"  # noqa: E501
        "    2    2 A W  H  S    0   0  191     28,-0.1     2,-0.7    29,-0.1    28,-0.0  -0.913 360.0-149.8-121.6 125.4  -12.2    3.5    9.9\n"  # noqa: E501
        "    3    3 A E  E         0   0  120      0, 0.0     0, 0.0     0, 0.0     0, 0.0   0.000 360.0 360.0 360.0  37.9  -15.5    3.4   11.9\n"  # noqa: E501
    )
    ss = _parse_dssp_ss(body)
    assert ss == "-HE"


def test_run_dssp_binary_falls_back_to_legacy_cli(monkeypatch, tmp_path):
    pdb = tmp_path / "input.pdb"
    out = tmp_path / "out.dssp"
    pdb.write_text("HEADER test\n")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "--output-format" in cmd:
            return subprocess.CompletedProcess(cmd, 1, "", "unknown option output-format")
        out.write_text("legacy dssp\n")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    _run_dssp_binary("mkdssp", pdb, out, "dssp")

    assert calls[0][:3] == ["mkdssp", "--output-format", "dssp"]
    assert calls[1] == ["mkdssp", "-i", str(pdb), "-o", str(out)]
    assert out.read_text() == "legacy dssp\n"


def test_output_round_trip():
    out = DSSPOutput(
        dssp_output_path="/tmp/x.dssp.json",
        secondary_structure_string="HHHEEE--",
        helix_fraction=0.375,
        sheet_fraction=0.375,
        coil_fraction=0.25,
        num_residues=8,
        runtime_ms=42,
    )
    rebuilt = DSSPOutput.model_validate(out.model_dump())
    assert rebuilt == out


@pytest.mark.skipif(not _binary_available(), reason="mkdssp/dssp binary unavailable")
@pytest.mark.skipif(not _FIXTURE.exists(), reason="6MRR fixture missing")
def test_run_on_6mrr(tmp_path):
    a = DSSPAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(DSSPInput(pdb_path=str(_FIXTURE), format="json"), null_context(job_dir))
    assert out.num_residues >= 30
    assert len(out.secondary_structure_string) == out.num_residues
    fractions = out.helix_fraction + out.sheet_fraction + out.coil_fraction
    assert abs(fractions - 1.0) < 1e-6
    assert out.helix_fraction > 0.0  # 6MRR is helix-rich.
    assert Path(out.dssp_output_path).exists()


@pytest.mark.skipif(not _binary_available(), reason="mkdssp/dssp binary unavailable")
@pytest.mark.skipif(not _FIXTURE.exists(), reason="6MRR fixture missing")
def test_run_classic_dssp_format(tmp_path):
    a = DSSPAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(DSSPInput(pdb_path=str(_FIXTURE), format="dssp"), null_context(job_dir))
    assert out.dssp_output_path.endswith(".dssp")
    assert "RESIDUE" in Path(out.dssp_output_path).read_text()
