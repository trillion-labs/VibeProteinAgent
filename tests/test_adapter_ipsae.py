"""Tests for the ipSAE adapter (CPU-only, vendored script).

The real test runs the full ipsae.py script end-to-end on a small AF2
multimer fixture. It is fast enough (sub-30s on a typical machine) not to
need the ``slow`` marker, but skips gracefully if the bundled gzipped PAE
JSON or the PDB are missing.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

import vibeproteinagent.adapters.ipsae.adapter as ipsae_adapter
from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.ipsae.adapter import (
    _IPSAE_SCRIPT,
    IpSaeAdapter,
    IpSaeInput,
    IpSaeOutput,
    _parse_scoreboard,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PAE_GZ = _PROJECT_ROOT / "tests" / "fixtures" / "ipsae_af2_pae.json.gz"
_PDB = _PROJECT_ROOT / "tests" / "fixtures" / "ipsae_af2_complex.pdb"


def test_descriptor_fields():
    d = IpSaeAdapter.descriptor
    assert d.name == "ipsae"
    assert d.category.value == "scoring"
    assert d.gpu_required is False
    assert d.pin_warm is True


def test_vendored_script_exists():
    assert _IPSAE_SCRIPT.exists(), f"missing {_IPSAE_SCRIPT}"


def test_input_defaults():
    p = IpSaeInput(pae_path="/tmp/a.json", pdb_path="/tmp/b.pdb", chain_a="A", chain_b="B")
    assert p.cutoff == 10.0
    assert p.dist_cutoff == 10.0


def test_input_rejects_empty_chain():
    with pytest.raises(ValidationError):
        IpSaeInput(pae_path="/tmp/a.json", pdb_path="/tmp/b.pdb", chain_a="", chain_b="B")


def test_output_round_trip():
    out = IpSaeOutput(
        ipsae_score=0.44,
        ipsae_asym_ab=0.42,
        ipsae_asym_ba=0.44,
        ipsae_max=0.44,
        interface_residues_a=[1, 2, 3],
        interface_residues_b=[10, 11],
        scoreboard_path="/tmp/s.txt",
        byres_path="/tmp/s_byres.txt",
        runtime_ms=12,
    )
    rebuilt = IpSaeOutput.model_validate(out.model_dump())
    assert rebuilt == out


def test_parse_scoreboard_minimal():
    text = (
        "Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  ipTM_d0chn  "
        "pDockQ     pDockQ2    LIS   n0res n0chn n0dom d0res d0chn "
        "d0dom nres1 nres2 dist1 dist2 Model\n"
        "A    B     10   10   asym  0.5759   0.83        0.69         0.53     0.36        "
        "0.20       0.04       0.36  289   1571  548   6.25  12.57 "
        "8.25  257   291    92    91   model1\n"
    )
    rows = _parse_scoreboard(text)
    assert len(rows) == 1
    assert rows[0]["Chn1"] == "A"
    assert rows[0]["Chn2"] == "B"
    assert rows[0]["Type"] == "asym"
    assert float(rows[0]["ipSAE"]) == 0.5759


def test_fractional_cutoffs_are_passed_to_upstream_without_rounding(tmp_path, monkeypatch):
    pae = tmp_path / "pae.json"
    pdb = tmp_path / "model.pdb"
    pae.write_text("{}")
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  "
        "1.00 90.00           C\n"
    )
    seen_cmd: list[str] = []

    def fake_run(cmd, **kwargs):
        seen_cmd.extend(cmd)
        staged_pdb = Path(cmd[3])
        pae_string = f"{int(float(cmd[4])):02d}"
        dist_string = f"{int(float(cmd[5])):02d}"
        stem = staged_pdb.with_suffix("")
        scoreboard = stem.parent / f"{stem.name}_{pae_string}_{dist_string}.txt"
        scoreboard.write_text(
            "Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom  ipTM_af  "
            "ipTM_d0chn  pDockQ     pDockQ2    LIS   n0res n0chn n0dom d0res d0chn "
            "d0dom nres1 nres2 dist1 dist2 Model\n"
            "A    B     10   12   asym  0.5759   0.83        0.69         0.53     "
            "0.36        0.20       0.04       0.36  289   1571  548   6.25  "
            "12.57 8.25  257   291    92    91   model1\n"
            "B    A     10   12   asym  0.6123   0.84        0.70         0.53     "
            "0.37        0.20       0.04       0.36  291   1571  548   6.27  "
            "12.57 8.25  257   291    92    91   model1\n"
            "A    B     10   12   max    0.6123   0.84        0.70         0.53     "
            "0.37        0.20       0.04       0.36  291   1571  548   6.27  "
            "12.57 8.25  257   291    92    91   model1\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(ipsae_adapter.subprocess, "run", fake_run)
    adapter = IpSaeAdapter(config={}, weights_dir=Path("/nonexistent"), device=torch.device("cpu"))
    out = adapter.run(
        IpSaeInput(
            pae_path=str(pae),
            pdb_path=str(pdb),
            chain_a="A",
            chain_b="B",
            cutoff=10.5,
            dist_cutoff=12.25,
        ),
        null_context(tmp_path / "job"),
    )

    assert seen_cmd[-2:] == ["10.5", "12.25"]
    assert Path(out.scoreboard_path).name == "model_10_12.txt"
    assert out.ipsae_score == pytest.approx(0.6123)
    assert out.ipsae_asym_ab == pytest.approx(0.5759)
    assert out.ipsae_asym_ba == pytest.approx(0.6123)
    assert out.ipsae_max == pytest.approx(0.6123)


@pytest.mark.skipif(
    not (_PAE_GZ.exists() and _PDB.exists()),
    reason="ipSAE fixtures not present",
)
def test_run_end_to_end(tmp_path):
    a = IpSaeAdapter(config={}, weights_dir=Path("/nonexistent"), device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(
        IpSaeInput(
            pae_path=str(_PAE_GZ),
            pdb_path=str(_PDB),
            chain_a="A",
            chain_b="B",
            cutoff=15.0,
            dist_cutoff=15.0,
        ),
        null_context(job_dir),
    )
    assert 0.0 <= out.ipsae_score <= 1.0
    assert out.ipsae_score == out.ipsae_max
    assert out.ipsae_max == max(out.ipsae_asym_ab, out.ipsae_asym_ba)
    # 257-ish interface residues on chain A in the AF2 fixture.
    assert len(out.interface_residues_a) > 0
    assert len(out.interface_residues_b) > 0
    assert Path(out.scoreboard_path).exists()
    assert Path(out.byres_path).exists()
    assert out.runtime_ms >= 0
