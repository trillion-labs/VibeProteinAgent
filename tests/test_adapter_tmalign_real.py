"""Tests for the TM-align binary-wrapper adapter."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.tmalign.adapter import (
    TMalignAdapter,
    TMalignInput,
    TMalignOutput,
    _find_binary,
    _parse_tmalign_stdout,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_A = _PROJECT_ROOT / "tests" / "fixtures" / "6MRR.pdb"
_FIXTURE_B = _PROJECT_ROOT / "tests" / "fixtures" / "4GYT.pdb"


def _binary_available() -> bool:
    try:
        _find_binary()
        return True
    except FileNotFoundError:
        return False


def test_descriptor_fields():
    d = TMalignAdapter.descriptor
    assert d.name == "tmalign"
    assert d.category.value == "scoring"
    assert d.backend.value == "binary"
    assert d.gpu_required is False
    assert d.pin_warm is True


def test_input_defaults():
    p = TMalignInput(pdb_a="/tmp/a.pdb", pdb_b="/tmp/b.pdb")
    assert p.normalize_by == "shorter"


def test_input_rejects_bad_normalize_by():
    with pytest.raises(ValidationError):
        TMalignInput(pdb_a="/tmp/a.pdb", pdb_b="/tmp/b.pdb", normalize_by="weird")


def test_parse_minimal_stdout():
    text = (
        "Length of Chain_1: 68 residues\n"
        "Length of Chain_2: 68 residues\n"
        "Aligned length= 68, RMSD=   0.00, Seq_ID=n_identical/n_aligned= 1.000\n"
        "TM-score= 1.00000 (if normalized by length of Chain_1, i.e., LN=68, d0=2.86)\n"
        "TM-score= 1.00000 (if normalized by length of Chain_2, i.e., LN=68, d0=2.86)\n"
    )
    parsed = _parse_tmalign_stdout(text)
    assert parsed["length_a"] == 68
    assert parsed["length_b"] == 68
    assert parsed["aligned_length"] == 68
    assert parsed["rmsd"] == 0.0
    assert parsed["seq_identity"] == 1.0
    assert parsed["tm_a"] == 1.0
    assert parsed["tm_b"] == 1.0


def test_output_round_trip():
    out = TMalignOutput(
        tm_score_normalized_by_a=0.9,
        tm_score_normalized_by_b=0.8,
        tm_score_normalized=0.8,
        rmsd=1.23,
        aligned_length=50,
        length_a=60,
        length_b=70,
        seq_identity=0.42,
        superposition_pdb=None,
        runtime_ms=12,
    )
    assert TMalignOutput.model_validate(out.model_dump()) == out


@pytest.mark.skipif(not _binary_available(), reason="TMalign/tmalign binary unavailable")
@pytest.mark.skipif(not _FIXTURE_A.exists(), reason="6MRR fixture missing")
def test_run_self_align(tmp_path):
    """6MRR vs itself: TM-score must be ~1.0, RMSD ~0, alignment full-length."""
    a = TMalignAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(
        TMalignInput(pdb_a=str(_FIXTURE_A), pdb_b=str(_FIXTURE_A)),
        null_context(job_dir),
    )
    assert abs(out.tm_score_normalized_by_a - 1.0) < 1e-3
    assert abs(out.tm_score_normalized_by_b - 1.0) < 1e-3
    assert out.rmsd < 0.1
    assert out.aligned_length == out.length_a == out.length_b
    assert out.length_a >= 30


@pytest.mark.skipif(not _binary_available(), reason="TMalign/tmalign binary unavailable")
@pytest.mark.skipif(
    not (_FIXTURE_A.exists() and _FIXTURE_B.exists()),
    reason="fixtures missing",
)
def test_run_different_proteins(tmp_path):
    """Two unrelated PDBs — TM-score < 1, alignment terminates cleanly."""
    a = TMalignAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(
        TMalignInput(pdb_a=str(_FIXTURE_A), pdb_b=str(_FIXTURE_B), normalize_by="average"),
        null_context(job_dir),
    )
    assert 0.0 <= out.tm_score_normalized_by_a <= 1.0
    assert 0.0 <= out.tm_score_normalized_by_b <= 1.0
    assert out.aligned_length >= 1
    # `average` selection chosen — must equal the mean of the two normalisations.
    expected = (out.tm_score_normalized_by_a + out.tm_score_normalized_by_b) / 2
    assert abs(out.tm_score_normalized - expected) < 1e-9
