"""Tests for the Foldseek binary-wrapper adapter."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.foldseek.adapter import (
    FoldseekAdapter,
    FoldseekInput,
    FoldseekOutput,
    _find_binary,
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
    d = FoldseekAdapter.descriptor
    assert d.name == "foldseek"
    assert d.category.value == "misc"
    assert d.backend.value == "binary"
    assert d.gpu_required is False
    assert d.pin_warm is True


def test_input_defaults():
    p = FoldseekInput(query_pdb="/tmp/q.pdb", target_db="/tmp/db")
    assert p.mode == "easy-search"
    assert p.alignment_type == 2
    assert p.e_value == 0.001
    assert p.max_seqs == 1000


def test_input_rejects_bad_mode():
    with pytest.raises(ValidationError):
        FoldseekInput(query_pdb="/tmp/q.pdb", target_db="/tmp/db", mode="bogus")


def test_output_round_trip():
    out = FoldseekOutput(hits_tsv="/tmp/h.tsv", num_hits=5, runtime_ms=99)
    assert FoldseekOutput.model_validate(out.model_dump()) == out


@pytest.mark.slow
@pytest.mark.skipif(not _binary_available(), reason="foldseek binary unavailable")
@pytest.mark.skipif(not _FIXTURE.exists(), reason="6MRR fixture missing")
def test_run_self_search(tmp_path):
    """End-to-end: easy-search 6MRR against itself."""
    a = FoldseekAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(
        FoldseekInput(query_pdb=str(_FIXTURE), target_db=str(_FIXTURE), max_seqs=10),
        null_context(job_dir),
    )
    assert Path(out.hits_tsv).is_file()
    assert out.num_hits >= 1
    assert out.runtime_ms >= 0
