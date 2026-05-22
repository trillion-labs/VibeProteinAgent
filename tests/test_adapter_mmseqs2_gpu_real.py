"""Tests for the MMseqs2-gpu binary-wrapper adapter."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.mmseqs2_gpu.adapter import (
    MMseqs2GpuAdapter,
    MMseqs2GpuInput,
    MMseqs2GpuOutput,
)


def _binary_available() -> bool:
    if os.environ.get("VPA_MMSEQS_BINARY"):
        return True
    if shutil.which("mmseqs-gpu") or shutil.which("mmseqs"):
        return True
    return Path("/opt/vpa-data/mmseqs2-gpu/bin/mmseqs").is_file()


def test_descriptor_fields():
    d = MMseqs2GpuAdapter.descriptor
    assert d.name == "mmseqs2_gpu"
    assert d.category.value == "misc"
    assert d.backend.value == "binary"
    assert d.gpu_required is True
    assert d.pin_warm is True
    assert d.vram_min_gib == 4


def test_input_defaults():
    p = MMseqs2GpuInput(query_fasta="/tmp/q.fa", target_db="/tmp/db.fa")
    assert p.min_seq_id == 0.3
    assert p.coverage == 0.8
    assert p.sensitivity == 7.5
    assert p.max_seqs == 300


def test_input_rejects_bad_sensitivity():
    with pytest.raises(ValidationError):
        MMseqs2GpuInput(query_fasta="/tmp/q.fa", target_db="/tmp/db.fa", sensitivity=20.0)


def test_output_round_trip():
    out = MMseqs2GpuOutput(hits_tsv="/tmp/h.tsv", num_hits=3, runtime_ms=42)
    assert MMseqs2GpuOutput.model_validate(out.model_dump()) == out


@pytest.mark.slow
@pytest.mark.skipif(not _binary_available(), reason="mmseqs binary not available")
def test_run_self_search(tmp_path):
    """End-to-end: search a tiny FASTA against itself; expect >=1 self-hit per query."""
    fasta = tmp_path / "q.fa"
    fasta.write_text(
        ">seq_a\nMKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFKKKR\n"
        ">seq_b\nMKKILLIASLLLAVLAYAQGGEYTRTRTKHTGDFEYNFGT\n"
    )
    a = MMseqs2GpuAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(
        MMseqs2GpuInput(
            query_fasta=str(fasta),
            target_db=str(fasta),
            min_seq_id=0.1,
            coverage=0.5,
            sensitivity=7.5,
            max_seqs=10,
        ),
        null_context(job_dir),
    )
    assert Path(out.hits_tsv).is_file()
    assert out.num_hits >= 2  # at least the two self-self hits
    assert out.runtime_ms >= 0
