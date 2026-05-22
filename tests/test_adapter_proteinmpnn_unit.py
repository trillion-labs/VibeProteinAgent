"""Unit-level tests for the Phase 4.1 ProteinMPNN adapter.

Fast: no GPU, no IPC. Exercises construction on CPU, descriptor metadata,
pydantic validation, and the weight-resolution fallback path.
"""
from __future__ import annotations

import hashlib

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.proteinmpnn.adapter import (
    _VENDOR_WEIGHTS_DEFAULT,
    ProteinMPNNAdapter,
    ProteinMPNNDesign,
    ProteinMPNNInput,
    ProteinMPNNOutput,
    _resolve_checkpoint,
)


def test_vendored_checkpoint_exists():
    assert _VENDOR_WEIGHTS_DEFAULT.exists(), (
        f"vendored ProteinMPNN checkpoint missing at {_VENDOR_WEIGHTS_DEFAULT!s}. "
        "See vendor/proteinmpnn/UPSTREAM.md to reproduce."
    )


def test_vendored_checkpoint_sha256_matches_descriptor():
    expected = "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
    h = hashlib.sha256(_VENDOR_WEIGHTS_DEFAULT.read_bytes()).hexdigest()
    assert h == expected
    declared = [w.sha256 for w in ProteinMPNNAdapter.descriptor.weight_uris or []]
    assert expected in declared


def test_descriptor_metadata_phase_4_1():
    d = ProteinMPNNAdapter.descriptor
    assert d.name == "proteinmpnn"
    assert d.version == "1.0.1"
    assert d.gpu_required is True
    assert d.pin_warm is True
    assert d.weight_uris is not None and len(d.weight_uris) == 1
    w = d.weight_uris[0]
    assert w.uri.endswith("v_48_020.pt")
    assert w.size_bytes == 6681301


def test_resolve_checkpoint_uses_explicit_dir(tmp_path):
    candidate = tmp_path / "v_48_020.pt"
    candidate.write_bytes(b"\x00")
    assert _resolve_checkpoint(tmp_path) == candidate


def test_resolve_checkpoint_falls_back_to_vendor(tmp_path):
    assert _resolve_checkpoint(tmp_path) == _VENDOR_WEIGHTS_DEFAULT


def test_construct_on_cpu_uses_vendor_fallback(tmp_path):
    a = ProteinMPNNAdapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    assert a.healthcheck() is True
    assert a.ckpt_path == _VENDOR_WEIGHTS_DEFAULT
    assert a.estimate_vram(ProteinMPNNInput(pdb_path="/tmp/x.pdb")) == 1024


def test_input_pydantic_defaults():
    p = ProteinMPNNInput(pdb_path="/tmp/x.pdb")
    assert p.num_designs == 8
    assert p.temperature == 0.1
    assert p.omit_aas == "X"
    assert p.chains is None
    assert p.seed is None


def test_input_pydantic_rejects_out_of_range():
    with pytest.raises(ValidationError):
        ProteinMPNNInput(pdb_path="/tmp/x.pdb", num_designs=0)
    with pytest.raises(ValidationError):
        ProteinMPNNInput(pdb_path="/tmp/x.pdb", num_designs=2048)
    with pytest.raises(ValidationError):
        ProteinMPNNInput(pdb_path="/tmp/x.pdb", temperature=0.0)
    with pytest.raises(ValidationError):
        ProteinMPNNInput(pdb_path="/tmp/x.pdb", temperature=3.0)


def test_output_pydantic_round_trip():
    out = ProteinMPNNOutput(
        designs=[ProteinMPNNDesign(sequence="AAA", score=0.5, seq_recovery=0.33)],
        designs_fasta="/tmp/d.fa",
        log_prob_npz="/tmp/d.npz",
        num_residues=3,
        runtime_ms=42,
    )
    blob = out.model_dump()
    assert blob["num_residues"] == 3
    assert blob["designs"][0]["sequence"] == "AAA"
    rebuilt = ProteinMPNNOutput.model_validate(blob)
    assert rebuilt == out


def test_run_raises_on_missing_pdb(tmp_path):
    a = ProteinMPNNAdapter(
        config={}, weights_dir=tmp_path, device=torch.device("cpu")
    )
    params = ProteinMPNNInput(pdb_path=str(tmp_path / "does-not-exist.pdb"), num_designs=1)
    job_dir = tmp_path / "jobs" / "job_0"
    job_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        a.run(params, null_context(job_dir))


def test_input_pydantic_accepts_seed_and_chains():
    p = ProteinMPNNInput(
        pdb_path="/tmp/x.pdb",
        chains=["A", "B"],
        num_designs=4,
        temperature=0.2,
        seed=42,
    )
    assert p.chains == ["A", "B"]
    assert p.seed == 42
