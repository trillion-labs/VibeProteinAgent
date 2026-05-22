"""Fast unit tests for the LigandMPNN adapter (no GPU, no heavy import)."""
from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.ligandmpnn.adapter import (
    _VENDOR_WEIGHTS_DIR,
    LigandMPNNAdapter,
    LigandMPNNDesign,
    LigandMPNNInput,
    LigandMPNNOutput,
    _resolve_checkpoint,
)


def test_descriptor_fields():
    d = LigandMPNNAdapter.descriptor
    assert d.name == "ligandmpnn"
    assert d.version == "1.0.0"
    assert d.category.value == "inverse_folding"
    assert d.gpu_required is True
    assert d.pin_warm is True
    assert d.weight_uris is not None and len(d.weight_uris) == 1
    w = d.weight_uris[0]
    assert w.uri.endswith("ligandmpnn_v_32_020_25.pt")


def test_vendored_checkpoint_sha256_matches_descriptor():
    ckpt = _VENDOR_WEIGHTS_DIR / "ligandmpnn_v_32_020_25.pt"
    if not ckpt.exists():
        pytest.skip("LigandMPNN checkpoint not staged — see vendor/ligandmpnn/UPSTREAM.md")
    expected = "42caaa5cdb380867d2b31c30de4ab53ea89171279b3a0732ddf94bc4d3cb6981"
    h = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert h == expected
    assert expected in [w.sha256 for w in LigandMPNNAdapter.descriptor.weight_uris or []]


def test_resolve_checkpoint_uses_explicit_dir(tmp_path):
    candidate = tmp_path / "ligandmpnn_v_32_020_25.pt"
    candidate.write_bytes(b"\x00")
    assert _resolve_checkpoint(tmp_path, "ligand_mpnn") == candidate


def test_resolve_checkpoint_unknown_model_type(tmp_path):
    with pytest.raises(ValueError):
        _resolve_checkpoint(tmp_path, "not_a_real_model")


def test_input_pydantic_defaults():
    p = LigandMPNNInput(pdb_path="/tmp/x.pdb")
    assert p.num_designs == 8
    assert p.temperature == 0.1
    assert p.omit_aas == "X"
    assert p.model_type == "ligand_mpnn"
    assert p.redesigned_residues is None
    assert p.seed is None


def test_input_pydantic_rejects_out_of_range():
    with pytest.raises(ValidationError):
        LigandMPNNInput(pdb_path="/tmp/x.pdb", num_designs=0)
    with pytest.raises(ValidationError):
        LigandMPNNInput(pdb_path="/tmp/x.pdb", num_designs=2048)
    with pytest.raises(ValidationError):
        LigandMPNNInput(pdb_path="/tmp/x.pdb", temperature=-0.1)
    with pytest.raises(ValidationError):
        LigandMPNNInput(pdb_path="/tmp/x.pdb", temperature=3.0)


def test_output_pydantic_round_trip():
    out = LigandMPNNOutput(
        designs=[LigandMPNNDesign(sequence="AAA", score=0.5, seq_recovery=0.33)],
        designs_fasta="/tmp/d.fa",
        num_residues=3,
        runtime_ms=42,
    )
    blob = out.model_dump()
    assert blob["num_residues"] == 3
    rebuilt = LigandMPNNOutput.model_validate(blob)
    assert rebuilt == out
