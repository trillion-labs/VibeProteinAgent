"""Tests for the ESM3 adapter (gated EvolutionaryScale model).

The end-to-end test is ``slow`` and additionally skips when the
``esm3_sm_open_v1`` weights can't be loaded (no GPU, no network, gated
repo without HF_TOKEN/VPA_HF_CREDENTIAL). Fast tests cover descriptor metadata, pydantic
validation, the model-name normalisation helper, and the gated-weight
heuristic — none of which need CUDA or the actual model.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.esm3.adapter import (
    ESM3Adapter,
    ESM3Input,
    ESM3Output,
    _ensure_shared_hf_cache_env,
    _looks_gated,
    _normalize_model_name,
)
from vibeproteinagent.core.config import model_cache_root
from vibeproteinagent.core.errors import DependencyError

_SEQ_38 = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRF"


def test_descriptor_fields():
    d = ESM3Adapter.descriptor
    assert d.name == "esm3"
    assert d.category.value == "embedding"
    assert d.gpu_required is True
    assert d.pin_warm is False
    assert d.startup_cost.value == "high"


def test_normalize_model_name_open_small():
    assert _normalize_model_name("esm3_sm_open_v1") == "esm3_sm_open_v1"
    assert _normalize_model_name("esm3-sm-open-v1") == "esm3_sm_open_v1"
    assert (
        _normalize_model_name("EvolutionaryScale/esm3-sm-open-v1") == "esm3_sm_open_v1"
    )


def test_normalize_model_name_passes_through_gated():
    assert (
        _normalize_model_name("EvolutionaryScale/esm3-medium-2024-08")
        == "esm3-medium-2024-08"
    )


def test_looks_gated_flags_medium_and_large():
    assert _looks_gated("esm3-medium-2024-08")
    assert _looks_gated("esm3-large-2024-12")
    assert not _looks_gated("esm3_sm_open_v1")
    assert not _looks_gated("esm3-sm-open-v1")


def test_input_defaults():
    p = ESM3Input(sequence="MKLL")
    assert p.model_name == "esm3_sm_open_v1"
    assert p.tracks == ["sequence", "structure"]
    assert p.num_steps == 20
    assert p.temperature == pytest.approx(0.5)


def test_input_rejects_empty_sequence():
    with pytest.raises(ValidationError):
        ESM3Input(sequence="")


def test_input_rejects_zero_num_steps():
    with pytest.raises(ValidationError):
        ESM3Input(sequence="MKLL", num_steps=0)


def test_output_round_trip():
    out = ESM3Output(
        per_residue_embeddings_npz="/tmp/e.npz",
        predicted_pdb="/tmp/p.pdb",
        confidence_scores={"mean_plddt": 0.78, "ptm": 0.30},
        num_residues=38,
        model_name="esm3_sm_open_v1",
        runtime_ms=4500,
    )
    rebuilt = ESM3Output.model_validate(out.model_dump())
    assert rebuilt.num_residues == 38
    assert rebuilt.confidence_scores["mean_plddt"] == 0.78


def test_estimate_vram_total():
    inst = ESM3Adapter.__new__(ESM3Adapter)
    assert inst.estimate_vram(ESM3Input(sequence="M")) == 8 * 1024


def test_shared_hf_cache_env_default(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    _ensure_shared_hf_cache_env()
    assert os.environ["HF_HOME"] == str(model_cache_root() / "huggingface")


def _can_load_open_small() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        from esm.models.esm3 import ESM3  # noqa: F401
    except Exception as exc:  # pragma: no cover - env-dependent
        return False, f"esm not importable: {exc!r}"
    try:
        from esm.utils.constants.esm3 import data_root

        weight = (
            data_root("esm3") / "data/weights/esm3_sm_open_v1.pth"
        )
        if not weight.exists():
            return False, f"weights not on disk at {weight}"
    except Exception as exc:  # pragma: no cover - env-dependent
        return False, f"weight lookup failed: {exc!r}"
    return True, ""


@pytest.mark.slow
def test_run_open_small_end_to_end(tmp_path):
    can_load, reason = _can_load_open_small()
    if not can_load:
        pytest.skip(f"ESM3 open-small not loadable: {reason}")

    device = torch.device("cuda")
    adapter = ESM3Adapter(
        config={"model_name": "esm3_sm_open_v1"}, weights_dir=Path("/tmp"), device=device
    )
    try:
        job_dir = tmp_path / "jobs" / "esm3"
        job_dir.mkdir(parents=True)
        out = adapter.run(
            ESM3Input(
                sequence=_SEQ_38, tracks=["sequence", "structure"], num_steps=1
            ),
            null_context(job_dir),
        )
        assert out.num_residues == 38
        assert out.model_name == "esm3_sm_open_v1"
        # Embeddings present and CLS+EOS stripped to [L, hidden].
        assert out.per_residue_embeddings_npz is not None
        npz = np.load(out.per_residue_embeddings_npz)
        assert npz["per_residue"].shape == (38, 1536)
        assert np.isfinite(npz["per_residue"]).all()
        # Structure track produces a PDB with finite plddt/ptm.
        assert out.predicted_pdb is not None and Path(out.predicted_pdb).exists()
        assert out.confidence_scores is not None
        assert 0.0 < out.confidence_scores["mean_plddt"] <= 1.0
    finally:
        adapter.unload()


def test_gated_failure_raises_dependency_error(monkeypatch, tmp_path):
    """Bogus gated model name should raise DependencyError with HF guidance."""
    # Patch ESM3.from_pretrained to always raise — simulates a gated/missing weight.
    from esm.models import esm3 as esm3_mod

    def _boom(*args, **kwargs):
        raise OSError("401 Client Error: Unauthorized for ...")

    monkeypatch.setattr(esm3_mod.ESM3, "from_pretrained", staticmethod(_boom))

    # Use _looks_gated-positive name so the error message points at the gated URL.
    with pytest.raises(DependencyError) as exc:
        ESM3Adapter(
            config={"model_name": "esm3-medium-2024-08"},
            weights_dir=Path("/tmp"),
            device=torch.device("cpu"),
        )
    msg = str(exc.value)
    assert "esm3-medium-2024-08" in msg
    assert "HF_TOKEN" in msg
    assert "VPA_HF_CREDENTIAL" in msg
    assert "huggingface.co/EvolutionaryScale" in msg
    assert exc.value.details["hf_credential_set"] is False
