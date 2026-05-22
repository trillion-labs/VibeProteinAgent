"""End-to-end exercise of the real ESM2 adapter.

Marked ``slow``: opt in with
``pytest -m slow tests/test_adapter_esm2_real.py``. Pulls a small ESM2 model
from HuggingFace on first run (cached locally afterwards).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.esm2.adapter import ESM2Adapter, ESM2Input

pytestmark = pytest.mark.slow


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def adapter() -> ESM2Adapter:
    return ESM2Adapter(config={}, weights_dir=Path("/nonexistent"), device=_device())


def test_per_residue_shape_matches_input_length(adapter: ESM2Adapter, tmp_path: Path):
    """CLS+EOS stripping must yield exactly [L, hidden]."""
    seq = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRF"
    L = len(seq)
    job_dir = tmp_path / "j"
    job_dir.mkdir()
    out = adapter.run(ESM2Input(sequence=seq), null_context(job_dir))

    assert out.num_residues == L
    arr = np.load(out.per_residue_embeddings_npz)["per_residue"]
    assert arr.shape == (L, adapter.hidden_size)
    assert np.isfinite(arr).all()
    assert len(out.pooled_embedding) == adapter.hidden_size
    assert all(np.isfinite(v) for v in out.pooled_embedding)
    assert out.runtime_ms >= 0
