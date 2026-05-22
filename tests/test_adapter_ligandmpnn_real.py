"""End-to-end exercise of the real LigandMPNN adapter (no daemon).

Marked ``slow``: opt in with
``pytest -m slow tests/test_adapter_ligandmpnn_real.py``. Runs on GPU when
CUDA is available; otherwise CPU.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.ligandmpnn.adapter import (
    _VENDOR_WEIGHTS_DIR,
    LigandMPNNAdapter,
    LigandMPNNInput,
)

pytestmark = pytest.mark.slow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "4GYT.pdb"


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def adapter() -> LigandMPNNAdapter:
    if not (_VENDOR_WEIGHTS_DIR / "ligandmpnn_v_32_020_25.pt").exists():
        pytest.skip("LigandMPNN checkpoint not staged — see vendor/ligandmpnn/UPSTREAM.md")
    return LigandMPNNAdapter(
        config={}, weights_dir=Path("/nonexistent"), device=_device()
    )


def test_descriptor_metadata():
    d = LigandMPNNAdapter.descriptor
    assert d.name == "ligandmpnn"
    assert d.category.value == "inverse_folding"
    assert d.gpu_required is True
    assert d.pin_warm is True


def test_run_produces_two_designs(adapter: LigandMPNNAdapter, tmp_path: Path):
    assert _FIXTURE.exists(), f"missing fixture {_FIXTURE}"
    params = LigandMPNNInput(
        pdb_path=str(_FIXTURE),
        chains=["A"],
        num_designs=2,
        temperature=0.1,
        seed=42,
    )
    job_dir = tmp_path / "jobs" / "job_real"
    job_dir.mkdir(parents=True)
    out = adapter.run(params, null_context(job_dir))

    assert len(out.designs) == 2
    L = out.num_residues
    assert L >= 30
    for d in out.designs:
        assert len(d.sequence) == L
        assert set(d.sequence).issubset(set("ACDEFGHIKLMNPQRSTVWYX"))
        assert np.isfinite(d.score)
        assert d.seq_recovery is not None and 0.0 <= d.seq_recovery <= 1.0
    fasta = Path(out.designs_fasta)
    assert fasta.exists()
    records = [
        line for line in fasta.read_text().splitlines() if line.startswith(">")
    ]
    # 1 input record + 2 designs
    assert len(records) == 3
    assert out.runtime_ms >= 0
