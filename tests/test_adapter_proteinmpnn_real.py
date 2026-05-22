"""End-to-end exercise of the real ProteinMPNN adapter (no daemon).

Runs on GPU when CUDA is available; otherwise on CPU. Marked ``slow``: opt in
with ``pytest -m slow tests/test_adapter_proteinmpnn_real.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.proteinmpnn.adapter import (
    ProteinMPNNAdapter,
    ProteinMPNNInput,
)

pytestmark = pytest.mark.slow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "6MRR.pdb"


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def adapter() -> ProteinMPNNAdapter:
    return ProteinMPNNAdapter(
        config={}, weights_dir=Path("/nonexistent"), device=_device()
    )


def test_run_produces_two_designs(adapter: ProteinMPNNAdapter, tmp_path: Path):
    assert _FIXTURE.exists(), f"missing fixture {_FIXTURE}"
    params = ProteinMPNNInput(
        pdb_path=str(_FIXTURE), num_designs=2, temperature=0.1, seed=42
    )
    job_dir = tmp_path / "jobs" / "job_real_2"
    job_dir.mkdir(parents=True)
    out = adapter.run(params, null_context(job_dir))

    assert len(out.designs) == 2
    L = out.num_residues
    assert L >= 30, f"unexpectedly short backbone: {L}"
    for d in out.designs:
        assert len(d.sequence) == L
        assert set(d.sequence).issubset(set("ACDEFGHIKLMNPQRSTVWY"))
        assert np.isfinite(d.score)
        assert d.seq_recovery is not None and 0.0 <= d.seq_recovery <= 1.0
    assert out.runtime_ms >= 0


def test_fasta_and_npz_are_written(adapter: ProteinMPNNAdapter, tmp_path: Path):
    params = ProteinMPNNInput(
        pdb_path=str(_FIXTURE), num_designs=2, temperature=0.1, seed=7
    )
    job_dir = tmp_path / "jobs" / "job_real_fa"
    job_dir.mkdir(parents=True)
    out = adapter.run(params, null_context(job_dir))

    fasta = Path(out.designs_fasta)
    assert fasta.exists()
    records = [
        line for line in fasta.read_text().splitlines() if line.startswith(">")
    ]
    # 1 input record + 2 design records
    assert len(records) == 3

    npz = np.load(out.log_prob_npz)
    assert "log_probs" in npz.files
    assert "S_sample" in npz.files
    log_probs = npz["log_probs"]
    assert log_probs.shape[0] == 2
    assert log_probs.shape[-1] == 21
    assert np.isfinite(log_probs).all()


def test_seed_determinism(adapter: ProteinMPNNAdapter, tmp_path: Path):
    """Same seed → same first sampled sequence."""
    params = ProteinMPNNInput(
        pdb_path=str(_FIXTURE), num_designs=1, temperature=0.1, seed=123
    )
    job_a = tmp_path / "a"
    job_b = tmp_path / "b"
    job_a.mkdir()
    job_b.mkdir()
    out_a = adapter.run(params, null_context(job_a))
    out_b = adapter.run(params, null_context(job_b))
    assert out_a.designs[0].sequence == out_b.designs[0].sequence


def test_progress_events_emitted(adapter: ProteinMPNNAdapter, tmp_path: Path):
    seen: list[float] = []
    params = ProteinMPNNInput(
        pdb_path=str(_FIXTURE), num_designs=2, temperature=0.1, seed=11
    )
    job_dir = tmp_path / "jobs" / "job_prog"
    job_dir.mkdir(parents=True)
    ctx = null_context(job_dir)
    # Swap in a capturing emit_progress.
    ctx_with_cap = type(ctx)(
        job_id=ctx.job_id,
        workspace_root=ctx.workspace_root,
        job_dir=ctx.job_dir,
        is_cancelled=ctx.is_cancelled,
        log=ctx.log,
        emit_progress=lambda v: seen.append(v),
        fusion_inputs=None,
    )
    adapter.run(params, ctx_with_cap)
    assert seen, "expected at least one progress event"
    assert max(seen) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= v <= 1.0 for v in seen)
