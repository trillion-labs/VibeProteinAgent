"""Slow: run the real Protenix v2 adapter end-to-end on the local model.

Skipped when:
  * CUDA isn't available (Protenix v1.0.4 needs a real GPU);
  * the Protenix tree isn't on disk;
  * the Protenix checkpoint isn't on disk.

Opt-in: ``pytest -m slow tests/test_adapter_protenix_v2_real.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.protenix_v2 import (
    PROTENIX_CKPT_DIR,
    PROTENIX_TREE,
    ProtenixV2Adapter,
    ProtenixV2Input,
)
from vibeproteinagent.adapters.protenix_v2.adapter import PROTENIX_MODEL_NAME

pytestmark = pytest.mark.slow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "protenix_v2_input_tiny.json"


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Protenix v2")
    if not PROTENIX_TREE.is_dir():
        pytest.skip(f"Protenix tree not found at {PROTENIX_TREE}")
    ckpt = PROTENIX_CKPT_DIR / f"{PROTENIX_MODEL_NAME}.pt"
    if not ckpt.is_file():
        pytest.skip(f"Protenix checkpoint not found at {ckpt}")


@pytest.fixture(scope="module")
def adapter() -> ProtenixV2Adapter:
    _skip_if_unavailable()
    return ProtenixV2Adapter(
        config={}, weights_dir=Path("/tmp"), device=torch.device("cuda:0")
    )


def test_run_produces_predicted_cif(adapter: ProtenixV2Adapter, tmp_path: Path) -> None:
    assert _FIXTURE.exists(), f"missing fixture {_FIXTURE}"
    params = ProtenixV2Input(
        input_json=str(_FIXTURE),
        seeds=[101],
        use_msa=False,
        use_template=False,
        use_rna_msa=False,
        n_sample=1,
        n_step=20,
        n_cycle=1,
    )
    job_dir = tmp_path / "jobs" / "job_real_protenix_v2"
    job_dir.mkdir(parents=True)
    out = adapter.run(params, null_context(job_dir))

    predicted_cif = Path(out.predicted_cif)
    assert predicted_cif.is_file(), f"predicted CIF missing: {predicted_cif}"
    assert predicted_cif.stat().st_size > 0
    text = predicted_cif.read_text()
    # CIF should declare an atom_site loop with at least one ATOM record.
    assert "_atom_site." in text
    assert "ATOM" in text

    assert out.runtime_ms > 0
    assert out.seeds_used == [101]
    assert len(out.all_predictions) >= 1
    assert out.predicted_pdb is None

    metrics = out.metrics
    for name in ("plddt", "iptm", "ptm", "ranking_score"):
        value = getattr(metrics, name)
        assert math.isfinite(value), f"metrics.{name} is not finite: {value!r}"
    # plddt is on the 0..100 scale for Protenix.
    assert 0.0 <= metrics.plddt <= 100.0
    # iptm/ptm/ranking_score are unit-interval scores.
    for name in ("iptm", "ptm", "ranking_score"):
        v = getattr(metrics, name)
        assert -0.05 <= v <= 1.05, f"metrics.{name} out of range: {v}"


def test_run_writes_summary_json(adapter: ProtenixV2Adapter, tmp_path: Path) -> None:
    params = ProtenixV2Input(
        input_json=str(_FIXTURE),
        seeds=[101],
        use_msa=False,
        n_sample=1,
        n_step=20,
        n_cycle=1,
    )
    job_dir = tmp_path / "jobs" / "job_real_protenix_v2_sum"
    job_dir.mkdir(parents=True)
    out = adapter.run(params, null_context(job_dir))

    assert out.metrics.summary_json is not None
    summary_path = Path(out.metrics.summary_json)
    assert summary_path.is_file()
    assert summary_path.suffix == ".json"
    # Per-prediction record should also point at a real summary file.
    for rec in out.all_predictions:
        assert Path(rec.summary_json).is_file()
        assert Path(rec.cif_path).is_file()


def test_progress_events_emitted(adapter: ProtenixV2Adapter, tmp_path: Path) -> None:
    seen: list[float] = []
    params = ProtenixV2Input(
        input_json=str(_FIXTURE),
        seeds=[101],
        use_msa=False,
        n_sample=1,
        n_step=20,
        n_cycle=1,
    )
    job_dir = tmp_path / "jobs" / "job_real_protenix_v2_prog"
    job_dir.mkdir(parents=True)
    ctx = null_context(job_dir)
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
