"""Slow: real Boltz-2 affinity adapter end-to-end on a tiny protein+caffeine fixture.

Skipped when:
  * CUDA isn't available;
  * the Boltz source tree isn't on disk at ``$VPA_BOLTZ_TREE``;
  * either Boltz-2 checkpoint is missing under ``$VPA_BOLTZ_CACHE``;
  * the extracted ``mols/`` CCD cache is missing.

Opt-in: ``pytest -m slow tests/test_adapter_boltz2_affinity_real.py``.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.boltz2_affinity import (
    BOLTZ_CACHE,
    BOLTZ_TREE,
    Boltz2AffinityAdapter,
    Boltz2AffinityInput,
)

pytestmark = pytest.mark.slow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "boltz2_affinity_input_tiny.yaml"


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Boltz-2 affinity")
    if not BOLTZ_TREE.is_dir():
        pytest.skip(f"Boltz tree not found at {BOLTZ_TREE}")
    for ckpt in ("boltz2_conf.ckpt", "boltz2_aff.ckpt"):
        if not (BOLTZ_CACHE / ckpt).is_file():
            pytest.skip(f"Boltz checkpoint missing: {BOLTZ_CACHE / ckpt}")
    if not (BOLTZ_CACHE / "mols").is_dir():
        pytest.skip(f"Boltz CCD cache missing: {BOLTZ_CACHE / 'mols'}")


@pytest.fixture(scope="module")
def adapter() -> Boltz2AffinityAdapter:
    _skip_if_unavailable()
    return Boltz2AffinityAdapter(
        config={}, weights_dir=Path("/tmp"), device=torch.device("cuda:0")
    )


def test_run_produces_predicted_cif_and_affinity(
    adapter: Boltz2AffinityAdapter, tmp_path: Path
) -> None:
    assert _FIXTURE.is_file(), f"missing fixture {_FIXTURE}"
    params = Boltz2AffinityInput(
        input_yaml=str(_FIXTURE),
        seeds=[42],
        # Knobs tuned for fastest-possible end-to-end on a tiny fixture; still
        # exercises both the confidence model and the affinity head.
        recycling_steps=1,
        sampling_steps=10,
        diffusion_samples=1,
        sampling_steps_affinity=20,
        diffusion_samples_affinity=1,
        use_msa=False,
    )
    job_dir = tmp_path / "jobs" / "job_boltz2_affinity_real"
    job_dir.mkdir(parents=True)
    out = adapter.run(params, null_context(job_dir))

    predicted_cif = Path(out.predicted_cif)
    assert predicted_cif.is_file(), f"predicted CIF missing: {predicted_cif}"
    assert predicted_cif.stat().st_size > 0
    cif_text = predicted_cif.read_text()
    assert "_atom_site." in cif_text
    assert "ATOM" in cif_text

    assert out.runtime_ms > 0
    assert out.seeds_used == [42]

    # Brief calls for finite affinity_pred_value (log10 IC50 in uM) and
    # affinity_probability_binary in [0, 1].
    assert math.isfinite(out.affinity_pred_value)
    assert 0.0 <= out.affinity_probability_binary <= 1.0
    assert math.isfinite(out.confidence_score)


def test_progress_events_emitted(
    adapter: Boltz2AffinityAdapter, tmp_path: Path
) -> None:
    seen: list[float] = []
    params = Boltz2AffinityInput(
        input_yaml=str(_FIXTURE),
        seeds=[42],
        recycling_steps=1,
        sampling_steps=10,
        diffusion_samples=1,
        sampling_steps_affinity=20,
        diffusion_samples_affinity=1,
        use_msa=False,
    )
    job_dir = tmp_path / "jobs" / "job_boltz2_affinity_prog"
    job_dir.mkdir(parents=True)
    ctx = null_context(job_dir)
    ctx_capt = type(ctx)(
        job_id=ctx.job_id,
        workspace_root=ctx.workspace_root,
        job_dir=ctx.job_dir,
        is_cancelled=ctx.is_cancelled,
        log=ctx.log,
        emit_progress=lambda v: seen.append(v),
        fusion_inputs=None,
    )
    adapter.run(params, ctx_capt)
    assert seen, "expected at least one progress event"
    assert max(seen) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= v <= 1.0 for v in seen)
