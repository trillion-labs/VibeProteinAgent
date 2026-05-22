"""Slow: real-runtime regression for PXDesign's `pipeline` mode.

Auto-skips when:

* ``VPA_PXDESIGN_PIPELINE_ENV`` is unset AND
  ``/opt/conda/envs/pxdesign-pipeline-vpa`` is missing, OR
* the resolved env directory has no ``bin/python``, OR
* the PXDesign in-process checkpoint isn't on disk (we still construct the
  adapter the canonical way, which needs the diffusion ckpt for warm load).

When all prerequisites are present, runs a tiny pipeline call on the
bundled `5o45.cif` fixture (1 sample x 5 steps, top-K = 1) and asserts:

* ``designs[0].sequence`` is non-empty and contains no ``X`` placeholders,
* ``designs[0].cif_path`` points to a real file on disk,
* ``designs[0].score`` is finite when present.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.pxdesign import (
    PROTENIX_PXD_VENDOR,
    PXDBENCH_VENDOR,
    PXDESIGN_CKPT_DIR,
    PXDESIGN_MODEL_NAME,
    PXDESIGN_VENDOR,
    PXDesignAdapter,
    PXDesignInput,
    availability_skip_reason,
)

pytestmark = pytest.mark.slow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_CIF = _PROJECT_ROOT / "tests" / "fixtures" / "5o45.cif"


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for PXDesign pipeline mode")
    if "VPA_PXDESIGN_PIPELINE_ENV" not in os.environ:
        skip = availability_skip_reason()
        if skip is not None:
            pytest.skip(skip)
    skip = availability_skip_reason()
    if skip is not None:
        pytest.skip(skip)
    for root in (PXDESIGN_VENDOR, PXDBENCH_VENDOR, PROTENIX_PXD_VENDOR):
        if not root.is_dir():
            pytest.skip(f"vendor tree missing: {root}")
    ckpt = PXDESIGN_CKPT_DIR / f"{PXDESIGN_MODEL_NAME}.pt"
    if not ckpt.is_file():
        pytest.skip(f"PXDesign checkpoint not found at {ckpt}")
    if not _FIXTURE_CIF.is_file():
        pytest.skip(f"fixture CIF missing: {_FIXTURE_CIF}")


def test_pipeline_mode_emits_real_amino_acid_sequence(tmp_path: Path) -> None:
    _skip_if_unavailable()

    adapter = PXDesignAdapter(
        config={}, weights_dir=Path("/tmp"), device=torch.device("cuda:0")
    )

    params = PXDesignInput(
        target_pdb=str(_FIXTURE_CIF),
        target_chain="A",
        binder_length=50,
        num_samples=1,
        num_steps=5,
        mode="pipeline",
        top_k_designs=1,
        seed=101,
    )
    ctx = null_context(tmp_path)
    out = adapter.run(params, ctx)

    assert out.designs, "pipeline mode returned zero designs"
    d = out.designs[0]
    assert d.sequence, "pipeline mode returned an empty sequence"
    assert "X" not in d.sequence, (
        f"pipeline mode returned placeholder sequence {d.sequence!r} — "
        f"expected real AA letters (no 'X')"
    )
    assert Path(d.cif_path).is_file(), f"missing CIF: {d.cif_path}"
    if d.score is not None:
        import math
        assert math.isfinite(d.score), f"score not finite: {d.score}"
