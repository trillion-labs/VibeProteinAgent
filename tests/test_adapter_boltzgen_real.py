"""Slow regression test for the BoltzGen adapter (GPU + weights required)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vibeproteinagent.adapters.boltzgen.adapter import (  # noqa: E402
    BOLTZGEN_CACHE,
    BOLTZGEN_TREE,
    BoltzGenAdapter,
    BoltzGenInput,
)
from vibeproteinagent.adapters.context import AdapterRunContext  # noqa: E402

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("BoltzGen real test requires CUDA")
    if not BOLTZGEN_TREE.is_dir():
        pytest.skip(f"BoltzGen source tree not present at {BOLTZGEN_TREE}")
    if not BOLTZGEN_CACHE.is_dir():
        pytest.skip(f"BoltzGen cache not present at {BOLTZGEN_CACHE}")
    weight_glob = list(
        BOLTZGEN_CACHE.glob("models--boltzgen--boltzgen-1/snapshots/*/boltzgen1_diverse.ckpt")
    )
    if not weight_glob:
        pytest.skip("BoltzGen diverse weight checkpoint not present in cache")


@pytest.mark.slow
def test_boltzgen_design_only_produces_at_least_one_cif(tmp_path):
    _skip_if_unavailable()

    target_pdb = _FIXTURE_DIR / "6MRR.pdb"
    assert target_pdb.is_file()

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    progress: list[float] = []

    ctx = AdapterRunContext(
        job_id="job_boltzgen_real_test",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: progress.append(frac),
    )

    params = BoltzGenInput(
        target_pdb=str(target_pdb),
        target_chain="A",
        binder_length=30,
        num_designs=1,
        num_diffusion_steps=10,
        design_only=True,
        seed=7,
    )

    adapter = BoltzGenAdapter(
        config={}, weights_dir=tmp_path / "weights", device=torch.device("cuda:0")
    )
    out = adapter.run(params, ctx)

    assert out.num_designs_returned >= 1
    assert len(out.designs) == out.num_designs_returned
    for d in out.designs:
        cif = Path(d.cif_path)
        assert cif.is_file(), f"design CIF missing: {cif}"
        assert cif.stat().st_size > 0
        assert d.length == params.binder_length
    assert out.runtime_ms > 0
    assert math.isfinite(out.runtime_ms)
    # progress monotonic and reaches 1.0
    assert progress and progress[-1] == 1.0
