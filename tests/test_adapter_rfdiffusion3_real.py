"""Slow real regression for the Foundry-backed RFdiffusion3 adapter."""
from __future__ import annotations

# ruff: noqa: E402
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.rfdiffusion3.adapter import (
    RFdiffusion3Adapter,
    RFdiffusion3Input,
    availability_skip_reason,
)


@pytest.mark.slow
def test_rfdiffusion3_runs_small_binder_smoke(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("RFdiffusion3 real test requires CUDA")
    reason = availability_skip_reason()
    if reason is not None:
        pytest.skip(reason)

    fixture = Path(__file__).parent / "fixtures" / "6MRR.pdb"
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    progress: list[float] = []
    ctx = AdapterRunContext(
        job_id="job_rfd3_real",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: progress.append(frac),
    )
    adapter = RFdiffusion3Adapter(
        config={}, weights_dir=tmp_path / "weights", device=torch.device("cuda:0")
    )
    out = adapter.run(
        RFdiffusion3Input(
            target_pdb=str(fixture),
            target_chain="A",
            binder_length=30,
            num_designs=1,
            num_diffusion_steps=10,
            seed=1,
        ),
        ctx,
    )

    assert out.num_designs_returned >= 1
    assert progress and progress[-1] == 1.0
    for design in out.designs:
        path = Path(design.pdb_path)
        assert path.is_file()
        assert path.stat().st_size > 0
