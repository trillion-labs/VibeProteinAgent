"""Slow real smoke test for the RosettaCommons RFdiffusion adapter."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vibeproteinagent.adapters.context import AdapterRunContext  # noqa: E402
from vibeproteinagent.adapters.rfdiffusion.adapter import (  # noqa: E402
    RFDIFFUSION_TREE,
    RFdiffusionAdapter,
    RFdiffusionInput,
    _required_weight,
    availability_report,
)


def _skip_if_unavailable(params: RFdiffusionInput) -> None:
    if os.environ.get("VPA_RFDIFFUSION_REAL_TESTS") != "1":
        pytest.skip("set VPA_RFDIFFUSION_REAL_TESTS=1 to run RFdiffusion real smoke")
    if not torch.cuda.is_available():
        pytest.skip("RFdiffusion real test requires CUDA")
    if not RFDIFFUSION_TREE.is_dir():
        pytest.skip(f"RFdiffusion tree not present at {RFDIFFUSION_TREE}")
    if not (RFDIFFUSION_TREE / "scripts" / "run_inference.py").is_file():
        pytest.skip("RFdiffusion run_inference.py missing")
    weight = _required_weight(params)
    if not weight.is_file():
        pytest.skip(f"RFdiffusion checkpoint not present at {weight}")
    report = availability_report()
    if not report.get("available"):
        pytest.skip(f"RFdiffusion unavailable: {report.get('reason')}")


@pytest.mark.slow
def test_rfdiffusion_unconditional_real_smoke(tmp_path):
    """Generate one tiny unconditional backbone through real RFdiffusion."""
    params = RFdiffusionInput(
        mode="unconditional",
        length_min=20,
        length_max=20,
        num_designs=1,
        num_diffusion_steps=15,
        seed=11,
    )
    _skip_if_unavailable(params)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    progress: list[float] = []
    ctx = AdapterRunContext(
        job_id="job_rfdiffusion_real_test",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: progress.append(frac),
    )

    adapter = RFdiffusionAdapter(
        config={}, weights_dir=tmp_path / "weights", device=torch.device("cuda:0")
    )
    out = adapter.run(params, ctx)

    assert out.mode == "unconditional"
    assert out.contigs == ["20-20"]
    assert out.num_designs_returned >= 1
    assert len(out.designs) == out.num_designs_returned
    for design in out.designs:
        pdb = Path(design.pdb_path)
        assert pdb.is_file(), f"design PDB missing: {pdb}"
        assert pdb.stat().st_size > 0
        assert design.length >= 1
    assert out.runtime_ms > 0
    assert progress and progress[-1] == 1.0
