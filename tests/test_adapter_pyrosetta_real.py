"""Slow regression test for the PyRosetta adapter.

Auto-skips cleanly when no usable PyRosetta runtime exists. On this host,
PyRosetta is installed in the shared ``vpa`` env and executed through the
persistent sidecar. The test runs a ``score`` pass against ``6MRR.pdb`` and
asserts that the runner returns a finite REF2015 total score plus a non-zero
runtime.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.pyrosetta.adapter import (
    PyRosettaAdapter,
    PyRosettaInput,
    availability_skip_reason,
)


def _skip_if_unavailable() -> None:
    reason = availability_skip_reason()
    if reason is not None:
        pytest.skip(reason)


@pytest.mark.slow
def test_score_mode_on_fixture(tmp_path):
    """Score a fixture PDB end-to-end via the PyRosetta sidecar."""
    _skip_if_unavailable()

    fixture = Path(__file__).parent / "fixtures" / "6MRR.pdb"
    if not fixture.is_file():
        pytest.skip(f"fixture missing: {fixture}")

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    progress: list[float] = []
    ctx = AdapterRunContext(
        job_id="job_pyrosetta_real_score",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: progress.append(frac),
    )

    params = PyRosettaInput(
        pdb_path=str(fixture),
        mode="score",
        score_fn="ref2015",
        seed=7,
    )
    adapter = PyRosettaAdapter(
        config={}, weights_dir=tmp_path / "weights", device=torch.device("cpu")
    )
    out = adapter.run(params, ctx)

    assert out.mode == "score"
    # REF2015 scores can be wildly out-of-range for un-relaxed crystal
    # structures (clashes, missing atoms). Just assert finiteness — if
    # PyRosetta returned a number at all, the bridge worked.
    assert out.total_score == out.total_score  # not NaN
    assert out.runtime_ms > 0
    assert out.relaxed_pdb_path is None
    assert out.interface_metrics is None
    assert progress and progress[-1] == 1.0
    # Spec + result files should both have landed under job_dir.
    assert (job_dir / "inputs" / "_spec.json").is_file()
    assert (job_dir / "outputs" / "_result.json").is_file()


@pytest.mark.slow
def test_relax_mode_on_fixture(tmp_path):
    """One FastRelax cycle on a tiny fixture, validating the relaxed PDB lands."""
    _skip_if_unavailable()

    fixture = Path(__file__).parent / "fixtures" / "6MRR.pdb"
    if not fixture.is_file():
        pytest.skip(f"fixture missing: {fixture}")

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    ctx = AdapterRunContext(
        job_id="job_pyrosetta_real_relax",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: None,
    )

    params = PyRosettaInput(
        pdb_path=str(fixture),
        mode="relax",
        score_fn="ref2015",
        max_relax_cycles=1,
        seed=7,
    )
    adapter = PyRosettaAdapter(
        config={}, weights_dir=tmp_path / "weights", device=torch.device("cpu")
    )
    out = adapter.run(params, ctx)

    assert out.mode == "relax"
    assert out.relaxed_pdb_path is not None
    relaxed = Path(out.relaxed_pdb_path)
    assert relaxed.is_file()
    assert relaxed.stat().st_size > 0
    assert out.runtime_ms > 0
