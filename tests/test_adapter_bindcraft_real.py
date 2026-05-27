"""Slow regression test for the BindCraft adapter (GPU + dedicated conda env)."""
from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vibeproteinagent.adapters.bindcraft.adapter import (  # noqa: E402
    BINDCRAFT_CONDA_ENV,
    BINDCRAFT_PARAMS,
    BINDCRAFT_TREE,
    BindCraftAdapter,
    BindCraftInput,
    _af2_params_present,
    _conda_env_exists,
    _find_conda_exe,
)
from vibeproteinagent.adapters.context import AdapterRunContext  # noqa: E402


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("BindCraft real test requires CUDA")
    if not BINDCRAFT_TREE.is_dir() or not (BINDCRAFT_TREE / "bindcraft.py").is_file():
        pytest.skip(f"BindCraft tree not present at {BINDCRAFT_TREE}")
    if not _af2_params_present():
        pytest.skip(
            f"AF2 params not present at {BINDCRAFT_PARAMS}. "
            f"BindCraft requires the AlphaFold2 2022-12-06 release (~5.3 GiB)."
        )
    conda = _find_conda_exe()
    if conda is None:
        pytest.skip("conda executable not on PATH ($CONDA_EXE unset)")
    if not _conda_env_exists(conda, BINDCRAFT_CONDA_ENV):
        pytest.skip(
            f"BindCraft conda env '{BINDCRAFT_CONDA_ENV}' not present. "
            f"Run vendor/bindcraft/install_bindcraft.sh (jax+cuda+pyrosetta+colabdesign "
            f"with numpy<2; must NOT overwrite the vpa env)."
        )


@pytest.mark.slow
def test_bindcraft_runs_smallest_trajectory_against_fixture(tmp_path):
    """Generate the smallest possible BindCraft binder design.

    This test asserts only that the subprocess returns success and that at
    least one generated PDB artifact is collected — i.e. the env, weights,
    and pipeline plumbing all work end-to-end without relying on filter
    acceptance.
    """
    _skip_if_unavailable()
    fixture = Path(__file__).parent / "fixtures" / "6MRR.pdb"
    if not fixture.is_file():
        pytest.skip(f"fixture missing: {fixture}")

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    progress: list[float] = []
    ctx = AdapterRunContext(
        job_id="job_bindcraft_real_test",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: progress.append(frac),
    )

    params = BindCraftInput(
        target_pdb=str(fixture),
        binder_length=30,
        num_designs=1,
        num_recycles=1,
        seed=675122,
        max_trajectories=1,
    )

    adapter = BindCraftAdapter(
        config={}, weights_dir=tmp_path / "weights", device=torch.device("cuda:0")
    )
    out = adapter.run(params, ctx)

    assert out.num_designs_returned >= 1
    for d in out.designs:
        pdb = Path(d.pdb_path)
        assert pdb.is_file(), f"design PDB missing: {pdb}"
        assert pdb.stat().st_size > 0
        assert d.binder_length == params.binder_length
        assert len(d.sequence) >= 1
    assert out.runtime_ms > 0
    assert progress and progress[-1] == 1.0
