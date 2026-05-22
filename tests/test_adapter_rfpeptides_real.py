"""Slow regression test for the RFpeptides adapter (GPU + weights required)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vibeproteinagent.adapters.context import AdapterRunContext  # noqa: E402
from vibeproteinagent.adapters.rfpeptides.adapter import (  # noqa: E402
    RFPEPTIDES_TREE,
    RFPeptidesAdapter,
    RFPeptidesInput,
    _import_dgl_for_rfpeptides,
    _required_weight,
)


def _skip_if_unavailable(target_pdb: str | None) -> None:
    if not torch.cuda.is_available():
        pytest.skip("RFpeptides real test requires CUDA")
    if not RFPEPTIDES_TREE.is_dir():
        pytest.skip(f"RFpeptides tree not present at {RFPEPTIDES_TREE}")
    if not (RFPEPTIDES_TREE / "scripts" / "run_inference.py").is_file():
        pytest.skip("RFpeptides run_inference.py missing")
    weight = _required_weight(target_pdb)
    if not weight.is_file():
        pytest.skip(f"RFpeptides weight checkpoint not present at {weight}")
    # RFpeptides needs DGL CUDA graph ops, while DGL's GraphBolt extension
    # does not currently match torch 2.7. Use the adapter import helper so
    # this probe mirrors the production subprocess path.
    try:
        dgl = _import_dgl_for_rfpeptides()
        g = dgl.graph(([0, 1], [1, 0])).to("cuda")
        _ = g.all_edges()
    except Exception as exc:
        pytest.skip(
            f"DGL GPU graph ops unavailable in this env "
            f"(install dgl==2.1.0+cu121 or rebuild DGL). RFpeptides cannot run. {exc!r}"
)


@pytest.mark.slow
def test_rfpeptides_monomer_macrocycle_produces_at_least_one_pdb(tmp_path):
    """Generate a single macrocyclic monomer (no target). Uses Base_ckpt.pt."""
    _skip_if_unavailable(target_pdb=None)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    progress: list[float] = []
    ctx = AdapterRunContext(
        job_id="job_rfpeptides_real_test",
        workspace_root=tmp_path,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=lambda line: None,
        emit_progress=lambda frac: progress.append(frac),
    )

    params = RFPeptidesInput(
        peptide_length=8,
        num_designs=1,
        num_diffusion_steps=15,
        seed=7,
    )

    adapter = RFPeptidesAdapter(
        config={}, weights_dir=tmp_path / "weights", device=torch.device("cuda:0")
    )
    out = adapter.run(params, ctx)

    assert out.num_designs_returned >= 1
    assert len(out.designs) == out.num_designs_returned
    for d in out.designs:
        pdb = Path(d.pdb_path)
        assert pdb.is_file(), f"design PDB missing: {pdb}"
        assert pdb.stat().st_size > 0
        assert d.length == params.peptide_length
    assert out.runtime_ms > 0
    assert math.isfinite(out.runtime_ms)
    assert progress and progress[-1] == 1.0
