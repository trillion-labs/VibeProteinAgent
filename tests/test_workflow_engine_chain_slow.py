from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_mpnn_then_refold_runs_end_to_end(tmp_path: Path) -> None:
    """Run the bundled ``mpnn_then_refold`` workflow against real adapters.

    Chains ProteinMPNN → mpnn_to_protenix_input → Protenix-v2 end-to-end.
    Skipped if no CUDA is available, since two of the three adapters are
    GPU-bound, and also if no `vpa daemon start`-style supervisor is running
    (we don't manage daemon lifecycle from inside pytest).
    """
    try:
        import torch
    except Exception:
        pytest.skip("torch unavailable")

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")

    fixture = Path(__file__).parent / "fixtures" / "6MRR.pdb"
    if not fixture.is_file():
        pytest.skip(f"fixture {fixture} not present")

    os.environ["VPA_HOME"] = str(tmp_path / "vpa_home")
    (tmp_path / "vpa_home").mkdir()
    os.environ["VPA_RUNTIME_DIR"] = str(tmp_path / "vpa_run")
    (tmp_path / "vpa_run").mkdir()

    from vibeproteinagent.adapters import ToolRegistry
    from vibeproteinagent.workers.supervisor_client import (
        SupervisorClient,
    )
    from vibeproteinagent.workflows import WorkflowExecutor, load_workflow
    from vibeproteinagent.workspace import Workspace

    ws = Workspace.create("wf_slow")
    registry = ToolRegistry()
    registry.discover()
    sup = SupervisorClient()
    if not sup.is_running():
        pytest.skip("supervisor not running; this test needs `vpa daemon start` first")

    descriptor = (
        Path(__file__).resolve().parents[1]
        / "workflows" / "examples" / "mpnn_then_refold.yaml"
    )
    wf = load_workflow(descriptor)
    ex = WorkflowExecutor(ws, registry, sup)
    wf_id = ex.submit(
        wf,
        {"pdb_path": str(fixture), "num_designs": 2, "top_k": 1},
    )
    final = ex.run_sync(wf_id, max_wait_s=1800, poll_s=1.0)
    assert final.state == "completed", final

    # Now verify each node ran (or was cached) and the refold leg produced a CIF.
    rows = {
        n.node_id: n
        for n in ex.store.list_nodes(wf_id)
    }
    # Expect: 'design', 'convert', 'refold' (base placeholder, skipped) and
    # 'refold[0]' for the single foreach expansion.
    assert "design" in rows
    assert "convert" in rows
    success = {"completed", "cached"}
    assert rows["design"].state in success
    assert rows["convert"].state in success
    # Find at least one foreach child of 'refold' that succeeded.
    refold_children = [
        n for nid, n in rows.items() if nid.startswith("refold[")
    ]
    assert refold_children, "expected at least one refold foreach child"
    completed_children = [n for n in refold_children if n.state in success]
    assert completed_children, "no refold child reached completed/cached"

    # The refold job's output must include a predicted_cif path that exists.
    last_job_id = completed_children[0].job_id
    assert last_job_id is not None
    refold_job = ex.jobs.get(last_job_id)
    assert isinstance(refold_job.output, dict), refold_job.output
    cif_path = refold_job.output.get("predicted_cif")
    assert isinstance(cif_path, str) and cif_path, refold_job.output
    assert Path(cif_path).is_file(), cif_path
