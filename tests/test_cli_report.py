from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.mcp.tools import __all__ as mcp_tool_exports
from vibeproteinagent.reporting import collect_report_model
from vibeproteinagent.workflows import WorkflowStore
from vibeproteinagent.workspace import ArtifactStore, JobState, JobStore, Workspace

PDB_TEXT = """\
ATOM      1  N   ALA A   1      11.104  13.207   9.447  1.00 80.00           N
ATOM      2  CA  ALA A   1      12.560  13.233   9.447  1.00 80.00           C
ATOM      3  C   ALA A   1      13.048  14.665   9.447  1.00 80.00           C
ATOM      4  O   ALA A   1      12.250  15.590   9.447  1.00 80.00           O
TER
END
"""


def test_report_structure_only_job_generates_offline_html(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text(PDB_TEXT)
    store = JobStore(tmp_workspace)
    rec = store.submit("protenix_v2", "1.0.0", {"input": "x"})
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED, output={"predicted_pdb": str(pdb)})
    ArtifactStore(tmp_workspace).register(
        rec.job_id,
        "predicted_pdb",
        pdb,
        kind="pdb",
        mime="chemical/x-pdb",
    )
    out = tmp_path / "report.html"

    res = runner.invoke(
        app,
        [
            "report",
            rec.job_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(out),
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["source"] == {"kind": "job", "id": rec.job_id}
    assert data["candidate_count"] == 1
    html = out.read_text()
    assert "Executive Summary" in html
    assert "Top Candidates" in html
    assert "VPA_OFFLINE_VIEWER" in html
    assert "https://3Dmol.org/build/3Dmol-min.js" not in html


def test_report_workflow_uses_ranked_candidates_qc_and_msa(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "ranked.pdb"
    pdb.write_text(PDB_TEXT)
    fasta = tmp_path / "designs.fa"
    fasta.write_text(">wt\nAAAA\n>design_1\nAAQA\n")
    job_store = JobStore(tmp_workspace)
    rank_job = job_store.submit("rank_designs", "1.0.0", {"top_k": 1})
    job_store.transition(rank_job.job_id, JobState.QUEUED)
    job_store.transition(rank_job.job_id, JobState.RUNNING)
    ranked_output = {
        "candidate_count": 2,
        "ranked_candidates": [
            {
                "rank": 1,
                "candidate_id": "design_1",
                "score": 1.23,
                "plddt": 87.2,
                "iptm": 0.83,
                "structure_path": str(pdb),
            },
            {
                "rank": 2,
                "candidate_id": "design_2",
                "score": 0.42,
                "plddt": 72.0,
                "sequence": "AAAA",
            }
        ],
    }
    job_store.transition(rank_job.job_id, JobState.COMPLETED, output=ranked_output)
    art = ArtifactStore(tmp_workspace)
    registered = art.register(rank_job.job_id, "ranked_structure", pdb, kind="pdb")
    art.register(rank_job.job_id, "designs_fasta", fasta, kind="fasta")
    pdb.unlink()

    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="report_demo",
        version="1.0",
        descriptor_yaml="name: report_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="rank",
        tool_name="rank_designs",
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    wf_store.update_node_state(
        wf_id,
        "rank",
        "completed",
        job_id=rank_job.job_id,
        output=ranked_output,
    )
    wf_store.update_state(wf_id, "completed", finished=True)
    out = tmp_path / "wf_report.html"

    res = runner.invoke(
        app,
        [
            "report",
            wf_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(out),
            "--top-k",
            "1",
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout
    data = json.loads(res.stdout)["data"]
    assert data["source"] == {"kind": "workflow", "id": wf_id}
    assert data["candidate_count"] == 2
    html = out.read_text()
    assert "design_1" in html
    assert "design_2" in html
    assert "Top #1" in html
    assert "Top #2" not in html
    assert registered.pool_path in html
    assert "pLDDT" in html or "plddt" in html
    assert "VPA_OFFLINE_VIEWER" in html
    # Per the updated MSA policy: the MSA section only renders when an
    # MSA-generating tool (boltz2_affinity / mmseqs2_gpu / template_select /
    # foldseek / boltzgen / protenix_v2) is part of the workflow. This
    # fixture's rank_designs run produces a designs FASTA (a sequence
    # collection, not a homolog alignment), so the section stays hidden —
    # the candidate cards and the All Designs table cover those sequences.
    assert "MSA / Sequence Context" not in html
    assert "Reproducibility" in html


def test_report_bundle_writes_directory(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "model.pdb"
    pdb.write_text(PDB_TEXT)
    store = JobStore(tmp_workspace)
    rec = store.submit("protenix_v2", "1.0.0", {"input": "x"})
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED, output={"predicted_pdb": str(pdb)})
    ArtifactStore(tmp_workspace).register(rec.job_id, "predicted_pdb", pdb, kind="pdb")
    out = tmp_path / "bundle"

    res = runner.invoke(
        app,
        [
            "report",
            rec.job_id,
            "--workspace",
            tmp_workspace.name,
            "--format",
            "bundle",
            "--out",
            str(out),
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout
    assert (out / "report.html").is_file()
    assert (out / "structures").is_dir()
    copied = list((out / "structures").glob("predicted_pdb_*.pdb"))
    assert len(copied) == 1
    assert str(copied[0]) in (out / "report.html").read_text()


def test_collect_report_model_rejects_bad_id(tmp_workspace: Workspace) -> None:
    with pytest.raises(Exception) as exc_info:
        collect_report_model(tmp_workspace, "bad_id")

    assert getattr(exc_info.value, "code", None) == "E_INVALID_PARAMS"


def test_report_is_not_mcp_or_adapter_surface(tmp_workspace: Workspace) -> None:
    from vibeproteinagent.adapters import ToolRegistry

    registry = ToolRegistry()
    registry.discover()

    assert "report" not in mcp_tool_exports
    assert "report" not in {desc.name for desc in registry.list()}
