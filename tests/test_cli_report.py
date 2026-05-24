from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.agents import AgentSessionStore
from vibeproteinagent.cli.main import app
from vibeproteinagent.mcp.tools import __all__ as mcp_tool_exports
from vibeproteinagent.reporting import collect_report_model, render_report_html
from vibeproteinagent.reporting.collect import _normalize_candidates
from vibeproteinagent.reporting.models import WorkflowNodeSummary
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


def test_report_does_not_auto_select_without_agent_budget_marker(
    tmp_workspace: Workspace,
) -> None:
    job_store = JobStore(tmp_workspace)
    rank_job = job_store.submit("rank_designs", "1.0.0", {"top_k": 1})
    job_store.transition(rank_job.job_id, JobState.QUEUED)
    job_store.transition(rank_job.job_id, JobState.RUNNING)
    ranked_output = {
        "candidate_count": 1,
        "ranked_candidates": [
            {"rank": 1, "candidate_id": "design_1", "score": 1.0}
        ],
    }
    job_store.transition(rank_job.job_id, JobState.COMPLETED, output=ranked_output)

    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="no_submission_demo",
        version="1.0",
        descriptor_yaml="name: no_submission_demo\nversion: '1.0'\nnodes: []\n",
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

    model = collect_report_model(tmp_workspace, wf_id)

    assert model.final_submission is None
    assert model.final_candidates == []
    assert [candidate.candidate_id for candidate in model.candidates] == ["design_1"]


def test_report_separates_explicit_final_submission(
    tmp_workspace: Workspace,
) -> None:
    job_store = JobStore(tmp_workspace)
    rank_job = job_store.submit("rank_designs", "1.0.0", {"top_k": 2})
    job_store.transition(rank_job.job_id, JobState.QUEUED)
    job_store.transition(rank_job.job_id, JobState.RUNNING)
    ranked_output = {
        "candidate_count": 2,
        "ranked_candidates": [
            {"rank": 1, "candidate_id": "design_1", "score": 1.0},
            {"rank": 2, "candidate_id": "design_2", "score": 0.5},
        ],
    }
    job_store.transition(rank_job.job_id, JobState.COMPLETED, output=ranked_output)

    submit_job = job_store.submit(
        "submit_candidate",
        "1.0.0",
        {"outcome": "accepted", "candidates": [], "reasoning": "x"},
    )
    job_store.transition(submit_job.job_id, JobState.QUEUED)
    job_store.transition(submit_job.job_id, JobState.RUNNING)
    submission = {
        "submission_id": "sub_final",
        "outcome": "accepted",
        "terminal": True,
        "candidates_resolved": [
            {
                "source_job_id": rank_job.job_id,
                "source_tool": "rank_designs",
                "candidate_id": "design_2",
                "rank": 2,
                "score": 0.5,
                "justification": "diverse backup",
                "scores": {"score": 0.5},
                "artifacts": [],
            }
        ],
        "reasoning": "selected diverse backup",
        "supersedes": None,
        "auto_selected": False,
        "runtime_ms": 1,
    }
    job_store.transition(submit_job.job_id, JobState.COMPLETED, output=submission)

    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="submission_demo",
        version="1.0",
        descriptor_yaml="name: submission_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    for node_id, tool, job, output in (
        ("rank", "rank_designs", rank_job, ranked_output),
        ("submit", "submit_candidate", submit_job, submission),
    ):
        wf_store.add_node(
            workflow_id=wf_id,
            node_id=node_id,
            tool_name=tool,
            tool_version="1.0.0",
            upstream=[],
            params_hash="abc",
            input_hash="def",
            cache_key="ghi",
        )
        wf_store.update_node_state(
            wf_id,
            node_id,
            "completed",
            job_id=job.job_id,
            output=output,
        )
    wf_store.update_state(wf_id, "completed", finished=True)

    model = collect_report_model(tmp_workspace, wf_id)

    assert model.final_submission["submission_id"] == "sub_final"
    assert [candidate.candidate_id for candidate in model.final_candidates] == ["design_2"]
    assert [candidate.candidate_id for candidate in model.candidates] == ["design_1", "design_2"]


def test_report_uses_latest_non_superseded_submission(
    tmp_workspace: Workspace,
) -> None:
    job_store = JobStore(tmp_workspace)
    rank_job = job_store.submit("rank_designs", "1.0.0", {"top_k": 2})
    job_store.transition(rank_job.job_id, JobState.QUEUED)
    job_store.transition(rank_job.job_id, JobState.RUNNING)
    ranked_output = {
        "candidate_count": 2,
        "ranked_candidates": [
            {"rank": 1, "candidate_id": "design_1", "score": 1.0},
            {"rank": 2, "candidate_id": "design_2", "score": 0.5},
        ],
    }
    job_store.transition(rank_job.job_id, JobState.COMPLETED, output=ranked_output)

    first = {
        "submission_id": "sub_first",
        "outcome": "accepted",
        "terminal": True,
        "candidates_resolved": [
            {
                "source_job_id": rank_job.job_id,
                "source_tool": "rank_designs",
                "candidate_id": "design_1",
                "rank": 1,
                "score": 1.0,
                "justification": "initial pick",
                "scores": {"score": 1.0},
                "artifacts": [],
            }
        ],
        "reasoning": "initial pick",
        "supersedes": None,
        "auto_selected": False,
        "runtime_ms": 1,
    }
    second = {
        **first,
        "submission_id": "sub_second",
        "candidates_resolved": [
            {
                **first["candidates_resolved"][0],
                "candidate_id": "design_2",
                "rank": 2,
                "score": 0.5,
                "justification": "superseding pick",
                "scores": {"score": 0.5},
            }
        ],
        "reasoning": "superseding pick",
        "supersedes": "sub_first",
    }
    submit_jobs = []
    for submission in (first, second):
        rec = job_store.submit(
            "submit_candidate",
            "1.0.0",
            {"outcome": "accepted", "candidates": [], "reasoning": "x"},
            salt=str(submission["submission_id"]),
        )
        job_store.transition(rec.job_id, JobState.QUEUED)
        job_store.transition(rec.job_id, JobState.RUNNING)
        job_store.transition(rec.job_id, JobState.COMPLETED, output=submission)
        submit_jobs.append((rec, submission))

    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="supersession_demo",
        version="1.0",
        descriptor_yaml="name: supersession_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    rows = [("rank", "rank_designs", rank_job, ranked_output)] + [
        (f"submit_{idx}", "submit_candidate", rec, submission)
        for idx, (rec, submission) in enumerate(submit_jobs, start=1)
    ]
    for node_id, tool, job, output in rows:
        wf_store.add_node(
            workflow_id=wf_id,
            node_id=node_id,
            tool_name=tool,
            tool_version="1.0.0",
            upstream=[],
            params_hash="abc",
            input_hash="def",
            cache_key="ghi",
        )
        wf_store.update_node_state(
            wf_id, node_id, "completed", job_id=job.job_id, output=output
        )
    wf_store.update_state(wf_id, "completed", finished=True)

    model = collect_report_model(tmp_workspace, wf_id)

    assert [submission["submission_id"] for submission in model.submissions] == [
        "sub_first",
        "sub_second",
    ]
    assert model.final_submission["submission_id"] == "sub_second"
    assert [candidate.candidate_id for candidate in model.final_candidates] == ["design_2"]


def test_report_auto_selects_only_after_agent_budget_marker(
    tmp_workspace: Workspace,
) -> None:
    job_store = JobStore(tmp_workspace)
    rank_job = job_store.submit("rank_designs", "1.0.0", {"top_k": 1})
    job_store.transition(rank_job.job_id, JobState.QUEUED)
    job_store.transition(rank_job.job_id, JobState.RUNNING)
    ranked_output = {
        "candidate_count": 1,
        "ranked_candidates": [
            {"rank": 1, "candidate_id": "design_1", "score": 1.0}
        ],
    }
    job_store.transition(rank_job.job_id, JobState.COMPLETED, output=ranked_output)
    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="auto_submission_demo",
        version="1.0",
        descriptor_yaml="name: auto_submission_demo\nversion: '1.0'\nnodes: []\n",
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
        wf_id, "rank", "completed", job_id=rank_job.job_id, output=ranked_output
    )
    wf_store.update_state(wf_id, "completed", finished=True)
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("agent max rounds reached")
    session.update_run(run.run_id, workflow_ids=[wf_id])
    session.add_decision(
        run.run_id,
        {
            "decision": "completed_review_metrics",
            "next_round": {"recommended": False, "reason": "max rounds reached"},
            "fallback_final_selection": {"auto_select": True},
        },
    )

    model = collect_report_model(tmp_workspace, wf_id)

    assert model.final_submission["auto_selected"] is True
    assert [candidate.candidate_id for candidate in model.final_candidates] == ["design_1"]
    assert any(w.source == "submit_candidate" for w in model.warnings)


def test_report_renders_no_acceptable_candidate_submission(
    tmp_workspace: Workspace,
) -> None:
    job_store = JobStore(tmp_workspace)
    submit_job = job_store.submit(
        "submit_candidate",
        "1.0.0",
        {
            "outcome": "no_acceptable_candidate",
            "candidates": [],
            "reasoning": "All explored candidates failed the acceptance criteria.",
        },
    )
    job_store.transition(submit_job.job_id, JobState.QUEUED)
    job_store.transition(submit_job.job_id, JobState.RUNNING)
    submission = {
        "submission_id": "sub_empty",
        "outcome": "no_acceptable_candidate",
        "terminal": True,
        "candidates_resolved": [],
        "reasoning": "All explored candidates failed the acceptance criteria.",
        "supersedes": None,
        "auto_selected": False,
        "runtime_ms": 1,
    }
    job_store.transition(submit_job.job_id, JobState.COMPLETED, output=submission)
    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="empty_submission_demo",
        version="1.0",
        descriptor_yaml="name: empty_submission_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="submit",
        tool_name="submit_candidate",
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    wf_store.update_node_state(
        wf_id, "submit", "completed", job_id=submit_job.job_id, output=submission
    )
    wf_store.update_state(wf_id, "completed", finished=True)

    model = collect_report_model(tmp_workspace, wf_id)
    html = render_report_html(model)

    assert model.final_submission["outcome"] == "no_acceptable_candidate"
    assert model.final_candidates == []
    assert "No acceptable final candidate was submitted." in html
    assert "All explored candidates failed the acceptance criteria." in html


def test_output_derived_flat_candidates_keep_source_metrics_separate() -> None:
    nodes = [
        WorkflowNodeSummary(
            node_id="boltz2_affinity",
            tool="boltz2_affinity",
            version="*",
            state="completed",
            job_id="job_boltz",
            output={
                "predicted_cif": "/tmp/boltz.cif",
                "affinity_pred_value": 1.2,
                "confidence_score": 0.8,
                "runtime_ms": 100,
            },
        ),
        WorkflowNodeSummary(
            node_id="boltzgen_design_only",
            tool="boltzgen",
            version="*",
            state="completed",
            job_id="job_boltzgen",
            output={
                "designs": [{"candidate_id": "boltzgen-1", "length": 20}],
                "runtime_ms": 200,
            },
        ),
        WorkflowNodeSummary(
            node_id="protenix_v2",
            tool="protenix_v2",
            version="*",
            state="completed",
            job_id="job_protenix",
            output={
                "predicted_cif": "/tmp/protenix.cif",
                "metrics": {"plddt": 81.0, "ptm": 0.61, "ranking_score": 0.14},
                "runtime_ms": 300,
            },
        ),
    ]

    candidates = _normalize_candidates(nodes, [])
    by_id = {candidate.candidate_id: candidate for candidate in candidates}

    assert "boltz2_affinity:affinity_pred_value" in by_id["boltz2_affinity-1"].scores
    assert "protenix_v2:plddt" not in by_id["boltz2_affinity-1"].scores
    assert "boltzgen:runtime_ms" not in by_id["boltz2_affinity-1"].scores

    assert "boltzgen:runtime_ms" in by_id["boltzgen-1"].scores
    assert "boltz2_affinity:affinity_pred_value" not in by_id["boltzgen-1"].scores
    assert "protenix_v2:plddt" not in by_id["boltzgen-1"].scores

    assert "protenix_v2:plddt" in by_id["protenix_v2-1"].scores
    assert "boltz2_affinity:affinity_pred_value" not in by_id["protenix_v2-1"].scores


def test_ranked_candidates_keep_global_qc_context() -> None:
    nodes = [
        WorkflowNodeSummary(
            node_id="rank",
            tool="rank_designs",
            version="*",
            state="completed",
            job_id="job_rank",
            output={
                "ranked_candidates": [
                    {"candidate_id": "design_1", "rank": 1, "score": 1.0, "plddt": 88.0}
                ]
            },
        ),
        WorkflowNodeSummary(
            node_id="qc",
            tool="protein_qc",
            version="*",
            state="completed",
            job_id="job_qc",
            output={
                "flags": ["high_instability_index"],
                "metrics": {"instability_index": 55.0},
            },
        ),
    ]

    candidates = _normalize_candidates(nodes, [])

    assert candidates[0].candidate_id == "design_1"
    assert "high_instability_index" in candidates[0].qc_flags
    assert candidates[0].scores["instability_index"] == 55.0


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
