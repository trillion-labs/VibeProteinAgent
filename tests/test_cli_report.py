from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.agents import AgentSessionStore
from vibeproteinagent.cli.main import app
from vibeproteinagent.mcp.tools import __all__ as mcp_tool_exports
from vibeproteinagent.reporting import collect_report_model, render_report_html, write_report
from vibeproteinagent.reporting.collect import (
    _candidate_merge_events_from_nodes,
    _normalize_candidates,
    _read_candidate_archive_snapshot,
)
from vibeproteinagent.reporting.models import ReportArtifact, WorkflowNodeSummary
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


def _workflow_with_raw_candidate(
    workspace: Workspace,
    *,
    candidate_id: str = "raw_candidate",
) -> tuple[str, str]:
    job_store = JobStore(workspace)
    job = job_store.submit("go_sequences", "1.0.0", {"candidate_id": candidate_id})
    job_store.transition(job.job_id, JobState.QUEUED)
    job_store.transition(job.job_id, JobState.RUNNING)
    output = {
        "candidates": [
            {
                "schema_version": "candidate_v1",
                "candidate_id": candidate_id,
                "sequence": "RAW",
                "scores": {"raw_tool": {"score": 0.1}},
            }
        ]
    }
    job_store.transition(job.job_id, JobState.COMPLETED, output=output)

    wf_store = WorkflowStore(workspace)
    wf_id = wf_store.create(
        name="archive_contract_demo",
        version="1.0",
        descriptor_yaml="name: archive_contract_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="retrieve",
        tool_name="go_sequences",
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    wf_store.update_node_state(
        wf_id,
        "retrieve",
        "completed",
        job_id=job.job_id,
        output=output,
    )
    wf_store.update_state(wf_id, "completed", finished=True)
    return wf_id, job.job_id


def _ensure_candidate_refs_column(workspace: Workspace) -> None:
    conn = workspace.connect()
    try:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(agent_candidate_snapshots)")
        }
        if "candidate_refs_json" not in columns:
            conn.execute(
                "ALTER TABLE agent_candidate_snapshots ADD COLUMN candidate_refs_json TEXT"
            )
        if "workflow_ids_json" not in columns:
            conn.execute(
                "ALTER TABLE agent_candidate_snapshots ADD COLUMN workflow_ids_json TEXT"
            )
    finally:
        conn.close()


def _insert_agent_candidate_record(
    workspace: Workspace,
    *,
    run_id: str,
    candidate_id: str,
    version_id: str,
    round_index: int,
    workflow_id: str,
    node_id: str,
    job_id: str,
    status: str,
    candidate: dict,
    created_at: int,
) -> None:
    conn = workspace.connect()
    try:
        conn.execute(
            """
            INSERT INTO agent_candidate_records (
              run_id, candidate_id, version_id, round_index, workflow_id,
              node_id, job_id, status, candidate_json, raw_source_json,
              validation_errors_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                candidate_id,
                version_id,
                round_index,
                workflow_id,
                node_id,
                job_id,
                status,
                json.dumps(candidate),
                json.dumps({"workflow_id": workflow_id, "node_id": node_id, "job_id": job_id}),
                "[]",
                created_at,
            ),
        )
    finally:
        conn.close()


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


def test_report_surfaces_bounded_candidate_availability_events(
    tmp_workspace: Workspace,
) -> None:
    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="availability_demo",
        version="1.0",
        descriptor_yaml="name: availability_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="resolver",
        tool_name="function_term_resolver",
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    wf_store.update_node_state(
        wf_id,
        "resolver",
        "completed",
        output={
            "empty_facets": [
                {
                    "event_type": "empty_candidate_event",
                    "intent_id": "facet_epimerase",
                    "label": "epimerase intent",
                    "query": "epimerase",
                    "selected_term": "GO:0047736",
                    "selected_reason": "best evidence term produced no candidates",
                    "candidate_count": 0,
                }
            ]
        },
    )
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="sequence_metrics",
        tool_name="sequence_analysis",
        tool_version="1.0.0",
        upstream=["resolver"],
        params_hash="jkl",
        input_hash="mno",
        cache_key="pqr",
    )
    wf_store.update_node_state(
        wf_id,
        "sequence_metrics",
        "completed",
        output={
            "candidate_capability_filter_event": {
                "status": "partitioned",
                "candidate_count": 8,
                "compatible_candidate_count": 3,
                "skipped_candidate_count": 5,
                "filtered_candidates": [
                    {
                        "candidate_id": f"cand_{idx}",
                        "source": {"tool": "go_sequences", "local_id": f"seq_{idx}"},
                        "capabilities": ["has_candidate_scores"],
                        "missing_capabilities": ["has_sequence"],
                    }
                    for idx in range(8)
                ],
            },
            "parameter_resolution_events": [
                {
                    "field": "pdb_path",
                    "source_before": "planner_literal",
                    "source_after": "upstream_candidate",
                    "reason": "protein_qc.pdb_path derived from upstream canonical candidate",
                    "value_summary": "/tmp/design.pdb",
                },
                {
                    "field": "temperature",
                    "source_before": "adapter_default",
                    "source_after": "adapter_default",
                    "reason": "adapter schema default will be used",
                    "value_summary": 0.1,
                },
            ],
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "seq_hit_1",
                    "sequence": "ACDE",
                    "scores": {
                        "semantic_relevance": {
                            "sequence_relevance_score": 0.91,
                            "term_relevance_score": 0.87,
                            "intent_id": "facet_dehydratase",
                            "intent": {
                                "intent_id": "facet_dehydratase",
                                "label": "dehydratase intent",
                                "query": "dehydratase",
                                "selected_term": "GO:0052855",
                            },
                            "selected_reason": "top relevant sequence for facet",
                        }
                    },
                }
            ],
        },
    )
    wf_store.update_state(wf_id, "completed", finished=True)

    model = collect_report_model(tmp_workspace, wf_id)
    html = render_report_html(model)

    # The collect layer still records all availability events on the model …
    assert model.empty_candidate_events[0]["label"] == "epimerase intent"
    assert model.candidate_capability_filter_events[0]["status"] == "partitioned"
    assert len(model.candidate_capability_filter_events[0]["filtered_candidates"]) == 5
    assert model.satisfied_semantic_facets[0]["label"] == "dehydratase intent"
    assert model.satisfied_semantic_facets[0]["candidate_count"] == 1
    assert model.parameter_resolution_events[0]["field"] == "pdb_path"
    assert model.graph.nodes[1].parameter_resolution_summary["derived_count"] == 1
    assert model.graph.nodes[1].parameter_resolution_summary["adapter_default_count"] == 1
    # … but the "Candidate Availability" report section is intentionally
    # suppressed in the HTML render (per user request): it was mostly no-op
    # parameter-default audit noise and added little to the report.
    assert "Candidate Availability" not in html
    assert "Satisfied facets" not in html
    assert "Parameter resolution" not in html


def test_report_surfaces_parameter_conflict_events(
    tmp_workspace: Workspace,
) -> None:
    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="parameter_conflict_demo",
        version="1.0",
        descriptor_yaml="name: parameter_conflict_demo\nversion: '1.0'\nnodes: []\n",
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
        "failed",
        error_code="invalid_params",
        error_message="rank_designs.candidates violates parameter ownership policy",
        error_details={
            "parameter_conflict_event": {
                "field": "candidates",
                "source": "planner_literal",
                "ownership": "candidate_stream",
                "status": "failed_planner_literal_not_allowed",
                "technical_failure": False,
            }
        },
    )
    wf_store.update_state(wf_id, "failed", finished=True)

    model = collect_report_model(tmp_workspace, wf_id)
    html = render_report_html(model)

    # The collect layer still records the parameter-conflict event …
    assert model.parameter_resolution_events[0]["field"] == "candidates"
    assert model.graph.nodes[0].parameter_resolution_summary["conflict_count"] == 1
    # … but the "Candidate Availability" / "Parameter resolution" report
    # section is intentionally suppressed in the HTML render (per user request).
    # (The raw event payload may still live in the embedded data blob, so we
    # only assert the visible section header is gone.)
    assert "Parameter resolution" not in html


def test_report_candidate_dedupe_preserves_scores_qc_liabilities_and_artifacts(
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "ranked.pdb"
    pdb.write_text(PDB_TEXT)
    artifact = ReportArtifact(
        field="ranked_structure",
        artifact_id="art_1",
        kind="pdb",
        mime="chemical/x-pdb",
        size_bytes=pdb.stat().st_size,
        sha256=None,
        path=str(pdb),
        job_id="job_rank",
    )
    nodes = [
        WorkflowNodeSummary(
            node_id="rank_sparse",
            tool="rank_designs",
            version="1.0.0",
            state="completed",
            job_id="job_sparse",
            output={
                "ranked_candidates": [
                    {
                        "candidate_id": "design_1",
                        "rank": 1,
                        "score": 0.4,
                        "sequence": "ACDE",
                        "qc_flags": ["low_plddt"],
                        "liabilities": [
                            {"type": "buried_polar", "chain_id": "A", "positions": [4]}
                        ],
                    }
                ]
            },
        ),
        WorkflowNodeSummary(
            node_id="rank_enriched",
            tool="rank_designs",
            version="1.0.0",
            state="completed",
            job_id="job_rank",
            output={
                "ranked_candidates": [
                    {
                        "candidate_id": "design_1",
                        "rank": 1,
                        "score": 0.8,
                        "plddt": 91.2,
                        "structure_path": str(pdb),
                        "qc_flags": ["high_instability_index"],
                        "liabilities": [
                            {"type": "exposed_hydrophobe", "chain_id": "A", "positions": [8]}
                        ],
                    }
                ]
            },
        ),
    ]

    candidates = _normalize_candidates(nodes, [artifact])

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.score == 0.8
    assert candidate.structure_path == str(pdb)
    assert candidate.scores["score"] == 0.8
    assert candidate.scores["plddt"] == 91.2
    assert set(candidate.qc_flags) == {"low_plddt", "high_instability_index"}
    assert len(candidate.liabilities) == 2
    assert [item.field for item in candidate.artifacts] == ["ranked_structure"]


def test_report_collects_and_renders_candidate_merge_audit(
    tmp_workspace: Workspace,
) -> None:
    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="merge_audit_demo",
        version="1.0",
        descriptor_yaml="name: merge_audit_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    merge_audit = {
        "duplicates_removed": 1,
        "duplicate_groups": [
            {
                "identity_key": "candidate_id:design_1",
                "identity_type": "candidate_id",
                "retained_candidate_id": "design_1",
                "merged_count": 2,
                "versions": [
                    {
                        "candidate_id": "design_1",
                        "source": {
                            "tool": "go_sequences",
                            "native_id": "GO_ACC_1",
                        },
                        "score_namespaces": ["semantic_relevance"],
                    },
                    {
                        "candidate_id": "design_1",
                        "source": {
                            "tool": "sequence_analysis",
                            "native_id": "ANALYSIS_ACC_1",
                        },
                        "score_namespaces": ["sequence_analysis"],
                    },
                ],
            }
        ],
        "conflicts": [],
    }
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="merge",
        tool_name="merge_candidates",
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    wf_store.update_node_state(
        wf_id,
        "merge",
        "completed",
        output={
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_1",
                    "sequence": "ACDE",
                    "scores": {"sequence_analysis": {"gravy": -0.1}},
                }
            ],
            "merge_audit": merge_audit,
        },
    )
    wf_store.update_state(wf_id, "completed", finished=True)

    model = collect_report_model(tmp_workspace, wf_id)
    html = render_report_html(model)

    assert model.candidate_merge_events[0]["duplicate_group_count"] == 1
    assert model.candidate_merge_events[0]["groups"][0]["source_tools"] == [
        "go_sequences",
        "sequence_analysis",
    ]
    assert "Candidate Merge Audit" in html
    assert "Duplicate Enrichment Merges" in html
    assert "design_1: type candidate_id; versions 2" in html
    assert "versions" not in str(model.candidates[0].source)


def test_candidate_merge_event_compaction_omits_raw_versions_from_event() -> None:
    node = WorkflowNodeSummary(
        node_id="merge",
        tool="merge_candidates",
        version="1.0.0",
        state="completed",
        job_id="job_merge",
        output={
            "merge_audit": {
                "duplicates_removed": 1,
                "duplicate_groups": [
                    {
                        "identity_key": "sequence:ACDE",
                        "identity_type": "sequence",
                        "retained_candidate_id": "go_hit",
                        "merged_count": 2,
                        "versions": [
                            {
                                "candidate_id": "go_hit",
                                "source": {
                                    "tool": "go_sequences",
                                    "native_id": "GO_ACC_1",
                                },
                                "score_namespaces": ["semantic_relevance"],
                            },
                            {
                                "candidate_id": "interpro_hit",
                                "source": {
                                    "tool": "interpro_sequences",
                                    "native_id": "IPR_ACC_1",
                                },
                                "score_namespaces": ["semantic_relevance"],
                            },
                        ],
                    }
                ],
            }
        },
    )

    events = _candidate_merge_events_from_nodes([node])

    assert events[0]["groups"][0]["native_ids"] == ["GO_ACC_1", "IPR_ACC_1"]
    assert "versions" not in events[0]["groups"][0]


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


def test_bundle_report_rewrites_final_candidate_structure_paths(
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    pdb = tmp_path / "final_candidate.pdb"
    pdb.write_text(PDB_TEXT)
    job_store = JobStore(tmp_workspace)
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
                "candidate_id": "design_1",
                "rank": 1,
                "structure_path": str(pdb),
                "scores": {"score": 1.0},
                "artifacts": [],
            }
        ],
        "reasoning": "selected final",
        "supersedes": None,
        "auto_selected": False,
    }
    job_store.transition(submit_job.job_id, JobState.COMPLETED, output=submission)
    ArtifactStore(tmp_workspace).register(
        submit_job.job_id,
        "final_candidate",
        pdb,
        kind="pdb",
        mime="chemical/x-pdb",
    )

    model = collect_report_model(tmp_workspace, submit_job.job_id)
    bundle_root = tmp_path / "bundle_report"
    report_path = write_report(model, bundle_root, report_format="bundle")
    html = report_path.read_text()

    assert str(pdb) not in html
    assert "structures/final_candidate_" in html


def test_report_uses_agent_standalone_submit_candidate_decision(
    tmp_workspace: Workspace,
) -> None:
    job_store = JobStore(tmp_workspace)
    rank_job = job_store.submit("rank_designs", "1.0.0", {"top_k": 1})
    job_store.transition(rank_job.job_id, JobState.QUEUED)
    job_store.transition(rank_job.job_id, JobState.RUNNING)
    ranked_output = {
        "ranked_candidates": [
            {"rank": 1, "candidate_id": "design_1", "sequence": "ACDE", "score": 1.0}
        ]
    }
    job_store.transition(rank_job.job_id, JobState.COMPLETED, output=ranked_output)

    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="post_round_submission_demo",
        version="1.0",
        descriptor_yaml="name: post_round_submission_demo\nversion: '1.0'\nnodes: []\n",
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

    submission = {
        "submission_id": "sub_post_round",
        "outcome": "accepted",
        "terminal": True,
        "candidates_resolved": [
            {
                "source_job_id": rank_job.job_id,
                "source_tool": "rank_designs",
                "candidate_id": "design_1",
                "rank": 1,
                "sequence": "ACDE",
                "score": 1.0,
                "justification": "best ranked",
                "scores": {"score": 1.0},
                "artifacts": [],
            }
        ],
        "reasoning": "post-round decision selected the top candidate",
        "supersedes": None,
        "auto_selected": False,
    }
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("agent post-round submission")
    session.update_run(run.run_id, workflow_ids=[wf_id], job_ids=[rank_job.job_id, "job_submit"])
    session.add_decision(
        run.run_id,
        {
            "decision": "submit_candidate",
            "submit_job_id": "job_submit",
            "submission": submission,
            "next_round": {"recommended": False, "reason": "ready"},
            "metrics_used": ["score"],
            "notes": [],
        },
    )

    model = collect_report_model(tmp_workspace, wf_id)

    assert model.final_submission["submission_id"] == "sub_post_round"
    assert [candidate.candidate_id for candidate in model.final_candidates] == ["design_1"]
    assert [candidate.candidate_id for candidate in model.candidates] == ["design_1"]


def test_report_prefers_persisted_candidate_archive_over_outputs_and_summaries(
    tmp_workspace: Workspace,
) -> None:
    raw_output = {
        "candidates": [
            {
                "schema_version": "candidate_v1",
                "candidate_id": "raw_candidate",
                "sequence": "RAW",
                "scores": {"raw_tool": {"score": 0.1}},
            }
        ]
    }
    job_store = JobStore(tmp_workspace)
    job = job_store.submit("go_sequences", "1.0.0", {})
    job_store.transition(job.job_id, JobState.QUEUED)
    job_store.transition(job.job_id, JobState.RUNNING)
    job_store.transition(job.job_id, JobState.COMPLETED, output=raw_output)

    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="archive_demo",
        version="1.0",
        descriptor_yaml="name: archive_demo\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="retrieve",
        tool_name="go_sequences",
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    wf_store.update_node_state(
        wf_id,
        "retrieve",
        "completed",
        job_id=job.job_id,
        output=raw_output,
    )
    wf_store.update_state(wf_id, "completed", finished=True)

    archived = {
        "schema_version": "candidate_v1",
        "candidate_id": "archive_candidate",
        "sequence": "ARCHIVE",
        "source": {
            "tool": "go_sequences",
            "node_id": "retrieve",
            "job_id": job.job_id,
            "local_id": "seq_0000",
        },
        "scores": {
            "semantic_relevance": {
                "sequence_relevance_score": 0.91,
                "score_breakdown": {"term_match_score": 0.8, "embedding_score": 0.7},
            },
            "go_sequences": {
                "term_relevance_score": 0.87,
                "evidence": {"go_id": "GO:0052855", "reviewed": True},
            },
        },
    }
    conn = tmp_workspace.connect()
    try:
        conn.execute(
            """
            CREATE TABLE candidate_records (
              workflow_id TEXT,
              job_id TEXT,
              node_id TEXT,
              candidate_json TEXT,
              created_at INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO candidate_records (
              workflow_id, job_id, node_id, candidate_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (wf_id, job.job_id, "retrieve", json.dumps(archived), 1),
        )
    finally:
        conn.close()

    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("archive report")
    session.update_run(run.run_id, workflow_ids=[wf_id], job_ids=[job.job_id])
    session.add_summary(
        run.run_id,
        {
            "workflow_id": wf_id,
            "nodes": [
                {
                    "node_id": "retrieve",
                    "tool": "go_sequences",
                    "output": {
                        "candidates": [
                            {
                                "candidate_id": "summary_only",
                                "scores": {"score_keys": ["semantic_relevance"]},
                            }
                        ]
                    },
                }
            ],
        },
        workflow_id=wf_id,
    )

    model = collect_report_model(tmp_workspace, wf_id)

    assert [candidate.candidate_id for candidate in model.candidates] == [
        "archive_candidate"
    ]
    scores = model.candidates[0].scores
    assert scores["semantic_relevance"]["score_breakdown"]["embedding_score"] == 0.7
    assert scores["go_sequences"]["evidence"]["reviewed"] is True
    assert "raw_candidate" not in {candidate.candidate_id for candidate in model.candidates}
    assert "summary_only" not in {candidate.candidate_id for candidate in model.candidates}


def test_report_falls_back_to_raw_outputs_when_candidate_archive_absent(
    tmp_workspace: Workspace,
) -> None:
    job_store = JobStore(tmp_workspace)
    job = job_store.submit("go_sequences", "1.0.0", {})
    job_store.transition(job.job_id, JobState.QUEUED)
    job_store.transition(job.job_id, JobState.RUNNING)
    output = {
        "candidates": [
            {"candidate_id": "raw_candidate", "sequence": "ACDE", "scores": {"score": 0.4}}
        ]
    }
    job_store.transition(job.job_id, JobState.COMPLETED, output=output)

    model = collect_report_model(tmp_workspace, job.job_id)

    assert [candidate.candidate_id for candidate in model.candidates] == ["raw_candidate"]


def test_report_prefers_latest_survivor_snapshot_refs(
    tmp_workspace: Workspace,
) -> None:
    wf_id, job_id = _workflow_with_raw_candidate(tmp_workspace)
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("survivor snapshot report")
    _ensure_candidate_refs_column(tmp_workspace)
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="current_candidate",
        version_id="cver_current",
        round_index=0,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="current",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "current_candidate",
            "sequence": "CURR",
            "scores": {"current_round": {"score": 0.5}},
        },
        created_at=1,
    )
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="survivor_candidate",
        version_id="cver_survivor",
        round_index=1,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="survivor",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "survivor_candidate",
            "sequence": "SURV",
            "scores": {"semantic_relevance": {"score_breakdown": {"embedding": 0.9}}},
        },
        created_at=2,
    )
    conn = tmp_workspace.connect()
    try:
        conn.execute(
            """
            INSERT INTO agent_candidate_snapshots (
              run_id, snapshot_id, round_index, kind, candidate_ids_json,
              candidate_refs_json, merge_audit_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                "csnap_current",
                0,
                "current_round",
                "[]",
                json.dumps([{"candidate_id": "current_candidate", "version_id": "cver_current"}]),
                "{}",
                10,
            ),
        )
        conn.execute(
            """
            INSERT INTO agent_candidate_snapshots (
              run_id, snapshot_id, round_index, kind, candidate_ids_json,
              candidate_refs_json, merge_audit_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                "csnap_survivor",
                1,
                "survivor",
                "[]",
                json.dumps(
                    [{"candidate_id": "survivor_candidate", "version_id": "cver_survivor"}]
                ),
                "{}",
                20,
            ),
        )
    finally:
        conn.close()

    model = collect_report_model(tmp_workspace, wf_id)

    assert [candidate.candidate_id for candidate in model.candidates] == [
        "survivor_candidate"
    ]
    candidate = model.candidates[0]
    assert candidate.scores["semantic_relevance"]["score_breakdown"]["embedding"] == 0.9
    assert candidate.source["round_index"] == 1
    assert candidate.source["source_node_id"] == "retrieve"
    assert candidate.source["source_job_id"] == job_id
    assert candidate.source["archive_snapshot_id"] == "csnap_survivor"
    assert candidate.source["archive_status"] == "survivor"


def test_report_filters_rejected_agent_candidate_records(
    tmp_workspace: Workspace,
) -> None:
    wf_id, job_id = _workflow_with_raw_candidate(tmp_workspace)
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("rejected filtering report")
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="valid_candidate",
        version_id="cver_valid",
        round_index=0,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="current",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "valid_candidate",
            "sequence": "GOOD",
            "scores": {"go_sequences": {"score": 0.7}},
        },
        created_at=1,
    )
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="bad_candidate",
        version_id="cver_bad",
        round_index=0,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="rejected",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "bad_candidate",
            "sequence": "BAD",
            "scores": {"go_sequences": {"score": 0.1}},
        },
        created_at=2,
    )

    model = collect_report_model(tmp_workspace, wf_id)

    assert [candidate.candidate_id for candidate in model.candidates] == [
        "valid_candidate"
    ]
    assert "bad_candidate" not in {candidate.candidate_id for candidate in model.candidates}


def test_report_snapshot_refs_use_exact_version_not_latest_same_id(
    tmp_workspace: Workspace,
) -> None:
    wf_id, job_id = _workflow_with_raw_candidate(tmp_workspace)
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("exact version report")
    _ensure_candidate_refs_column(tmp_workspace)
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="same_candidate",
        version_id="cver_old",
        round_index=0,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="current",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "same_candidate",
            "sequence": "OLD",
            "scores": {"go_sequences": {"score": 0.2}},
        },
        created_at=1,
    )
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="same_candidate",
        version_id="cver_new",
        round_index=1,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="current",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "same_candidate",
            "sequence": "NEW",
            "scores": {"go_sequences": {"score": 0.9}},
        },
        created_at=2,
    )
    conn = tmp_workspace.connect()
    try:
        conn.execute(
            """
            INSERT INTO agent_candidate_snapshots (
              run_id, snapshot_id, round_index, kind, candidate_ids_json,
              candidate_refs_json, merge_audit_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                "csnap_exact",
                0,
                "current_round",
                "[]",
                json.dumps([{"candidate_id": "same_candidate", "version_id": "cver_old"}]),
                "{}",
                10,
            ),
        )
    finally:
        conn.close()

    model = collect_report_model(tmp_workspace, wf_id)

    assert [candidate.candidate_id for candidate in model.candidates] == [
        "same_candidate"
    ]
    assert model.candidates[0].sequence == "OLD"
    assert model.candidates[0].scores["go_sequences"]["score"] == 0.2


def test_report_filters_snapshot_records_to_requested_workflow_scope(
    tmp_workspace: Workspace,
) -> None:
    wf_1, job_1 = _workflow_with_raw_candidate(tmp_workspace, candidate_id="raw_1")
    wf_2, job_2 = _workflow_with_raw_candidate(tmp_workspace, candidate_id="raw_2")
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("scoped survivor report")
    _ensure_candidate_refs_column(tmp_workspace)
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="scoped_1",
        version_id="cver_scoped_1",
        round_index=0,
        workflow_id=wf_1,
        node_id="retrieve",
        job_id=job_1,
        status="survivor",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "scoped_1",
            "sequence": "ONE",
        },
        created_at=1,
    )
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="scoped_2",
        version_id="cver_scoped_2",
        round_index=1,
        workflow_id=wf_2,
        node_id="retrieve",
        job_id=job_2,
        status="survivor",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "scoped_2",
            "sequence": "TWO",
        },
        created_at=2,
    )
    conn = tmp_workspace.connect()
    try:
        conn.execute(
            """
            INSERT INTO agent_candidate_snapshots (
              run_id, snapshot_id, round_index, kind, candidate_ids_json,
              candidate_refs_json, merge_audit_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                "csnap_mixed_scope",
                1,
                "survivor",
                "[]",
                json.dumps(
                    [
                        {"candidate_id": "scoped_1", "version_id": "cver_scoped_1"},
                        {"candidate_id": "scoped_2", "version_id": "cver_scoped_2"},
                    ]
                ),
                "{}",
                20,
            ),
        )
    finally:
        conn.close()

    model = collect_report_model(tmp_workspace, wf_1)

    assert [candidate.candidate_id for candidate in model.candidates] == ["scoped_1"]


def test_report_candidate_ids_legacy_fallback_is_report_only(
    tmp_workspace: Workspace,
) -> None:
    wf_id, job_id = _workflow_with_raw_candidate(tmp_workspace)
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("legacy candidate ids report")
    _ensure_candidate_refs_column(tmp_workspace)
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="legacy_same",
        version_id="cver_legacy_old",
        round_index=0,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="current",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "legacy_same",
            "sequence": "OLD",
        },
        created_at=1,
    )
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="legacy_same",
        version_id="cver_legacy_new",
        round_index=1,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="current",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "legacy_same",
            "sequence": "NEW",
        },
        created_at=2,
    )
    conn = tmp_workspace.connect()
    try:
        conn.execute(
            """
            INSERT INTO agent_candidate_snapshots (
              run_id, snapshot_id, round_index, kind, candidate_ids_json,
              candidate_refs_json, merge_audit_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                "csnap_legacy_ids",
                0,
                "current_round",
                json.dumps(["legacy_same"]),
                "[]",
                "{}",
                10,
            ),
        )
    finally:
        conn.close()

    model = collect_report_model(tmp_workspace, wf_id)

    assert [candidate.candidate_id for candidate in model.candidates] == ["legacy_same"]
    assert model.candidates[0].sequence == "NEW"


def test_report_candidate_refs_require_exact_version(
    tmp_workspace: Workspace,
) -> None:
    wf_id, job_id = _workflow_with_raw_candidate(tmp_workspace)
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("exact refs only report")
    _ensure_candidate_refs_column(tmp_workspace)
    _insert_agent_candidate_record(
        tmp_workspace,
        run_id=run.run_id,
        candidate_id="ref_without_version",
        version_id="cver_ref_latest",
        round_index=0,
        workflow_id=wf_id,
        node_id="retrieve",
        job_id=job_id,
        status="current",
        candidate={
            "schema_version": "candidate_v1",
            "candidate_id": "ref_without_version",
            "sequence": "LATEST",
        },
        created_at=1,
    )
    conn = tmp_workspace.connect()
    try:
        conn.execute(
            """
            INSERT INTO agent_candidate_snapshots (
              run_id, snapshot_id, round_index, kind, candidate_ids_json,
              candidate_refs_json, merge_audit_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                "csnap_bad_refs",
                0,
                "current_round",
                "[]",
                json.dumps([{"candidate_id": "ref_without_version"}]),
                "{}",
                10,
            ),
        )
    finally:
        conn.close()

    records = _read_candidate_archive_snapshot(
        tmp_workspace,
        workflow_id=wf_id,
        job_ids=[job_id],
        node_ids=["retrieve"],
        kind="current_round",
    )

    assert records == []


def test_report_never_uses_agent_summaries_as_candidate_source(
    tmp_workspace: Workspace,
) -> None:
    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="summary_only_candidate",
        version="1.0",
        descriptor_yaml="name: summary_only_candidate\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    wf_store.update_state(wf_id, "completed", finished=True)
    session = AgentSessionStore(tmp_workspace)
    run = session.create_run("summary candidate should not report")
    session.add_summary(
        run.run_id,
        {
            "workflow_id": wf_id,
            "nodes": [
                {
                    "node_id": "summary",
                    "tool": "summary",
                    "output": {
                        "candidates": [
                            {
                                "schema_version": "candidate_v1",
                                "candidate_id": "summary_only",
                                "sequence": "SUMMARY",
                            }
                        ],
                        "survivor_pool": [
                            {
                                "schema_version": "candidate_v1",
                                "candidate_id": "summary_survivor",
                                "sequence": "SURVIVOR",
                            }
                        ],
                        "current_round_candidates": [
                            {
                                "schema_version": "candidate_v1",
                                "candidate_id": "summary_current",
                                "sequence": "CURRENT",
                            }
                        ],
                    },
                }
            ],
        },
        workflow_id=wf_id,
    )

    model = collect_report_model(tmp_workspace, wf_id)

    assert model.candidates == []


def test_report_legacy_snapshot_payload_fallback_is_archive_snapshot_only(
    tmp_workspace: Workspace,
) -> None:
    wf_id, _job_id = _workflow_with_raw_candidate(tmp_workspace)
    conn = tmp_workspace.connect()
    try:
        conn.execute(
            """
            CREATE TABLE candidate_archive_snapshots (
              snapshot_id TEXT PRIMARY KEY,
              workflow_id TEXT,
              kind TEXT,
              payload_json TEXT,
              created_at INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO candidate_archive_snapshots (
              snapshot_id, workflow_id, kind, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "legacy_payload_snapshot",
                wf_id,
                "current_round",
                json.dumps(
                    {
                        "current_round_candidates": [
                            {
                                "schema_version": "candidate_v1",
                                "candidate_id": "legacy_payload_candidate",
                                "sequence": "LEGACY",
                            }
                        ]
                    }
                ),
                1,
            ),
        )
    finally:
        conn.close()

    model = collect_report_model(tmp_workspace, wf_id)

    assert [candidate.candidate_id for candidate in model.candidates] == [
        "legacy_payload_candidate"
    ]


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


def test_report_candidate_dedupe_keeps_richer_duplicate() -> None:
    nodes = [
        WorkflowNodeSummary(
            node_id="source",
            tool="go_sequences",
            version="*",
            state="completed",
            job_id="job_source",
            output={
                "candidates": [
                    {"candidate_id": "cand_x", "sequence": "ACDE", "score": 0.1}
                ]
            },
        ),
        WorkflowNodeSummary(
            node_id="analysis",
            tool="sequence_analysis",
            version="*",
            state="completed",
            job_id="job_analysis",
            output={
                "candidates": [
                    {
                        "candidate_id": "cand_x",
                        "sequence": "ACDE",
                        "score": 0.2,
                        "plddt": 88.0,
                    }
                ]
            },
        ),
    ]

    candidates = _normalize_candidates(nodes, [])

    assert len(candidates) == 1
    assert candidates[0].source["node_id"] == "analysis"
    assert candidates[0].scores["score"] == 0.2
    assert candidates[0].scores["plddt"] == 88.0


def test_ranked_report_dedupe_preserves_score_keys() -> None:
    nodes = [
        WorkflowNodeSummary(
            node_id="rank",
            tool="rank_designs",
            version="*",
            state="completed",
            job_id="job_rank",
            output={
                "ranked_candidates": [
                    {"candidate_id": "cand_x", "rank": 1, "score": 0.1},
                    {
                        "candidate_id": "cand_x",
                        "rank": 2,
                        "score": 0.2,
                        "plddt": 88.0,
                    },
                ]
            },
        ),
    ]

    candidates = _normalize_candidates(nodes, [])

    assert len(candidates) == 1
    assert candidates[0].rank == 2
    assert candidates[0].scores["score"] == 0.2
    assert candidates[0].scores["plddt"] == 88.0


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
