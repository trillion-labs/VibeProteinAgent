from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.reporting import collect_report_model, render_report_html
from vibeproteinagent.reporting.enrich import ensure_sequence_analysis_enrichment
from vibeproteinagent.reporting.review import build_review_model
from vibeproteinagent.workflows import WorkflowStore
from vibeproteinagent.workspace import JobState, JobStore, TrajectoryStore, Workspace


def _complete_job(
    workspace: Workspace,
    tool: str,
    output: dict,
    *,
    params: dict | None = None,
    trajectory_id: str | None = None,
) -> str:
    store = JobStore(workspace)
    job = store.submit(tool, "1.0.0", params or {}, trajectory_id=trajectory_id)
    store.transition(job.job_id, JobState.QUEUED)
    store.transition(job.job_id, JobState.RUNNING)
    store.transition(job.job_id, JobState.COMPLETED, output=output)
    return job.job_id


def _workflow_with_node(
    workspace: Workspace,
    *,
    node_id: str,
    tool: str,
    job_id: str,
    output: dict,
    descriptor_yaml: str | None = None,
    trajectory_id: str | None = None,
) -> str:
    store = WorkflowStore(workspace)
    wf_id = store.create(
        name="report_contract_demo",
        version="1.0",
        descriptor_yaml=descriptor_yaml
        or "name: report_contract_demo\nversion: '1.0'\nnodes: []\n",
        params={},
        trajectory_id=trajectory_id,
    )
    store.add_node(
        workflow_id=wf_id,
        node_id=node_id,
        tool_name=tool,
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    store.update_node_state(wf_id, node_id, "completed", job_id=job_id, output=output)
    store.update_state(wf_id, "completed", finished=True)
    return wf_id


def test_report_job_uses_only_job_state(tmp_workspace: Workspace) -> None:
    output = {
        "candidates": [
            {
                "schema_version": "candidate_v1",
                "candidate_id": "cand_job",
                "sequence": "ACDE",
                "scores": {"score": 0.4},
            }
        ]
    }
    job_id = _complete_job(tmp_workspace, "go_sequences", output)

    model = collect_report_model(tmp_workspace, job_id)
    html = render_report_html(model)

    assert model.summary.kind == "job"
    assert [candidate.candidate_id for candidate in model.candidates] == ["cand_job"]
    assert "cand_job" in html


def test_report_workflow_requires_explicit_submit_candidate(
    tmp_workspace: Workspace,
) -> None:
    ranked_output = {
        "ranked_candidates": [
            {"rank": 1, "candidate_id": "design_1", "sequence": "ACDE", "score": 1.0}
        ]
    }
    rank_job = _complete_job(tmp_workspace, "rank_designs", ranked_output)
    wf_id = _workflow_with_node(
        tmp_workspace,
        node_id="rank",
        tool="rank_designs",
        job_id=rank_job,
        output=ranked_output,
    )

    model = collect_report_model(tmp_workspace, wf_id)
    assert model.summary.kind == "workflow"
    assert model.final_submission is None
    assert model.final_candidates == []


def test_report_workflow_uses_recorded_user_query_in_html(
    tmp_workspace: Workspace,
) -> None:
    ranked_output = {
        "ranked_candidates": [
            {"rank": 1, "candidate_id": "design_1", "sequence": "ACDE", "score": 1.0}
        ]
    }
    rank_job = _complete_job(tmp_workspace, "rank_designs", ranked_output)
    wf_id = _workflow_with_node(
        tmp_workspace,
        node_id="rank",
        tool="rank_designs",
        job_id=rank_job,
        output=ranked_output,
        descriptor_yaml=(
            "name: report_contract_demo\n"
            "version: '1.0'\n"
            "user_query: Design fixed-backbone sequences for chain A.\n"
            "objective: Accept designs that satisfy explicit confidence thresholds.\n"
            "nodes: []\n"
        ),
    )

    model = collect_report_model(tmp_workspace, wf_id)
    html = render_report_html(model)

    assert model.raw_summary["user_query"] == "Design fixed-backbone sequences for chain A."
    assert (
        model.raw_summary["objective"]
        == "Accept designs that satisfy explicit confidence thresholds."
    )
    assert "Design fixed-backbone sequences for chain A." in html
    assert "User Intent</p>\n      <p>Design fixed-backbone sequences for chain A.</p>" in html
    assert "claude_opus48_design" not in html


def test_report_workflow_uses_explicit_submit_candidate(
    tmp_workspace: Workspace,
) -> None:
    submission = {
        "submission_id": "sub_accepted",
        "outcome": "accepted",
        "terminal": True,
        "candidates_resolved": [
            {
                "candidate_id": "design_1",
                "sequence": "ACDE",
                "score": 1.0,
                "justification": "passes stated threshold",
                "scores": {"score": 1.0},
            }
        ],
        "reasoning": "accepted against the objective threshold",
        "auto_selected": False,
    }
    submit_job = _complete_job(tmp_workspace, "submit_candidate", submission)
    wf_id = _workflow_with_node(
        tmp_workspace,
        node_id="submit",
        tool="submit_candidate",
        job_id=submit_job,
        output=submission,
    )

    model = collect_report_model(tmp_workspace, wf_id)
    render_report_html(model)

    assert model.final_submission["submission_id"] == "sub_accepted"
    assert model.final_submission["reasoning"] == "accepted against the objective threshold"
    assert [candidate.candidate_id for candidate in model.final_candidates] == ["design_1"]


def test_report_submit_job_expands_direct_run_trajectory(
    tmp_workspace: Workspace,
) -> None:
    _complete_job(
        tmp_workspace,
        "boltzgen",
        {
            "candidates": [
                {"candidate_id": "design_1", "sequence": "ACDE", "score": 0.4}
            ]
        },
        params={"query": "design binder"},
    )
    _complete_job(
        tmp_workspace,
        "protenix_v2",
        {
            "predictions": [
                {
                    "candidate_id": "design_1",
                    "sequence": "ACDE",
                    "predicted_cif": "/tmp/design_1.cif",
                    "confidence_scores": {"iptm": 0.64, "mean_plddt": 82.0},
                }
            ]
        },
        params={"candidate_id": "design_1"},
    )
    _complete_job(
        tmp_workspace,
        "ipsae",
        {
            "candidates": [
                {
                    "candidate_id": "design_1",
                    "sequence": "ACDE",
                    "scores": {
                        "ipsae": {
                            "ipsae_score": 0.61,
                            "ipsae_max": 0.61,
                            "interface_pae": 0.61,
                        }
                    },
                }
            ]
        },
        params={"candidate_id": "design_1"},
    )
    submission = {
        "submission_id": "sub_direct",
        "outcome": "accepted",
        "terminal": True,
        "candidates_resolved": [
            {
                "candidate_id": "design_1",
                "sequence": "ACDE",
                "score": 0.64,
                "scores": {"protenix_v2": {"iptm": 0.64, "plddt": 82.0}},
            }
        ],
        "reasoning": "accepted from direct tool trajectory",
    }
    submit_job = _complete_job(tmp_workspace, "submit_candidate", submission)

    model = collect_report_model(tmp_workspace, submit_job)
    html = render_report_html(model)

    assert model.summary.kind == "job"
    assert model.summary.name == "direct-run trajectory"
    assert [node.tool for node in model.graph.nodes] == [
        "boltzgen",
        "protenix_v2",
        "ipsae",
        "submit_candidate",
    ]
    review = build_review_model(model)
    ipsae_metric = [
        metric
        for bundle in review.bundles
        for group in bundle.score_groups
        for metric in group.metrics
        if metric[0] == "ipSAE"
    ]
    assert ipsae_metric[0][1] == "0.61"
    assert "ipSAE" in html
    assert model.final_submission["submission_id"] == "sub_direct"
    assert [candidate.candidate_id for candidate in model.final_candidates] == ["design_1"]
    assert "Recorded tool calls" in html
    assert "BoltzGen" in html
    assert "Protenix v2" in html


def test_report_functional_trajectory_separates_reference_hits_from_design_board(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="design a G alpha protein")
    _complete_job(
        tmp_workspace,
        "interpro_sequences",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": f"ipr_hit_{idx}",
                    "source": {"tool": "interpro_sequences", "local_id": f"hit_{idx}"},
                    "sequence": "REFERENCE",
                    "scores": {"interpro_sequences": {"score": 0.9}},
                }
                for idx in range(65)
            ]
        },
        trajectory_id=traj.trajectory_id,
    )
    ranked_output = {
        "candidate_count": 2,
        "ranked_candidates": [
            {
                "rank": 1,
                "candidate_id": "v40",
                "score": 2.4,
                "sequence": "ACDEFA",
                "structure_path": "/tmp/v40.cif",
                "scores": {
                    "esm2": {"score": 0.8},
                    "protenix_v2": {"plddt": 79.512, "ptm": 0.745},
                    "protein_qc": {
                        "plddt": 79.512,
                        "ca_bfactor_plddt": 83.363,
                        "flags": [],
                    },
                    "tmalign": {"tm_score_normalized": 0.88058},
                },
            },
            {
                "rank": 2,
                "candidate_id": "v25",
                "score": 2.0,
                "sequence": "ACDEFG",
                "structure_path": "/tmp/v25.cif",
                "scores": {
                    "esm2": {"score": 0.6},
                    "protenix_v2": {"plddt": 75.7, "ptm": 0.661},
                },
            },
        ],
    }
    _complete_job(
        tmp_workspace,
        "rank_designs",
        ranked_output,
        trajectory_id=traj.trajectory_id,
    )
    submission = {
        "submission_id": "sub_galpha",
        "outcome": "accepted",
        "terminal": True,
        "candidates_resolved": [
            {
                "candidate_id": "v40",
                "rank": 1,
                "score": 2.4,
                "sequence": "ACDEFA",
                "structure_path": "/tmp/v40.cif",
                "scores": ranked_output["ranked_candidates"][0]["scores"],
                "justification": "best validation evidence",
            }
        ],
        "reasoning": "accepted ranked design",
        "auto_selected": False,
    }
    submit_job = _complete_job(
        tmp_workspace,
        "submit_candidate",
        submission,
        trajectory_id=traj.trajectory_id,
    )

    model = collect_report_model(tmp_workspace, submit_job)
    html = render_report_html(model)

    assert [candidate.candidate_id for candidate in model.candidates] == ["v40", "v25"]
    assert all(not candidate.candidate_id.startswith("ipr_hit_") for candidate in model.candidates)
    assert "interpro_sequences" in {node.tool for node in model.graph.nodes}
    assert model.final_candidates[0].candidate_id == "v40"
    assert model.final_candidates[0].structure_path == "/tmp/v40.cif"
    assert model.final_candidates[0].scores["protenix_v2"]["plddt"] == pytest.approx(79.512)
    assert model.final_candidates[0].scores["protein_qc"]["ca_bfactor_plddt"] == pytest.approx(
        83.363
    )
    review = build_review_model(model)
    shown_metrics = {
        metric[0]
        for bundle in review.bundles
        for group in bundle.score_groups
        for metric in group.metrics
    }
    assert "CA B-factor pLDDT" in shown_metrics
    assert "v40" in html
    assert "v25" in html
    assert "submit_candidate-1" not in html


def test_report_keeps_generated_only_candidates_when_ranked_subset_exists(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="rank a validated subset")
    _complete_job(
        tmp_workspace,
        "bindcraft",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": f"design_{idx}",
                    "sequence": f"ACDE{idx}",
                    "scores": {"bindcraft": {"iptm": 0.1 + idx / 10}},
                }
                for idx in range(3)
            ],
            "candidate_selection_event": {
                "input_candidate_count": 3,
                "selected_candidate_ids": ["design_0"],
                "skipped_candidate_ids": ["design_1", "design_2"],
                "selection_reason": "top preliminary iptm",
                "selection_metrics": {"iptm": "descending"},
                "downstream_tool": "protenix_v2",
            },
        },
        trajectory_id=traj.trajectory_id,
    )
    ranked_output = {
        "ranked_candidates": [
            {
                "rank": 1,
                "candidate_id": "design_0",
                "sequence": "ACDE0",
                "structure_path": "/tmp/design_0.cif",
                "scores": {
                    "protenix_v2": {"plddt": 90.0},
                    "protein_qc": {"passes_filters": True},
                },
            }
        ]
    }
    _complete_job(
        tmp_workspace,
        "rank_designs",
        ranked_output,
        trajectory_id=traj.trajectory_id,
    )
    submit_job = _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_subset",
            "outcome": "accepted",
            "terminal": True,
            "candidates_resolved": [
                {
                    "candidate_id": "design_0",
                    "rank": 1,
                    "sequence": "ACDE0",
                    "structure_path": "/tmp/design_0.cif",
                    "scores": ranked_output["ranked_candidates"][0]["scores"],
                    "justification": "ranked subset winner",
                }
            ],
            "reasoning": "accepted rank 1 from evaluated subset",
        },
        trajectory_id=traj.trajectory_id,
    )

    model = collect_report_model(tmp_workspace, submit_job)
    review = build_review_model(model)
    statuses = {bundle.candidate.candidate_id: bundle.status for bundle in review.bundles}

    assert {candidate.candidate_id for candidate in model.candidates} == {
        "design_0",
        "design_1",
        "design_2",
    }
    assert statuses["design_0"] == "final"
    assert statuses["design_1"] == "generated_not_evaluated"
    assert statuses["design_2"] == "generated_not_evaluated"
    assert model.candidate_capability_filter_events[0]["event_type"] == "candidate_selection_event"
    assert model.candidate_capability_filter_events[0]["downstream_tool"] == "protenix_v2"

    html = render_report_html(model)
    assert "Generated, Not Evaluated" in html
    assert "top preliminary iptm" in html
    assert "design_1" in html
    assert "design_2" in html


def test_report_trajectory_excludes_unrelated_workspace_jobs(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="design only design_1")
    _complete_job(
        tmp_workspace,
        "boltzgen",
        {"candidates": [{"candidate_id": "unrelated", "sequence": "AAAA"}]},
        params={"query": "other query"},
    )
    _complete_job(
        tmp_workspace,
        "boltzgen",
        {"candidates": [{"candidate_id": "design_1", "sequence": "ACDE"}]},
        params={"query": "design only design_1"},
        trajectory_id=traj.trajectory_id,
    )
    _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_accepted",
            "outcome": "accepted",
            "terminal": True,
            "candidates_resolved": [
                {
                    "candidate_id": "design_1",
                    "sequence": "ACDE",
                    "justification": "passes criteria",
                }
            ],
            "reasoning": "accepted",
            "auto_selected": False,
        },
        trajectory_id=traj.trajectory_id,
    )

    model = collect_report_model(tmp_workspace, traj.trajectory_id)
    html = render_report_html(model)

    assert model.summary.kind == "trajectory"
    assert model.raw_summary["scope"] == "trajectory"
    assert "design_1" in html
    assert "unrelated" not in html


def test_report_trajectory_orders_direct_and_workflow_events(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="direct workflow direct")
    first_job = _complete_job(
        tmp_workspace,
        "boltzgen",
        {"candidates": [{"candidate_id": "first", "sequence": "AAAA"}]},
        params={"query": "first"},
        trajectory_id=traj.trajectory_id,
    )
    wf_job = _complete_job(
        tmp_workspace,
        "protein_qc",
        {"candidates": [{"candidate_id": "middle", "sequence": "CCCC"}]},
        params={"candidate_id": "middle"},
        trajectory_id=traj.trajectory_id,
    )
    wf_id = _workflow_with_node(
        tmp_workspace,
        node_id="qc",
        tool="protein_qc",
        job_id=wf_job,
        output={"candidates": [{"candidate_id": "middle", "sequence": "CCCC"}]},
        trajectory_id=traj.trajectory_id,
    )
    final_job = _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_final",
            "outcome": "accepted",
            "terminal": True,
            "candidates_resolved": [
                {"candidate_id": "middle", "sequence": "CCCC", "justification": "best"}
            ],
            "reasoning": "accepted",
            "auto_selected": False,
        },
        trajectory_id=traj.trajectory_id,
    )

    model = collect_report_model(tmp_workspace, traj.trajectory_id)

    events = model.raw_summary["top_level_events"]
    assert [(e["kind"], e["id"]) for e in events] == [
        ("job", first_job),
        ("job", wf_job),
        ("workflow", wf_id),
        ("job", final_job),
    ]
    assert all(not node.upstream for node in model.graph.nodes if node.workflow_id is None)


def test_trajectory_report_record_is_single_final_report(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="one report")
    _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
        trajectory_id=traj.trajectory_id,
    )
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"

    result = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(first),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    status = TrajectoryStore(tmp_workspace).status(traj.trajectory_id)
    assert status["report_count"] == 1
    assert status["report_path"] == str(first)
    assert status["report_matches_final_submit"] is True
    assert status["stale_report"] is False
    assert status["complete"] is True

    same = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(first),
            "--json",
        ],
    )
    assert same.exit_code == 0, same.stdout

    duplicate = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(second),
            "--json",
        ],
    )
    assert duplicate.exit_code == 1, duplicate.stdout
    assert json.loads(duplicate.stdout)["error"]["code"] == "E_CONFLICT"
    assert not second.exists()

    replaced = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(second),
            "--replace-report",
            "--json",
        ],
    )
    assert replaced.exit_code == 0, replaced.stdout
    status = TrajectoryStore(tmp_workspace).status(traj.trajectory_id)
    assert status["report_count"] == 1
    assert status["report_path"] == str(second)
    assert status["report_matches_final_submit"] is True
    assert status["complete"] is True


def test_trajectory_report_relative_path_is_idempotent(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="relative report")
    _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
        trajectory_id=traj.trajectory_id,
    )
    monkeypatch.chdir(tmp_path)

    args = [
        "report",
        traj.trajectory_id,
        "--workspace",
        tmp_workspace.name,
        "--out",
        "rel.html",
        "--json",
    ]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    status = TrajectoryStore(tmp_workspace).status(traj.trajectory_id)
    assert status["report_path"] == str((tmp_path / "rel.html").resolve())
    assert status["complete"] is True


def test_trajectory_bundle_report_uses_canonical_html_path(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="bundle report")
    _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
        trajectory_id=traj.trajectory_id,
    )
    monkeypatch.chdir(tmp_path)

    first = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--format",
            "bundle",
            "--out",
            "bundle.html",
            "--json",
        ],
    )
    second = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--format",
            "bundle",
            "--out",
            "bundle",
            "--json",
        ],
    )

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    status = TrajectoryStore(tmp_workspace).status(traj.trajectory_id)
    assert status["report_path"] == str((tmp_path / "bundle" / "report.html").resolve())
    assert status["complete"] is True


def test_pre_submit_report_cannot_complete_trajectory_after_submit(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="stale report")
    first = tmp_path / "pre_submit.html"
    result = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(first),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout

    _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_final",
            "outcome": "no_acceptable_candidate",
            "terminal": True,
            "candidates_resolved": [],
            "reasoning": "none passed",
            "auto_selected": False,
        },
        trajectory_id=traj.trajectory_id,
    )
    status = TrajectoryStore(tmp_workspace).status(traj.trajectory_id)

    assert status["complete"] is False
    assert status["report_count"] == 1
    assert status["report_final_submit_job_id"] is None
    assert status["report_matches_final_submit"] is False
    assert status["stale_report"] is True


def test_report_enriches_missing_sequence_analysis_for_candidates(
    tmp_workspace: Workspace,
) -> None:
    job_id = _complete_job(
        tmp_workspace,
        "boltzgen",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_1",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "scores": {"boltzgen": {"confidence": 0.73}},
                },
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_2",
                    "sequence": "GGGGACDEFGHIKLMNPQRS",
                    "scores": {"sequence_analysis": {"gravy": -0.1}},
                },
            ]
        },
        params={"query": "Design a compact binder for target X."},
    )
    model = collect_report_model(tmp_workspace, job_id)

    enrichment_ids = ensure_sequence_analysis_enrichment(
        tmp_workspace,
        model.summary.id,
        model,
    )
    enriched = collect_report_model(tmp_workspace, job_id)
    html = render_report_html(enriched)
    repeated_ids = ensure_sequence_analysis_enrichment(
        tmp_workspace,
        enriched.summary.id,
        enriched,
    )

    assert len(enrichment_ids) == 1
    assert repeated_ids == []
    assert [node.tool for node in enriched.graph.nodes] == ["boltzgen", "sequence_analysis"]
    by_id = {candidate.candidate_id: candidate for candidate in enriched.candidates}
    assert "sequence_analysis" in by_id["design_1"].scores
    assert "sequence_analysis" in by_id["design_2"].scores
    assert "GRAVY" in html
    assert "Instability" in html


def test_report_cli_default_does_not_run_sequence_analysis_enrichment(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    job_id = _complete_job(
        tmp_workspace,
        "boltzgen",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_1",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "scores": {"boltzgen": {"confidence": 0.73}},
                }
            ]
        },
    )
    out = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            job_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(out),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)["data"]
    assert payload["enrichment_enabled"] is False
    assert payload["enrichment_job_ids"] == []
    model = collect_report_model(tmp_workspace, job_id)
    assert "sequence_analysis" not in {node.tool for node in model.graph.nodes}


def test_report_cli_runs_sequence_analysis_enrichment_when_requested(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    job_id = _complete_job(
        tmp_workspace,
        "boltzgen",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_1",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "scores": {"boltzgen": {"confidence": 0.73}},
                }
            ]
        },
    )
    out = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            job_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(out),
            "--enrich-sequence-analysis",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)["data"]
    assert payload["enrichment_enabled"] is True
    assert len(payload["enrichment_job_ids"]) == 1
    html = out.read_text()
    assert "Sequence" in html
    assert "GRAVY" in html
    model = collect_report_model(tmp_workspace, job_id)
    assert "sequence_analysis" in {node.tool for node in model.graph.nodes}


def test_sequence_analysis_enrichment_is_idempotent_with_separate_enriched_record(
    tmp_workspace: Workspace,
) -> None:
    job_id = _complete_job(
        tmp_workspace,
        "boltzgen",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_1",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "scores": {"boltzgen": {"confidence": 0.73}},
                },
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "design_1",
                    "sequence": "ACDEFGHIKLMNPQRSTVWY",
                    "scores": {"sequence_analysis": {"gravy": -0.1}},
                    "source": {"report_enrichment": "sequence_analysis"},
                },
            ]
        },
    )
    model = collect_report_model(tmp_workspace, job_id)

    enrichment_ids = ensure_sequence_analysis_enrichment(
        tmp_workspace,
        model.summary.id,
        model,
    )

    assert enrichment_ids == []


def test_direct_submit_report_does_not_use_tool_name_as_user_query(
    tmp_workspace: Workspace,
) -> None:
    submit_job = _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_direct",
            "outcome": "accepted",
            "terminal": True,
            "candidates_resolved": [
                {
                    "candidate_id": "design_1",
                    "sequence": "ACDE",
                    "scores": {"score": 1.0},
                }
            ],
        },
    )

    model = collect_report_model(tmp_workspace, submit_job)
    html = render_report_html(model)

    assert "User query not recorded in workspace metadata." in html
    assert "User Query</p>\n      <p>submit_candidate</p>" not in html


def test_report_workflow_no_acceptable_candidate_has_no_final_candidates(
    tmp_workspace: Workspace,
) -> None:
    submission = {
        "submission_id": "sub_none",
        "outcome": "no_acceptable_candidate",
        "terminal": True,
        "candidates_resolved": [],
        "reasoning": "no design met the stated objective threshold",
        "auto_selected": False,
    }
    submit_job = _complete_job(tmp_workspace, "submit_candidate", submission)
    wf_id = _workflow_with_node(
        tmp_workspace,
        node_id="submit",
        tool="submit_candidate",
        job_id=submit_job,
        output=submission,
    )

    model = collect_report_model(tmp_workspace, wf_id)

    assert model.final_submission["outcome"] == "no_acceptable_candidate"
    assert model.final_candidates == []


def test_fresh_workspace_schema_has_no_active_agent_tables(
    tmp_workspace: Workspace,
) -> None:
    conn = tmp_workspace.connect()
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()

    assert not {
        "agent_runs",
        "agent_messages",
        "agent_tool_calls",
        "agent_plans",
        "agent_summaries",
        "agent_decisions",
    } & tables


def test_collect_report_model_rejects_bad_id(tmp_workspace: Workspace) -> None:
    with pytest.raises(InvalidParamsError):
        collect_report_model(tmp_workspace, "not_a_report_id")

    with pytest.raises(InvalidParamsError):
        collect_report_model(tmp_workspace, "agent_legacy")
