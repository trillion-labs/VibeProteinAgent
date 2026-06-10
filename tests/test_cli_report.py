from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vibeproteinagent.cli.main import app
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.reporting import collect_report_model, render_report_html
from vibeproteinagent.reporting.collect import (
    _dedupe_candidates,
    _merge_objective_ranked_aliases_into_candidates,
)
from vibeproteinagent.reporting.enrich import (
    _candidate_to_canonical,
    _sequence_analysis_candidates,
    ensure_sequence_analysis_enrichment,
)
from vibeproteinagent.reporting.models import (
    ReportCandidate,
    ReportModel,
    RunSummary,
    WorkflowGraph,
)
from vibeproteinagent.reporting.review import build_review_model
from vibeproteinagent.reporting.view_models import build_candidate_board_view_model
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
    rank_job = _complete_job(tmp_workspace, "objective_rank", ranked_output)
    wf_id = _workflow_with_node(
        tmp_workspace,
        node_id="rank",
        tool="objective_rank",
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
    rank_job = _complete_job(tmp_workspace, "objective_rank", ranked_output)
    wf_id = _workflow_with_node(
        tmp_workspace,
        node_id="rank",
        tool="objective_rank",
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
    assert "User Query</p>" in html
    assert (
        '<div class="overview-summary-body"><p>Design fixed-backbone sequences for chain A.</p></div>'
        in html
    )
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


def test_pyrosetta_score_pair_becomes_reportable_ddg(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(
        user_query="Improve stability with S138L Q189N",
        objective="PyRosetta ddG < 0 against WT",
    )
    sidecar_logs = tmp_workspace.root / "writeup_S138L_Q189N" / "logs"
    sidecar_logs.mkdir(parents=True)
    (sidecar_logs / "combo.log").write_text(
        "=== combo confirm (flex-local ddG, n=3; neg=stabilizing) ===\n"
        "LN     ddG= -15.00 sd= 0.57 raw=[-15.8, -14.6, -14.6]\n"
    )
    _complete_job(
        tmp_workspace,
        "pyrosetta",
        {
            "mode": "score",
            "total_score": 126.3968805839819,
            "candidates": [
                {
                    "candidate_id": "wt_score",
                    "structure_path": "/tmp/sb_WT.pdb",
                    "scores": {
                        "pyrosetta": {
                            "mode": "score",
                            "total_score": 126.3968805839819,
                        }
                    },
                }
            ],
        },
        params={"pdb_path": "/tmp/sb_WT.pdb", "mode": "score"},
        trajectory_id=traj.trajectory_id,
    )
    _complete_job(
        tmp_workspace,
        "pyrosetta",
        {
            "mode": "score",
            "total_score": 110.89846864172092,
            "candidates": [
                {
                    "candidate_id": "ln_score",
                    "structure_path": "/tmp/sb_LN.pdb",
                    "scores": {
                        "pyrosetta": {
                            "mode": "score",
                            "total_score": 110.89846864172092,
                        }
                    },
                }
            ],
        },
        params={"pdb_path": "/tmp/sb_LN.pdb", "mode": "score"},
        trajectory_id=traj.trajectory_id,
    )
    submission = {
        "terminal": True,
        "outcome": "accepted",
        "submission_id": "sub_stable",
        "candidates_resolved": [
            {
                "candidate_id": "design_LN",
                "sequence": "ACDE",
                "justification": "Accepted from structured PyRosetta score outputs.",
            }
        ],
        "reasoning": "Result: design meets the stated stability criterion.",
        "auto_selected": False,
    }
    _complete_job(
        tmp_workspace,
        "submit_candidate",
        submission,
        trajectory_id=traj.trajectory_id,
    )

    model = collect_report_model(tmp_workspace, traj.trajectory_id)
    html = render_report_html(model)

    scores = model.final_candidates[0].scores["pyrosetta"]
    assert scores["wild_type_total_score"] == pytest.approx(126.3968805839819)
    assert scores["mutant_total_score"] == pytest.approx(110.89846864172092)
    assert scores["ddg_reu"] == pytest.approx(-15.498411942260972)
    assert scores["mean_ddg"] == pytest.approx(-15.0)
    assert scores["mean_ddg_sd"] == pytest.approx(0.57)
    assert "Stability scoring" in html
    assert "ddG" in html
    assert "-15.5" in html
    assert "Mean ddG" in html
    assert "-15" in html


def test_report_aliases_final_submission_to_objective_rank_row(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(
        user_query="Improve fold stability with Y139F C215V",
        objective="Select the best stable mutant after objective_rank.",
    )
    _complete_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "candidate_id": "cand_final",
                    "sequence": "ACDE",
                    "scores": {"protenix_v2": {"plddt": 89.84, "ptm": 0.943}},
                    "candidate_artifacts": {
                        "protenix_v2": {
                            "predicted_pdb": "/tmp/c139_215_sample_0.pdb"
                        }
                    },
                },
                {
                    "candidate_id": "cand_mpnn",
                    "sequence": "ACDF",
                    "structure_path": "/tmp/mpnnMin_sample_0.pdb",
                    "scores": {"protenix_v2": {"plddt": 90.07, "ptm": 0.947}},
                },
            ]
        },
        trajectory_id=traj.trajectory_id,
    )
    _complete_job(
        tmp_workspace,
        "objective_rank",
        {
            "ranking_intent": "objective",
            "objective_name": "fold_stability",
            "formula_text": "-1 * ${pyrosetta.mean_ddg}",
            "score_direction": "higher",
            "gate_summary": ["pyrosetta.mean_ddg < 0"],
            "ranked_candidates": [
                {
                    "candidate_id": "mpnnMin",
                    "rank": 1,
                    "score": 1.13745,
                    "objective_score": 1.13745,
                    "objective_name": "fold_stability",
                    "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                    "score_direction": "higher",
                    "passed_gates": [],
                    "failed_gates": [],
                    "scores": {
                        "objective_rank": {
                            "rank": 1,
                            "objective_score": 1.13745,
                            "objective_name": "fold_stability",
                            "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                            "score_direction": "higher",
                            "passed_gates": [],
                            "failed_gates": [],
                        }
                    },
                },
                {
                    "candidate_id": "c139_215",
                    "rank": 7,
                    "score": 1.13415,
                    "objective_score": 1.13415,
                    "objective_name": "fold_stability",
                    "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                    "score_direction": "higher",
                    "passed_gates": [],
                    "failed_gates": [],
                    "plddt": 89.84,
                    "ptm": 0.943,
                    "scores": {
                        "objective_rank": {
                            "rank": 7,
                            "objective_score": 1.13415,
                            "objective_name": "fold_stability",
                            "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                            "score_direction": "higher",
                            "passed_gates": [],
                            "failed_gates": [],
                        }
                    },
                },
            ]
        },
        trajectory_id=traj.trajectory_id,
    )
    _complete_job(
        tmp_workspace,
        "pyrosetta",
        {
            "mode": "relax",
            "candidates": [
                {
                    "candidate_id": "wt_relax",
                    "structure_path": "/tmp/wt_sample_0.relax.pdb",
                    "scores": {
                        "pyrosetta": {"mode": "relax", "total_score": -100.0}
                    },
                }
            ],
        },
        params={"label": "wt_relax"},
        trajectory_id=traj.trajectory_id,
    )
    _complete_job(
        tmp_workspace,
        "pyrosetta",
        {
            "mode": "relax",
            "candidates": [
                {
                    "candidate_id": "mut_relax",
                    "structure_path": "/tmp/c139_215_sample_0.relax.pdb",
                    "scores": {
                        "pyrosetta": {"mode": "relax", "total_score": -112.0}
                    },
                },
                {
                    "candidate_id": "mut_relax_seed2",
                    "structure_path": "/tmp/c139_215_sample_0.relax.pdb",
                    "scores": {
                        "pyrosetta": {"mode": "relax", "total_score": -114.0}
                    },
                }
            ],
        },
        params={"label": "mut_relax"},
        trajectory_id=traj.trajectory_id,
    )
    _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "terminal": True,
            "outcome": "accepted",
            "submission_id": "sub_alias",
            "candidates_resolved": [
                {
                    "candidate_id": "cand_final",
                    "sequence": "ACDE",
                    "candidate_artifacts": {
                        "protenix_v2": {
                            "predicted_pdb": "/tmp/c139_215_sample_0.pdb"
                        }
                    },
                    "justification": "accepted alias-matched final candidate",
                }
            ],
            "reasoning": "accepted alias-matched final candidate",
            "auto_selected": False,
        },
        trajectory_id=traj.trajectory_id,
    )

    model = collect_report_model(tmp_workspace, traj.trajectory_id)
    review = build_review_model(model)
    board = build_candidate_board_view_model(review)

    final = model.final_candidates[0]
    assert final.rank == 7
    assert final.score == pytest.approx(1.13415)
    assert final.scores["objective_rank"]["objective_score"] == pytest.approx(1.13415)
    assert final.scores["pyrosetta"]["mean_ddg"] == pytest.approx(-13.0)
    current = next(c for c in model.candidates if c.candidate_id == "cand_final")
    assert current.rank == 7
    assert current.score == pytest.approx(1.13415)
    assert current.scores["objective_rank"]["rank"] == 7
    assert current.scores["pyrosetta"]["mean_total_score"] == pytest.approx(-113.0)
    alternative = next(c for c in model.candidates if c.candidate_id == "cand_mpnn")
    assert alternative.sequence == "ACDF"
    assert alternative.structure_path == "/tmp/mpnnMin_sample_0.pdb"
    assert alternative.rank == 1
    assert alternative.score == pytest.approx(1.13745)
    assert alternative.scores["objective_rank"]["objective_score"] == pytest.approx(1.13745)
    assert not any(
        (c.source or {}).get("tool") == "objective_rank"
        for c in model.candidates
    )
    final_row = next(row for row in board.rows if row.status == "final")
    assert final_row.rank == 7
    assert "objective_score 1.134" in final_row.short_verdict
    assert "missing stability scoring metrics" not in final_row.short_verdict
    alternative_row = next(row for row in board.rows if row.candidate_id == "cand_mpnn")
    assert alternative_row.status == "ranked_alternative"
    assert "objective_score 1.137" in alternative_row.short_verdict


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


def test_report_collects_protenix_complex_input_json_as_binder_sequence(
    tmp_workspace: Workspace,
) -> None:
    input_json = tmp_workspace.root / "complex.json"
    input_json.write_text(
        json.dumps(
            [
                {
                    "name": "cplx_design",
                    "sequences": [
                        {"proteinChain": {"sequence": "A" * 809, "count": 1}},
                        {"proteinChain": {"sequence": "ACDEF" * 14, "count": 1}},
                    ],
                    "modelSeeds": [101],
                }
            ]
        )
    )
    job_id = _complete_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_complex",
                    "sequence": "A" * 809,
                    "structure_path": "/tmp/complex.cif",
                    "scores": {"protenix_v2": {"iptm": 0.65}},
                }
            ]
        },
        params={"input_json": str(input_json)},
    )

    model = collect_report_model(tmp_workspace, job_id)

    assert model.candidates[0].sequence == "ACDEF" * 14
    assert (
        model.candidates[0].source["candidate_artifacts"]["protenix_v2"][
            "complex_input_mode"
        ]
        == "input_json_target_sequence_binder_sequence"
    )


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
        "objective_rank",
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
        "objective_rank",
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


def test_report_warns_when_final_submit_skips_ranked_validation_merge(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="design miniprotein binder")
    _complete_job(
        tmp_workspace,
        "bindcraft",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_bindcraft",
                    "sequence": "ACDEFGHIK",
                    "structure_path": "/tmp/designed_pose.pdb",
                    "scores": {"bindcraft": {"iptm": 0.78}},
                }
            ]
        },
        trajectory_id=traj.trajectory_id,
    )
    _complete_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_protenix_standalone",
                    "sequence": "ACDEFGHIK",
                    "structure_path": "/tmp/full_target_complex.cif",
                    "scores": {"protenix_v2": {"iptm": 0.31, "plddt": 83.7}},
                }
            ]
        },
        trajectory_id=traj.trajectory_id,
    )
    submit_job = _complete_job(
        tmp_workspace,
        "submit_candidate",
        {
            "submission_id": "sub_direct_unmerged",
            "outcome": "accepted",
            "terminal": True,
            "candidates_resolved": [
                {
                    "candidate_id": "cand_bindcraft",
                    "sequence": "ACDEFGHIK",
                    "structure_path": "/tmp/designed_pose.pdb",
                    "scores": {"bindcraft": {"iptm": 0.78}},
                    "justification": "direct generator candidate",
                }
            ],
            "reasoning": "accepted direct generator candidate",
        },
        trajectory_id=traj.trajectory_id,
    )

    model = collect_report_model(tmp_workspace, submit_job)
    messages = {warning.message for warning in model.warnings}

    assert (
        "The agent submitted candidates without objective_rank evidence. "
        "Selection rationale may not reflect explicit gates and formula scoring."
    ) in messages
    assert (
        "The agent submitted candidates before merging validation evidence. "
        "Some Protenix/ipSAE/QC results may be shown as unmatched evidence."
    ) in messages


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


def test_report_default_path_is_workspace_root(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="default report path")
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

    result = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    expected = tmp_workspace.root / f"{traj.trajectory_id}_report.html"
    assert payload["data"]["report_html"] == str(expected)
    assert expected.exists()
    assert not (tmp_path / f"{traj.trajectory_id}_report.html").exists()
    status = TrajectoryStore(tmp_workspace).status(traj.trajectory_id)
    assert status["report_path"] == str(expected)


def test_bundle_report_default_path_is_workspace_root(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="default bundle path")
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

    result = runner.invoke(
        app,
        [
            "report",
            traj.trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--format",
            "bundle",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    expected = tmp_workspace.root / f"{traj.trajectory_id}_report_bundle" / "report.html"
    assert payload["data"]["report_html"] == str(expected)
    assert expected.exists()
    assert not (tmp_path / f"{traj.trajectory_id}_report_bundle").exists()
    status = TrajectoryStore(tmp_workspace).status(traj.trajectory_id)
    assert status["report_path"] == str(expected)


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


def test_report_cli_default_runs_sequence_analysis_enrichment(
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
    assert payload["enrichment_enabled"] is True
    assert len(payload["enrichment_job_ids"]) == 1
    html = out.read_text()
    assert "GRAVY" in html
    model = collect_report_model(tmp_workspace, job_id)
    assert "sequence_analysis" in {node.tool for node in model.graph.nodes}


@pytest.mark.parametrize(
    "removed_flag",
    ["--no-enrich-sequence-analysis", "--enrich-sequence-analysis"],
)
def test_report_cli_rejects_sequence_analysis_enrichment_flags(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
    removed_flag: str,
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
            removed_flag,
            "--json",
        ],
    )

    assert result.exit_code != 0
    assert not out.exists()
    model = collect_report_model(tmp_workspace, job_id)
    assert "sequence_analysis" not in {node.tool for node in model.graph.nodes}


def test_report_cli_enriches_sequence_analysis_for_trajectory_candidates(
    runner: CliRunner,
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    trajectory_id = TrajectoryStore(tmp_workspace).create(
        user_query="render trajectory report"
    ).trajectory_id
    _complete_job(
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
        trajectory_id=trajectory_id,
    )
    out = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            trajectory_id,
            "--workspace",
            tmp_workspace.name,
            "--out",
            str(out),
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
    model = collect_report_model(tmp_workspace, trajectory_id)
    assert "sequence_analysis" in {node.tool for node in model.graph.nodes}
    by_id = {candidate.candidate_id: candidate for candidate in model.candidates}
    assert "sequence_analysis" in by_id["design_1"].scores


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


def test_report_sequence_analysis_enrichment_strips_flat_scores_from_input() -> None:
    payload = _candidate_to_canonical(
        ReportCandidate(
            candidate_id="design_1",
            sequence="ACDEFGHIKLMNPQRSTVWY",
            source={"tool": "objective_rank", "kind": "ranked_candidates"},
            scores={
                "ligandmpnn:runtime_ms": 9263.0,
                "objective_rank": {"objective_score": 1.2},
            },
        )
    )

    assert payload["scores"] == {"objective_rank": {"objective_score": 1.2}}
    assert payload["source"] == {"tool": "objective_rank", "local_id": "design_1"}


def test_report_sequence_analysis_enrichment_keeps_same_sequence_candidate_ids() -> None:
    report_model = ReportModel(
        summary=RunSummary(kind="job", id="job_same_seq", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None),
        candidates=[
            ReportCandidate(candidate_id="design_a", sequence="ACDEFGHIK"),
            ReportCandidate(candidate_id="design_b", sequence="ACDEFGHIK"),
        ],
    )

    payloads = _sequence_analysis_candidates(report_model)

    assert [payload["candidate_id"] for payload in payloads] == ["design_a", "design_b"]


def test_objective_rank_merge_preserves_duplicate_logical_candidates_by_sequence() -> None:
    ranked = [
        ReportCandidate(
            candidate_id="dup",
            sequence="AAAA",
            rank=1,
            score=2.0,
            scores={
                "objective_rank": {
                    "rank": 1,
                    "objective_score": 2.0,
                    "ranked_candidate_id": "dup__v1_a1b2c3d4",
                }
            },
            source={
                "tool": "objective_rank",
                "ranked_candidate_id": "dup__v1_a1b2c3d4",
                "material_key": "mat_a",
            },
        ),
        ReportCandidate(
            candidate_id="dup",
            sequence="CCCC",
            rank=2,
            score=1.0,
            scores={
                "objective_rank": {
                    "rank": 2,
                    "objective_score": 1.0,
                    "ranked_candidate_id": "dup__v2_e5f6a7b8",
                }
            },
            source={
                "tool": "objective_rank",
                "ranked_candidate_id": "dup__v2_e5f6a7b8",
                "material_key": "mat_c",
            },
        ),
    ]
    candidates = [
        ReportCandidate(candidate_id="dup", sequence="AAAA"),
        ReportCandidate(candidate_id="dup", sequence="CCCC"),
    ]

    merged = _merge_objective_ranked_aliases_into_candidates(ranked, candidates)

    assert len(merged) == 2
    by_sequence = {candidate.sequence: candidate for candidate in merged}
    assert by_sequence["AAAA"].rank == 1
    assert by_sequence["AAAA"].scores["objective_rank"]["ranked_candidate_id"] == "dup__v1_a1b2c3d4"
    assert by_sequence["AAAA"].source["objective_rank_evidence"]["ranked_candidate_id"] == "dup__v1_a1b2c3d4"
    assert by_sequence["CCCC"].rank == 2
    assert by_sequence["CCCC"].scores["objective_rank"]["ranked_candidate_id"] == "dup__v2_e5f6a7b8"
    assert by_sequence["CCCC"].source["objective_rank_evidence"]["ranked_candidate_id"] == "dup__v2_e5f6a7b8"


def test_objective_rank_merge_keeps_ambiguous_alias_rows_separate() -> None:
    ranked = [
        ReportCandidate(
            candidate_id="dup",
            rank=1,
            score=2.0,
            scores={"objective_rank": {"rank": 1, "ranked_candidate_id": "dup__v1"}},
            source={"tool": "objective_rank", "ranked_candidate_id": "dup__v1"},
        ),
        ReportCandidate(
            candidate_id="dup",
            rank=2,
            score=1.0,
            scores={"objective_rank": {"rank": 2, "ranked_candidate_id": "dup__v2"}},
            source={"tool": "objective_rank", "ranked_candidate_id": "dup__v2"},
        ),
    ]
    candidates = [
        ReportCandidate(candidate_id="dup"),
        ReportCandidate(candidate_id="dup"),
    ]

    merged = _merge_objective_ranked_aliases_into_candidates(ranked, candidates)

    assert len(merged) == 4
    assert [candidate.scores for candidate in merged[:2]] == [{}, {}]
    assert {
        candidate.source.get("ranked_candidate_id")
        for candidate in merged[2:]
        if isinstance(candidate.source, dict)
    } == {"dup__v1", "dup__v2"}


def test_report_dedupe_keeps_distinct_objective_ranked_identities() -> None:
    candidates = [
        ReportCandidate(
            candidate_id="dup",
            sequence="AAAA",
            scores={"objective_rank": {"ranked_candidate_id": "dup__v1"}},
            source={
                "tool": "objective_rank",
                "ranked_candidate_id": "dup__v1",
                "local_id": "dup__v1",
            },
        ),
        ReportCandidate(
            candidate_id="dup",
            sequence="CCCC",
            scores={"objective_rank": {"ranked_candidate_id": "dup__v2"}},
            source={
                "tool": "objective_rank",
                "ranked_candidate_id": "dup__v2",
                "local_id": "dup__v2",
            },
        ),
    ]

    deduped = _dedupe_candidates(candidates)

    assert len(deduped) == 2
    assert {candidate.sequence for candidate in deduped} == {"AAAA", "CCCC"}


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
    long_reasoning = (
        "NO ACCEPTABLE CANDIDATE for a Beta-agarase binder centered on V452.\n\n"
        "OBJECTIVE & ACCEPTANCE CRITERIA: binder must directly contact target "
        "residue V452 with heavy-atom <=5.0 A; supporting targets ipSAE>=0.7, "
        "binder pLDDT>=80.\n\n"
        "KEY FINDING - V452 IS BURIED: V452 has relative solvent accessibility "
        "5.6% and 17 heavy-atom neighbors.\n\n"
        "EXPLORATION (18 binders independently validated): pxdesign and "
        "RFdiffusion explored many hotspot strategies with a very long account "
        "that should not dominate the candidate review banner.\n\n"
        "RESULT: 0/18 binders contact V452. Closest approach 17.4 A; most "
        "re-dock 20-52 A away.\n\n"
        "CONCLUSION: The literal objective is not achievable as stated - V452 "
        "is not solvent-accessible enough to serve as an interface center."
    )
    submission = {
        "submission_id": "sub_none",
        "outcome": "no_acceptable_candidate",
        "terminal": True,
        "candidates_resolved": [],
        "reasoning": long_reasoning,
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
    html = render_report_html(model)

    assert model.final_submission["outcome"] == "no_acceptable_candidate"
    assert model.final_candidates == []
    assert "The literal objective is not achievable as stated" in html
    assert "Full recorded submission reasoning" in html
    assert "EXPLORATION (18 binders independently validated)" in html
    banner = html.split('<div class="console-banner">', 1)[1].split(
        '<details class="console-banner-details">',
        1,
    )[0]
    assert "very long account that should not dominate" not in banner


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
