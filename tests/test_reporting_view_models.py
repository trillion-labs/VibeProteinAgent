"""Tests for derived SaaS report view models.

The view-model layer is presentation-only: it reshapes ``ReportModel`` and
``ReportReviewModel`` data without changing collection, review classification,
or rendering behavior.
"""

from __future__ import annotations

from dataclasses import replace

from vibeproteinagent.reporting.models import (
    CandidateStatus,
    Provenance,
    ReportCandidate,
    ReportModel,
    ReportWarning,
    RunSummary,
    WorkflowGraph,
    WorkflowNodeSummary,
)
from vibeproteinagent.reporting.review import build_review_model
from vibeproteinagent.reporting.view_models import (
    build_agent_rationale_view_model,
    build_candidate_board_view_model,
    build_candidate_dossier_view_model,
    build_overview_view_model,
    build_report_view_model,
    build_workflow_review_preset,
    build_workflow_trace_view_model,
)


def _node(node_id: str, tool: str, *, upstream=None, state="completed"):
    return WorkflowNodeSummary(
        node_id=node_id,
        tool=tool,
        version="1",
        state=state,
        job_id=f"job_{node_id}",
        upstream=list(upstream or []),
    )


def _model(
    *,
    nodes=None,
    candidates=None,
    finals=None,
    warnings=None,
    raw_summary=None,
    candidate_capability_filter_events=None,
):
    return ReportModel(
        summary=RunSummary(
            kind="workflow",
            id="wf_view",
            workspace="ws",
            state="completed",
            submitted_at=100,
            finished_at=200,
        ),
        graph=WorkflowGraph(workflow_id="wf_view", nodes=list(nodes or [])),
        candidates=list(candidates or []),
        final_candidates=list(finals or []),
        warnings=list(warnings or []),
        candidate_capability_filter_events=list(candidate_capability_filter_events or []),
        raw_summary=dict(raw_summary or {}),
    )


def test_overview_counts_and_provenance_from_review_model():
    candidates = [
        ReportCandidate(
            candidate_id="cand_final",
            rank=1,
            sequence="ACDEFGHIK",
            structure_path="/tmp/final.pdb",
            scores={
                "esm3": {"mean_plddt": 0.91, "ptm": 0.8},
                "boltz2_affinity": {"iptm": 0.72},
                "sequence_analysis": {"gravy": -0.2},
            },
        ),
        ReportCandidate(
            candidate_id="cand_failed",
            rank=2,
            sequence="ACDE",
            qc_flags=["low_complexity"],
            scores={"protein_qc": {"passes_filters": False}},
        ),
        ReportCandidate(candidate_id="cand_incomplete", rank=3, sequence="ACDEFG"),
    ]
    model = _model(
        nodes=[_node("n1", "pxdesign"), _node("n2", "protein_qc"), _node("n3", "esm3")],
        candidates=candidates,
        finals=[ReportCandidate(candidate_id="cand_final", sequence="ACDEFGHIK")],
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    counts = {count.label: count for count in overview.counts}
    assert counts["Generated"].value == 3
    assert counts["Passed QC"].value == 2
    assert counts["Fully Evaluated"].value == 1
    assert counts["Recommended"].value == 1
    assert counts["Recommended"].provenance == Provenance.EXACT
    assert overview.display_status == "Completed"
    assert overview.workflow_id == "wf_view"


def test_overview_counts_use_unified_review_population_when_final_is_separate():
    model = _model(
        nodes=[_node("n1", "pxdesign"), _node("n2", "esm3"), _node("n3", "rank_designs")],
        candidates=[
            ReportCandidate(
                candidate_id="cand_alt",
                rank=2,
                sequence="ACDEFG",
                scores={
                    "esm3": {"mean_plddt": 0.8, "ptm": 0.7},
                    "sequence_analysis": {"gravy": -0.2},
                    "boltz2_affinity": {"iptm": 0.6},
                },
            )
        ],
        finals=[
            ReportCandidate(
                candidate_id="cand_final",
                rank=1,
                sequence="ACDEFA",
                structure_path="/tmp/final.pdb",
                scores={
                    "esm3": {"mean_plddt": 0.9, "ptm": 0.8},
                    "sequence_analysis": {"gravy": -0.3},
                    "boltz2_affinity": {"iptm": 0.7},
                },
            )
        ],
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    counts = {count.label: count.value for count in overview.counts}
    assert counts["Generated"] == 2
    assert counts["Passed QC"] <= counts["Generated"]
    assert counts["Fully Evaluated"] <= counts["Passed QC"]
    assert counts["Recommended"] <= counts["Generated"]


def test_overview_counts_deduplicate_repeated_candidate_records():
    model = _model(
        nodes=[_node("n1", "pxdesign"), _node("n2", "protein_qc")],
        candidates=[
            ReportCandidate(
                candidate_id="cand_dup",
                rank=1,
                sequence="ACDEFG",
                scores={
                    "protein_qc": {"passes_filters": True},
                    "sequence_analysis": {"gravy": -0.2},
                },
            ),
            ReportCandidate(
                candidate_id="cand_dup",
                rank=2,
                sequence="ACDEFG",
                scores={
                    "protein_qc": {"passes_filters": True},
                    "sequence_analysis": {"gravy": -0.2},
                },
            ),
        ],
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    counts = {count.label: count.value for count in overview.counts}
    assert len(review.bundles) == 2
    assert counts["Generated"] == 1
    assert counts["Passed QC"] == 1
    assert counts["Fully Evaluated"] <= counts["Passed QC"]


def test_overview_prefers_recorded_warning_as_main_caveat():
    model = _model(
        candidates=[ReportCandidate(candidate_id="c1", sequence="ACDE")],
        warnings=[ReportWarning(level="warning", message="Structure prediction did not complete.")],
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert overview.main_caveat == "Structure prediction did not complete."


def test_overview_explains_subset_evaluation_without_selection_reason():
    model = _model(
        nodes=[_node("n1", "pxdesign"), _node("n2", "rank_designs")],
        candidates=[
            ReportCandidate(
                candidate_id="selected",
                rank=1,
                sequence="ACDE",
                scores={"protein_qc": {"passes_filters": True}},
            ),
            ReportCandidate(candidate_id="not_evaluated", sequence="WXYZ"),
        ],
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert overview.main_caveat == (
        "The agent evaluated, ranked, or submitted only a subset of generated "
        "candidates. No subset selection reason was recorded, so the remaining "
        "candidates are shown as generated but not evaluated."
    )


def test_overview_explains_recorded_subset_evaluation():
    model = _model(
        nodes=[_node("n1", "pxdesign"), _node("n2", "rank_designs")],
        candidates=[
            ReportCandidate(
                candidate_id="selected",
                rank=1,
                sequence="ACDE",
                scores={"protein_qc": {"passes_filters": True}},
            ),
            ReportCandidate(candidate_id="not_evaluated", sequence="WXYZ"),
        ],
        candidate_capability_filter_events=[
            {
                "event_type": "candidate_selection_event",
                "selected_candidate_ids": ["selected"],
                "skipped_candidate_ids": ["not_evaluated"],
                "reason": "top preliminary score",
            }
        ],
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert overview.main_caveat == (
        "The agent evaluated, ranked, or submitted only a subset of generated "
        "candidates. The remaining candidates are shown as generated but not "
        "evaluated."
    )


def test_agent_rationale_uses_recorded_workflow_query_and_graph_plan():
    model = _model(
        nodes=[_node("n1", "pxdesign"), _node("n2", "rank_designs")],
        raw_summary={
            "user_query": "Design a 70 aa binder.",
            "objective": "Accept high-confidence designs.",
        },
    )
    review = build_review_model(model)

    rationale = build_agent_rationale_view_model(model, review)

    assert rationale.user_query == "Design a 70 aa binder."
    assert rationale.stage_plan == ["Sequence Generation", "Ranking"]
    assert rationale.major_decisions == []
    assert rationale.has_full_transcript is False


def test_candidate_board_rows_map_status_metric_chips_and_download_flags():
    candidate = ReportCandidate(
        candidate_id="cand_1",
        rank=1,
        sequence="ACDEFGHIK",
        structure_path="/tmp/cand_1.pdb",
        scores={
            "esm3": {"mean_plddt": 0.92, "ptm": 0.81},
            "boltz2_affinity": {"iptm": 0.7},
            "sequence_analysis": {"gravy": -0.1, "net_charge": 1.0},
        },
    )
    model = _model(
        nodes=[_node("n1", "pxdesign"), _node("n2", "esm3"), _node("n3", "boltz2_affinity")],
        candidates=[candidate],
        finals=[ReportCandidate(candidate_id="cand_1", sequence="ACDEFGHIK")],
    )
    review = build_review_model(model)

    board = build_candidate_board_view_model(review)

    assert board.total_count == 1
    assert board.recommended_count == 1
    row = board.rows[0]
    assert row.candidate_id == "cand_1"
    assert row.status == CandidateStatus.FINAL
    assert row.recommendation_label == "Recommended"
    assert "strong recorded interface evidence" in row.short_verdict
    assert row.has_structure is True
    assert row.has_downloads is True
    assert {chip.label for chip in row.metric_chips} >= {
        "pLDDT",
        "pTM",
        "ipTM",
        "GRAVY",
        "Net charge",
        "QC",
    }
    assert [chip.label for chip in row.metric_chips[:3]] == ["ipTM", "pLDDT", "pTM"]


def test_candidate_board_renumbers_duplicate_raw_ranks_by_review_order():
    candidates = [
        ReportCandidate(
            candidate_id="cand_a",
            rank=1,
            sequence="ACDEFG",
            structure_path="/tmp/cand_a.pdb",
            scores={"esm3": {"mean_plddt": 0.90, "ptm": 0.78}},
        ),
        ReportCandidate(
            candidate_id="cand_b",
            rank=1,
            sequence="ACDEFA",
            structure_path="/tmp/cand_b.pdb",
            scores={"esm3": {"mean_plddt": 0.82, "ptm": 0.70}},
        ),
        ReportCandidate(
            candidate_id="cand_c",
            rank=1,
            sequence="ACDEFC",
            structure_path="/tmp/cand_c.pdb",
            scores={"esm3": {"mean_plddt": 0.74, "ptm": 0.62}},
        ),
    ]
    model = _model(
        nodes=[_node("design", "esm3")],
        candidates=candidates,
        finals=[ReportCandidate(candidate_id="cand_a", sequence="ACDEFG")],
    )
    review = build_review_model(model)

    board = build_candidate_board_view_model(review)

    assert [row.candidate_id for row in board.rows[:3]] == ["cand_a", "cand_b", "cand_c"]
    assert [row.rank for row in board.rows[:3]] == [1, 2, 3]


def test_candidate_dossier_surfaces_missing_evidence_and_risks():
    candidate = ReportCandidate(
        candidate_id="cand_missing",
        sequence="ACDEFGHIK",
        qc_flags=["polybasic_patch"],
        scores={"protein_qc": {"passes_filters": False}},
    )
    model = _model(nodes=[_node("n1", "pxdesign")], candidates=[candidate])
    review = build_review_model(model)

    dossier = build_candidate_dossier_view_model(review.bundles[0])

    assert dossier.status == CandidateStatus.FAILED_QC
    assert "polybasic patch" in dossier.risk_notes
    assert {item.label for item in dossier.missing_evidence} >= {
        "structure",
        "interface and binding metrics",
        "structure confidence metrics",
    }


def test_workflow_trace_groups_nodes_into_product_stages():
    model = _model(
        nodes=[
            _node("n1", "pxdesign"),
            _node("n2", "protein_qc", upstream=["n1"]),
            _node("n3", "esm3", upstream=["n2"]),
            _node("n4", "boltz2_affinity", upstream=["n3"]),
            _node("n5", "rank_designs", upstream=["n4"]),
            _node("n6", "submit_candidate", upstream=["n5"]),
        ]
    )
    review = build_review_model(model)

    trace = build_workflow_trace_view_model(model, review)

    by_key = {stage.key: stage for stage in trace.stages}
    assert by_key["generation"].tools == ["pxdesign"]
    assert by_key["qc"].tools == ["protein_qc"]
    assert by_key["structure"].tools == ["esm3"]
    assert by_key["interface"].tools == ["boltz2_affinity"]
    assert by_key["ranking"].tools == ["rank_designs"]
    assert by_key["recommendation"].tools == ["submit_candidate"]


def test_build_report_view_model_composes_sections():
    model = _model(
        nodes=[_node("n1", "search_sequence")],
        candidates=[ReportCandidate(candidate_id="hit_1", sequence="ACDEFG")],
    )
    review = build_review_model(model)

    vm = build_report_view_model(model, review)

    assert vm.overview.workflow_id == "wf_view"
    assert vm.candidate_board.total_count == 1
    assert len(vm.candidate_dossiers) == 1
    assert vm.workflow_trace.workflow_type == review.workflow_type


def test_agent_rationale_thresholds_use_short_reasoning_excerpt():
    reasoning = (
        "NO ACCEPTABLE CANDIDATE.\n\n"
        "OBJECTIVE & ACCEPTANCE CRITERIA: binder must contact V452; "
        "ipSAE>=0.7 and pLDDT>=80.\n\n"
        "EXPLORATION: this long section should not be copied into the "
        "threshold summary because it is not a threshold."
    )
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
        raw_summary={"objective": "Design a V452 binder."},
    )
    model = replace(
        model,
        final_submission={
            "outcome": "no_acceptable_candidate",
            "reasoning": reasoning,
        },
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert "binder must contact V452" in overview.agent_objective
    assert "ipSAE>=0.7" in overview.agent_objective
    assert "this long section should not be copied" not in overview.agent_objective


def test_agent_objective_prefers_raw_objective_over_submission_metadata():
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
        raw_summary={"objective": "Design a thermostable V452 binder."},
    )
    model = replace(
        model,
        final_submission={
            "outcome": "accepted",
            "objective": "Submission-level objective should not win.",
            "acceptance_criteria": "ipSAE>=0.7",
            "reasoning": (
                "OBJECTIVE & ACCEPTANCE CRITERIA: reasoning objective should "
                "not win either."
            ),
        },
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert "Objective: Design a thermostable V452 binder." in overview.agent_objective
    assert "Submission-level objective should not win" not in overview.agent_objective
    assert "reasoning objective should not win" not in overview.agent_objective


def test_agent_objective_uses_structured_final_submission_fields_when_raw_missing():
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
    )
    model = replace(
        model,
        final_submission={
            "outcome": "accepted",
            "design_objective": "Design a short binder for residue V452.",
            "acceptance_criteria": {"ipSAE": ">=0.7", "pLDDT": ">=80"},
        },
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert "Objective: Design a short binder for residue V452." in overview.agent_objective
    assert '"ipSAE": ">=0.7"' in overview.agent_objective
    assert "Objective metadata was not recorded" not in overview.agent_objective


def test_agent_rationale_uses_final_submission_as_major_decisions():
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
    )
    model = replace(
        model,
        final_submission={
            "outcome": "accepted",
            "reasoning": "Selected because PyRosetta ddG improved and fold confidence held.",
            "candidates_resolved": [
                {
                    "candidate_id": "cand",
                    "justification": "ddG -15.5 REU with TM-score 0.94.",
                }
            ],
        },
    )
    review = build_review_model(model)

    rationale = build_agent_rationale_view_model(model, review)

    labels = [item.label for item in rationale.major_decisions]
    assert labels == ["Final submission", "Selection rationale", "Candidate justification"]
    assert rationale.major_decisions[0].value == "Accepted"
    assert "PyRosetta ddG improved" in rationale.major_decisions[1].value
    assert "ddG -15.5 REU" in rationale.major_decisions[2].value


def test_agent_rationale_major_decisions_are_not_truncated():
    long_reasoning = " ".join(["selection rationale detail"] * 30) + " terminal-marker"
    long_justification = " ".join(["candidate justification detail"] * 30) + " final-marker"
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
    )
    model = replace(
        model,
        final_submission={
            "outcome": "accepted",
            "reasoning": long_reasoning,
            "candidates_resolved": [
                {
                    "candidate_id": "cand",
                    "justification": long_justification,
                }
            ],
        },
    )
    review = build_review_model(model)

    rationale = build_agent_rationale_view_model(model, review)

    assert rationale.major_decisions[1].value.endswith("terminal-marker")
    assert rationale.major_decisions[2].value.endswith("final-marker")
    assert "..." not in rationale.major_decisions[1].value
    assert "..." not in rationale.major_decisions[2].value


def test_agent_objective_uses_structured_final_submission_criteria_when_objective_missing():
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
    )
    model = replace(
        model,
        final_submission={
            "outcome": "no_acceptable_candidate",
            "acceptance_criteria": "binder contact at V452 and ipSAE>=0.7",
        },
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert (
        "Objective: Acceptance criteria: binder contact at V452 and ipSAE>=0.7"
        in overview.agent_objective
    )
    assert "No explicit numeric thresholds were captured" not in overview.agent_objective


def test_agent_objective_falls_back_to_explicit_reasoning_objective_section():
    reasoning = (
        "NO ACCEPTABLE CANDIDATE.\n\n"
        "OBJECTIVE: design a binder that contacts V452 and passes ipSAE>=0.7.\n\n"
        "EXPLORATION: this should stay out of the objective summary."
    )
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
    )
    model = replace(
        model,
        final_submission={
            "outcome": "no_acceptable_candidate",
            "reasoning": reasoning,
        },
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert (
        "Objective: design a binder that contacts V452 and passes ipSAE>=0.7."
        in overview.agent_objective
    )
    assert "this should stay out" not in overview.agent_objective


def test_agent_objective_uses_metadata_not_recorded_fallback_without_explicit_section():
    model = _model(
        nodes=[_node("submit", "submit_candidate")],
        candidates=[ReportCandidate(candidate_id="cand", sequence="ACDE")],
    )
    model = replace(
        model,
        final_submission={
            "outcome": "no_acceptable_candidate",
            "reasoning": (
                "No candidate passed the informal cutoff, but no explicit "
                "objective section was recorded."
            ),
        },
    )
    review = build_review_model(model)

    overview = build_overview_view_model(model, review)

    assert (
        "Objective metadata was not recorded; report summarizes recorded evidence."
        in overview.agent_objective
    )
    assert "No separate objective field" not in overview.agent_objective


def test_workflow_review_preset_classifies_binder_from_interface_evidence():
    model = _model(
        nodes=[
            _node("n1", "pxdesign"),
            _node("n2", "esm3"),
            _node("n3", "boltz2_affinity"),
        ],
        candidates=[
            ReportCandidate(
                candidate_id="binder_1",
                sequence="ACDEFG",
                structure_path="/tmp/binder.pdb",
                scores={"boltz2_affinity": {"iptm": 0.73}},
            )
        ],
    )
    review = build_review_model(model)

    preset = build_workflow_review_preset(model, review)

    assert preset.workflow_type == "binder"
    assert preset.confidence >= 0.9
    assert preset.metric_priority[:3] == ["ipSAE", "ipTM", "pLDDT"]
    assert preset.cross_candidate_metrics[:3] == ["ipSAE", "ipTM", "pLDDT"]
    assert preset.evidence_family_order[0] == "Interface & binding"


def test_workflow_review_preset_prioritizes_stability_objective_over_incidental_iptm():
    model = _model(
        raw_summary={
            "user_query": (
                "Protein stability engineering: improve stability while preserving "
                "the existing fold."
            )
        },
        nodes=[
            _node("n1", "sequence_edit"),
            _node("n2", "esm3"),
            _node("n3", "protenix_v2"),
            _node("n4", "pyrosetta"),
            _node("n5", "sequence_analysis"),
        ],
        candidates=[
            ReportCandidate(
                candidate_id="stab_1",
                sequence="ACDEFG",
                structure_path="/tmp/stab.pdb",
                scores={
                    "esm3": {"mean_plddt": 0.88, "ptm": 0.76},
                    "protenix_v2": {"iptm": 0.42},
                    "pyrosetta": {"ddg": -2.4, "total_score": -128.0},
                    "sequence_analysis": {
                        "gravy": -0.32,
                        "net_charge": -1.0,
                        "instability_index": 27.4,
                    },
                },
            )
        ],
    )
    review = build_review_model(model)

    preset = build_workflow_review_preset(model, review)
    board = build_candidate_board_view_model(review, preset)

    assert preset.workflow_type == "denovo"
    assert preset.label == "Stability engineering review"
    assert preset.metric_priority[:4] == [
        "ddG",
        "Mean ddG",
        "Rosetta total score",
        "pLDDT",
    ]
    assert preset.cross_candidate_metrics[:3] == [
        "ddG",
        "Mean ddG",
        "Rosetta total score",
    ]
    assert preset.evidence_family_order[0] == "Stability scoring"
    assert [chip.label for chip in board.rows[0].metric_chips[:5]] == [
        "ddG",
        "Rosetta total score",
        "pLDDT",
        "pTM",
        "Instability",
    ]
    assert "GRAVY" in [chip.label for chip in board.rows[0].metric_chips]
    assert "ipTM" not in [chip.label for chip in board.rows[0].metric_chips[:5]]


def test_workflow_review_preset_orders_denovo_for_foldability():
    model = _model(
        nodes=[_node("n1", "esm3"), _node("n2", "protein_qc")],
        candidates=[
            ReportCandidate(
                candidate_id="scaffold_1",
                sequence="ACDEFG",
                structure_path="/tmp/scaffold.pdb",
                scores={
                    "esm3": {"mean_plddt": 0.88, "ptm": 0.76},
                    "sequence_analysis": {"gravy": -0.2},
                },
            )
        ],
    )
    review = build_review_model(model)

    preset = build_workflow_review_preset(model, review)
    board = build_candidate_board_view_model(review, preset)

    assert preset.workflow_type == "denovo"
    assert preset.metric_priority[:3] == ["pLDDT", "pTM", "QC"]
    assert [chip.label for chip in board.rows[0].metric_chips[:3]] == ["pLDDT", "pTM", "GRAVY"]
    assert "structure confidence" in board.rows[0].short_verdict


def test_retrieval_preset_avoids_design_specific_missing_evidence():
    model = _model(
        nodes=[_node("n1", "search_sequence"), _node("n2", "rank_designs")],
        candidates=[ReportCandidate(candidate_id="hit_1", sequence="ACDEFG")],
    )
    review = build_review_model(model)

    preset = build_workflow_review_preset(model, review)
    dossier = build_candidate_dossier_view_model(review.bundles[0], preset)

    assert preset.workflow_type == "retrieval"
    assert "What evidence supports the answer?" == preset.scientific_question
    assert {item.label for item in dossier.missing_evidence} == {
        "retrieved evidence or source coverage"
    }
