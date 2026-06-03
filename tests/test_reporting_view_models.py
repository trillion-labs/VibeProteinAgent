"""Tests for derived SaaS report view models.

The view-model layer is presentation-only: it reshapes ``ReportModel`` and
``ReportReviewModel`` data without changing collection, review classification,
or rendering behavior.
"""

from __future__ import annotations

from vibeproteinagent.reporting.models import (
    AgentDecision,
    AgentTranscript,
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


def _model(*, nodes=None, candidates=None, finals=None, transcript=None, warnings=None):
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
        agent_transcript=transcript,
        warnings=list(warnings or []),
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


def test_agent_rationale_uses_recorded_query_plan_and_decisions():
    transcript = AgentTranscript(
        run_id="agent_1",
        user_query="Design a 70 aa binder.",
        plan_structured={"steps": [{"name": "Generate"}, {"name": "Rank"}]},
        decisions=[
            AgentDecision(
                round_index=1,
                decision_label="completed_review_metrics",
                reason="Enough candidates passed review.",
            )
        ],
        tool_calls=[],
    )
    model = _model(transcript=transcript, nodes=[_node("n1", "pxdesign"), _node("n2", "rank_designs")])
    review = build_review_model(model)

    rationale = build_agent_rationale_view_model(model, review)

    assert rationale.user_query == "Design a 70 aa binder."
    assert rationale.stage_plan == ["Generate", "Rank"]
    assert rationale.major_decisions[0].provenance == Provenance.EXACT
    assert "Enough candidates" in rationale.major_decisions[0].value


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
    assert {chip.label for chip in row.metric_chips} >= {"pLDDT", "pTM", "ipTM", "GRAVY", "Net charge", "QC"}
    assert [chip.label for chip in row.metric_chips[:3]] == ["ipTM", "pLDDT", "pTM"]


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
    assert preset.metric_priority[:3] == ["ipTM", "pLDDT", "pTM"]
    assert preset.cross_candidate_metrics[:3] == ["ipTM", "pLDDT", "pTM"]
    assert preset.evidence_family_order[0] == "Interface & binding"


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
