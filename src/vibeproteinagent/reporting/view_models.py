"""Derived SaaS report view models.

This module is intentionally presentation-only. It translates the collected
``ReportModel`` plus the candidate-centric ``ReportReviewModel`` into stable
section data for the future report redesign. It does not collect new data, run
tools, mutate the report model, or render HTML.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .models import (
    CandidateEvidenceBundle,
    CandidateStatus,
    MSABlock,
    Provenance,
    ReportModel,
    ReportReviewModel,
    RunSummary,
    ScoreGroup,
    ToolRole,
    WorkflowGraph,
    WorkflowNodeSummary,
    WorkflowType,
)
from .review import classify_tool_role


@dataclass(frozen=True)
class CountSummary:
    label: str
    value: int
    detail: str
    provenance: str = Provenance.DERIVED


@dataclass(frozen=True)
class OverviewViewModel:
    display_status: str
    workflow_id: str
    timestamp_ms: int | None
    design_objective: str
    agent_interpretation: str
    agent_objective: str
    workflow_summary: str
    counts: list[CountSummary] = field(default_factory=list)
    main_caveat: str | None = None
    caveat_provenance: str = Provenance.DERIVED
    count_warning: str | None = None


@dataclass(frozen=True)
class AgentRationaleItem:
    label: str
    value: str
    provenance: str


@dataclass(frozen=True)
class AgentRationaleViewModel:
    user_query: str | None = None
    intent_summary: str | None = None
    workflow_selection: str | None = None
    stage_plan: list[str] = field(default_factory=list)
    major_decisions: list[AgentRationaleItem] = field(default_factory=list)
    tool_call_count: int = 0
    has_full_transcript: bool = False


@dataclass(frozen=True)
class MetricChip:
    label: str
    value: str
    unit: str | None = None
    source: str = ""
    family: str = ""
    provenance: str = Provenance.DERIVED


@dataclass(frozen=True)
class CandidateBoardRow:
    candidate_id: str
    rank: int | None
    status: str
    status_label: str
    recommendation_label: str
    short_verdict: str
    metric_chips: list[MetricChip] = field(default_factory=list)
    has_structure: bool = False
    has_downloads: bool = False
    status_provenance: str = Provenance.DERIVED


@dataclass(frozen=True)
class CandidateBoardViewModel:
    rows: list[CandidateBoardRow] = field(default_factory=list)
    recommended_count: int = 0
    total_count: int = 0


@dataclass(frozen=True)
class MissingEvidenceItem:
    label: str
    provenance: str = Provenance.DERIVED


@dataclass(frozen=True)
class CandidateDossierViewModel:
    candidate_id: str
    sequence: str | None
    status: str
    status_label: str
    structure_path: str | None
    structure_variant_count: int = 0
    download_count: int = 0
    msa_blocks: list[MSABlock] = field(default_factory=list)
    score_groups: list[ScoreGroup] = field(default_factory=list)
    metric_chips: list[MetricChip] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    missing_evidence: list[MissingEvidenceItem] = field(default_factory=list)
    provenance: str = Provenance.DERIVED


@dataclass(frozen=True)
class WorkflowStageViewModel:
    key: str
    label: str
    tools: list[str] = field(default_factory=list)
    node_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    evidence_label: str = ""
    provenance: str = Provenance.DERIVED


@dataclass(frozen=True)
class WorkflowTraceViewModel:
    workflow_type: str
    stages: list[WorkflowStageViewModel] = field(default_factory=list)


@dataclass(frozen=True)
class ReproducibilityViewModel:
    ids: list[str] = field(default_factory=list)
    replay_commands: list[str] = field(default_factory=list)
    tool_version_count: int = 0
    params_hash_count: int = 0
    artifact_hash_count: int = 0


@dataclass(frozen=True)
class WorkflowReviewPreset:
    workflow_type: str
    label: str
    confidence: float
    confidence_basis: str
    scientific_question: str
    overview_focus: str
    candidate_review_guidance: str
    comparison_note: str
    cross_candidate_focus: str
    evidence_family_order: list[str] = field(default_factory=list)
    metric_priority: list[str] = field(default_factory=list)
    cross_candidate_metrics: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportSaaSViewModel:
    review_preset: WorkflowReviewPreset
    overview: OverviewViewModel
    agent_rationale: AgentRationaleViewModel
    candidate_board: CandidateBoardViewModel
    candidate_dossiers: list[CandidateDossierViewModel]
    workflow_trace: WorkflowTraceViewModel
    reproducibility: ReproducibilityViewModel


_STATUS_LABELS = {
    CandidateStatus.FINAL: "Recommended",
    CandidateStatus.RANKED_ALTERNATIVE: "Alternative",
    CandidateStatus.EVALUATED_NOT_SELECTED: "Evaluated",
    CandidateStatus.GENERATED_NOT_EVALUATED: "Generated Only",
    CandidateStatus.REFERENCE_GROUNDING: "Reference",
    CandidateStatus.FAILED_OR_REJECTED: "Failed/Rejected",
}


_WORKFLOW_LABELS = {
    WorkflowType.BINDER: "Binder design",
    WorkflowType.DENOVO: "De novo design",
    WorkflowType.RETRIEVAL: "Retrieval",
    WorkflowType.GENERIC: "Generic workflow",
}

_OBJECTIVE_METADATA_FALLBACK = (
    "Objective metadata was not recorded; report summarizes recorded evidence."
)


def build_report_view_model(
    model: ReportModel, review: ReportReviewModel
) -> ReportSaaSViewModel:
    """Build the top-level derived view model for the redesigned report."""

    preset = build_workflow_review_preset(model, review)
    return ReportSaaSViewModel(
        review_preset=preset,
        overview=build_overview_view_model(model, review, preset),
        agent_rationale=build_agent_rationale_view_model(model, review, preset),
        candidate_board=build_candidate_board_view_model(review, preset),
        candidate_dossiers=[
            build_candidate_dossier_view_model(b, preset) for b in _ordered_bundles(review)
        ],
        workflow_trace=build_workflow_trace_view_model(model, review, preset),
        reproducibility=build_reproducibility_view_model(model),
    )


def build_workflow_review_preset(
    model: ReportModel, review: ReportReviewModel
) -> WorkflowReviewPreset:
    """Classify the scientific review task and return display priorities.

    This layer is deliberately derived from recorded execution evidence: graph
    tools, review workflow type, candidate score families, and linked
    structures. It does not inspect user text unless the upstream review layer
    has already classified the workflow from tool/evidence signals.
    """

    tools = {
        (node.tool or "").strip().lower()
        for node in model.graph.nodes
        if node.tool
    }
    families = {
        group.family
        for bundle in review.bundles
        for group in bundle.score_groups
    }
    has_structure = any(
        bool((bundle.candidate.structure_path or "").strip())
        for bundle in review.bundles
    )
    has_interface = "Interface & binding" in families or bool(
        tools & {"boltz2_affinity", "ipsae", "prodigy", "pxmeter", "vina"}
    )
    has_stability = "Stability scoring" in families or "pyrosetta" in tools
    has_sequence_biophysics = "Sequence biophysics" in families or bool(
        tools & {"sequence_analysis", "protein_qc"}
    )
    has_retrieval = bool(
        tools
        & {
            "search_sequence",
            "sequence_search",
            "semantic_search",
            "retrieve_sequence",
            "evidence_retrieval",
            "literature_search",
        }
    )
    objective_focus = _objective_focus(model)
    if objective_focus == "stability" or (
        has_stability and objective_focus != "binding"
    ):
        confidence = 0.92 if has_stability else 0.76
        return WorkflowReviewPreset(
            workflow_type=WorkflowType.DENOVO,
            label="Stability engineering review",
            confidence=confidence,
            confidence_basis=(
                "stability objective text and/or PyRosetta stability-scoring evidence"
            ),
            scientific_question=(
                "Which candidates best improve protein stability while retaining a plausible fold?"
            ),
            overview_focus=(
                "Stability engineering: prioritize recorded ddG/Rosetta stability "
                "scores, fold confidence, QC, and sequence biophysics. Interface "
                "metrics are shown only when recorded, not treated as primary."
            ),
            candidate_review_guidance=(
                "Review candidates as stability-engineering designs. ddG/Rosetta "
                "stability evidence is primary when present; pLDDT/pTM support fold "
                "plausibility, and GRAVY/net charge/instability summarize sequence risk."
            ),
            comparison_note=(
                "For stability engineering, compare ddG/Rosetta stability scores, "
                "fold confidence, QC, sequence biophysics, and recorded liabilities."
            ),
            cross_candidate_focus=(
                "Stability patterns emphasize ddG/Rosetta score ranges, pLDDT/pTM, "
                "sequence biophysics, and liability prevalence."
            ),
            evidence_family_order=[
                "Stability scoring",
                "Structure confidence",
                "Sequence biophysics",
                "Flags & liabilities",
                "Novelty & similarity",
                "Interface & binding",
            ],
            metric_priority=[
                "ddG",
                "Mean ddG",
                "Rosetta total score",
                "pLDDT",
                "pTM",
                "Instability",
                "GRAVY",
                "Net charge",
                "pI",
                "QC",
                "ipTM",
                "ipSAE",
            ],
            cross_candidate_metrics=[
                "ddG",
                "Mean ddG",
                "Rosetta total score",
                "pLDDT",
                "pTM",
                "Instability",
                "GRAVY",
                "Net charge",
                "pI",
            ],
        )
    if objective_focus == "solubility" or (
        has_sequence_biophysics and objective_focus == "developability"
    ):
        confidence = 0.88 if has_sequence_biophysics else 0.72
        return WorkflowReviewPreset(
            workflow_type=WorkflowType.DENOVO,
            label="Solubility engineering review",
            confidence=confidence,
            confidence_basis="solubility/developability objective text and sequence-biophysics evidence",
            scientific_question=(
                "Which candidates best improve solubility and sequence developability?"
            ),
            overview_focus=(
                "Solubility engineering: prioritize GRAVY, net charge, instability, "
                "pI, QC liabilities, and fold confidence when available."
            ),
            candidate_review_guidance=(
                "Review candidates as solubility/developability designs. GRAVY and "
                "charge-related sequence metrics are primary; structure confidence "
                "is supporting evidence."
            ),
            comparison_note=(
                "For solubility engineering, compare GRAVY, net charge, instability, "
                "pI, QC status, liabilities, and fold confidence where recorded."
            ),
            cross_candidate_focus=(
                "Solubility patterns emphasize hydropathy, charge, instability, pI, "
                "QC state, and liability prevalence."
            ),
            evidence_family_order=[
                "Sequence biophysics",
                "Flags & liabilities",
                "Structure confidence",
                "Stability scoring",
                "Novelty & similarity",
                "Interface & binding",
            ],
            metric_priority=[
                "GRAVY",
                "Net charge",
                "Instability",
                "pI",
                "QC",
                "pLDDT",
                "pTM",
                "ddG",
                "Mean ddG",
                "ipTM",
            ],
            cross_candidate_metrics=[
                "GRAVY",
                "Net charge",
                "Instability",
                "pI",
                "pLDDT",
                "pTM",
            ],
        )
    if review.workflow_type == WorkflowType.BINDER or has_interface or objective_focus == "binding":
        confidence = 0.92 if has_interface and has_structure else 0.78
        basis = "interface/binding tools or evidence plus structure review signals"
        return WorkflowReviewPreset(
            workflow_type=WorkflowType.BINDER,
            label="Binder design review",
            confidence=confidence,
            confidence_basis=basis,
            scientific_question="Which candidates are most likely to bind the target?",
            overview_focus=(
                "Binder design: prioritize interface evidence, complex structure "
                "confidence, pTM, affinity/contact estimates, liabilities, and novelty."
            ),
            candidate_review_guidance=(
                "Review candidates as target-binding designs. ipTM/interface evidence "
                "and complex structure confidence are primary; pLDDT is presented as "
                "model confidence, not binding evidence."
            ),
            comparison_note=(
                "For binder selection, prioritize ipTM/interface evidence, complex "
                "structure confidence, pTM, recorded affinity/contact estimates, "
                "liabilities, and panel diversity where recorded."
            ),
            cross_candidate_focus=(
                "Binder-level patterns emphasize ipTM, complex pLDDT/model confidence, "
                "pTM, QC status, missing structure/evidence prevalence, and sequence "
                "biophysics where recorded."
            ),
            evidence_family_order=[
                "Interface & binding",
                "Structure confidence",
                "Sequence biophysics",
                "Flags & liabilities",
                "Novelty & similarity",
            ],
            metric_priority=[
                "ipSAE",
                "ipTM",
                "pLDDT",
                "pTM",
                "Affinity",
                "Delta G",
                "Interface contacts",
                "GRAVY",
                "Net charge",
                "Instability",
                "pI",
                "QC",
            ],
            cross_candidate_metrics=[
                "ipSAE",
                "ipTM",
                "pLDDT",
                "pTM",
                "GRAVY",
                "Net charge",
                "Instability",
                "pI",
                "Affinity",
                "Delta G",
                "Interface contacts",
            ],
        )
    if review.workflow_type == WorkflowType.DENOVO or has_structure:
        confidence = 0.88 if has_structure else 0.72
        return WorkflowReviewPreset(
            workflow_type=WorkflowType.DENOVO,
            label="De novo design review",
            confidence=confidence,
            confidence_basis=(
                "structure-generation or structure artifact evidence without "
                "binding-specific scoring"
            ),
            scientific_question="Which candidates are most likely to fold and remain stable?",
            overview_focus=(
                "De novo design: prioritize fold confidence, QC, sequence biophysics, "
                "developability risks, and novelty."
            ),
            candidate_review_guidance=(
                "Review candidates as foldability/stability designs. Structure "
                "confidence and QC are weighted before other metrics."
            ),
            comparison_note=(
                "For de novo selection, prioritize pLDDT, pTM, QC, sequence "
                "biophysics, and recorded liabilities."
            ),
            cross_candidate_focus=(
                "De novo patterns emphasize pLDDT/pTM distributions, QC state, "
                "sequence biophysics, and liability prevalence."
            ),
            evidence_family_order=[
                "Structure confidence",
                "Flags & liabilities",
                "Sequence biophysics",
                "Novelty & similarity",
                "Interface & binding",
            ],
            metric_priority=[
                "pLDDT",
                "pTM",
                "QC",
                "Instability",
                "GRAVY",
                "Net charge",
                "pI",
                "Delta G",
                "ipTM",
            ],
            cross_candidate_metrics=["pLDDT", "pTM", "Instability", "GRAVY", "Net charge", "pI"],
        )
    if review.workflow_type == WorkflowType.RETRIEVAL or has_retrieval:
        return WorkflowReviewPreset(
            workflow_type=WorkflowType.RETRIEVAL,
            label="Retrieval evidence review",
            confidence=0.86 if has_retrieval else 0.68,
            confidence_basis="retrieval/search tools without structure-design evidence",
            scientific_question="What evidence supports the answer?",
            overview_focus=(
                "Retrieval workflow: prioritize retrieved evidence, source confidence, "
                "coverage, and consistency."
            ),
            candidate_review_guidance=(
                "Review retrieved hits as evidence records. Evidence availability and "
                "source context are weighted before design metrics."
            ),
            comparison_note=(
                "For retrieval review, compare evidence coverage, missing-source gaps, "
                "and recorded confidence before treating hits as ranked designs."
            ),
            cross_candidate_focus=(
                "Retrieval patterns emphasize evidence availability, missing-source "
                "gaps, sequence coverage, and consistency signals when recorded."
            ),
            evidence_family_order=[
                "Novelty & similarity",
                "Sequence biophysics",
                "Flags & liabilities",
                "Structure confidence",
                "Interface & binding",
            ],
            metric_priority=[
                "QC",
                "MSA mean identity",
                "Seq recovery",
                "GRAVY",
                "Net charge",
                "pLDDT",
                "pTM",
                "ipTM",
            ],
            cross_candidate_metrics=["MSA mean identity", "Seq recovery", "GRAVY", "Net charge"],
        )
    return WorkflowReviewPreset(
        workflow_type=WorkflowType.GENERIC,
        label="Generic review",
        confidence=0.50,
        confidence_basis="no workflow-specific tool/evidence pattern dominated the report",
        scientific_question="Which records have the strongest available evidence?",
        overview_focus=(
            "Generic workflow: surface available candidate evidence without assuming "
            "a task-specific metric hierarchy."
        ),
        candidate_review_guidance=(
            "Review candidates using the available recorded evidence. "
            "No workflow-specific priority was inferred."
        ),
        comparison_note=(
            "Use the available metrics and explicit missing-evidence markers; "
            "no workflow-specific priority was inferred."
        ),
        cross_candidate_focus=(
            "Generic patterns show available metric ranges and missing evidence "
            "without workflow-specific assumptions."
        ),
        evidence_family_order=[
            "Structure confidence",
            "Interface & binding",
            "Sequence biophysics",
            "Flags & liabilities",
            "Novelty & similarity",
        ],
        metric_priority=[
            "pLDDT",
            "pTM",
            "ipTM",
            "ipSAE",
            "Delta G",
            "Affinity",
            "GRAVY",
            "Net charge",
            "Instability",
            "pI",
            "QC",
        ],
        cross_candidate_metrics=[
            "pLDDT",
            "pTM",
            "ipTM",
            "ipSAE",
            "GRAVY",
            "Net charge",
            "Instability",
            "pI",
        ],
    )


def build_overview_view_model(
    model: ReportModel, review: ReportReviewModel, preset: WorkflowReviewPreset | None = None
) -> OverviewViewModel:
    preset = preset or build_workflow_review_preset(model, review)
    recorded_query = _recorded_user_query(model)
    objective = recorded_query or "User query not recorded in workspace metadata."
    interpretation = _agent_interpretation(model, review)
    generated_population = _count_population(review)
    passed_population = [
        bundle for bundle in generated_population if _qc_passes(bundle)
    ]
    evaluated_population = [
        bundle for bundle in passed_population if _is_fully_evaluated(bundle)
    ]
    recommended_population = [
        bundle for bundle in generated_population if bundle.status == CandidateStatus.FINAL
    ]
    structured_population = [
        bundle for bundle in generated_population if _has_structure_evidence(bundle)
    ]
    interface_population = [
        bundle for bundle in generated_population if _has_interface_evidence(bundle)
    ]
    qc_population = [
        bundle for bundle in generated_population if _has_qc_or_sequence_evidence(bundle)
    ]
    ranked_population = [
        bundle for bundle in generated_population if _has_rank_evidence(bundle)
    ]
    counts = [
        CountSummary(
            label="Generated",
            value=len(generated_population),
            detail="Unique review candidates",
            provenance=Provenance.DERIVED,
        ),
        CountSummary(
            label="Passed QC",
            value=len(passed_population),
            detail="Generated candidates passing QC",
            provenance=Provenance.DERIVED,
        ),
        CountSummary(
            label="Fully Evaluated",
            value=len(evaluated_population),
            detail="Passed QC with workflow evidence",
            provenance=Provenance.DERIVED,
        ),
        CountSummary(
            label="Structure",
            value=len(structured_population),
            detail="Candidates with recorded structures",
            provenance=Provenance.DERIVED,
        ),
        CountSummary(
            label="Interface/Binding",
            value=len(interface_population),
            detail="Candidates with binding/interface evidence",
            provenance=Provenance.DERIVED,
        ),
        CountSummary(
            label="QC/Sequence",
            value=len(qc_population),
            detail="Candidates with QC or sequence biophysics",
            provenance=Provenance.DERIVED,
        ),
        CountSummary(
            label="Ranked",
            value=len(ranked_population),
            detail="Candidates with recorded rank",
            provenance=Provenance.DERIVED,
        ),
        CountSummary(
            label="Recommended",
            value=len(recommended_population),
            detail="Final submitted candidates",
            provenance=Provenance.EXACT
            if model.final_candidates
            else Provenance.DERIVED,
        ),
    ]
    return OverviewViewModel(
        display_status=_display_status(model.summary.state),
        workflow_id=model.graph.workflow_id or model.summary.id,
        timestamp_ms=model.summary.finished_at or model.summary.submitted_at,
        design_objective=objective,
        agent_interpretation=_clip(f"{interpretation} {preset.overview_focus}", 260),
        agent_objective=_agent_objective_summary(model, preset),
        workflow_summary=preset.overview_focus,
        counts=counts,
        main_caveat=_main_caveat(model, review),
        count_warning=_count_warning(model, generated_population),
    )


def build_agent_rationale_view_model(
    model: ReportModel,
    review: ReportReviewModel,
    preset: WorkflowReviewPreset | None = None,
) -> AgentRationaleViewModel:
    preset = preset or build_workflow_review_preset(model, review)
    workflow_selection = (
        f"{preset.label}: {preset.scientific_question}"
    )
    recorded_query = _recorded_user_query(model)
    recorded_objective = _recorded_objective(model)
    intent = _first_nonempty(recorded_query, recorded_objective)
    return AgentRationaleViewModel(
        user_query=intent or None,
        intent_summary=_clip(intent, 220) or None,
        workflow_selection=workflow_selection,
        stage_plan=_stage_plan_from_graph(model.graph.nodes),
        major_decisions=_major_decisions_from_submission(model),
        has_full_transcript=False,
    )


def build_candidate_board_view_model(
    review: ReportReviewModel, preset: WorkflowReviewPreset | None = None
) -> CandidateBoardViewModel:
    preset = preset or _preset_from_review_only(review)
    bundles = _ordered_bundles(review)
    rows = [
        build_candidate_board_row(bundle, preset, display_rank=index)
        for index, bundle in enumerate(bundles, start=1)
    ]
    return CandidateBoardViewModel(
        rows=rows,
        recommended_count=sum(1 for row in rows if row.status == CandidateStatus.FINAL),
        total_count=len(rows),
    )


def build_candidate_board_row(
    bundle: CandidateEvidenceBundle,
    preset: WorkflowReviewPreset | None = None,
    display_rank: int | None = None,
) -> CandidateBoardRow:
    preset = preset or _preset_for_workflow_type(bundle.workflow_type)
    candidate = bundle.candidate
    label = _status_label(bundle.status)
    return CandidateBoardRow(
        candidate_id=candidate.candidate_id,
        rank=display_rank if display_rank is not None else (
            bundle.display_rank if bundle.display_rank is not None else candidate.rank
        ),
        status=bundle.status,
        status_label=label,
        recommendation_label=label,
        short_verdict=_candidate_verdict(bundle, preset),
        metric_chips=_metric_chips(bundle, preset),
        has_structure=bool(candidate.structure_path),
        has_downloads=bool(candidate.structure_path or candidate.artifacts or bundle.artifacts),
        status_provenance=bundle.status_basis,
    )


def build_candidate_dossier_view_model(
    bundle: CandidateEvidenceBundle,
    preset: WorkflowReviewPreset | None = None,
) -> CandidateDossierViewModel:
    preset = preset or _preset_for_workflow_type(bundle.workflow_type)
    candidate = bundle.candidate
    return CandidateDossierViewModel(
        candidate_id=candidate.candidate_id,
        sequence=candidate.sequence,
        status=bundle.status,
        status_label=_status_label(bundle.status),
        structure_path=candidate.structure_path,
        structure_variant_count=len(bundle.structure_variants),
        download_count=(
            len(candidate.artifacts)
            + len(bundle.artifacts)
            + (1 if candidate.structure_path else 0)
        ),
        msa_blocks=list(bundle.msa_blocks),
        score_groups=_ordered_score_groups(bundle.score_groups, preset),
        metric_chips=_metric_chips(bundle, preset),
        risk_notes=_risk_notes(bundle),
        missing_evidence=[
            MissingEvidenceItem(label=item, provenance=Provenance.DERIVED)
            for item in _missing_evidence(bundle, preset)
        ],
        provenance=bundle.status_basis,
    )


def build_workflow_trace_view_model(
    model: ReportModel,
    review: ReportReviewModel,
    preset: WorkflowReviewPreset | None = None,
) -> WorkflowTraceViewModel:
    preset = preset or build_workflow_review_preset(model, review)
    stage_nodes: dict[str, list[WorkflowNodeSummary]] = {
        "engineering": [],
        "generation": [],
        "qc": [],
        "structure": [],
        "structure_comparison": [],
        "energy": [],
        "interface": [],
        "ranking": [],
        "recommendation": [],
        "other": [],
    }
    for node in model.graph.nodes:
        stage_nodes[_stage_key(node)].append(node)
    stages = [
        _stage_view(
            "engineering",
            "Engineering",
            stage_nodes["engineering"],
            "Edited candidate sequences",
        ),
        _stage_view(
            "generation",
            "Sequence Generation",
            stage_nodes["generation"],
            "Sequence model output",
        ),
        _stage_view("qc", "Sequence QC", stage_nodes["qc"], "QC flags and biophysical filters"),
        _stage_view(
            "structure",
            "Structure Prediction",
            stage_nodes["structure"],
            "Structure confidence",
        ),
        _stage_view(
            "structure_comparison",
            "Structure Comparison",
            stage_nodes["structure_comparison"],
            "Fold preservation evidence",
        ),
        _stage_view(
            "energy",
            "Energy Analysis",
            stage_nodes["energy"],
            "Rosetta energy and stability evidence",
        ),
        _stage_view(
            "interface",
            "Interface Scoring",
            stage_nodes["interface"],
            "Binding/interface evidence",
        ),
        _stage_view("ranking", "Ranking", stage_nodes["ranking"], "Ranked candidate scores"),
        _stage_view(
            "recommendation",
            "Recommendation",
            stage_nodes["recommendation"],
            "Final candidate selection",
        ),
    ]
    if stage_nodes["other"]:
        stages.append(
            _stage_view(
                "other",
                "Other Evidence",
                stage_nodes["other"],
                "Additional workflow evidence",
            )
        )
    return WorkflowTraceViewModel(workflow_type=preset.workflow_type, stages=stages)


def build_reproducibility_view_model(model: ReportModel) -> ReproducibilityViewModel:
    repro = model.reproducibility
    if repro is None:
        return ReproducibilityViewModel(ids=[model.summary.id])
    return ReproducibilityViewModel(
        ids=list(repro.ids),
        replay_commands=list(repro.replay_commands),
        tool_version_count=len(repro.tool_versions),
        params_hash_count=len(repro.params_hashes),
        artifact_hash_count=len(repro.artifact_sha256),
    )


def _ordered_bundles(review: ReportReviewModel) -> list[CandidateEvidenceBundle]:
    bundles: list[CandidateEvidenceBundle] = []
    for group in review.board_groups:
        bundles.extend(group.bundles)
    represented = {id(bundle) for bundle in bundles}
    bundles.extend(bundle for bundle in review.bundles if id(bundle) not in represented)
    return bundles


def _count_population(review: ReportReviewModel) -> list[CandidateEvidenceBundle]:
    """Unique candidate population used by all Overview funnel counts.

    The renderer keeps final picks and explored candidates in separate lists
    after display deduplication. ``ReportReviewModel`` is the only layer that
    has already reconciled those records into one candidate board population,
    including final-only records. Use that single population for every Overview
    count so the funnel cannot report impossible combinations such as more
    passed-QC candidates than generated candidates.
    """

    population: list[CandidateEvidenceBundle] = []
    seen: set[str] = set()
    for bundle in _ordered_bundles(review):
        candidate = bundle.candidate
        key = candidate.candidate_id or f"seq:{candidate.sequence or id(bundle)}"
        if key in seen:
            continue
        seen.add(key)
        population.append(bundle)
    return population


def _score_families(bundle: CandidateEvidenceBundle) -> set[str]:
    return {group.family for group in bundle.score_groups}


def _has_structure_evidence(bundle: CandidateEvidenceBundle) -> bool:
    return bool(bundle.candidate.structure_path) or "Structure confidence" in _score_families(
        bundle
    )


def _has_interface_evidence(bundle: CandidateEvidenceBundle) -> bool:
    return "Interface & binding" in _score_families(bundle)


def _has_qc_or_sequence_evidence(bundle: CandidateEvidenceBundle) -> bool:
    families = _score_families(bundle)
    return bool({"Sequence biophysics", "Flags & liabilities"} & families) or "protein_qc" in (
        bundle.candidate.scores or {}
    )


def _has_rank_evidence(bundle: CandidateEvidenceBundle) -> bool:
    return bundle.display_rank is not None or bundle.candidate.rank is not None


def _metric_chips(
    bundle: CandidateEvidenceBundle,
    preset: WorkflowReviewPreset | None = None,
    *,
    limit: int = 7,
) -> list[MetricChip]:
    preset = preset or _preset_for_workflow_type(bundle.workflow_type)
    preferred = list(preset.metric_priority)
    chips: list[MetricChip] = []
    by_label: dict[str, MetricChip] = {}
    for group in bundle.score_groups:
        for label, value, unit, source in group.metrics:
            normalized = _metric_label(label)
            chip = MetricChip(
                label=normalized,
                value=str(value),
                unit=unit,
                source=source,
                family=group.family,
                provenance=group.provenance,
            )
            by_label.setdefault(normalized, chip)
    for label in preferred:
        chip = by_label.get(label)
        if chip is not None:
            chips.append(chip)
    if bundle.status == CandidateStatus.FAILED_QC:
        chips.append(MetricChip(label="QC", value="Fail", family="Flags & liabilities"))
    elif bundle.candidate.qc_flags or bundle.candidate.liabilities:
        chips.append(MetricChip(label="QC", value="Review", family="Flags & liabilities"))
    else:
        chips.append(MetricChip(label="QC", value="Pass", family="Flags & liabilities"))
    return chips[:limit]


def _metric_label(label: str) -> str:
    if label == "\u0394G":
        return "Delta G"
    if label.lower() in {"net charge", "charge"}:
        return "Net charge"
    return label


def _candidate_verdict(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> str:
    preset = preset or _preset_for_workflow_type(bundle.workflow_type)
    strengths = _candidate_strength_phrases(bundle, preset)
    missing = _missing_evidence(bundle, preset)
    if bundle.status == CandidateStatus.FINAL:
        if preset.workflow_type == WorkflowType.BINDER:
            primary = _binder_primary_evidence_summary(strengths)
            if primary:
                return f"Recommended due to {primary}{_missing_tail(missing)}."
        if strengths:
            return f"Recommended final candidate with {strengths[0]}{_missing_tail(missing)}."
        metric = _headline_metric(bundle)
        return (
            "Recommended final candidate"
            f"{': ' + metric if metric else ''}{_missing_tail(missing)}."
        )
    if bundle.status == CandidateStatus.RANKED_ALTERNATIVE:
        if strengths:
            return f"Alternative with {strengths[0]}{_missing_tail(missing)}."
        metric = _headline_metric(bundle)
        return f"Alternative{': ' + metric if metric else ''}{_missing_tail(missing)}."
    if bundle.status == CandidateStatus.EVALUATED_NOT_SELECTED:
        if strengths:
            return f"Evaluated but not selected: {strengths[0]}{_missing_tail(missing)}."
        metric = _headline_metric(bundle)
        return (
            "Evaluated but not selected"
            f"{': ' + metric if metric else ''}{_missing_tail(missing)}."
        )
    if bundle.status == CandidateStatus.GENERATED_NOT_EVALUATED:
        return "Generated candidate; downstream evaluation was not recorded."
    if bundle.status == CandidateStatus.REFERENCE_GROUNDING:
        return "Reference or grounding evidence, not a submitted design candidate."
    if bundle.status == CandidateStatus.FAILED_OR_REJECTED:
        flags = _risk_notes(bundle)
        return (
            f"Failed or rejected: {flags[0]}."
            if flags
            else "Failed or rejected during recorded review."
        )
    return "Candidate evidence available for review."


def _binder_primary_evidence_summary(strengths: list[str]) -> str:
    interface = next((s for s in strengths if "interface evidence" in s), "")
    structure = next((s for s in strengths if "structure confidence" in s), "")
    if interface and structure:
        return f"{interface} and {structure}"
    return interface or structure


def _candidate_strength_phrases(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> list[str]:
    preset = preset or _preset_for_workflow_type(bundle.workflow_type)
    chips = _metric_chips(bundle, preset, limit=8)
    by_label = {chip.label: chip for chip in chips}
    phrases: list[str] = []
    ipsae = _chip_float(by_label.get("ipSAE"))
    iptm = _chip_float(by_label.get("ipTM"))
    if preset.workflow_type == WorkflowType.BINDER and ipsae is not None:
        if ipsae >= 0.50:
            phrases.append("strong recorded interface evidence")
        elif ipsae >= 0.30:
            phrases.append("usable recorded interface evidence")
    elif preset.workflow_type == WorkflowType.BINDER and iptm is not None:
        if iptm >= 0.70:
            phrases.append("strong recorded interface evidence")
        elif iptm >= 0.50:
            phrases.append("usable recorded interface evidence")
    affinity = by_label.get("Affinity")
    if preset.workflow_type == WorkflowType.BINDER and affinity is not None:
        phrases.append("recorded affinity estimate")
    delta_g = by_label.get("Delta G")
    if preset.workflow_type == WorkflowType.BINDER and delta_g is not None:
        phrases.append("recorded binding energy estimate")
    ddg = _chip_float(by_label.get("ddG")) or _chip_float(by_label.get("Mean ddG"))
    if ddg is not None and "stability" in preset.label.lower():
        if ddg < 0:
            phrases.append("stabilizing recorded ddG/Rosetta evidence")
        else:
            phrases.append("recorded ddG/Rosetta stability evidence")
    plddt = _chip_float(by_label.get("pLDDT"))
    if plddt is not None:
        if plddt >= 90:
            if preset.workflow_type == WorkflowType.BINDER:
                phrases.append("high complex structure confidence")
            else:
                phrases.append("high predicted structure confidence")
        elif plddt >= 70:
            if preset.workflow_type == WorkflowType.BINDER:
                phrases.append("moderate complex structure confidence")
            else:
                phrases.append("moderate predicted structure confidence")
    ptm = _chip_float(by_label.get("pTM"))
    if ptm is not None and ptm >= 0.75:
        if preset.workflow_type == WorkflowType.DENOVO:
            phrases.append("strong foldability signal")
        else:
            phrases.append("strong global fold confidence")
    qc = by_label.get("QC")
    if qc and qc.value == "Pass":
        phrases.append("QC pass and no recorded liabilities")
    if preset.workflow_type == WorkflowType.RETRIEVAL:
        evidence_count = len(bundle.score_groups) + len(bundle.artifacts) + len(bundle.msa_blocks)
        if evidence_count:
            phrases.insert(0, "recorded retrieval evidence")
        elif bundle.status == CandidateStatus.FINAL:
            phrases.insert(0, "recorded final retrieval hit")
    return _dedup(phrases)


def _missing_tail(missing: list[str]) -> str:
    if not missing:
        return ""
    return f"; missing {missing[0]}"


def _chip_float(chip: MetricChip | None) -> float | None:
    if chip is None:
        return None
    try:
        return float(str(chip.value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _headline_metric(bundle: CandidateEvidenceBundle) -> str:
    for chip in _metric_chips(bundle, limit=4):
        if chip.label == "QC":
            continue
        unit = f" {chip.unit}" if chip.unit else ""
        return f"{chip.label} {chip.value}{unit}"
    if bundle.candidate.score is not None:
        return f"score {bundle.candidate.score:g}"
    return ""


def _ordered_score_groups(
    groups: list[ScoreGroup], preset: WorkflowReviewPreset
) -> list[ScoreGroup]:
    order = {family: idx for idx, family in enumerate(preset.evidence_family_order)}
    return sorted(groups, key=lambda group: (order.get(group.family, 99), group.family))


def _risk_notes(bundle: CandidateEvidenceBundle) -> list[str]:
    notes: list[str] = []
    for flag in bundle.candidate.qc_flags:
        notes.append(str(flag).replace("_", " "))
    for liability in bundle.candidate.liabilities:
        if isinstance(liability, dict):
            label = liability.get("label") or liability.get("type") or liability.get("name")
            if label:
                notes.append(str(label).replace("_", " "))
        elif liability:
            notes.append(str(liability).replace("_", " "))
    if bundle.status == CandidateStatus.GENERATED_NOT_EVALUATED:
        notes.extend(_missing_evidence(bundle))
    return _dedup(notes)[:6]


def _missing_evidence(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> list[str]:
    workflow_type = preset.workflow_type if preset else bundle.workflow_type
    families = {group.family for group in bundle.score_groups}
    missing: list[str] = []
    if workflow_type == WorkflowType.RETRIEVAL:
        if not (
            bundle.score_groups
            or bundle.artifacts
            or bundle.tool_evidence
            or bundle.msa_blocks
        ):
            missing.append("retrieved evidence or source coverage")
        return _dedup(missing)
    if (
        workflow_type in {WorkflowType.BINDER, WorkflowType.DENOVO}
        and not bundle.candidate.structure_path
    ):
        missing.append("structure")
    if workflow_type == WorkflowType.BINDER and "Interface & binding" not in families:
        missing.append("interface and binding metrics")
    if (
        preset is not None
        and "stability" in preset.label.lower()
        and "Stability scoring" not in families
    ):
        missing.append("stability scoring metrics")
    if (
        workflow_type in {WorkflowType.BINDER, WorkflowType.DENOVO}
        and "Structure confidence" not in families
    ):
        missing.append("structure confidence metrics")
    if "Sequence biophysics" not in families:
        missing.append("sequence biophysics metrics")
    if (
        workflow_type != WorkflowType.GENERIC
        and not bundle.msa_blocks
        and "Novelty & similarity" not in families
    ):
        missing.append("novelty or similarity evidence")
    if (
        bundle.status not in {CandidateStatus.FINAL, CandidateStatus.RANKED_ALTERNATIVE}
        and _has_rank_evidence(bundle) is False
    ):
        missing.append("rank")
    return _dedup(missing)


def _is_fully_evaluated(bundle: CandidateEvidenceBundle) -> bool:
    if bundle.status in {
        CandidateStatus.FAILED_OR_REJECTED,
        CandidateStatus.GENERATED_NOT_EVALUATED,
    }:
        return False
    blocking = {
        "structure",
        "interface and binding metrics",
        "structure confidence metrics",
        "sequence biophysics metrics",
    }
    return not any(item in blocking for item in _missing_evidence(bundle))


def _qc_passes(bundle: CandidateEvidenceBundle) -> bool:
    return bundle.status != CandidateStatus.FAILED_QC


def _count_warning(
    model: ReportModel, generated_population: list[CandidateEvidenceBundle]
) -> str | None:
    raw_unique_ids = {
        candidate.candidate_id
        for candidate in [*model.candidates, *model.final_candidates]
        if candidate.candidate_id
    }
    if raw_unique_ids and len(raw_unique_ids) != len(generated_population):
        return (
            "Counts use the unified candidate review population because final "
            "picks and explored records are stored separately."
        )
    return None


def _main_caveat(model: ReportModel, review: ReportReviewModel) -> str | None:
    if model.warnings:
        return _clip(model.warnings[0].message, 180)
    failed = sum(1 for b in review.bundles if b.status == CandidateStatus.FAILED_OR_REJECTED)
    generated_only = sum(
        1 for b in review.bundles if b.status == CandidateStatus.GENERATED_NOT_EVALUATED
    )
    subset_evaluated = sum(
        1
        for b in review.bundles
        if b.status
        in {
            CandidateStatus.FINAL,
            CandidateStatus.RANKED_ALTERNATIVE,
            CandidateStatus.EVALUATED_NOT_SELECTED,
            CandidateStatus.FAILED_OR_REJECTED,
        }
    )
    if failed:
        return (
            f"{failed} candidate{'s' if failed != 1 else ''} failed or were rejected "
            "by recorded checks."
        )
    if generated_only and subset_evaluated:
        suffix = (
            "The remaining candidates are shown as generated but not evaluated."
            if model.candidate_capability_filter_events
            else (
                "No subset selection reason was recorded, so the remaining candidates "
                "are shown as generated but not evaluated."
            )
        )
        return (
            "The agent evaluated, ranked, or submitted only a subset of generated "
            f"candidates. {suffix}"
        )
    if generated_only:
        return (
            f"{generated_only} candidate{'s' if generated_only != 1 else ''} "
            "were generated but not downstream-evaluated."
        )
    finals = sum(1 for b in review.bundles if b.status == CandidateStatus.FINAL)
    if finals == 0 and review.bundles:
        return "No final submitted candidate was recorded."
    return None


def _agent_interpretation(model: ReportModel, review: ReportReviewModel) -> str:
    return _workflow_summary(review)


def _agent_objective_summary(
    model: ReportModel,
    preset: WorkflowReviewPreset,
) -> str:
    del preset  # Objective metadata is reported from recorded fields only.
    objective = _recorded_objective(model)
    thresholds = _recorded_thresholds(model)
    parts: list[str] = []
    if objective:
        parts.append(f"Objective: {objective}")
    else:
        parts.append(_OBJECTIVE_METADATA_FALLBACK)
    if thresholds:
        if thresholds != objective and not (
            _objective_text_is_criteria(objective) and thresholds in objective
        ):
            parts.append(f"Thresholds/criteria: {thresholds}")
    elif not _objective_text_is_criteria(objective):
        parts.append("No explicit numeric thresholds were captured in VPA metadata.")
    return " ".join(parts)


def _workflow_summary(review: ReportReviewModel) -> str:
    label = _WORKFLOW_LABELS.get(review.workflow_type, "Workflow")
    if review.workflow_type == WorkflowType.BINDER:
        return f"{label}: generation -> QC -> structure prediction -> interface scoring -> ranking."
    if review.workflow_type == WorkflowType.DENOVO:
        return f"{label}: generation -> QC -> structure prediction -> ranking."
    if review.workflow_type == WorkflowType.RETRIEVAL:
        return f"{label}: search -> filtering -> candidate review."
    return f"{label}: workflow evidence grouped by candidate."


def _preset_from_review_only(review: ReportReviewModel) -> WorkflowReviewPreset:
    return _preset_for_workflow_type(review.workflow_type)


def _preset_for_workflow_type(workflow_type: str) -> WorkflowReviewPreset:
    dummy_model = ReportModel(
        summary=RunSummary(kind="workflow", id="preset", workspace="", state="unknown"),
        graph=WorkflowGraph(workflow_id="preset"),
    )
    dummy_review = ReportReviewModel(
        base=dummy_model,
        workflow_type=workflow_type,
        bundles=[],
        board_groups=[],
    )
    return build_workflow_review_preset(dummy_model, dummy_review)


def _stage_plan_from_graph(nodes: list[WorkflowNodeSummary]) -> list[str]:
    stages = []
    for key in ("generation", "qc", "structure", "interface", "ranking", "recommendation", "other"):
        if any(_stage_key(node) == key for node in nodes):
            stages.append(_stage_label(key))
    return stages


def _major_decisions_from_submission(model: ReportModel) -> list[AgentRationaleItem]:
    final_submission = model.final_submission if isinstance(model.final_submission, dict) else {}
    if not final_submission:
        return []
    decisions: list[AgentRationaleItem] = []
    outcome = str(final_submission.get("outcome") or "").replace("_", " ").strip()
    if outcome:
        decisions.append(
            AgentRationaleItem(
                label="Final submission",
                value=outcome.title(),
                provenance=Provenance.EXACT,
            )
        )
    reasoning = final_submission.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        decisions.append(
            AgentRationaleItem(
                label="Selection rationale",
                value=" ".join(reasoning.split()),
                provenance=Provenance.EXACT,
            )
        )
    resolved = final_submission.get("candidates_resolved")
    if isinstance(resolved, list):
        justifications = [
            str(row.get("justification")).strip()
            for row in resolved
            if isinstance(row, dict) and row.get("justification")
        ]
        if justifications:
            decisions.append(
                AgentRationaleItem(
                    label="Candidate justification",
                    value=" ".join(justifications[0].split()),
                    provenance=Provenance.EXACT,
                )
            )
    return decisions[:3]


def _stage_key(node: WorkflowNodeSummary) -> str:
    tool = (node.tool or "").strip().lower()
    explicit = {
        "sequence_edit": "engineering",
        "esm2": "generation",
        "protein_qc": "qc",
        "sequence_analysis": "qc",
        "protenix_v2": "structure",
        "structure_analysis": "structure",
        "tmalign": "structure_comparison",
        "pyrosetta": "energy",
    }
    if tool in explicit:
        return explicit[tool]
    role = classify_tool_role(tool)
    if role == ToolRole.GENERATOR:
        return "structure" if tool in {"esm3", "boltzgen"} else "generation"
    if role == ToolRole.RANKER:
        return "ranking"
    if role == ToolRole.SUBMISSION:
        return "recommendation"
    if role == ToolRole.EVALUATOR:
        if tool in {"protein_qc", "sequence_analysis"}:
            return "qc"
        if tool in {"esm3", "protenix_v2", "structure_analysis"}:
            return "structure"
        if tool in {"tmalign"}:
            return "structure_comparison"
        if tool in {"pyrosetta"}:
            return "energy"
        if tool in {"boltz2_affinity", "pxmeter", "ipsae", "prodigy", "vina"}:
            return "interface"
        return "other"
    return "other"


def _stage_view(
    key: str, label: str, nodes: list[WorkflowNodeSummary], evidence_label: str
) -> WorkflowStageViewModel:
    tools = _dedup([node.tool for node in nodes if node.tool])
    completed = sum(1 for node in nodes if str(node.state).lower() == "completed")
    failed = sum(1 for node in nodes if str(node.state).lower() == "failed")
    return WorkflowStageViewModel(
        key=key,
        label=label,
        tools=tools,
        node_count=len(nodes),
        completed_count=completed,
        failed_count=failed,
        evidence_label=evidence_label,
        provenance=Provenance.DERIVED,
    )


def _stage_label(key: str) -> str:
    return {
        "engineering": "Engineering",
        "generation": "Sequence Generation",
        "qc": "Sequence QC",
        "structure": "Structure Prediction",
        "structure_comparison": "Structure Comparison",
        "energy": "Energy Analysis",
        "interface": "Interface Scoring",
        "ranking": "Ranking",
        "recommendation": "Recommendation",
        "other": "Other Evidence",
    }.get(key, "Other Evidence")


def _status_label(status: str) -> str:
    return _STATUS_LABELS.get(status, status.replace("_", " ").title())


def _display_status(state: str | None) -> str:
    value = (state or "unknown").replace("_", " ").strip()
    return value.title() if value else "Unknown"


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


def _recorded_user_query(model: ReportModel) -> str:
    raw = model.raw_summary if isinstance(model.raw_summary, dict) else {}
    return _first_nonempty(
        str(raw.get("user_query")) if raw.get("user_query") is not None else None,
        _workflow_context_from_raw(raw, "user_query"),
    )


def _recorded_objective(model: ReportModel) -> str:
    raw = model.raw_summary if isinstance(model.raw_summary, dict) else {}
    final_submission = model.final_submission if isinstance(model.final_submission, dict) else {}
    reasoning = final_submission.get("reasoning")
    return _first_nonempty(
        str(raw.get("objective")) if raw.get("objective") is not None else None,
        str(raw.get("design_objective")) if raw.get("design_objective") is not None else None,
        _workflow_context_from_raw(raw, "objective"),
        _final_submission_objective_metadata(final_submission),
        _objective_excerpt_from_reasoning(reasoning) if isinstance(reasoning, str) else None,
    )


def _recorded_thresholds(model: ReportModel) -> str:
    raw = model.raw_summary if isinstance(model.raw_summary, dict) else {}
    values: list[Any] = []
    for key in (
        "thresholds",
        "acceptance_thresholds",
        "acceptance_criteria",
        "success_criteria",
        "criteria",
    ):
        if raw.get(key) not in (None, "", {}, []):
            values.append(raw.get(key))
    final_submission = model.final_submission if isinstance(model.final_submission, dict) else {}
    for key in (
        "acceptance_criteria",
        "success_criteria",
        "criteria",
        "thresholds",
        "acceptance_thresholds",
    ):
        if final_submission.get(key) not in (None, "", {}, []):
            values.append(final_submission.get(key))
    reasoning = final_submission.get("reasoning")
    if not values and isinstance(reasoning, str):
        values.append(_threshold_excerpt_from_reasoning(reasoning))
    return "; ".join(
        _format_threshold_value(value) for value in values if value not in (None, "", {}, [])
    )


def _objective_focus(model: ReportModel) -> str | None:
    text = " ".join(
        part
        for part in (
            _recorded_user_query(model),
            _recorded_objective(model),
            _recorded_thresholds(model),
        )
        if part
    ).lower()
    if not text:
        return None
    stability_terms = (
        "stability",
        "stabil",
        "thermostability",
        "thermal stability",
        "melting temperature",
        "ddg",
        "delta delta g",
        "delta_delta_g",
        "rosetta",
        "pyrosetta",
    )
    solubility_terms = (
        "solubility",
        "soluble",
        "aggregation",
        "aggregate",
        "gravy",
        "hydrophobic patch",
        "hydropathy",
        "developability",
    )
    binding_terms = (
        "binder",
        "binding",
        "bind ",
        "interface",
        "affinity",
        "substrate",
        "ligand",
        "enzyme-substrate",
        "complex",
    )
    if any(term in text for term in stability_terms):
        return "stability"
    if any(term in text for term in solubility_terms):
        return "solubility"
    if any(term in text for term in binding_terms):
        return "binding"
    if "developability" in text:
        return "developability"
    return None


def _final_submission_objective_metadata(final_submission: dict[str, Any]) -> str:
    for key in ("objective", "design_objective"):
        if final_submission.get(key) not in (None, "", {}, []):
            return _format_threshold_value(final_submission.get(key))
    labels = {
        "acceptance_criteria": "Acceptance criteria",
        "success_criteria": "Success criteria",
        "criteria": "Criteria",
        "thresholds": "Thresholds",
        "acceptance_thresholds": "Acceptance thresholds",
    }
    for key, label in labels.items():
        if final_submission.get(key) not in (None, "", {}, []):
            return f"{label}: {_format_threshold_value(final_submission.get(key))}"
    return ""


def _objective_text_is_criteria(objective: str) -> bool:
    return objective.startswith(
        (
            "Acceptance criteria:",
            "Success criteria:",
            "Criteria:",
            "Thresholds:",
            "Acceptance thresholds:",
        )
    )


def _objective_excerpt_from_reasoning(reasoning: str) -> str:
    return _explicit_objective_or_criteria_section(reasoning, limit=260)


def _threshold_excerpt_from_reasoning(reasoning: str) -> str:
    return _explicit_objective_or_criteria_section(reasoning, limit=260)


def _explicit_objective_or_criteria_section(reasoning: str, *, limit: int) -> str:
    text = " ".join(str(reasoning or "").split())
    if not text:
        return ""
    for marker in (
        "OBJECTIVE & ACCEPTANCE CRITERIA:",
        "OBJECTIVE AND ACCEPTANCE CRITERIA:",
        "OBJECTIVE / ACCEPTANCE CRITERIA:",
        "DESIGN OBJECTIVE:",
        "OBJECTIVE:",
        "ACCEPTANCE CRITERIA:",
        "SUCCESS CRITERIA:",
        "CRITERIA:",
    ):
        index = text.upper().find(marker)
        if index >= 0:
            section = text[index + len(marker):]
            next_heading = re.search(
                r"\b[A-Z][A-Z &()0-9/-]{3,}\s*[-:]",
                section,
            )
            if next_heading:
                section = section[: next_heading.start()]
            return _clip(section, limit)
    return ""


def _format_threshold_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


def _workflow_context_from_raw(raw: dict[str, Any], key: str) -> str | None:
    workflows = raw.get("workflows")
    if not isinstance(workflows, list):
        return None
    for workflow in workflows:
        if isinstance(workflow, dict) and workflow.get(key):
            return str(workflow.get(key))
    return None


def _clip(value: str, limit: int) -> str:
    value = " ".join(str(value or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "..."


def _dedup(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        out.append(clean)
    return out


__all__ = [
    "AgentRationaleItem",
    "AgentRationaleViewModel",
    "CandidateBoardRow",
    "CandidateBoardViewModel",
    "CandidateDossierViewModel",
    "CountSummary",
    "MetricChip",
    "MissingEvidenceItem",
    "OverviewViewModel",
    "ReportSaaSViewModel",
    "ReproducibilityViewModel",
    "WorkflowReviewPreset",
    "WorkflowStageViewModel",
    "WorkflowTraceViewModel",
    "build_agent_rationale_view_model",
    "build_candidate_board_row",
    "build_candidate_board_view_model",
    "build_candidate_dossier_view_model",
    "build_overview_view_model",
    "build_report_view_model",
    "build_reproducibility_view_model",
    "build_workflow_review_preset",
    "build_workflow_trace_view_model",
]
