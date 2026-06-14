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

from vibeproteinagent.core.errors import InvalidParamsError

from .metrics import (
    SCORE_DEFINITIONS,
    display_metric_label,
    metric_matches,
    metric_variant_rank,
)
from .models import (
    CandidateEvidenceBundle,
    CandidateStatus,
    MSABlock,
    Provenance,
    ReportModel,
    ReportReviewModel,
    ReportWarning,
    RunSummary,
    ScoreGroup,
    WorkflowGraph,
    WorkflowNodeSummary,
    WorkflowType,
)
from .review import classify_workflow_stage_key


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
    raw_label: str = ""


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
class RankingBasisViewModel:
    method: str = ""
    objective_name: str = ""
    gates: list[str] = field(default_factory=list)
    score_formula: str = ""
    score_direction: str = ""
    fallback_text: str = ""


@dataclass(frozen=True)
class CandidateBoardViewModel:
    rows: list[CandidateBoardRow] = field(default_factory=list)
    recommended_count: int = 0
    total_count: int = 0
    ranking_basis_caption: str = ""
    ranking_basis: RankingBasisViewModel | None = None
    objective_override_note: str = ""
    has_objective_rank: bool = False


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


def build_report_view_model(model: ReportModel, review: ReportReviewModel) -> ReportSaaSViewModel:
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

    tools = {(node.tool or "").strip().lower() for node in model.graph.nodes if node.tool}
    families = {group.family for bundle in review.bundles for group in bundle.score_groups}
    has_structure = any(
        bool((bundle.candidate.structure_path or "").strip()) for bundle in review.bundles
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
    binder_signal = (
        review.workflow_type == WorkflowType.BINDER or has_interface or objective_focus == "binding"
    )
    if objective_focus == "stability" or (
        has_stability and not binder_signal and objective_focus != "binding"
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
                "Stability review: compare stabilization evidence, fold confidence, "
                "and sequence risk."
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
                "Objective ranking",
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
    if not binder_signal and (
        objective_focus == "solubility"
        or (has_sequence_biophysics and objective_focus == "developability")
    ):
        confidence = 0.88 if has_sequence_biophysics else 0.72
        return WorkflowReviewPreset(
            workflow_type=WorkflowType.DENOVO,
            label="Solubility engineering review",
            confidence=confidence,
            confidence_basis=(
                "solubility/developability objective text and sequence-biophysics evidence"
            ),
            scientific_question=(
                "Which candidates best improve solubility and sequence developability?"
            ),
            overview_focus=(
                "Solubility engineering: prioritize GRAVY, net charge, instability, "
                "pI, QC liabilities, and fold confidence when available."
            ),
            candidate_review_guidance=(
                "Developability review: compare hydropathy, charge, liabilities, "
                "and fold confidence."
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
                "Objective ranking",
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
    if binder_signal:
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
                "Review the submitted candidate and alternatives using the recorded evidence."
            ),
            comparison_note=(
                "Compare candidates by final status, recorded evidence, missing data, "
                "and practical follow-up needs."
            ),
            cross_candidate_focus=(
                "Adjust the filters to narrow candidates by recorded criteria; matching "
                "candidates and downloads update with the current view."
            ),
            evidence_family_order=[
                "Interface & binding",
                "Objective ranking",
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
                "Binder probability",
                "DockQ",
                "Interaction PAE",
                "ipAE",
                "ipLDDT",
                "Ligand buried fraction",
                "Ligand contacts",
                "Ligand contact residues",
                "Ligand atom contacts",
                "Delta G",
                "Interface contacts",
                "Confidence",
                "Generation score",
                "Refold RMSD",
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
                "Affinity",
                "Binder probability",
                "DockQ",
                "Interaction PAE",
                "Ligand buried fraction",
                "Ligand contacts",
                "Ligand contact residues",
                "Ligand atom contacts",
                "GRAVY",
                "Net charge",
                "Instability",
                "pI",
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
                "Foldability review: compare pLDDT, pTM, QC, and sequence risk."
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
                "Objective ranking",
                "Flags & liabilities",
                "Sequence biophysics",
                "Novelty & similarity",
                "Interface & binding",
            ],
            metric_priority=[
                "pLDDT",
                "pTM",
                "QC",
                "Confidence",
                "lDDT",
                "Interface lDDT",
                "Scaffold pLDDT",
                "Designability",
                "Generation score",
                "Refold RMSD",
                "Instability",
                "GRAVY",
                "Net charge",
                "pI",
                "Delta G",
                "ipTM",
            ],
            cross_candidate_metrics=[
                "pLDDT",
                "pTM",
                "Confidence",
                "lDDT",
                "Designability",
                "Instability",
                "GRAVY",
                "Net charge",
                "pI",
            ],
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
                "Evidence review: compare source coverage, confidence, and gaps."
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
                "Objective ranking",
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
        candidate_review_guidance=("Review candidates using the available recorded evidence."),
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
            "Objective ranking",
            "Sequence biophysics",
            "Flags & liabilities",
            "Novelty & similarity",
        ],
        metric_priority=[
            "pLDDT",
            "pTM",
            "ipTM",
            "ipSAE",
            "Affinity",
            "Binder probability",
            "DockQ",
            "Confidence",
            "Delta G",
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
            "Affinity",
            "Binder probability",
            "DockQ",
            "Confidence",
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
    passed_population = [bundle for bundle in generated_population if _qc_passes(bundle)]
    evaluated_population = [bundle for bundle in passed_population if _is_fully_evaluated(bundle)]
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
    ranked_population = [bundle for bundle in generated_population if _has_rank_evidence(bundle)]
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
            provenance=Provenance.EXACT if model.final_candidates else Provenance.DERIVED,
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
    workflow_selection = f"{preset.label}: {preset.scientific_question}"
    llm_rationale = _llm_agent_rationale_view_model(
        model,
        workflow_selection=workflow_selection,
    )
    if llm_rationale is not None:
        return llm_rationale
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
    has_objective_rank = _has_objective_rank_evidence(bundles)
    renumber = _should_renumber_board_rows(bundles)
    rows = [
        build_candidate_board_row(
            bundle,
            preset,
            display_rank=index if renumber else None,
            show_rank=has_objective_rank or bundle.status == CandidateStatus.FINAL,
            force_recommendation_label=(
                "Alternative"
                if not has_objective_rank and bundle.status != CandidateStatus.FINAL
                else None
            ),
        )
        for index, bundle in enumerate(bundles, start=1)
    ]
    return CandidateBoardViewModel(
        rows=rows,
        recommended_count=sum(1 for row in rows if row.status == CandidateStatus.FINAL),
        total_count=len(rows),
        ranking_basis_caption=_ranking_basis_caption(bundles),
        ranking_basis=_ranking_basis_view_model(bundles),
        objective_override_note=_objective_override_note(bundles),
        has_objective_rank=has_objective_rank,
    )


def _has_objective_rank_evidence(bundles: list[CandidateEvidenceBundle]) -> bool:
    return any(
        isinstance(bundle.candidate.scores, dict)
        and isinstance(bundle.candidate.scores.get("objective_rank"), dict)
        for bundle in bundles
    )


def _should_renumber_board_rows(bundles: list[CandidateEvidenceBundle]) -> bool:
    if _has_objective_rank_evidence(bundles):
        return False
    ranks = [
        bundle.display_rank if bundle.display_rank is not None else bundle.candidate.rank
        for bundle in bundles
    ]
    numeric = [rank for rank in ranks if rank is not None]
    return len(numeric) != len(bundles) or len(set(numeric)) != len(numeric)


def build_candidate_board_row(
    bundle: CandidateEvidenceBundle,
    preset: WorkflowReviewPreset | None = None,
    display_rank: int | None = None,
    show_rank: bool = True,
    force_recommendation_label: str | None = None,
) -> CandidateBoardRow:
    preset = preset or _preset_for_workflow_type(bundle.workflow_type)
    candidate = bundle.candidate
    label = _candidate_status_label(bundle)
    rank = (
        display_rank
        if display_rank is not None
        else (bundle.display_rank if bundle.display_rank is not None else candidate.rank)
    )
    return CandidateBoardRow(
        candidate_id=candidate.candidate_id,
        rank=rank if show_rank else None,
        status=bundle.status,
        status_label=label,
        recommendation_label=force_recommendation_label or label,
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
        status_label=_candidate_status_label(bundle),
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
        "backbone_generation": [],
        "binder_generation": [],
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
            "backbone_generation",
            "Backbone Generation",
            stage_nodes["backbone_generation"],
            "Generated backbone scaffolds",
        ),
        _stage_view(
            "binder_generation",
            "Binder Generation",
            stage_nodes["binder_generation"],
            "Generated target-conditioned binder candidates",
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
    observed_chips: list[tuple[str, MetricChip]] = []
    for group in bundle.score_groups:
        for label, value, unit, source in group.metrics:
            normalized = display_metric_label(label)
            chip = MetricChip(
                label=normalized,
                value=str(value),
                unit=unit,
                source=source,
                family=group.family,
                provenance=group.provenance,
                raw_label=str(label),
            )
            observed_chips.append((str(label), chip))
    chips: list[MetricChip] = []
    used: set[int] = set()
    for label in preferred:
        ranked_matches = [
            (metric_variant_rank(raw_label, label), index, chip)
            for index, (raw_label, chip) in enumerate(observed_chips)
            if index not in used and metric_matches(raw_label, label)
        ]
        if not ranked_matches:
            continue
        _rank, index, chip = min(ranked_matches, key=lambda item: (item[0], item[1]))
        used.add(index)
        chips.append(chip)
    if bundle.status == CandidateStatus.FAILED_QC:
        chips.append(MetricChip(label="QC", value="Fail", family="Flags & liabilities"))
    elif bundle.candidate.qc_flags or bundle.candidate.liabilities:
        chips.append(MetricChip(label="QC", value="Review", family="Flags & liabilities"))
    else:
        chips.append(MetricChip(label="QC", value="Pass", family="Flags & liabilities"))
    return chips[:limit]

def _candidate_verdict(
    bundle: CandidateEvidenceBundle, preset: WorkflowReviewPreset | None = None
) -> str:
    preset = preset or _preset_for_workflow_type(bundle.workflow_type)
    strengths = _candidate_strength_phrases(bundle, preset)
    missing = _missing_evidence(bundle, preset)
    if bundle.status == CandidateStatus.FINAL:
        objective = _objective_rank_summary(bundle)
        if objective:
            return f"Final: {objective}{_missing_tail(missing)}."
        metric = _headline_metric(bundle)
        if preset.workflow_type == WorkflowType.BINDER:
            primary = _binder_primary_evidence_summary(strengths)
            if primary:
                return f"Recommended due to {primary}{_missing_tail(missing)}."
        if strengths:
            return f"Recommended final candidate with {strengths[0]}{_missing_tail(missing)}."
        return (
            f"Recommended final candidate{': ' + metric if metric else ''}{_missing_tail(missing)}."
        )
    if bundle.status == CandidateStatus.RANKED_ALTERNATIVE:
        objective = _objective_rank_summary(bundle)
        if objective:
            return f"Alternative: {objective}{_missing_tail(missing)}."
        metric = _headline_metric(bundle)
        if metric:
            return f"Alternative: {metric}{_missing_tail(missing)}."
        if strengths:
            return f"Alternative with {strengths[0]}{_missing_tail(missing)}."
        return f"Alternative{_missing_tail(missing)}."
    if bundle.status == CandidateStatus.EVALUATED_NOT_SELECTED:
        if strengths:
            return f"Evaluated but not selected: {strengths[0]}{_missing_tail(missing)}."
        metric = _headline_metric(bundle)
        return (
            f"Evaluated but not selected{': ' + metric if metric else ''}{_missing_tail(missing)}."
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
    score_metric = _candidate_score_headline(bundle.candidate)
    if score_metric and _candidate_score_tool(bundle.candidate) == "objective_rank":
        return score_metric
    for chip in _metric_chips(bundle, limit=4):
        if chip.label == "QC":
            continue
        unit = f" {chip.unit}" if chip.unit else ""
        return f"{chip.label} {chip.value}{unit}"
    return score_metric


def _candidate_score_headline(candidate: Any) -> str:
    objective = _objective_rank_block(candidate)
    if objective and objective.get("objective_score") is not None:
        return f"objective_score {_fmt_score(objective.get('objective_score'))}"
    if candidate.score is None:
        return ""
    return f"{_candidate_score_display_name(candidate)} {candidate.score:g}"


def _candidate_score_display_name(candidate: Any) -> str:
    tool = _candidate_score_tool(candidate)
    if tool:
        return f"{tool} candidate.score"
    return "candidate.score"


def _candidate_score_tool(candidate: Any) -> str:
    scores = getattr(candidate, "scores", None)
    if isinstance(scores, dict):
        ordered_tools = [tool for tool in ("objective_rank",) if tool in scores] + [
            tool for tool in scores if tool != "objective_rank"
        ]
        for tool in ordered_tools:
            block = scores.get(tool)
            tool_key = str(tool or "").strip().lower()
            if tool_key not in SCORE_DEFINITIONS or not isinstance(block, dict):
                continue
            if any(
                key in block
                for key in ("objective_score", "score", "ranking_score", "candidate_score")
            ):
                return tool_key
    source = getattr(candidate, "source", None)
    if isinstance(source, dict):
        tool = str(source.get("tool") or "").strip().lower()
        if tool in SCORE_DEFINITIONS:
            return tool
    return ""


def _ranking_basis_caption(bundles: list[CandidateEvidenceBundle]) -> str:
    basis = _ranking_basis_view_model(bundles)
    if basis is None:
        return ""
    if basis.fallback_text:
        return basis.fallback_text
    gates = "; ".join(basis.gates) if basis.gates else "none"
    return (
        f"Ranking: objective_rank for {basis.objective_name}. Gates: {gates}. "
        f"Score: {basis.score_formula}; {basis.score_direction} is better."
    )


def _ranking_basis_view_model(
    bundles: list[CandidateEvidenceBundle],
) -> RankingBasisViewModel | None:
    if not bundles:
        return None
    objective = _first_objective_rank_block(bundles)
    if objective:
        name = str(objective.get("objective_name") or "objective")
        formula = str(
            objective.get("score_formula") or objective.get("formula_text") or "recorded formula"
        )
        formula = _human_objective_formula(formula)
        direction = str(objective.get("score_direction") or "higher")
        return RankingBasisViewModel(
            method="objective_rank",
            objective_name=name,
            gates=_compact_gate_list(objective),
            score_formula=formula,
            score_direction=direction,
        )
    if any(bundle.status == CandidateStatus.FINAL for bundle in bundles):
        return RankingBasisViewModel(
            fallback_text=(
                "No objective ranking formula was run. Ranking source: "
                "submit_candidate (agent-selected). No objective_rank gates were "
                "recorded; final is the accepted submission, and alternatives are "
                "shown for evidence only."
            )
        )
    ranked_count = sum(
        1
        for bundle in bundles
        if (bundle.display_rank if bundle.display_rank is not None else bundle.candidate.rank)
        is not None
    )
    if ranked_count > 1:
        return RankingBasisViewModel(
            fallback_text=(
                "No objective ranking formula was run. Ranking source: recorded "
                "tool-native candidate ranks. No objective_rank gates were recorded; "
                "ranks are shown as provenance, not as objective-gated selection evidence."
            )
        )
    if any(
        (bundle.display_rank if bundle.display_rank is not None else bundle.candidate.rank)
        is not None
        for bundle in bundles
    ):
        return RankingBasisViewModel(
            fallback_text=(
                "No objective ranking formula was run. Final is the submitted candidate; "
                "alternatives are shown for evidence only."
            )
        )
    return RankingBasisViewModel(
        fallback_text=(
            "No objective ranking source was recorded. Candidates are shown for evidence only."
        )
    )


def _objective_override_note(bundles: list[CandidateEvidenceBundle]) -> str:
    has_rank_one_final = any(
        bundle.status == CandidateStatus.FINAL
        and (bundle.display_rank if bundle.display_rank is not None else bundle.candidate.rank)
        == 1
        for bundle in bundles
    )
    if has_rank_one_final:
        return ""
    for bundle in bundles:
        if bundle.status != CandidateStatus.FINAL or bundle.submitted_despite_rank is None:
            continue
        summary = _objective_rank_summary(bundle) or "submitted candidate was not objective rank #1"
        return f"Submitted objective rank #{bundle.submitted_despite_rank}: {summary}."
    return ""


def _objective_rank_block(candidate: Any) -> dict[str, Any]:
    scores = getattr(candidate, "scores", None)
    if isinstance(scores, dict):
        block = scores.get("objective_rank")
        if isinstance(block, dict):
            return block
    source = getattr(candidate, "source", None)
    if isinstance(source, dict):
        evidence = source.get("selection_evidence")
        if isinstance(evidence, dict) and evidence.get("status") == "objective_ranked":
            return evidence
    return {}


def _first_objective_rank_block(
    bundles: list[CandidateEvidenceBundle],
) -> dict[str, Any]:
    for bundle in bundles:
        block = _objective_rank_block(bundle.candidate)
        if block:
            return block
    return {}


def _objective_rank_summary(bundle: CandidateEvidenceBundle) -> str:
    block = _objective_rank_block(bundle.candidate)
    if not block:
        return ""
    score = block.get("objective_score")
    score_text = (
        f"objective_score {_fmt_score(score)}" if score is not None else "objective_score missing"
    )
    failed = block.get("failed_gates") if isinstance(block.get("failed_gates"), list) else []
    passed = block.get("passed_gates") if isinstance(block.get("passed_gates"), list) else []
    if failed:
        return f"{score_text}; {len(failed)} gate{'s' if len(failed) != 1 else ''} failed"
    if passed:
        key_metrics = _objective_key_metrics(block)
        tail = f"; {key_metrics}" if key_metrics else ""
        return f"{score_text}; gates passed{tail}"
    return score_text


def _objective_key_metrics(block: dict[str, Any], *, limit: int = 2) -> str:
    components = block.get("score_components")
    if not isinstance(components, list):
        return ""
    bits: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        metric = component.get("metric")
        value = component.get("value")
        if metric is None or value is None:
            continue
        bits.append(f"{_human_metric_label(str(metric))} {_fmt_score(value)}")
        if len(bits) >= limit:
            break
    return ", ".join(bits)


def _compact_gate_summary(block: dict[str, Any], *, limit: int = 3) -> str:
    return _join_limited(_compact_gate_list(block), limit)


def _compact_gate_list(block: dict[str, Any]) -> list[str]:
    gates = []
    for key in ("passed_gates", "failed_gates"):
        values = block.get(key)
        if isinstance(values, list):
            gates.extend(item for item in values if isinstance(item, dict))
    labels: list[str] = []
    for gate in gates:
        description = str(gate.get("description") or "").strip()
        if description:
            labels.append(description)
            continue
        metric = gate.get("metric")
        op = gate.get("op")
        threshold = gate.get("threshold")
        if metric and op and threshold is not None:
            labels.append(f"{_human_metric_label(str(metric))} {op} {_fmt_score(threshold)}")
    if labels:
        return _dedup(labels)
    explicit = block.get("gate_summary")
    if isinstance(explicit, list) and explicit:
        return [_human_gate_text(str(item)) for item in explicit if item]
    return []


def _human_objective_formula(formula: str) -> str:
    text = str(formula or "").strip()
    if not text:
        return text
    return re.sub(
        r"\$\{([^}]+)\}",
        lambda match: _human_metric_label(match.group(1)),
        text,
    )


def _human_gate_text(text: str) -> str:
    return re.sub(
        r"\b([A-Za-z_][A-Za-z0-9_.-]*)\b(?=\s*(?:<=|>=|==|!=|<|>))",
        lambda match: _human_metric_label(match.group(1)),
        str(text or ""),
    )


_OBJECTIVE_METRIC_LABELS = {
    "m.aff_log_ic50": "predicted log IC50",
    "m.aff_prob": "affinity probability",
    "m.complex_conf": "complex confidence",
    "m.complex_plddt": "complex pLDDT",
    "m.instability": "instability index",
    "m.mono_plddt": "standalone monomer pLDDT",
    "m.mono_ptm": "monomer pTM",
    "m.pocket": "pocket contacts",
}


_METRIC_TOKEN_LABELS = {
    "aff": "affinity",
    "conf": "confidence",
    "ddg": "ddG",
    "ic50": "IC50",
    "iptm": "ipTM",
    "log": "log",
    "mono": "monomer",
    "pae": "PAE",
    "plddt": "pLDDT",
    "ptm": "pTM",
    "qc": "QC",
    "rmsd": "RMSD",
    "rsa": "RSA",
    "tm": "TM",
}


def _human_metric_label(metric: str) -> str:
    raw = str(metric or "").strip()
    if not raw:
        return raw
    normalized = raw.lower()
    if normalized in _OBJECTIVE_METRIC_LABELS:
        return _OBJECTIVE_METRIC_LABELS[normalized]
    leaf = raw.split(".")[-1]
    tokens = [part for part in re.split(r"[_\s.-]+", leaf) if part]
    if not tokens:
        return raw
    labels = [_METRIC_TOKEN_LABELS.get(token.lower(), token) for token in tokens]
    return " ".join(labels)


def _join_limited(values: list[str], limit: int) -> str:
    if len(values) <= limit:
        return "; ".join(values)
    return "; ".join(values[:limit]) + f"; +{len(values) - limit} more"


def _fmt_score(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(num - round(num)) < 1e-9:
        return str(round(num))
    return f"{num:.4g}"


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
            bundle.score_groups or bundle.artifacts or bundle.tool_evidence or bundle.msa_blocks
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
    proteina_warnings = [
        item
        for item in model.warnings
        if _is_proteina_complexa_warning(item)
        and _is_executive_caveat_warning(item, model.graph.nodes)
    ]
    if proteina_warnings:
        return _proteina_complexa_warning_caveat(proteina_warnings)
    warning = next(
        (item for item in model.warnings if _is_executive_caveat_warning(item, model.graph.nodes)),
        None,
    )
    if warning is not None:
        return _executive_warning_caveat(warning)
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


def _executive_warning_caveat(warning: ReportWarning) -> str:
    message = str(warning.message or "").strip()
    source = str(warning.source or "").strip().lower()
    lower = message.lower()
    if "complexa design sidecar failed" in lower or source.startswith("proteina_complexa"):
        if "target_ligand.pdb" in lower and "target_ligand_heavy" not in lower:
            return (
                "An initial Proteina-Complexa ligand-binder run failed during ligand "
                "featurization, likely because the ligand PDB atom list did not match "
                "the SMILES heavy-atom graph. Later recorded evidence should be read "
                "from recovered structures and downstream validation."
            )
        if (
            "py_ssize_t_clean" in lower
            or "file opened in 'text' mode" in lower
            or "dnn_support != nullptr" in lower
        ):
            return (
                "Proteina-Complexa generation produced structures, but its evaluation "
                "backend failed in the recorded run. Ranking therefore relies on "
                "downstream recorded validators rather than Proteina-Complexa reward "
                "metrics."
            )
        return (
            "A Proteina-Complexa node failed during recorded execution; inspect the "
            "Workflow audit for the full command and traceback. Final evidence is "
            "limited to successfully recorded downstream validators."
        )
    return message


def _is_proteina_complexa_warning(warning: ReportWarning) -> bool:
    source = str(warning.source or "").strip().lower()
    message = str(warning.message or "").strip().lower()
    return source.startswith("proteina_complexa") or "proteina-complexa" in message


def _proteina_complexa_warning_caveat(warnings: list[ReportWarning]) -> str:
    text = "\n".join(str(item.message or "") for item in warnings).lower()
    has_ligand_atom_mismatch = any(
        marker in text
        for marker in (
            "rdkit mol and atom array do not match",
            "operands could not be broadcast",
            "expected size",
            "target_ligand.pdb",
        )
    )
    has_eval_runtime_failure = any(
        marker in text
        for marker in (
            "target_ligand_heavy",
            "py_ssize_t_clean",
            "file opened in 'text' mode",
            "dnn_support != nullptr",
            "jaxruntimeerror",
        )
    )
    if has_ligand_atom_mismatch and has_eval_runtime_failure:
        return (
            "Proteina-Complexa had recorded failures in both setup and evaluation: "
            "an initial ligand atom-count/featurization mismatch, then an evaluation "
            "runtime/backend failure after heavy-atom ligand inputs were used. Final "
            "ranking should be read from successfully recorded downstream validators."
        )
    if has_eval_runtime_failure:
        return (
            "Proteina-Complexa generated structures, but its evaluation backend failed "
            "in the recorded run. Final ranking should be read from downstream recorded "
            "validators rather than Proteina-Complexa reward metrics."
        )
    if has_ligand_atom_mismatch:
        return (
            "A Proteina-Complexa ligand-binder run failed during ligand featurization, "
            "likely because the ligand PDB atom list did not match the SMILES heavy-atom "
            "graph. Later recorded validators determine the reportable evidence."
        )
    return _executive_warning_caveat(warnings[0])


def _is_executive_caveat_warning(
    warning: ReportWarning,
    nodes: list[WorkflowNodeSummary] | None = None,
) -> bool:
    message = str(warning.message or "").strip()
    if not message:
        return False
    lower = message.lower()
    # pydssp is the intended secondary-structure assignment fallback. Keep it
    # available in the audit trail, but do not promote it to the executive caveat.
    if "secondary_structure" in lower and "pydssp" in lower:
        return False
    if _is_recovered_tool_failure_warning(warning, nodes or []):
        return False
    return True


def _is_recovered_tool_failure_warning(
    warning: ReportWarning,
    nodes: list[WorkflowNodeSummary],
) -> bool:
    if str(warning.level or "").strip().lower() != "error":
        return False
    source = str(warning.source or "").strip()
    if not source:
        return False
    failed_index: int | None = None
    failed_tool = ""
    for index, node in enumerate(nodes):
        if node.node_id != source:
            continue
        if str(node.state or "").strip().lower() != "failed":
            return False
        if not (node.error_code or node.error_message):
            return False
        failed_index = index
        failed_tool = str(node.tool or "").strip().lower()
        break
    if failed_index is None or not failed_tool:
        return False
    return any(
        str(node.tool or "").strip().lower() == failed_tool
        and str(node.state or "").strip().lower() == "completed"
        for node in nodes[failed_index + 1 :]
    )


def _agent_interpretation(model: ReportModel, review: ReportReviewModel) -> str:
    return _workflow_summary(review)


def _agent_objective_summary(
    model: ReportModel,
    preset: WorkflowReviewPreset,
) -> str:
    del preset  # Objective metadata is reported from recorded fields only.
    objective, embedded_criteria = _split_recorded_objective_and_criteria(
        _recorded_objective(model)
    )
    thresholds = _recorded_thresholds(model)
    criteria = _first_nonempty(thresholds, embedded_criteria)
    parts: list[str] = []
    if objective:
        parts.append(f"Objective: {objective}")
    else:
        parts.append(_OBJECTIVE_METADATA_FALLBACK)
    if criteria:
        if criteria != objective and not (
            _objective_text_is_criteria(objective) and criteria in objective
        ):
            parts.append(f"Acceptance criteria: {criteria}")
    elif not _objective_text_is_criteria(objective):
        parts.append("No explicit numeric thresholds were captured in VPA metadata.")
    return " ".join(parts)


def _split_recorded_objective_and_criteria(objective: str) -> tuple[str, str]:
    text = _strip_embedded_objective_metadata_fallback(
        _strip_leading_objective_label(objective)
    )
    if not text:
        return "", ""
    match = _objective_criteria_marker(text)
    if match:
        objective_text = text[: match.start()].strip(" ;,")
        criteria_text = _strip_embedded_objective_metadata_fallback(
            text[match.end() :].strip(" ;,")
        )
        if not objective_text:
            return "", criteria_text
        return objective_text, criteria_text
    objective_text, criteria_text = _split_parenthetical_criteria(text)
    if criteria_text:
        return objective_text, criteria_text
    return text, ""


def _objective_criteria_marker(text: str) -> re.Match[str] | None:
    explicit_label = re.search(
        r"(?i)\b(?:acceptance criteria|acceptance|thresholds/criteria|thresholds|criteria)\s*:\s*",
        text,
    )
    accept_if = re.search(r"(?i)\baccept(?:\s+only)?\s+if\b\s*", text)
    matches = [explicit_label] if explicit_label else []
    if accept_if and _has_objective_clause_boundary(text, accept_if.start()):
        matches.append(accept_if)
    if not matches:
        return None
    return min(matches, key=lambda match: match.start())


def _has_objective_clause_boundary(text: str, index: int) -> bool:
    prefix = text[:index].rstrip()
    if not prefix:
        return True
    return prefix[-1] in ".;:!?)]}"


def _strip_embedded_objective_metadata_fallback(text: str) -> str:
    text = re.sub(
        r"(?i)(?:^|(?<=[.!?;]))\s*No explicit numeric "
        r"(?:acceptance criteria|thresholds?) were captured in VPA metadata\.?",
        " ",
        str(text or ""),
    )
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ;,")


def _split_parenthetical_criteria(text: str) -> tuple[str, str]:
    criteria: list[str] = []

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if _looks_like_acceptance_criteria(inner):
            criteria.append(inner)
            return ""
        return match.group(0)

    objective = re.sub(r"\(([^()]*)\)", replace, text)
    objective = re.sub(r"\s{2,}", " ", objective)
    objective = re.sub(r"\s+([,.;:])", r"\1", objective).strip(" ;,")
    return objective or text, "; ".join(criteria)


def _looks_like_acceptance_criteria(text: str) -> bool:
    lower = text.lower()
    return bool(re.search(r"(?:[<>]=?|=)\s*-?\d", text)) or any(
        token in lower
        for token in (
            "plddt",
            "iptm",
            "ipsae",
            "tm-score",
            "qc pass",
            "passes_filters",
            "robust",
            "interface contact",
        )
    )


def _strip_leading_objective_label(text: str) -> str:
    return re.sub(
        r"(?i)^\s*(?:objective|design objective)\s*:?\s*",
        "",
        str(text or "").strip(),
        count=1,
    )


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
    for key in (
        "backbone_generation",
        "binder_generation",
        "generation",
        "qc",
        "structure",
        "interface",
        "ranking",
        "recommendation",
        "other",
    ):
        if any(_stage_key(node) == key for node in nodes):
            stages.append(_stage_label(key))
    return stages


_LLM_NARRATIVE_KEYS = (
    "llm_narrative",
    "report_narrative",
    "agent_narrative",
)
_LLM_NARRATIVE_PROVENANCE = "llm_generated"


def _llm_agent_rationale_view_model(
    model: ReportModel,
    *,
    workflow_selection: str,
) -> AgentRationaleViewModel | None:
    narrative = _recorded_llm_narrative(model)
    if narrative is None:
        return None
    _validate_narrative_refs(narrative, model)
    user_intent = _narrative_text(
        narrative.get("user_intent"),
        narrative.get("intent_summary"),
    )
    review_framing = _narrative_text(
        narrative.get("review_framing"),
        narrative.get("review_frame"),
    )
    major_decisions = _llm_major_decisions(narrative, model)
    if not any((user_intent, review_framing, major_decisions)):
        return None
    recorded_query = _recorded_user_query(model)
    recorded_objective = _recorded_objective(model)
    fallback_intent = _first_nonempty(recorded_query, recorded_objective)
    return AgentRationaleViewModel(
        user_query=user_intent or fallback_intent or None,
        intent_summary=_clip(user_intent or fallback_intent, 220) or None,
        workflow_selection=review_framing or workflow_selection,
        stage_plan=_stage_plan_from_graph(model.graph.nodes),
        major_decisions=major_decisions,
        tool_call_count=len(model.graph.nodes),
        has_full_transcript=False,
    )


def _recorded_llm_narrative(model: ReportModel) -> dict[str, Any] | None:
    containers: list[dict[str, Any]] = []
    final_submission = model.final_submission if isinstance(model.final_submission, dict) else {}
    raw_summary = model.raw_summary if isinstance(model.raw_summary, dict) else {}
    containers.extend([final_submission, raw_summary])
    for container in containers:
        for key in _LLM_NARRATIVE_KEYS:
            value = container.get(key)
            narrative = _coerce_narrative_dict(value)
            if narrative is not None:
                return narrative
    return None


def _coerce_narrative_dict(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InvalidParamsError(
                "invalid LLM report narrative JSON",
                details={"error": str(exc)},
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise InvalidParamsError(
            "invalid LLM report narrative JSON",
            details={"expected": "object", "actual": type(parsed).__name__},
        )
    return None


def _llm_major_decisions(
    narrative: dict[str, Any],
    model: ReportModel,
) -> list[AgentRationaleItem]:
    decisions: list[AgentRationaleItem] = []
    raw_decisions = narrative.get("major_decisions")
    if isinstance(raw_decisions, list):
        for idx, entry in enumerate(raw_decisions):
            item = _llm_decision_item(entry, index=idx)
            if item is not None:
                _validate_narrative_refs(
                    entry if isinstance(entry, dict) else {},
                    model,
                    context=f"major_decisions[{idx}]",
                )
                decisions.append(item)
    if not _has_decision_label(decisions, "selection rationale"):
        selection = _narrative_text(narrative.get("selection_rationale"))
        if selection:
            _validate_narrative_refs(
                narrative.get("selection_rationale")
                if isinstance(narrative.get("selection_rationale"), dict)
                else {},
                model,
                context="selection_rationale",
            )
            decisions.append(
                AgentRationaleItem(
                    label="Selection rationale",
                    value=selection,
                    provenance=_LLM_NARRATIVE_PROVENANCE,
                )
            )
    if not _has_decision_label(decisions, "candidate justification"):
        justification = _candidate_justification_text(narrative)
        if justification:
            _validate_candidate_justification_refs(narrative, model)
            decisions.append(
                AgentRationaleItem(
                    label="Candidate justification",
                    value=justification,
                    provenance=_LLM_NARRATIVE_PROVENANCE,
                )
            )
    return decisions[:6]


def _llm_decision_item(entry: Any, *, index: int) -> AgentRationaleItem | None:
    if isinstance(entry, str):
        value = " ".join(entry.split())
        if not value:
            return None
        return AgentRationaleItem(
            label=f"Decision {index + 1}",
            value=value,
            provenance=_LLM_NARRATIVE_PROVENANCE,
        )
    if not isinstance(entry, dict):
        return None
    label = _narrative_text(entry.get("label"), entry.get("title"), entry.get("type"))
    value = _narrative_text(entry.get("value"), entry.get("body"), entry.get("summary"))
    if not value:
        return None
    return AgentRationaleItem(
        label=label or f"Decision {index + 1}",
        value=value,
        provenance=_LLM_NARRATIVE_PROVENANCE,
    )


def _candidate_justification_text(narrative: dict[str, Any]) -> str:
    direct = _narrative_text(narrative.get("candidate_justification"))
    if direct:
        return direct
    justifications = narrative.get("candidate_justifications")
    if isinstance(justifications, dict):
        for value in justifications.values():
            text = _narrative_text(value)
            if text:
                return text
    if isinstance(justifications, list):
        for value in justifications:
            text = _narrative_text(value)
            if text:
                return text
    return ""


def _validate_candidate_justification_refs(
    narrative: dict[str, Any],
    model: ReportModel,
) -> None:
    for key in ("candidate_justification", "candidate_justifications"):
        value = narrative.get(key)
        if isinstance(value, dict):
            if key == "candidate_justifications":
                for candidate_id, entry in value.items():
                    payload = entry if isinstance(entry, dict) else {}
                    payload = {**payload, "candidate_refs": [str(candidate_id)]}
                    _validate_narrative_refs(payload, model, context=f"{key}.{candidate_id}")
            else:
                _validate_narrative_refs(value, model, context=key)
        elif isinstance(value, list):
            for idx, entry in enumerate(value):
                _validate_narrative_refs(
                    entry if isinstance(entry, dict) else {},
                    model,
                    context=f"{key}[{idx}]",
                )


def _has_decision_label(items: list[AgentRationaleItem], label: str) -> bool:
    needle = label.strip().lower()
    return any(item.label.strip().lower() == needle for item in items)


def _narrative_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str):
            text = " ".join(value.split())
            if text:
                return text
        if isinstance(value, dict):
            text = _narrative_text(
                value.get("text"),
                value.get("body"),
                value.get("summary"),
                value.get("value"),
            )
            if text:
                return text
            parts: list[str] = []
            for key in ("strengths", "caveats"):
                raw = value.get(key)
                if isinstance(raw, list):
                    parts.extend(str(item).strip() for item in raw if str(item).strip())
            if parts:
                return " ".join(parts)
    return ""


def _validate_narrative_refs(
    payload: dict[str, Any],
    model: ReportModel,
    *,
    context: str = "llm_narrative",
) -> None:
    if not payload:
        return
    known_candidates = _known_candidate_refs(model)
    unknown_candidates = [
        ref
        for ref in _ref_list(payload.get("candidate_refs"))
        if ref not in known_candidates
    ]
    known_metrics = _known_metric_refs(model)
    unknown_metrics = [
        ref for ref in _ref_list(payload.get("metric_refs")) if ref not in known_metrics
    ]
    known_artifacts = _known_artifact_refs(model)
    unknown_artifacts = [
        ref
        for ref in _ref_list(payload.get("artifact_refs"))
        if ref not in known_artifacts
    ]
    if unknown_candidates or unknown_metrics or unknown_artifacts:
        raise InvalidParamsError(
            "LLM report narrative references unknown evidence",
            details={
                "context": context,
                "unknown_candidate_refs": unknown_candidates,
                "unknown_metric_refs": unknown_metrics,
                "unknown_artifact_refs": unknown_artifacts,
            },
        )


def _ref_list(value: Any) -> list[str]:
    if value in (None, "", [], ()):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _known_candidate_refs(model: ReportModel) -> set[str]:
    refs: set[str] = set()
    for candidate in [*model.candidates, *model.final_candidates]:
        if candidate.candidate_id:
            refs.add(str(candidate.candidate_id))
        source = candidate.source if isinstance(candidate.source, dict) else {}
        for key in ("ranked_candidate_id", "local_id", "material_key"):
            value = source.get(key)
            if value not in (None, ""):
                refs.add(str(value))
        objective = (
            candidate.scores.get("objective_rank")
            if isinstance(candidate.scores, dict)
            else None
        )
        if isinstance(objective, dict):
            for key in ("ranked_candidate_id", "material_key"):
                value = objective.get(key)
                if value not in (None, ""):
                    refs.add(str(value))
    return refs


def _known_metric_refs(model: ReportModel) -> set[str]:
    refs: set[str] = set()
    for candidate in [*model.candidates, *model.final_candidates]:
        scores = candidate.scores if isinstance(candidate.scores, dict) else {}
        for key, value in scores.items():
            key_text = str(key)
            refs.add(key_text)
            if isinstance(value, dict):
                for metric in value:
                    metric_text = str(metric)
                    refs.add(f"{key_text}.{metric_text}")
                    refs.add(f"{key_text}:{metric_text}")
            elif ":" in key_text:
                namespace, metric = key_text.split(":", 1)
                refs.add(f"{namespace}.{metric}")
    return refs


def _known_artifact_refs(model: ReportModel) -> set[str]:
    refs: set[str] = set()
    for artifact in model.artifacts:
        for value in (artifact.artifact_id, artifact.field, artifact.path):
            if value not in (None, ""):
                refs.add(str(value))
    for candidate in [*model.candidates, *model.final_candidates]:
        if candidate.structure_path:
            refs.add(str(candidate.structure_path))
        for artifact in candidate.artifacts:
            for value in (artifact.artifact_id, artifact.field, artifact.path):
                if value not in (None, ""):
                    refs.add(str(value))
    return refs


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
    return classify_workflow_stage_key(node)


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
        "backbone_generation": "Backbone Generation",
        "binder_generation": "Binder Generation",
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


def _candidate_status_label(bundle: CandidateEvidenceBundle) -> str:
    if bundle.status == CandidateStatus.FINAL and bundle.submitted_despite_rank is not None:
        return "Accepted alternative"
    return _status_label(bundle.status)


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
    return _first_nonempty(
        str(raw.get("objective")) if raw.get("objective") is not None else None,
        str(raw.get("design_objective")) if raw.get("design_objective") is not None else None,
        _workflow_context_from_raw(raw, "objective"),
        _final_submission_objective_only(final_submission),
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


def _final_submission_objective_only(final_submission: dict[str, Any]) -> str:
    for key in ("objective", "design_objective"):
        if final_submission.get(key) not in (None, "", {}, []):
            return _format_threshold_value(final_submission.get(key))
    return ""


def _objective_text_is_criteria(objective: str) -> bool:
    if objective.startswith(
        (
            "Acceptance criteria:",
            "Success criteria:",
            "Criteria:",
            "Thresholds:",
            "Acceptance thresholds:",
        )
    ):
        return True
    return bool(re.search(r"(?:[<>]=?|=)\s*-?\d", objective))


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
