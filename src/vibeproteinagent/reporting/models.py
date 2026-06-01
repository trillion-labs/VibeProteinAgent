from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReportArtifact:
    field: str
    artifact_id: str | None
    kind: str
    mime: str | None
    size_bytes: int | None
    sha256: str | None
    path: str
    job_id: str | None = None


@dataclass(frozen=True)
class ReportWarning:
    level: str
    message: str
    source: str | None = None


@dataclass(frozen=True)
class RunSummary:
    kind: str
    id: str
    workspace: str
    state: str
    name: str | None = None
    version: str | None = None
    submitted_at: int | None = None
    finished_at: int | None = None
    runtime_ms: int | None = None


@dataclass(frozen=True)
class WorkflowNodeSummary:
    node_id: str
    tool: str
    version: str
    state: str
    job_id: str | None
    upstream: list[str] = field(default_factory=list)
    params_hash: str | None = None
    # Resolved input parameters the node ran with — threaded through from the
    # backing job record so the report can show *how* each step was invoked,
    # not just a params hash. May be empty for older payloads.
    params: dict[str, Any] = field(default_factory=dict)
    cache_key: str | None = None
    output: Any | None = None
    parameter_resolution_events: list[dict[str, Any]] = field(default_factory=list)
    parameter_resolution_summary: dict[str, int] = field(default_factory=dict)
    artifacts: list[ReportArtifact] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    error_details: dict[str, Any] | None = None


@dataclass(frozen=True)
class WorkflowGraph:
    workflow_id: str | None
    nodes: list[WorkflowNodeSummary] = field(default_factory=list)


@dataclass(frozen=True)
class ReportCandidate:
    candidate_id: str
    rank: int | None = None
    score: float | None = None
    sequence: str | None = None
    structure_path: str | None = None
    scores: dict[str, Any] = field(default_factory=dict)
    qc_flags: list[str] = field(default_factory=list)
    liabilities: list[Any] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ReportArtifact] = field(default_factory=list)


@dataclass(frozen=True)
class MSABlock:
    """One alignment block ready to render.

    ``sequences`` are (label, aligned_seq) tuples, with ``sequences[0]`` being
    the canonical query row (first record of the source file). ``identities``
    is a parallel list of fractions (0..1) giving each row's % identity vs
    the query — computed over non-gap query columns and used both for the
    visual similarity column and for top-K sorting.

    ``mean_identity`` is the global mean over *all* alignment rows in the
    source file (not just the truncated top-K), so the header line ("68% mean
    identity") still reflects the underlying MSA, not the displayed subset.
    """

    path: str
    label: str
    sequence_count: int
    sequences: list[tuple[str, str]]
    identities: list[float] = field(default_factory=list)
    mean_identity: float = 0.0
    column_count: int = 0
    truncated: bool = False
    source_tool: str | None = None


@dataclass(frozen=True)
class PlotSpec:
    title: str
    kind: str
    svg: str


@dataclass(frozen=True)
class ReproducibilityBlock:
    ids: list[str]
    tool_versions: dict[str, str]
    params_hashes: dict[str, str]
    artifact_sha256: dict[str, str]
    replay_commands: list[str]


@dataclass(frozen=True)
class AgentMessage:
    """One row of the agent's chat transcript.

    ``role`` is the standard chat role (``"user"`` / ``"assistant"`` /
    ``"system"`` / ``"tool"``). ``content`` is the raw message text — the
    renderer is responsible for prettifying / collapsing tool blobs.
    ``timestamp_ms`` is wall-clock epoch-milliseconds when the message was
    recorded; ``model`` carries the model id for assistant rows when known.
    """

    role: str
    content: str
    timestamp_ms: int | None = None
    model: str | None = None
    token_count: int | None = None


@dataclass(frozen=True)
class AgentToolCall:
    """One tool invocation the agent made during the run.

    Rendered as a compact chip interleaved with the chat bubbles in the
    full-transcript view, in chronological order with messages. ``args``
    and ``result`` are arbitrary JSON-compatible dicts — the renderer
    pretty-prints them on demand.
    """

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    ok: bool = True
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class AgentDecision:
    """One ``agent_decisions`` row — the agent's reasoning between rounds.

    Populated by the planner / DecisionEngine after each workflow round
    in a multi-round run. ``decision_label`` is the short verdict string
    (``"completed_review_metrics"``, ``"workflow_not_completed"``, ...);
    ``next_round_recommended`` + ``reason`` carry the "should we go
    again?" call; ``notes`` captures any warnings the engine attached
    (e.g. LLM-fallback notices).
    """

    round_index: int
    decision_label: str
    next_round_recommended: bool = False
    reason: str = ""
    notes: list[Any] = field(default_factory=list)
    metrics_used: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class AgentStepSummary:
    """One ``agent_summaries`` row — what the agent saw after a step ran.

    Each step is either a workflow or a single job. ``kind`` says which.
    ``state`` (``"completed"`` / ``"failed"`` / ...) drives the
    timeline-card colour; ``tool`` is the adapter name for job-kind
    summaries; ``name`` is the workflow name when kind=="workflow".
    """

    kind: str  # "job" | "workflow"
    workflow_id: str | None = None
    job_id: str | None = None
    tool: str | None = None
    state: str | None = None
    name: str | None = None
    error_message: str | None = None
    metrics_used: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int | None = None


@dataclass(frozen=True)
class AgentTranscript:
    """Textual artifacts of an LLM-agent-orchestrated run.

    Surfaces the conversation that produced the workflow: the user's
    original natural-language request, the agent's plan response, the
    final answer / summary, plus three audit layers (tool calls,
    per-round decisions, per-step summaries). ``None`` on
    ``ReportModel`` means the run was *not* LLM-planned and the section
    should not render at all.
    """

    run_id: str
    user_query: str
    plan_text: str | None = None
    # Structured plan from agent_plans (last validated entry, if any).
    # Kept separately from plan_text so the renderer can format it as a
    # structured outline rather than a raw JSON blob.
    plan_structured: dict[str, Any] | None = None
    final_response: str | None = None
    messages: list[AgentMessage] = field(default_factory=list)
    # Interleaved with messages in chronological order for the
    # full-transcript view; also surfaced as a count chip in the header.
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    # Multi-round agent reasoning (one entry per round). Only rendered
    # when there's at least one decision row.
    decisions: list[AgentDecision] = field(default_factory=list)
    # Per-step summaries the agent recorded after each workflow/job —
    # drives the execution-timeline section.
    step_summaries: list[AgentStepSummary] = field(default_factory=list)
    # Optional metadata for the panel header.
    model: str | None = None
    provider: str | None = None
    started_at_ms: int | None = None
    finished_at_ms: int | None = None


# --------------------------------------------------------------------------- #
# Candidate review console (review layer)
#
# A pure, additive presentation model derived from ``ReportModel`` by
# ``reporting.review.build_review_model``. The report UI is re-framed as a
# *candidate review surface*: a ranked board of candidates, each backed by an
# evidence dossier. ``ReportModel`` itself is left untouched so the collection
# layer and its tests are unaffected.
#
# Honesty contract: every evidence atom carries a ``Provenance`` tag so the UI
# never presents a derived/inferred value as if it were recorded fact.
# --------------------------------------------------------------------------- #


class Provenance:
    """Provenance vocabulary for a single evidence atom (string constants).

    ``EXACT``        directly recorded (artifact link, final_submission ref,
                     resolved source_node_id).
    ``DERIVED``      deterministically computed by the report from recorded
                     metrics (QC pulled from candidate.scores, a rejection
                     reason composed from rank + flags).
    ``INFERRED``     a heuristic guess from a weak hint (native_id parse, role
                     classification). Must be visually distinct; never a default.
    ``NOT_RECORDED`` no data — render an empty slot, never fabricate.
    """

    EXACT = "exact"
    DERIVED = "derived"
    INFERRED = "inferred"
    NOT_RECORDED = "not_recorded"


class CandidateStatus:
    """Lifecycle bucket a candidate falls into on the board (string constants)."""

    FINAL = "final"  # referenced by final_submission
    RANKED_ALTERNATIVE = "ranked_alternative"  # ranked, not submitted, evidence ok
    INCOMPLETE_EVIDENCE = "incomplete_evidence"  # missing key evidence (shards land here)
    FAILED_QC = "failed_qc"  # protein_qc passes_filters is False / severe flags
    REJECTED = "rejected"  # explicitly dropped (capability filter / merge)


class ToolRole:
    """Role a tool plays in the candidate pipeline (string constants)."""

    GENERATOR = "generator"
    EVALUATOR = "evaluator"
    RANKER = "ranker"
    SUBMISSION = "submission"
    PLUMBING = "plumbing"  # merge / format bridge / msa-prep — not evidence
    OTHER = "other"  # unknown tool — surfaced as "Other evidence", never mis-roled


class WorkflowType:
    """Coarse workflow shape, used to pick board fields + detail tabs."""

    BINDER = "binder"
    DENOVO = "denovo"
    RETRIEVAL = "retrieval"
    GENERIC = "generic"


@dataclass(frozen=True)
class EvidenceStep:
    """One node of the shared pipeline, with this candidate's evidence overlay.

    The DAG is a *shared* pipeline that every candidate traverses, so the node
    topology is workflow-level truth (``Provenance.EXACT`` at node granularity).
    Per-candidate attribution onto a node (``produced this candidate``,
    ``this candidate's metrics``) is often partial in the recorded data, so each
    step carries its own provenance and may be ``NOT_RECORDED``.
    """

    node_id: str
    tool: str
    role: str  # ToolRole.*
    state: str | None = None
    job_id: str | None = None
    # Short, candidate-scoped evidence line (e.g. "pLDDT 97.3 · pTM 0.83"),
    # empty when nothing is attributable to this candidate at this node.
    evidence_summary: str = ""
    provenance: str = Provenance.NOT_RECORDED


@dataclass(frozen=True)
class EvidencePath:
    """The candidate's path through the pipeline, grouped by role.

    ``producer`` is the single generator node credited with producing the
    candidate (``None`` + provenance NOT_RECORDED when attribution is missing).
    ``evaluators`` / ``ranker`` / ``submission`` capture the rest of the lineage.
    ``shared_pipeline`` is the full ordered node list (workflow-level truth)
    rendered as the backdrop the overlay sits on.
    """

    shared_pipeline: list[EvidenceStep] = field(default_factory=list)
    producer: EvidenceStep | None = None
    producer_provenance: str = Provenance.NOT_RECORDED
    evaluators: list[EvidenceStep] = field(default_factory=list)
    ranker: EvidenceStep | None = None
    submission: EvidenceStep | None = None


@dataclass(frozen=True)
class ScoreGroup:
    """A metric family (not a tool dump): e.g. "Structure confidence".

    ``metrics`` are ``(label, value, unit|None, source)`` tuples already coerced
    for display, where ``source`` is the short producing-tool name (e.g.
    "ESM-3", "Boltz-2", "Protein QC") or ``""`` when no single tool owns it.
    Empty groups are dropped by the builder so the UI never shows a blank card.
    """

    family: str
    metrics: list[tuple[str, str, str | None, str]] = field(default_factory=list)
    provenance: str = Provenance.DERIVED


@dataclass(frozen=True)
class CandidateEvidenceBundle:
    """Everything needed to render one candidate's dossier + board card."""

    candidate: ReportCandidate
    workflow_type: str  # WorkflowType.*
    status: str  # CandidateStatus.*
    status_basis: str = Provenance.DERIVED
    # Authoritative display rank: the design's rank from the ranking tool
    # (``rank_designs``) when available, else the recorded ``candidate.rank``.
    # Drives both the board badge and the per-tool selector card so they agree
    # with the Design-ranking widget (which reads the same node output).
    display_rank: int | None = None
    # When this is the FINAL pick but the ranking tool scored a *different*
    # design #1, this carries this pick's own (worse) scoring rank so the UI can
    # surface a "submitted despite Design-ranking #N" divergence note. ``None``
    # when the final pick is also the #1-ranked design (or no ranking exists).
    submitted_despite_rank: int | None = None
    # Structured, deterministic rule inputs behind the verdict (not prose).
    selection_basis: dict[str, Any] = field(default_factory=dict)
    non_selection_basis: dict[str, Any] = field(default_factory=dict)
    evidence_path: EvidencePath = field(default_factory=EvidencePath)
    score_groups: list[ScoreGroup] = field(default_factory=list)
    # Zero or more structures for the same design (alternate predictions kept
    # as variants rather than collapsed).
    structure_variants: list[ReportArtifact] = field(default_factory=list)
    msa_blocks: list[MSABlock] = field(default_factory=list)
    # Tool-keyed per-candidate evidence pulled from candidate.scores (the join
    # is free here); node.output filtering is layered on only where reliable.
    tool_evidence: dict[str, Any] = field(default_factory=dict)
    artifacts: list[ReportArtifact] = field(default_factory=list)


@dataclass(frozen=True)
class BoardGroup:
    """A titled, ordered cluster of candidates on the board."""

    status: str  # CandidateStatus.*
    title: str
    bundles: list[CandidateEvidenceBundle] = field(default_factory=list)
    collapsed: bool = False  # failed/rejected default-collapsed


@dataclass(frozen=True)
class ReportReviewModel:
    """Candidate-centric presentation model derived from ``ReportModel``.

    ``grouping_policy`` is a forward hook: ``"raw"`` (MVP) keeps one bundle per
    recorded candidate; a future ``"design_family"`` can roll shards of one
    design into a single bundle without changing this contract or the UI.
    """

    base: ReportModel
    workflow_type: str  # WorkflowType.*
    board_groups: list[BoardGroup] = field(default_factory=list)
    bundles: list[CandidateEvidenceBundle] = field(default_factory=list)
    grouping_policy: str = "raw"


@dataclass(frozen=True)
class ReportModel:
    summary: RunSummary
    graph: WorkflowGraph
    candidates: list[ReportCandidate] = field(default_factory=list)
    final_candidates: list[ReportCandidate] = field(default_factory=list)
    final_submission: dict[str, Any] | None = None
    submissions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[ReportArtifact] = field(default_factory=list)
    msa_blocks: list[MSABlock] = field(default_factory=list)
    plots: list[PlotSpec] = field(default_factory=list)
    warnings: list[ReportWarning] = field(default_factory=list)
    reproducibility: ReproducibilityBlock | None = None
    agent_transcript: AgentTranscript | None = None
    satisfied_semantic_facets: list[dict[str, Any]] = field(default_factory=list)
    empty_candidate_events: list[dict[str, Any]] = field(default_factory=list)
    candidate_capability_filter_events: list[dict[str, Any]] = field(default_factory=list)
    candidate_merge_events: list[dict[str, Any]] = field(default_factory=list)
    parameter_resolution_events: list[dict[str, Any]] = field(default_factory=list)
    raw_summary: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AgentDecision",
    "AgentMessage",
    "AgentStepSummary",
    "AgentToolCall",
    "AgentTranscript",
    "BoardGroup",
    "CandidateEvidenceBundle",
    "CandidateStatus",
    "EvidencePath",
    "EvidenceStep",
    "MSABlock",
    "PlotSpec",
    "Provenance",
    "ReportArtifact",
    "ReportCandidate",
    "ReportModel",
    "ReportReviewModel",
    "ReportWarning",
    "ReproducibilityBlock",
    "RunSummary",
    "ScoreGroup",
    "ToolRole",
    "WorkflowGraph",
    "WorkflowNodeSummary",
    "WorkflowType",
]
