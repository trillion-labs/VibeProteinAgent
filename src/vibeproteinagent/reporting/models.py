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
    cache_key: str | None = None
    output: Any | None = None
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
    scores: dict[str, float | int | str | None] = field(default_factory=dict)
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
class ReportModel:
    summary: RunSummary
    graph: WorkflowGraph
    candidates: list[ReportCandidate] = field(default_factory=list)
    artifacts: list[ReportArtifact] = field(default_factory=list)
    msa_blocks: list[MSABlock] = field(default_factory=list)
    plots: list[PlotSpec] = field(default_factory=list)
    warnings: list[ReportWarning] = field(default_factory=list)
    reproducibility: ReproducibilityBlock | None = None
    raw_summary: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "MSABlock",
    "PlotSpec",
    "ReportArtifact",
    "ReportCandidate",
    "ReportModel",
    "ReportWarning",
    "ReproducibilityBlock",
    "RunSummary",
    "WorkflowGraph",
    "WorkflowNodeSummary",
]
