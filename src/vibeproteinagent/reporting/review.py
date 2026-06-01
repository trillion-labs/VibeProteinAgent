"""Candidate review console: derive a candidate-centric presentation model.

``build_review_model`` is a pure function of an already-collected
:class:`~vibeproteinagent.reporting.models.ReportModel`. It re-frames the report
as a *candidate review surface* — a ranked board of candidates, each backed by
an evidence dossier (status, pipeline path, score families, structure, MSA).

Design rules locked with the team:

* The workflow DAG is a **shared** pipeline every candidate traverses, so node
  topology is workflow-level truth. Per-candidate attribution onto a node is
  often partial in the recorded data, so each overlay atom carries a
  :class:`Provenance` tag and may be ``NOT_RECORDED`` — the UI never fabricates.
* Tool is **not** primary navigation. Per-candidate evidence comes from
  ``candidate.scores`` (already keyed by tool), regrouped into researcher-facing
  metric *families*, not tool dumps.
* ``grouping_policy="raw"`` keeps one bundle per recorded candidate (MVP). A
  future ``"design_family"`` can roll shards of one design together without
  changing this contract.

This module is part of the report rendering layer only; it never touches the
execution engine.
"""

from __future__ import annotations

from typing import Any

from .collect import _as_float, _as_int, _string_list
from .models import (
    BoardGroup,
    CandidateEvidenceBundle,
    CandidateStatus,
    EvidencePath,
    EvidenceStep,
    MSABlock,
    Provenance,
    ReportCandidate,
    ReportModel,
    ReportReviewModel,
    ScoreGroup,
    ToolRole,
    WorkflowNodeSummary,
    WorkflowType,
)

# --------------------------------------------------------------------------- #
# Tool classification (report-layer heuristic; unknown -> OTHER, never mis-roled)
# --------------------------------------------------------------------------- #

_GENERATOR_TOOLS = frozenset(
    {
        "rfdiffusion",
        "rfdiffusion3",
        "rfpeptides",
        "pxdesign",
        "bindcraft",
        "proteinmpnn",
        "ligandmpnn",
        "boltzgen",
        "sequence_edit",
        "proteina_complexa",
        # retrieval generators (produce candidate sequences)
        "search_sequence",
        "go_sequences",
        "interpro_sequences",
    }
)

_EVALUATOR_TOOLS = frozenset(
    {
        "esm3",
        "esm2",
        "boltz2_affinity",
        "protenix_v2",
        "structure_analysis",
        "sequence_analysis",
        "protein_qc",
        "pxmeter",
        "ipsae",
        "prodigy",
        "tmalign",
        "vina",
    }
)

_RANKER_TOOLS = frozenset({"rank_designs"})
_SUBMISSION_TOOLS = frozenset({"submit_candidate"})
_PLUMBING_TOOLS = frozenset(
    {
        "merge_candidates",
        "mpnn_to_protenix_input",
        "sequence_to_protenix_input",
        "noop",
        "mmseqs2_gpu",
        "foldseek",
        "template_select",
    }
)

# Structure-producing generators — used only for workflow-type detection.
_STRUCTURE_GENERATOR_TOOLS = frozenset(
    {"rfdiffusion", "rfdiffusion3", "rfpeptides", "pxdesign", "bindcraft", "boltzgen"}
)
_BINDER_TOOLS = frozenset({"pxdesign", "bindcraft"})
_RETRIEVAL_TOOLS = frozenset({"search_sequence", "go_sequences", "interpro_sequences"})

# Evaluator tools whose presence in candidate.scores counts as real evaluation
# evidence (drives RANKED_ALTERNATIVE vs INCOMPLETE_EVIDENCE).
_EVIDENCE_EVALUATORS = _EVALUATOR_TOOLS


def classify_tool_role(tool: str | None) -> str:
    """Map a tool name to its pipeline role (string constant)."""
    name = (tool or "").strip().lower()
    if not name:
        return ToolRole.OTHER
    if name in _RANKER_TOOLS:
        return ToolRole.RANKER
    if name in _SUBMISSION_TOOLS:
        return ToolRole.SUBMISSION
    if name in _PLUMBING_TOOLS:
        return ToolRole.PLUMBING
    if name in _EVALUATOR_TOOLS:
        return ToolRole.EVALUATOR
    if name in _GENERATOR_TOOLS:
        return ToolRole.GENERATOR
    return ToolRole.OTHER


def detect_workflow_type(model: ReportModel) -> str:
    """Infer the coarse workflow shape from the tool set + structure presence."""
    tools = {(node.tool or "").strip().lower() for node in model.graph.nodes}
    has_structures = any(
        (c.structure_path or "").strip() for c in model.candidates
    ) or any(c.structure_path for c in model.final_candidates)
    if tools & _BINDER_TOOLS:
        return WorkflowType.BINDER
    if (
        (tools & _RETRIEVAL_TOOLS)
        and not (tools & _STRUCTURE_GENERATOR_TOOLS)
        and not has_structures
    ):
        return WorkflowType.RETRIEVAL
    if has_structures or (tools & _STRUCTURE_GENERATOR_TOOLS):
        return WorkflowType.DENOVO
    return WorkflowType.GENERIC


# --------------------------------------------------------------------------- #
# Small display helpers
# --------------------------------------------------------------------------- #


def _plddt_100(value: Any) -> float | None:
    """Normalise a pLDDT-family value to the [0, 100] scale for display.

    The recorded data mixes [0, 1] and [0, 100] conventions; a value <= 1.0 is
    treated as a fraction and scaled up.
    """
    num = _as_float(value)
    if num is None:
        return None
    if 0.0 <= num <= 1.0:
        return round(num * 100.0, 1)
    return round(num, 1)


def _plddt_disp(value: Any) -> str | None:
    """Normalised pLDDT as a one-decimal display string (keeps the 0..100 scale
    visually distinct from a 0..1 fraction)."""
    num = _plddt_100(value)
    return None if num is None else f"{num:.1f}"


def _fmt(value: Any, digits: int = 2) -> str | None:
    num = _as_float(value)
    if num is None:
        return None
    if abs(num - round(num)) < 1e-9:
        return str(round(num))
    return f"{num:.{digits}f}"


def _degap(seq: str) -> str:
    return "".join(ch for ch in seq if ch.isalpha()).upper()


def _topo_order(nodes: list[WorkflowNodeSummary]) -> list[WorkflowNodeSummary]:
    """Kahn topological sort over node.upstream; stable, cycle-safe."""
    by_id = {n.node_id: n for n in nodes}
    indeg: dict[str, int] = {n.node_id: 0 for n in nodes}
    for n in nodes:
        for up in n.upstream:
            if up in by_id:
                indeg[n.node_id] += 1
    order: list[WorkflowNodeSummary] = []
    ready = [n for n in nodes if indeg[n.node_id] == 0]
    seen: set[str] = set()
    while ready:
        node = ready.pop(0)
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        order.append(node)
        for other in nodes:
            if node.node_id in other.upstream and other.node_id not in seen:
                indeg[other.node_id] -= 1
                if indeg[other.node_id] <= 0:
                    ready.append(other)
    # Append any nodes left out by cycles, preserving input order.
    for n in nodes:
        if n.node_id not in seen:
            order.append(n)
    return order


# --------------------------------------------------------------------------- #
# Score families (metric-family view, not a tool dump)
# --------------------------------------------------------------------------- #

# (tool, metric_key, label, unit, transform)
_FAMILY_SPECS: list[tuple[str, list[tuple[str, str, str, str | None, Any]]]] = [
    (
        "Structure confidence",
        [
            ("esm3", "mean_plddt", "pLDDT", None, _plddt_disp),
            ("esm3", "ptm", "pTM", None, _fmt),
            ("boltz2_affinity", "plddt", "pLDDT", None, _plddt_disp),
            ("boltz2_affinity", "ptm", "pTM", None, _fmt),
            ("boltz2_affinity", "pae", "PAE", "Å", _fmt),
            ("protenix_v2", "plddt", "pLDDT", None, _plddt_disp),
            ("protein_qc", "plddt", "pLDDT", None, _plddt_disp),
        ],
    ),
    (
        "Interface & binding",
        [
            ("boltz2_affinity", "iptm", "ipTM", None, _fmt),
            ("boltz2_affinity", "affinity", "Affinity", None, _fmt),
            ("boltz2_affinity", "affinity_pred_value", "Affinity", None, _fmt),
            ("ipsae", "ipsae", "ipSAE", None, _fmt),
            ("prodigy", "binding_affinity", "ΔG", "kcal/mol", _fmt),
            ("structure_analysis", "interface_contacts", "Interface contacts", None, _fmt),
        ],
    ),
    (
        "Sequence biophysics",
        [
            ("sequence_analysis", "gravy", "GRAVY", None, _fmt),
            ("sequence_analysis", "pI", "pI", None, _fmt),
            ("sequence_analysis", "net_charge", "Net charge", None, _fmt),
            ("sequence_analysis", "instability_index", "Instability", None, _fmt),
            ("protein_qc", "instability_index", "Instability", None, _fmt),
            ("protein_qc", "pi", "pI", None, _fmt),
            ("protein_qc", "gravy", "GRAVY", None, _fmt),
        ],
    ),
    (
        "Novelty & similarity",
        [
            ("proteinmpnn", "seq_recovery", "Seq recovery", "%", lambda v: _fmt(_pct(v))),
            ("proteinmpnn", "score", "MPNN score", None, _fmt),
        ],
    ),
]


def _pct(value: Any) -> float | None:
    num = _as_float(value)
    if num is None:
        return None
    return round(num * 100.0, 1) if 0.0 <= num <= 1.0 else round(num, 1)


def _score_namespace(candidate: ReportCandidate, tool: str) -> dict[str, Any]:
    block = candidate.scores.get(tool) if isinstance(candidate.scores, dict) else None
    return block if isinstance(block, dict) else {}


# Short, human-readable source-tool labels for metric provenance. Every score
# in a family is read from a specific tool's namespace; surfacing that tool
# (e.g. "pLDDT — ESM-3", "ipTM — Boltz-2") makes a confidence number traceable
# to the model that produced it rather than reading as a free-floating value.
_METRIC_SOURCE_NAME: dict[str, str] = {
    "esm3": "ESM-3",
    "esm2": "ESM-2",
    "boltz2_affinity": "Boltz-2",
    "protenix_v2": "Protenix",
    "protein_qc": "Protein QC",
    "sequence_analysis": "Seq. analysis",
    "structure_analysis": "Struct. analysis",
    "proteinmpnn": "ProteinMPNN",
    "ipsae": "ipSAE",
    "prodigy": "PRODIGY",
}


def _source_name(tool: str) -> str:
    return _METRIC_SOURCE_NAME.get(tool, tool.replace("_", " ").title() if tool else "")


def _build_score_groups(
    candidate: ReportCandidate, msa_blocks: list[MSABlock]
) -> list[ScoreGroup]:
    groups: list[ScoreGroup] = []
    for family, specs in _FAMILY_SPECS:
        # (label, value, unit, source_tool) — source carries the producing tool
        # so the renderer can tag each confidence metric with its origin.
        metrics: list[tuple[str, str, str | None, str]] = []
        for tool, key, label, unit, transform in specs:
            block = _score_namespace(candidate, tool)
            if key not in block:
                continue
            shown = transform(block[key]) if transform else _fmt(block[key])
            if shown is None:
                continue
            metrics.append((label, shown, unit, _source_name(tool)))
        if family == "Novelty & similarity" and msa_blocks:
            ident = max((mb.mean_identity for mb in msa_blocks), default=0.0)
            if ident:
                metrics.append(("MSA mean identity", _fmt(ident * 100.0) or "0", "%", "MSA"))
        if metrics:
            groups.append(ScoreGroup(family=family, metrics=metrics, provenance=Provenance.DERIVED))

    # Flags & liabilities family — always shown when present.
    flag_metrics: list[tuple[str, str, str | None, str]] = []
    qc = _score_namespace(candidate, "protein_qc")
    if "passes_filters" in qc:
        flag_metrics.append(("QC filters", "pass" if qc.get("passes_filters") else "fail", None, "Protein QC"))
    flags = _string_list(candidate.qc_flags) or _string_list(qc.get("flags"))
    if flags:
        flag_metrics.append(("Flags", ", ".join(flags), None, "Protein QC"))
    liab = _liability_labels(candidate)
    if liab:
        flag_metrics.append(("Liabilities", ", ".join(liab), None, ""))
    if flag_metrics:
        groups.append(
            ScoreGroup(
                family="Flags & liabilities",
                metrics=flag_metrics,
                provenance=Provenance.DERIVED,
            )
        )
    return groups


def _liability_labels(candidate: ReportCandidate) -> list[str]:
    out: list[str] = []
    for item in candidate.liabilities or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            label = item.get("type") or item.get("name") or item.get("label")
            if label:
                out.append(str(label))
    seq = _score_namespace(candidate, "sequence_analysis")
    out.extend(str(x) for x in _string_list(seq.get("liability_types")))
    # de-dup preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _has_evaluation_evidence(candidate: ReportCandidate) -> bool:
    if not isinstance(candidate.scores, dict):
        return bool((candidate.structure_path or "").strip())
    for tool in candidate.scores:
        if (tool or "").strip().lower() in _EVIDENCE_EVALUATORS:
            return True
    return bool((candidate.structure_path or "").strip())


# --------------------------------------------------------------------------- #
# Status classification + selection basis
# --------------------------------------------------------------------------- #


def _qc_fails(candidate: ReportCandidate) -> bool:
    qc = _score_namespace(candidate, "protein_qc")
    if "passes_filters" in qc:
        return not bool(qc.get("passes_filters"))
    return False


def _scoring_rank_map(model: ReportModel) -> dict[str, int]:
    """Map ``candidate_id -> authoritative scoring rank`` from ``rank_designs``.

    The board's ``candidate.rank`` is collection order (the final pick is hoisted
    to the front of ``model.candidates``, so it reads ``#1`` regardless of how the
    ranker actually scored it). The Design-ranking widget, by contrast, renders
    the ``rank_designs`` node's own ``ranked_candidates[].rank`` — the real
    scoring rank. The two disagree.

    This reads the *last* ``rank_designs`` node (the final ranking in a
    multi-round run) and returns its candidate_id -> rank map so the review layer
    can present a single, consistent rank everywhere. Returns ``{}`` when the run
    has no ``rank_designs`` node (e.g. some binder runs), in which case callers
    fall back to ``candidate.rank``.
    """
    rank_nodes = [
        n for n in model.graph.nodes if (n.tool or "").strip().lower() == "rank_designs"
    ]
    if not rank_nodes:
        return {}
    out: dict[str, int] = {}
    for node in rank_nodes:  # later nodes win (final ranking)
        output = node.output if isinstance(node.output, dict) else {}
        rows = output.get("ranked_candidates")
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cid = row.get("candidate_id")
            if not isinstance(cid, str) or not cid:
                continue
            rank = _as_int(row.get("rank"))
            out[cid] = rank if rank is not None else idx + 1
    return out


def _rank_sort_key(candidate: ReportCandidate) -> tuple[float, float]:
    """Order key preferring a recorded rank, then a higher score."""
    rank = _as_int(candidate.rank)
    score = _as_float(candidate.score)
    return (
        float(rank) if rank is not None else float("inf"),
        -score if score is not None else float("inf"),
    )


def _resolve_final_records(model: ReportModel) -> dict[int, str]:
    """Map ``model.candidates`` indices that are genuinely final → provenance.

    Fragmented designs share one degapped sequence across several candidate
    records with distinct ``candidate_id``s and no lineage key, so a naive
    sequence match would promote *every* shard. The resolution is therefore:

    1. **candidate_id match** (``Provenance.EXACT``): a recorded record whose id
       equals a final pick's id is the submitted record.
    2. **sequence fallback** (``Provenance.INFERRED``): only for final picks that
       matched no record by id, credit the *single* best-ranked record sharing
       the degapped sequence — never the whole shard set.
    """
    final_ids: set[str] = {
        str(fc.candidate_id) for fc in model.final_candidates if fc.candidate_id
    }
    resolved: dict[int, str] = {}

    # Pass 1: exact candidate_id match.
    matched_ids: set[str] = set()
    for idx, cand in enumerate(model.candidates):
        cid = str(cand.candidate_id or "")
        if cid and cid in final_ids:
            resolved[idx] = Provenance.EXACT
            matched_ids.add(cid)

    # Pass 2: sequence fallback, only for final picks unresolved by id.
    unresolved_seqs: set[str] = set()
    for fc in model.final_candidates:
        fcid = str(fc.candidate_id or "")
        if fcid and fcid in matched_ids:
            continue
        seq = _degap(fc.sequence or "")
        if seq:
            unresolved_seqs.add(seq)
    for seq in unresolved_seqs:
        candidates = [
            idx
            for idx, cand in enumerate(model.candidates)
            if idx not in resolved and _degap(cand.sequence or "") == seq
        ]
        if not candidates:
            continue
        best = min(candidates, key=lambda i: _rank_sort_key(model.candidates[i]))
        resolved[best] = Provenance.INFERRED
    return resolved


def _classify_status(candidate: ReportCandidate, is_final: bool) -> str:
    if is_final:
        return CandidateStatus.FINAL
    if _qc_fails(candidate):
        return CandidateStatus.FAILED_QC
    if _has_evaluation_evidence(candidate):
        return CandidateStatus.RANKED_ALTERNATIVE
    return CandidateStatus.INCOMPLETE_EVIDENCE


def _missing_evidence(candidate: ReportCandidate) -> list[str]:
    missing: list[str] = []
    if not (candidate.structure_path or "").strip():
        missing.append("structure")
    if "protein_qc" not in (candidate.scores or {}):
        missing.append("qc")
    if not _has_evaluation_evidence(candidate):
        missing.append("evaluation")
    return missing


def _selection_basis(candidate: ReportCandidate, status: str) -> dict[str, Any]:
    if status != CandidateStatus.FINAL:
        return {}
    src = candidate.source if isinstance(candidate.source, dict) else {}
    return {
        "submitted": True,
        "rank": _as_int(candidate.rank),
        "score": _as_float(candidate.score),
        "auto_selected": bool(src.get("auto_selected", False)),
        "justification": src.get("justification") or src.get("reasoning"),
    }


def _non_selection_basis(candidate: ReportCandidate, status: str) -> dict[str, Any]:
    if status == CandidateStatus.FINAL:
        return {}
    return {
        "submitted": False,
        "rank": _as_int(candidate.rank),
        "score": _as_float(candidate.score),
        "qc_flags": _string_list(candidate.qc_flags),
        "missing_evidence": _missing_evidence(candidate),
    }


# --------------------------------------------------------------------------- #
# Evidence path (shared pipeline + candidate overlay)
# --------------------------------------------------------------------------- #


def _node_evidence_summary(candidate: ReportCandidate, tool: str) -> str:
    """One short candidate-scoped line for a node, from candidate.scores[tool]."""
    block = _score_namespace(candidate, tool)
    if not block:
        return ""
    bits: list[str] = []
    name = (tool or "").lower()
    if name == "esm3":
        p = _plddt_100(block.get("mean_plddt"))
        if p is not None:
            bits.append(f"pLDDT {p}")
        t = _fmt(block.get("ptm"))
        if t:
            bits.append(f"pTM {t}")
    elif name == "protein_qc":
        flags = _string_list(block.get("flags"))
        bits.append("QC pass" if block.get("passes_filters") else "QC fail")
        if flags:
            bits.append(f"{len(flags)} flag" + ("s" if len(flags) != 1 else ""))
    elif name == "sequence_analysis":
        g = _fmt(block.get("gravy"))
        if g:
            bits.append(f"GRAVY {g}")
    elif name == "proteinmpnn":
        s = _fmt(block.get("score"))
        if s:
            bits.append(f"MPNN {s}")
    elif name in {"boltz2_affinity", "protenix_v2"}:
        p = _plddt_100(block.get("plddt"))
        if p is not None:
            bits.append(f"pLDDT {p}")
        i = _fmt(block.get("iptm"))
        if i:
            bits.append(f"ipTM {i}")
    return " · ".join(bits)


def _build_evidence_path(
    candidate: ReportCandidate,
    ordered_nodes: list[WorkflowNodeSummary],
    status: str,
) -> EvidencePath:
    src = candidate.source if isinstance(candidate.source, dict) else {}
    src_node_id = src.get("source_node_id") or src.get("node_id")
    src_tool = (src.get("tool") or "").strip().lower()

    steps: list[EvidenceStep] = []
    generators: list[EvidenceStep] = []
    evaluators: list[EvidenceStep] = []
    ranker: EvidenceStep | None = None
    submission: EvidenceStep | None = None

    for node in ordered_nodes:
        role = classify_tool_role(node.tool)
        if role == ToolRole.PLUMBING:
            continue  # plumbing is not evidence; keep it out of the candidate path
        tool = (node.tool or "").lower()
        summary = _node_evidence_summary(candidate, tool)
        # provenance for this node's overlay onto the candidate
        if node.node_id and node.node_id == src_node_id:
            prov = Provenance.EXACT
        elif summary:
            prov = Provenance.DERIVED  # this candidate's metrics are present
        else:
            prov = Provenance.NOT_RECORDED
        step = EvidenceStep(
            node_id=node.node_id,
            tool=node.tool,
            role=role,
            state=node.state,
            job_id=node.job_id,
            evidence_summary=summary,
            provenance=prov,
        )
        steps.append(step)
        if role == ToolRole.GENERATOR:
            generators.append(step)
        elif role == ToolRole.EVALUATOR and summary:
            evaluators.append(step)
        elif role == ToolRole.RANKER and ranker is None:
            ranker = step
        elif role == ToolRole.SUBMISSION and submission is None:
            submission = step

    # Producer attribution.
    producer: EvidenceStep | None = None
    producer_prov = Provenance.NOT_RECORDED
    exact_gen = [s for s in generators if s.node_id == src_node_id]
    if exact_gen:
        producer, producer_prov = exact_gen[0], Provenance.EXACT
    elif len(generators) == 1:
        # Single generator in the pipeline — credit it, flagged as inferred.
        producer, producer_prov = generators[0], Provenance.INFERRED
    elif src_tool:
        tool_gen = [s for s in generators if (s.tool or "").lower() == src_tool]
        if tool_gen:
            producer, producer_prov = tool_gen[0], Provenance.INFERRED

    if status == CandidateStatus.FINAL and submission is None:
        submission = EvidenceStep(
            node_id="final_submission",
            tool="submit_candidate",
            role=ToolRole.SUBMISSION,
            evidence_summary="selected",
            provenance=Provenance.EXACT,
        )

    return EvidencePath(
        shared_pipeline=steps,
        producer=producer,
        producer_provenance=producer_prov,
        evaluators=evaluators,
        ranker=ranker,
        submission=submission,
    )


# --------------------------------------------------------------------------- #
# MSA / structure association
# --------------------------------------------------------------------------- #


def _msa_for_candidate(candidate: ReportCandidate, msa_blocks: list[MSABlock]) -> list[MSABlock]:
    seq = _degap(candidate.sequence or "")
    if not seq or not msa_blocks:
        return []
    matched: list[MSABlock] = []
    for block in msa_blocks:
        for _label, aligned in block.sequences[:1]:  # query row only
            if _degap(aligned) == seq:
                matched.append(block)
                break
    return matched


def _structure_variants(candidate: ReportCandidate) -> list:
    seen: set[str] = set()
    out = []
    for art in candidate.artifacts:
        kind = (art.kind or "").lower()
        if kind in {"pdb", "cif", "mmcif", "ent"} and art.path not in seen:
            seen.add(art.path)
            out.append(art)
    return out


# --------------------------------------------------------------------------- #
# Bundle + board assembly
# --------------------------------------------------------------------------- #


def _build_bundle(
    candidate: ReportCandidate,
    *,
    workflow_type: str,
    ordered_nodes: list[WorkflowNodeSummary],
    msa_blocks: list[MSABlock],
    final_provenance: str | None,
    scoring_rank: int | None = None,
    top_ranked_id: str | None = None,
) -> CandidateEvidenceBundle:
    is_final = final_provenance is not None
    status = _classify_status(candidate, is_final)
    cand_msa = _msa_for_candidate(candidate, msa_blocks)
    # Authoritative rank for the board badge + selector card: the ranker's own
    # rank when known, else the recorded collection-order rank.
    display_rank = scoring_rank if scoring_rank is not None else _as_int(candidate.rank)
    # Flag the case where the agent submitted a design the ranker did NOT score
    # #1, so the board badge (#N) doesn't read as a contradiction.
    submitted_despite_rank = (
        display_rank
        if (
            is_final
            and scoring_rank is not None
            and scoring_rank != 1
            and top_ranked_id is not None
            and top_ranked_id != candidate.candidate_id
        )
        else None
    )
    return CandidateEvidenceBundle(
        candidate=candidate,
        workflow_type=workflow_type,
        status=status,
        status_basis=final_provenance if is_final else Provenance.DERIVED,
        display_rank=display_rank,
        submitted_despite_rank=submitted_despite_rank,
        selection_basis=_selection_basis(candidate, status),
        non_selection_basis=_non_selection_basis(candidate, status),
        evidence_path=_build_evidence_path(candidate, ordered_nodes, status),
        score_groups=_build_score_groups(candidate, cand_msa),
        structure_variants=_structure_variants(candidate),
        msa_blocks=cand_msa,
        tool_evidence={
            tool: block
            for tool, block in (candidate.scores or {}).items()
            if isinstance(block, dict)
        },
        artifacts=list(candidate.artifacts),
    )


# Board groups, in display order. Failed / rejected are collapsed by default.
_GROUP_ORDER: list[tuple[str, str, bool]] = [
    (CandidateStatus.FINAL, "Final / Submitted", False),
    (CandidateStatus.RANKED_ALTERNATIVE, "Ranked alternatives", False),
    (CandidateStatus.INCOMPLETE_EVIDENCE, "Incomplete evidence", True),
    (CandidateStatus.FAILED_QC, "Failed QC", True),
    (CandidateStatus.REJECTED, "Rejected records", True),
]


def _rank_key(bundle: CandidateEvidenceBundle) -> tuple[float, float]:
    # Prefer the authoritative display rank (ranker's rank) so the board order
    # matches the Design-ranking widget; fall back to the recorded rank.
    rank = bundle.display_rank if bundle.display_rank is not None else _as_int(bundle.candidate.rank)
    score = _as_float(bundle.candidate.score)
    return (
        float(rank) if rank is not None else float("inf"),
        -score if score is not None else float("inf"),
    )


def _board_group_title(status: str, default_title: str, workflow_type: str) -> str:
    """Group title, with a retrieval-aware override for unscored hits.

    In a retrieval workflow the "incomplete evidence" bucket isn't a quality
    failure: these are database sequences pulled by a query that were never
    individually folded or scored, so they legitimately lack evaluator
    evidence. Calling them "Incomplete evidence" reads as a defect; relabel so
    the reader understands the bucket is "retrieved but not evaluated".
    """
    if (
        workflow_type == WorkflowType.RETRIEVAL
        and status == CandidateStatus.INCOMPLETE_EVIDENCE
    ):
        return "Retrieved — not individually evaluated"
    return default_title


def _build_board_groups(
    bundles: list[CandidateEvidenceBundle],
    workflow_type: str = WorkflowType.GENERIC,
) -> list[BoardGroup]:
    by_status: dict[str, list[CandidateEvidenceBundle]] = {}
    for b in bundles:
        by_status.setdefault(b.status, []).append(b)
    groups: list[BoardGroup] = []
    for status, title, collapsed in _GROUP_ORDER:
        items = by_status.get(status)
        if not items:
            continue
        items.sort(key=_rank_key)
        groups.append(
            BoardGroup(
                status=status,
                title=_board_group_title(status, title, workflow_type),
                bundles=items,
                collapsed=collapsed,
            )
        )
    return groups


def build_review_model(model: ReportModel) -> ReportReviewModel:
    """Derive the candidate-centric :class:`ReportReviewModel` from a report."""
    workflow_type = detect_workflow_type(model)
    ordered_nodes = _topo_order(list(model.graph.nodes))
    final_map = _resolve_final_records(model)
    scoring_ranks = _scoring_rank_map(model)
    top_ranked_id = next(
        (cid for cid, rank in scoring_ranks.items() if rank == 1), None
    )

    bundles = [
        _build_bundle(
            candidate,
            workflow_type=workflow_type,
            ordered_nodes=ordered_nodes,
            msa_blocks=model.msa_blocks,
            final_provenance=final_map.get(idx),
            scoring_rank=scoring_ranks.get(candidate.candidate_id),
            top_ranked_id=top_ranked_id,
        )
        for idx, candidate in enumerate(model.candidates)
    ]

    # A final pick may be recorded as its own candidate record that never appears
    # in ``model.candidates`` (distinct id *and* sequence — a genuinely disjoint
    # submission record). The submitted design must never be missing from the
    # board, so synthesise a FINAL bundle directly from any such record.
    represented_ids = {
        str(model.candidates[i].candidate_id or "") for i in final_map
    } - {""}
    represented_seqs = {
        _degap(model.candidates[i].sequence or "") for i in final_map
    } - {""}
    for fc in model.final_candidates:
        fcid = str(fc.candidate_id or "")
        fseq = _degap(fc.sequence or "")
        if (fcid and fcid in represented_ids) or (fseq and fseq in represented_seqs):
            continue
        bundles.append(
            _build_bundle(
                fc,
                workflow_type=workflow_type,
                ordered_nodes=ordered_nodes,
                msa_blocks=model.msa_blocks,
                final_provenance=Provenance.EXACT,
                scoring_rank=scoring_ranks.get(fc.candidate_id),
                top_ranked_id=top_ranked_id,
            )
        )
        if fcid:
            represented_ids.add(fcid)
        if fseq:
            represented_seqs.add(fseq)

    board_groups = _build_board_groups(bundles, workflow_type)
    return ReportReviewModel(
        base=model,
        workflow_type=workflow_type,
        board_groups=board_groups,
        bundles=bundles,
        grouping_policy="raw",
    )


__all__ = [
    "build_review_model",
    "classify_tool_role",
    "detect_workflow_type",
]
