"""Unit tests for the candidate review model (``reporting.review``).

These lock the *contract* of ``build_review_model``: status classification,
final-pick resolution (no fragment over-promotion, single-shard fallback,
disjoint-record synthesis), workflow-type detection, tool-role classification,
score families, and evidence-path producer attribution. The review layer is a
pure function of a ``ReportModel``; none of these tests touch the engine.
"""

from __future__ import annotations

from vibeproteinagent.reporting.models import (
    CandidateStatus,
    MSABlock,
    Provenance,
    ReportCandidate,
    ReportModel,
    RunSummary,
    ToolRole,
    WorkflowGraph,
    WorkflowNodeSummary,
    WorkflowType,
)
from vibeproteinagent.reporting.review import (
    build_review_model,
    classify_tool_role,
    detect_workflow_type,
)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _node(node_id: str, tool: str, *, upstream=None, state="completed", job_id=None):
    return WorkflowNodeSummary(
        node_id=node_id,
        tool=tool,
        version="1",
        state=state,
        job_id=job_id or f"job_{node_id}",
        upstream=list(upstream or []),
    )


def _model(*, nodes=None, candidates=None, finals=None, msa_blocks=None):
    return ReportModel(
        summary=RunSummary(kind="workflow", id="wf_x", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id="wf_x", nodes=list(nodes or [])),
        candidates=list(candidates or []),
        final_candidates=list(finals or []),
        msa_blocks=list(msa_blocks or []),
    )


def _bundle_for(review, candidate_id):
    for b in review.bundles:
        if b.candidate.candidate_id == candidate_id:
            return b
    return None


def _status_counts(review):
    counts: dict[str, int] = {}
    for b in review.bundles:
        counts[b.status] = counts.get(b.status, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Tool-role classification
# --------------------------------------------------------------------------- #


def test_classify_tool_role_known_roles():
    assert classify_tool_role("pxdesign") == ToolRole.GENERATOR
    assert classify_tool_role("proteinmpnn") == ToolRole.GENERATOR
    assert classify_tool_role("esm3") == ToolRole.EVALUATOR
    assert classify_tool_role("protein_qc") == ToolRole.EVALUATOR
    assert classify_tool_role("rank_designs") == ToolRole.RANKER
    assert classify_tool_role("submit_candidate") == ToolRole.SUBMISSION
    assert classify_tool_role("merge_candidates") == ToolRole.PLUMBING


def test_classify_tool_role_unknown_is_other_not_evaluator():
    # Locked decision: an unknown tool is OTHER, never silently an evaluator.
    assert classify_tool_role("totally_new_tool") == ToolRole.OTHER
    assert classify_tool_role("") == ToolRole.OTHER
    assert classify_tool_role(None) == ToolRole.OTHER


def test_classify_tool_role_is_case_insensitive():
    assert classify_tool_role("ESM3") == ToolRole.EVALUATOR
    assert classify_tool_role("  PXDesign ") == ToolRole.GENERATOR


# --------------------------------------------------------------------------- #
# Workflow-type detection
# --------------------------------------------------------------------------- #


def test_detect_binder_when_binder_tool_present():
    model = _model(nodes=[_node("n1", "pxdesign"), _node("n2", "esm3")])
    assert detect_workflow_type(model) == WorkflowType.BINDER


def test_detect_retrieval_only_when_no_structures():
    model = _model(
        nodes=[_node("n1", "search_sequence"), _node("n2", "sequence_analysis")],
        candidates=[ReportCandidate(candidate_id="c1", sequence="ACDEFG")],
    )
    assert detect_workflow_type(model) == WorkflowType.RETRIEVAL


def test_detect_denovo_when_structure_generator_present():
    model = _model(nodes=[_node("n1", "rfdiffusion"), _node("n2", "proteinmpnn")])
    assert detect_workflow_type(model) == WorkflowType.DENOVO


def test_detect_denovo_when_candidate_has_structure_even_with_retrieval_tool():
    # A retrieval tool plus a structure means it is not a pure retrieval run.
    model = _model(
        nodes=[_node("n1", "search_sequence")],
        candidates=[ReportCandidate(candidate_id="c1", sequence="ACDE", structure_path="/tmp/x.pdb")],
    )
    assert detect_workflow_type(model) == WorkflowType.DENOVO


def test_detect_generic_when_nothing_matches():
    model = _model(nodes=[_node("n1", "sequence_analysis")])
    assert detect_workflow_type(model) == WorkflowType.GENERIC


# --------------------------------------------------------------------------- #
# Final-pick resolution — the core regression contract
# --------------------------------------------------------------------------- #


def test_fragmented_design_does_not_over_promote_to_final():
    """Three shards share one sequence; only the id-matched shard is FINAL.

    Regression for the bug where a sequence match promoted *every* shard of a
    fragmented design to FINAL.
    """
    seq = "SAAKREELYKFIVELLKEEG"
    shards = [
        ReportCandidate(candidate_id="cand_a", sequence=seq, rank=1,
                        scores={"esm3": {"mean_plddt": 0.9}}),
        ReportCandidate(candidate_id="cand_b", sequence=seq,
                        scores={"esm3": {"mean_plddt": 0.8}}),
        ReportCandidate(candidate_id="cand_c", sequence=seq),
    ]
    final = ReportCandidate(candidate_id="cand_a", sequence=seq, rank=1)
    review = build_review_model(_model(candidates=shards, finals=[final]))

    finals = [b for b in review.bundles if b.status == CandidateStatus.FINAL]
    assert len(finals) == 1
    assert finals[0].candidate.candidate_id == "cand_a"
    assert finals[0].status_basis == Provenance.EXACT
    # The other two shards must NOT be FINAL.
    assert _bundle_for(review, "cand_b").status != CandidateStatus.FINAL
    assert _bundle_for(review, "cand_c").status != CandidateStatus.FINAL


def test_sequence_fallback_promotes_single_best_shard_when_no_id_match():
    """Final pick has no id among candidates -> single best-ranked shard FINAL."""
    seq = "MKLVINGKTLKG"
    shards = [
        ReportCandidate(candidate_id="cand_x", sequence=seq, rank=3),
        ReportCandidate(candidate_id="cand_y", sequence=seq, rank=1),  # best rank
        ReportCandidate(candidate_id="cand_z", sequence=seq, rank=2),
    ]
    final = ReportCandidate(candidate_id="not_in_candidates", sequence=seq)
    review = build_review_model(_model(candidates=shards, finals=[final]))

    finals = [b for b in review.bundles if b.status == CandidateStatus.FINAL]
    assert len(finals) == 1
    assert finals[0].candidate.candidate_id == "cand_y"
    assert finals[0].status_basis == Provenance.INFERRED


def test_disjoint_final_record_is_synthesized_onto_board():
    """A final pick absent from candidates (distinct id AND seq) still shows."""
    explored = [ReportCandidate(candidate_id=f"c{i}", sequence="AAAA" + str(i)) for i in range(3)]
    final = ReportCandidate(candidate_id="final_only", sequence="WYWYWYWY", rank=1)
    review = build_review_model(_model(candidates=explored, finals=[final]))

    finals = [b for b in review.bundles if b.status == CandidateStatus.FINAL]
    assert len(finals) == 1
    assert finals[0].candidate.candidate_id == "final_only"
    assert finals[0].status_basis == Provenance.EXACT
    # The synthesized bundle is in addition to the explored candidates.
    assert len(review.bundles) == len(explored) + 1


def test_no_final_candidates_means_no_final_status():
    explored = [
        ReportCandidate(candidate_id="c1", sequence="AAAA", scores={"esm3": {"ptm": 0.7}}),
        ReportCandidate(candidate_id="c2", sequence="CCCC"),
    ]
    review = build_review_model(_model(candidates=explored, finals=[]))
    assert all(b.status != CandidateStatus.FINAL for b in review.bundles)


# --------------------------------------------------------------------------- #
# Non-final status classification
# --------------------------------------------------------------------------- #


def test_failed_qc_status():
    cand = ReportCandidate(
        candidate_id="c1", sequence="AAAA",
        scores={"protein_qc": {"passes_filters": False}},
    )
    review = build_review_model(_model(candidates=[cand]))
    assert _bundle_for(review, "c1").status == CandidateStatus.FAILED_QC


def test_ranked_alternative_when_evaluation_evidence_present():
    cand = ReportCandidate(
        candidate_id="c1", sequence="AAAA",
        scores={"esm3": {"mean_plddt": 0.91, "ptm": 0.8}},
    )
    review = build_review_model(_model(candidates=[cand]))
    assert _bundle_for(review, "c1").status == CandidateStatus.RANKED_ALTERNATIVE


def test_incomplete_evidence_when_no_evaluation():
    cand = ReportCandidate(candidate_id="c1", sequence="AAAA")
    review = build_review_model(_model(candidates=[cand]))
    assert _bundle_for(review, "c1").status == CandidateStatus.INCOMPLETE_EVIDENCE


def _group_for(review, status):
    for g in review.board_groups:
        if g.status == status:
            return g
    return None


def test_retrieval_relabels_incomplete_evidence_group_title():
    # In a retrieval workflow, unscored DB hits aren't a quality failure — they
    # were retrieved but never individually folded/scored. The board group must
    # say so rather than reading as a defect ("Incomplete evidence").
    cands = [ReportCandidate(candidate_id=f"c{i}", sequence="AAAA") for i in range(3)]
    review = build_review_model(
        _model(nodes=[_node("retrieve", "search_sequence")], candidates=cands)
    )
    assert review.workflow_type == WorkflowType.RETRIEVAL
    group = _group_for(review, CandidateStatus.INCOMPLETE_EVIDENCE)
    assert group is not None
    assert group.title == "Retrieved — not individually evaluated"


def test_non_retrieval_keeps_default_incomplete_title():
    # A generic workflow keeps the original wording.
    cand = ReportCandidate(candidate_id="c1", sequence="AAAA")
    review = build_review_model(_model(candidates=[cand]))
    assert review.workflow_type != WorkflowType.RETRIEVAL
    group = _group_for(review, CandidateStatus.INCOMPLETE_EVIDENCE)
    assert group is not None
    assert group.title == "Incomplete evidence"


# --------------------------------------------------------------------------- #
# Score families
# --------------------------------------------------------------------------- #


def test_score_groups_build_structure_confidence_family_with_plddt_normalized():
    cand = ReportCandidate(
        candidate_id="c1", sequence="AAAA",
        scores={"esm3": {"mean_plddt": 0.92, "ptm": 0.81}},
    )
    review = build_review_model(_model(candidates=[cand]))
    groups = {g.family: g for g in _bundle_for(review, "c1").score_groups}
    assert "Structure confidence" in groups
    sc = groups["Structure confidence"]
    metrics = dict((label, value) for label, value, _unit, _src in sc.metrics)
    # 0.92 fraction -> 92.0 on the 0..100 scale.
    assert metrics["pLDDT"] == "92.0"
    assert metrics["pTM"] == "0.81"
    # Each confidence metric is tagged with its producing tool.
    sources = {label: src for label, _v, _u, src in sc.metrics}
    assert sources["pLDDT"] == "ESM-3"
    assert sources["pTM"] == "ESM-3"


def test_score_groups_flags_family_from_qc_fail():
    cand = ReportCandidate(
        candidate_id="c1", sequence="AAAA",
        scores={"protein_qc": {"passes_filters": False}},
    )
    review = build_review_model(_model(candidates=[cand]))
    families = {g.family for g in _bundle_for(review, "c1").score_groups}
    assert "Flags & liabilities" in families


# --------------------------------------------------------------------------- #
# Evidence path producer attribution
# --------------------------------------------------------------------------- #


def test_producer_attribution_exact_when_source_node_matches():
    nodes = [_node("gen1", "rfdiffusion"), _node("eval1", "esm3", upstream=["gen1"])]
    cand = ReportCandidate(
        candidate_id="c1", sequence="AAAA",
        scores={"esm3": {"mean_plddt": 0.9}},
        source={"source_node_id": "gen1", "tool": "rfdiffusion"},
    )
    review = build_review_model(_model(nodes=nodes, candidates=[cand]))
    path = _bundle_for(review, "c1").evidence_path
    assert path.producer is not None
    assert path.producer.node_id == "gen1"
    assert path.producer_provenance == Provenance.EXACT


def test_producer_attribution_inferred_when_single_generator():
    nodes = [_node("gen1", "rfdiffusion"), _node("eval1", "esm3", upstream=["gen1"])]
    cand = ReportCandidate(
        candidate_id="c1", sequence="AAAA",
        scores={"esm3": {"mean_plddt": 0.9}},
        source={},  # no source_node_id
    )
    review = build_review_model(_model(nodes=nodes, candidates=[cand]))
    path = _bundle_for(review, "c1").evidence_path
    assert path.producer is not None
    assert path.producer.node_id == "gen1"
    assert path.producer_provenance == Provenance.INFERRED


def test_plumbing_nodes_excluded_from_evidence_path():
    nodes = [
        _node("gen1", "rfdiffusion"),
        _node("merge1", "merge_candidates", upstream=["gen1"]),
        _node("eval1", "esm3", upstream=["merge1"]),
    ]
    cand = ReportCandidate(candidate_id="c1", sequence="AAAA", scores={"esm3": {"ptm": 0.8}})
    review = build_review_model(_model(nodes=nodes, candidates=[cand]))
    path = _bundle_for(review, "c1").evidence_path
    tools = {step.tool for step in path.shared_pipeline}
    assert "merge_candidates" not in tools
    assert "rfdiffusion" in tools and "esm3" in tools


# --------------------------------------------------------------------------- #
# Board grouping
# --------------------------------------------------------------------------- #


def test_board_groups_ordered_and_failed_collapsed():
    cands = [
        ReportCandidate(candidate_id="final", sequence="FFFF", rank=1),
        ReportCandidate(candidate_id="alt", sequence="AAAA",
                        scores={"esm3": {"ptm": 0.7}}),
        ReportCandidate(candidate_id="bad", sequence="BBBB",
                        scores={"protein_qc": {"passes_filters": False}}),
    ]
    final = ReportCandidate(candidate_id="final", sequence="FFFF")
    review = build_review_model(_model(candidates=cands, finals=[final]))

    titles = [g.title for g in review.board_groups]
    # Final must come first; failed QC after ranked alternatives.
    assert titles.index("Final / Submitted") == 0
    assert "Ranked alternatives" in titles
    failed = next(g for g in review.board_groups if g.status == CandidateStatus.FAILED_QC)
    assert failed.collapsed is True
    final_group = next(g for g in review.board_groups if g.status == CandidateStatus.FINAL)
    assert final_group.collapsed is False


def _rank_designs_node(ranked):
    """A rank_designs node whose output carries a ranked_candidates list."""
    return WorkflowNodeSummary(
        node_id="rank",
        tool="rank_designs",
        version="1",
        state="completed",
        job_id="job_rank",
        output={"ranked_candidates": ranked},
    )


def test_display_rank_follows_ranker_not_collection_order():
    # The submitted design (hoisted to the front of model.candidates → would
    # read as collection-order #1) was actually scored #5 by rank_designs. The
    # board badge must follow the ranker, and the divergence must be flagged.
    cands = [
        ReportCandidate(candidate_id="cand_final", sequence="FFFF"),  # collection #1
        ReportCandidate(candidate_id="cand_best", sequence="BBBB",
                        scores={"esm3": {"ptm": 0.9}}),
        ReportCandidate(candidate_id="cand_mid", sequence="MMMM",
                        scores={"esm3": {"ptm": 0.8}}),
    ]
    ranked = [
        {"candidate_id": "cand_final", "rank": 5},
        {"candidate_id": "cand_best", "rank": 1},
        {"candidate_id": "cand_mid", "rank": 2},
    ]
    final = ReportCandidate(candidate_id="cand_final", sequence="FFFF")
    review = build_review_model(
        _model(nodes=[_rank_designs_node(ranked)], candidates=cands, finals=[final])
    )
    final_b = _bundle_for(review, "cand_final")
    best_b = _bundle_for(review, "cand_best")
    # Authoritative rank comes from the ranker, not the list position.
    assert final_b.display_rank == 5
    assert best_b.display_rank == 1
    # The final pick wasn't the #1-ranked design → divergence note surfaced.
    assert final_b.submitted_despite_rank == 5
    # A design the ranker actually scored #1 carries no divergence note.
    assert best_b.submitted_despite_rank is None


def test_display_rank_falls_back_to_recorded_rank_without_ranker():
    # No rank_designs node → display_rank is the recorded candidate.rank, and no
    # divergence note is ever attached.
    cands = [
        ReportCandidate(candidate_id="c1", sequence="AAAA", rank=2,
                        scores={"esm3": {"ptm": 0.7}}),
    ]
    final = ReportCandidate(candidate_id="c1", sequence="AAAA", rank=2)
    review = build_review_model(_model(candidates=cands, finals=[final]))
    b = _bundle_for(review, "c1")
    assert b.display_rank == 2
    assert b.submitted_despite_rank is None


def test_msa_attached_to_candidate_when_query_matches():
    seq = "ACDEFGHIKL"
    msa = MSABlock(
        path="/tmp/a.a3m", label="hit", sequence_count=2,
        sequences=[("query", seq), ("hit1", "ACDEFGHIKL")],
        mean_identity=0.8,
    )
    cand = ReportCandidate(candidate_id="c1", sequence=seq, scores={"esm3": {"ptm": 0.7}})
    review = build_review_model(_model(candidates=[cand], msa_blocks=[msa]))
    assert len(_bundle_for(review, "c1").msa_blocks) == 1


def test_review_model_exposes_base_and_workflow_type():
    model = _model(nodes=[_node("n1", "pxdesign")])
    review = build_review_model(model)
    assert review.base is model
    assert review.workflow_type == WorkflowType.BINDER
    assert review.grouping_policy == "raw"
