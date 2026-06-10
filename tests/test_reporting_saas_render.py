"""Rendering smoke tests for the SaaS report hierarchy.

These tests intentionally check semantic section presence and protected
structure controls without snapshotting the full HTML.
"""

from __future__ import annotations

import json
from pathlib import Path

from vibeproteinagent.reporting.models import (
    MSABlock,
    ReportCandidate,
    ReportModel,
    ReproducibilityBlock,
    RunSummary,
    WorkflowGraph,
    WorkflowNodeSummary,
)
from vibeproteinagent.reporting.render import render_report_html


def _pdb(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 90.00           N",
                "ATOM      2  CA  ALA A   1      12.560  13.207   2.100  1.00 91.00           C",
                "ATOM      3  CB  ALA A   1      13.000  14.500   2.900  1.00 92.00           C",
                "ATOM      4  C   ALA A   1      13.000  11.900   2.800  1.00 90.00           C",
                "ATOM      5  O   ALA A   1      14.100  11.600   2.500  1.00 90.00           O",
                "ATOM      6  N   CYS A   2      12.100  11.000   3.600  1.00 88.00           N",
                "ATOM      7  CA  CYS A   2      12.400   9.700   4.200  1.00 87.00           C",
                "ATOM      8  CB  CYS A   2      11.100   8.900   4.600  1.00 86.00           C",
                "ATOM      9  C   CYS A   2      13.400   9.900   5.300  1.00 87.00           C",
                "ATOM     10  O   CYS A   2      14.300   9.100   5.400  1.00 87.00           O",
                "TER",
                "END",
            ]
        )
    )
    return path


def _binder_complex_pdb(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  MET B   1       0.000   0.000   0.000  1.00 90.00           C",
                "ATOM      2  CA  ALA B   2       1.000   0.000   0.000  1.00 90.00           C",
                "ATOM      3  CA  GLY A   1       5.000   0.000   0.000  1.00 90.00           C",
                "ATOM      4  CA  LEU A   2       6.000   0.000   0.000  1.00 90.00           C",
                "TER",
                "END",
            ]
        )
    )
    return path


def _node(node_id: str, tool: str, *, upstream=None):
    return WorkflowNodeSummary(
        node_id=node_id,
        tool=tool,
        version="1",
        state="completed",
        job_id=f"job_{node_id}",
        upstream=list(upstream or []),
    )


def test_saas_report_sections_render_and_structure_controls_are_preserved(tmp_path: Path):
    structure = _pdb(tmp_path / "cand_1.pdb")
    alt_structure = _pdb(tmp_path / "cand_2.pdb")
    candidate = ReportCandidate(
        candidate_id="cand_1",
        rank=1,
        sequence="AC",
        structure_path=str(structure),
        scores={
            "esm3": {"mean_plddt": 0.91, "ptm": 0.82},
            "boltz2_affinity": {"iptm": 0.7},
            "ipsae": {"ipsae": 0.68},
            "sequence_analysis": {"gravy": -0.2},
        },
    )
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_saas", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_saas",
            nodes=[
                _node("n1", "pxdesign"),
                _node("n2", "esm3", upstream=["n1"]),
                _node("n3", "boltz2_affinity", upstream=["n2"]),
                _node("n4", "objective_rank", upstream=["n3"]),
            ],
        ),
        candidates=[
            candidate,
            ReportCandidate(
                candidate_id="cand_2",
                rank=2,
                sequence="AD",
                structure_path=str(alt_structure),
                scores={
                    "esm3": {"mean_plddt": 0.81, "ptm": 0.72},
                    "boltz2_affinity": {"iptm": 0.55},
                    "ipsae": {"ipsae": 0.44},
                    "sequence_analysis": {"gravy": -0.1},
                },
            ),
            ReportCandidate(
                candidate_id="cand_3",
                rank=3,
                sequence="AE",
                scores={"protein_qc": {"passes_filters": True}},
            ),
        ],
        final_candidates=[
            ReportCandidate(
                candidate_id="cand_1",
                sequence="AC",
                structure_path=str(structure),
                scores={"esm3": {"mean_plddt": 0.91, "ptm": 0.82}},
            )
        ],
        reproducibility=ReproducibilityBlock(
            ids=["wf_saas"],
            tool_versions={},
            params_hashes={},
            artifact_sha256={},
            replay_commands=["vpa report --workflow-id wf_saas"],
        ),
    )

    html = render_report_html(model)

    for marker in (
        'id="overview"',
        'id="candidates"',
        'id="candidate-comparison"',
        'id="structure-comparison"',
        'id="candidate-dossiers"',
        'id="cross-candidate-analysis"',
        'id="workflow-audit"',
        'id="agent-rationale"',
    ):
        assert marker in html
    assert html.index('id="overview"') < html.index('id="candidate-dossiers"')
    assert html.index('id="candidate-dossiers"') < html.index('id="candidates"')
    assert html.index('id="candidates"') < html.index('id="structure-comparison"')
    assert html.index('id="structure-comparison"') < html.index('id="candidate-comparison"')
    assert html.index('id="cross-candidate-analysis"') < html.index('id="workflow-audit"')
    assert html.index('id="workflow-audit"') < html.index('id="agent-rationale"')
    assert html.index('id="candidates"') < html.index('id="agent-rationale"')
    assert 'data-export-report' in html
    assert 'data-share-report' not in html
    assert ">Share<" not in html
    assert "sidebar-toggle" in html
    assert "data-sidebar-toggle" in html
    assert "sidebar-collapsed" in html
    assert "vpa-report-sidebar-collapsed" in html
    assert (
        '<button class="top-bar-action top-bar-action-primary" type="button" '
        "data-export-report>Export Report</button>"
    ) in html
    assert "Download Data" not in html
    assert "saas-icon-card" in html
    assert "overview-card-icon" in html
    assert "Design review summary" in html
    assert "Evidence Summary" in html
    assert "Top candidate evidence at a glance" in html
    assert (
        "Overview of the leading candidate evidence package before opening the "
        "full candidate review board."
    ) in html
    assert "Structure inspection, risks, downloads, and provenance remain" not in html
    assert "Candidate evidence at a glance" not in html
    assert "ipSAE" in html
    assert "<strong>0.68</strong><span>ipSAE</span>" in html
    assert "snapshot-bar is-missing" not in html
    assert "<b>--</b>" not in html
    assert "Candidate review" in html
    assert "Open dossier" in html
    assert "candidate-board-row" in html
    assert "candidate-board-card" in html
    assert "Recommended" in html
    assert "Evaluated, Not Selected" in html or "Evaluated" in html
    assert "Strong Alternatives" not in html
    assert "Strong alternative" not in html
    assert "Why this candidate?" not in html
    assert "Binder design review" in html
    assert "Which candidates are most likely to bind the target?" in html
    assert (
        "Recommended due to strong recorded interface evidence and high complex "
        "structure confidence"
    ) in html
    assert "ipTM" in html
    assert "candidate-dossier-primary" in html
    assert '<div class="dossier-summary-panel">' not in html
    assert "dossier-alert-grid" in html
    assert "candidate-comparison-matrix" in html
    assert "Decision matrix" in html
    assert (
        "Compare candidates by final status, recorded evidence, missing data, "
        "and practical follow-up needs."
    ) in html
    assert "For binder selection" not in html
    assert "Compare candidate structures side by side" in html
    assert "Drag candidate panels to reorder them" in html
    assert (
        "Drag candidate panels into the order that makes the most useful "
        "structural comparison."
    ) in html
    assert "keep candidate 1 beside candidate 9" not in html
    assert "structure-comparison-grid" in html
    assert "viewer-cand_1__compare_1" in html
    assert "viewer-cand_2__compare_2" in html
    assert "data-structure-compare-grid" in html
    assert "data-structure-compare-card" in html
    assert "setupStructureCompareGrid" in html
    assert "data-structure-comparison-fullscreen" in html
    assert "structure-comparison-fullscreen-btn" in html
    assert "structure-comparison-subtitle-row" in html
    assert "Expand comparison" in html
    assert "min-height:36px" in html
    assert "toggleStructureComparisonFullscreen" in html
    assert ".structure-comparison-panel:fullscreen" in html
    assert "grid-auto-rows:minmax(1040px, auto)" in html
    assert "min-height:1040px" in html
    assert "min-height:820px" in html
    assert "data-structure-compare-fullscreen" not in html
    assert 'draggable="true"' in html
    assert "grid-template-columns:repeat(3, minmax(0, 1fr))" in html
    assert "max-height:clamp(500px, 68vh, 620px)" in html
    assert 'id="workflow-trace"' not in html
    assert "Why this workflow?" not in html
    assert '<ol class="workflow-stage-strip' not in html
    assert '<section id="workflow-audit"' in html
    assert "Workflow execution graph" in html
    assert "audit-stage-strip" in html
    assert "Recorded tool calls" in html
    assert "PXDesign" in html
    assert "Boltz-2" in html
    assert "Advanced node DAG" not in html
    assert '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' in html
    assert "Patterns across candidates" in html
    assert (
        "Adjust the filters to narrow candidates by recorded criteria; matching "
        "candidates and downloads update with the current view."
    ) in html
    assert "Binder-level patterns" not in html
    assert "Filter candidates by metric ranges" in html
    assert 'data-candidate-range-filter' in html
    assert 'class="metric-filter-min"' in html
    assert 'class="metric-filter-max"' in html
    assert "dataset.filterTouched = 'false'" in html
    assert "dataset.resolvedDefaultMin = minInput.value" in html
    assert 'class="candidate-filter-result"' in html
    assert "candidate-filter-primary-metric" in html
    assert ".saas-bars > span" in html
    assert "flex:0 0 auto" in html
    assert "height:24px" in html
    assert "white-space:nowrap" in html
    assert "ipSAE 0.68" in html

    assert html.index("cand_1") < html.index("cand_2")
    assert "Binder length (aa)" in html
    assert "Diversity clustering is not recorded" in html
    assert "Advanced candidate evidence" not in html
    assert "metric-bar-row" in html
    assert "metric-bar-track" in html
    assert "metric-value-pill" in html
    assert '<details id="details" class="card appendix-panel">' not in html
    assert '<details id="workflow" class="card appendix-panel">' not in html
    assert "<span>Workflow DAG</span>" not in html
    assert "Workflow IDs</span><strong>" not in html
    assert "Tool versions</span><strong>" not in html
    assert "Parameter hashes</span><strong>" not in html
    assert "Artifact hashes</span><strong>" not in html
    assert 'id="reproducibility"' not in html
    assert "Replay and audit package" not in html
    assert 'id="appendix"' not in html
    assert '<span class="side-nav-divider">Appendix</span>' not in html
    assert "<summary><span>Per-tool results</span>" not in html
    assert '<h2 class="card-title">Per-tool widgets</h2>' not in html
    assert '<details id="plots"' not in html
    assert "Per-residue pLDDT ·" not in html
    assert "pLDDT vs ipTM" not in html
    assert "<summary><span>Plots</span>" not in html
    assert "overview-summary-card-user-query" in html
    assert "overview-summary-card-agent-objective" in html
    assert "overview-summary-body" in html
    assert "max-height:220px" in html
    assert "max-height:clamp(760px, 84vh, 1120px)" in html

    assert "height:clamp(580px, 64vh, 840px)" in html
    assert "max-height:clamp(360px, 55vh, 680px)" in html
    assert "max-height:clamp(240px, 34vh, 520px)" in html
    assert "overscroll-behavior:contain" in html
    assert 'data-open-cand="viewer-cand_1"' in html
    assert "viewer-cand_2" in html
    cross = html[html.index('id="cross-candidate-analysis"'):]
    assert 'class="asset-pill is-on asset-download-link" href="data:chemical/x-pdb;base64,' in cross
    comparison = html[html.index('id="candidate-comparison"'):html.index('id="cross-candidate-analysis"')]
    assert 'asset-download-link" href="data:chemical/x-pdb;base64,' not in comparison
    structure_comparison = html[html.index('id="structure-comparison"'):html.index('id="candidate-comparison"')]
    assert 'asset-download-link" href="data:chemical/x-pdb;base64,' not in structure_comparison
    assert '<span class="asset-pill is-missing" aria-disabled="true">No structure</span>' in html
    assert '<span class="asset-pill is-missing" aria-disabled="true">No downloads</span>' in html

    # Protected molecular review controls remain present in the candidate
    # detail template after the SaaS hierarchy changes.
    assert "3D structure" in html
    assert "pLDDT (Boltz-2 affinity)" in html
    assert "N to C spectrum" in html
    assert ">Chain<" in html
    assert "Sidechains" in html
    assert "Pick" in html
    assert "Expand" in html
    assert "structure-viewer" in html
    assert "structure-residue-info" in html
    assert "structure-health" in html
    assert "Structure viewer needs a refresh" not in html
    assert "webglcontextlost" in html
    assert "A context-loss event alone is not a user-facing failure" in html
    assert "showViewerFailure(container, '3Dmol could not load the structure model.')" in html
    assert "scheduleViewerResize" in html
    assert "antialias: true" in html
    assert "cartoonQuality: 8" in html
    assert "cartoonVisualSpec" in html
    assert "thickness: 0.58" in html
    assert "width: 1.25" in html
    assert "arrows: true" in html
    assert "zoomViewerToStructure" in html
    assert "backboneTraceSpec" not in html
    assert "addPolymerTraceOverlay" not in html
    assert "vpa-structures" in html


def test_candidate_review_renders_ranking_basis_caption():
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_rank_caption", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_rank_caption",
            nodes=[
                WorkflowNodeSummary(
                    node_id="rank",
                    tool="objective_rank",
                    version="1",
                    state="completed",
                    job_id="job_rank",
                    output={
                        "ranking_intent": "objective",
                        "objective_name": "fold_stability",
                        "formula_text": "-1 * ${pyrosetta.mean_ddg}",
                        "score_direction": "higher",
                        "gate_summary": ["pyrosetta.mean_ddg < 0"],
                        "ranked_candidates": [{"candidate_id": "cand_alt", "rank": 1}],
                    },
                )
            ],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="cand_alt",
                rank=1,
                score=1.13745,
                sequence="ACDE",
                source={"tool": "objective_rank"},
                scores={
                    "objective_rank": {
                        "rank": 1,
                        "objective_score": 1.13745,
                        "objective_name": "fold_stability",
                        "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                        "score_direction": "higher",
                        "passed_gates": [
                            {
                                "id": "stabilizing",
                                "metric": "pyrosetta.mean_ddg",
                                "op": "<",
                                "threshold": 0,
                            }
                        ],
                        "failed_gates": [],
                    }
                },
            )
        ],
    )

    html = render_report_html(model)

    assert "Ranking: objective_rank for fold_stability." in html
    assert "Score: -1 * ${pyrosetta.mean_ddg}; higher is better." in html
    assert "Alternative: objective_score 1.137; gates passed" in html


def test_candidate_review_renders_single_objective_rank_override_note():
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_rank_override", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_rank_override",
            nodes=[
                WorkflowNodeSummary(
                    node_id="rank",
                    tool="objective_rank",
                    version="1",
                    state="completed",
                    job_id="job_rank",
                    output={
                        "ranking_intent": "objective",
                        "objective_name": "fold_stability",
                        "formula_text": "-1 * ${pyrosetta.mean_ddg}",
                        "score_direction": "higher",
                        "gate_summary": ["pyrosetta.mean_ddg < 0"],
                        "ranked_candidates": [
                            {"candidate_id": "cand_best", "rank": 1},
                            {"candidate_id": "cand_final", "rank": 7},
                        ],
                    },
                )
            ],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="cand_final",
                rank=7,
                score=1.13415,
                sequence="ACDE",
                scores={
                    "objective_rank": {
                        "rank": 7,
                        "objective_score": 1.13415,
                        "objective_name": "fold_stability",
                        "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                        "formula_text": "-1 * ${pyrosetta.mean_ddg}",
                        "score_direction": "higher",
                        "gate_summary": ["pyrosetta.mean_ddg < 0"],
                        "passed_gates": [],
                        "failed_gates": [],
                    }
                },
            )
        ],
        final_candidates=[ReportCandidate(candidate_id="cand_final", sequence="ACDE")],
    )

    html = render_report_html(model)

    note = "Submitted objective rank #7: objective_score 1.134."
    assert html.count(note) == 1
    assert "Design" + "-ranking" not in html
    assert "Design" + " ranking" not in html


def test_liability_metrics_render_as_wrapping_tokens():
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_liability", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id="wf_liability"),
        candidates=[
            ReportCandidate(
                candidate_id="cand_liability",
                sequence="ACDEFG",
                liabilities=[
                    "Exposed_hydrophobic_patch",
                    "Buried_polar_candidate",
                ],
            )
        ],
    )

    html = render_report_html(model)

    assert "metric-token-list" in html
    assert '<span class="metric-token">Exposed_hydrophobic_patch</span>' in html
    assert '<span class="metric-token">Buried_polar_candidate</span>' in html
    assert ".metric-bar-row.is-categorical .metric-bar-top" in html
    assert "overflow-wrap:anywhere" in html


def test_overview_keeps_full_long_query_and_formats_segments() -> None:
    long_query = (
        "Enhance fold stability of a 426-aa protein by mutating only buried polar residues. "
        "Keep protected metal residues unchanged; preserve same length; report all metrics fully."
    )
    long_objective = (
        "Improve predicted fold stability while preserving length=426 and protected residues. "
        "Thresholds: ddG < 0, pLDDT within 3 points, no extra mutations."
    )
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_long", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[]),
        raw_summary={"user_query": long_query, "objective": long_objective},
    )

    html = render_report_html(model)

    assert long_query in html
    assert "Improve predicted fold stability while preserving length=426 and protected residues." in html
    assert "pLDDT within 3 points" in html
    assert "overview-objective-block" in html
    assert "Enhance fold stability of a 426-aa protein by mutating only buried polar residues..." not in html


def test_overview_does_not_split_single_query_at_with() -> None:
    query = (
        "Design a protein sequence with diacylglycerol O-acyltransferase activity "
        "(acyl-CoA + 1,2-diacylglycerol -> CoA + triacylglycerol) participating "
        "in triglyceride biosynthetic process"
    )
    objective = (
        "Objective: design a sequence that plausibly adopts a DGAT-compatible fold "
        "Thresholds/criteria: single-seq fold pLDDT>=80, TM-score>=0.5 vs natural scaffold"
    )
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_semantic", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[]),
        raw_summary={"user_query": query, "objective": objective},
    )

    html = render_report_html(model)
    overview = html[html.index('id="overview"'):html.index('id="agent-rationale"')]

    assert (
        "Design a protein sequence with diacylglycerol O-acyltransferase activity "
        "(acyl-CoA + 1,2-diacylglycerol -&gt; CoA + triacylglycerol) participating "
        "in triglyceride biosynthetic process</p>"
    ) in overview
    assert "<li>Design a protein sequence</li>" not in overview
    assert "<li>with diacylglycerol" not in overview
    assert "<b>Objective</b>" in overview
    assert "design a sequence that plausibly adopts a DGAT-compatible fold" in overview
    assert "<b>Acceptance criteria</b>" in overview
    assert "single-seq fold pLDDT&gt;=80, TM-score&gt;=0.5 vs natural scaffold" in overview
    assert "<li>Objective: design a sequence" not in overview
    assert "<li>Thresholds/criteria: single-seq fold" not in overview


def test_agent_objective_renders_objective_and_acceptance_criteria_blocks() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_objective_blocks", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[]),
        raw_summary={
            "objective": (
                "Design an amino-acid sequence that binds Gp23 specifically "
                "and plausibly folds as a standalone protein."
            ),
            "acceptance_criteria": [
                "Complex interface: ipSAE > 0.6 and iptm robust across multiple seeds",
                "Standalone monomer pLDDT >= 80",
                "Sequence QC pass",
            ],
        },
    )

    html = render_report_html(model)
    overview = html[html.index('id="overview"'):html.index('id="agent-rationale"')]

    assert "overview-objective-body" in overview
    assert overview.count("overview-objective-block") == 2
    assert "<b>Objective</b>" in overview
    assert "<b>Acceptance criteria</b>" in overview
    assert "Complex interface" in overview
    assert "Standalone monomer pLDDT" in overview
    assert "Thresholds/criteria" not in overview


def test_agent_objective_extracts_parenthetical_gate_text_into_acceptance_block() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_parenthetical_gates", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[]),
        raw_summary={
            "objective": (
                "Design a sequence that binds Gp23 specifically "
                "(multi-seed-robust interface: ipSAE>0.6 and robust iptm) "
                "and folds standalone (single-sequence monomer pLDDT>=80, sequence QC pass)."
            ),
        },
    )

    html = render_report_html(model)
    overview = html[html.index('id="overview"'):html.index('id="agent-rationale"')]

    assert overview.count("overview-objective-block") == 2
    assert "Design a sequence that binds Gp23 specifically and folds standalone." in overview
    assert "multi-seed-robust interface: ipSAE&gt;0.6 and robust iptm" in overview
    assert "single-sequence monomer pLDDT&gt;=80, sequence QC pass" in overview


def test_agent_rationale_structures_labeled_selection_rationale() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_reasoning", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[_node("submit", "submit_candidate")]),
        final_submission={
            "outcome": "accepted",
            "reasoning": (
                "OBJECTIVE: design a DGAT-like enzyme. "
                "APPROACH: compare fold confidence, topology, and motif retention. "
                "RESULT: selected dgat_1 because it retained the HPHG motif."
            ),
            "candidates_resolved": [
                {"candidate_id": "dgat_1", "justification": "Best recorded evidence."}
            ],
        },
    )

    html = render_report_html(model)
    rationale = html[html.index('id="agent-rationale"'):]

    assert "decision-text-structured" in rationale
    assert 'class="decision-label">Selection rationale</span>' in rationale
    assert 'class="decision-label">Candidate justification</span>' in rationale
    assert "decision-justification-card" in rationale
    assert "<b>Objective</b>" in rationale
    assert "<b>Approach</b>" in rationale
    assert "<b>Result</b>" in rationale
    assert "OBJECTIVE: design a DGAT-like enzyme. APPROACH:" not in rationale
    assert "prov-exact" not in rationale


def test_agent_rationale_structures_heading_with_parenthetical_context() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_reasoning_parenthetical", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[_node("submit", "submit_candidate")]),
        final_submission={
            "outcome": "accepted",
            "reasoning": (
                "OBJECTIVE: design a DGAT-like enzyme. "
                "APPROACH (GO-grounded scaffold redesign): search reviewed DGAT sequences. "
                "BACKBONE VALIDATION: fold the native scaffold with MSA. "
                "RESULT: selected the strongest fold-compatible design."
            ),
        },
    )

    html = render_report_html(model)
    rationale = html[html.index('id="agent-rationale"'):]

    assert "decision-text-structured" in rationale
    assert "<b>Objective</b>" in rationale
    assert "<b>Approach</b>" in rationale
    assert "<b>Backbone Validation</b>" in rationale
    assert "<b>Result</b>" in rationale
    assert "GO-grounded scaffold redesign" not in rationale


def test_agent_rationale_structures_title_case_selection_rationale() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_gp23_reasoning", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[_node("submit", "submit_candidate")]),
        final_submission={
            "outcome": "accepted",
            "reasoning": (
                "Objective: de novo amino-acid sequence that binds Gp23 specifically. "
                "Pipeline: generated binders, screened complexes, then validated monomers. "
                "Acceptance gates: ipSAE>=0.6, robust iptm, standalone pLDDT>=80. "
                "Validated alternatives: bg_18 and bg_11 remain usable profiles."
            ),
        },
    )

    html = render_report_html(model)
    rationale = html[html.index('id="agent-rationale"'):]

    assert "decision-text-structured" in rationale
    assert "<b>Objective</b>" in rationale
    assert "<b>Pipeline</b>" in rationale
    assert "<b>Acceptance Gates</b>" in rationale
    assert "<b>Validated Alternatives</b>" in rationale
    assert "Objective: de novo amino-acid sequence" not in rationale


def test_agent_rationale_structures_single_heading_with_sentence_cues() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_solubility_reasoning", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[_node("submit", "submit_candidate")]),
        final_submission={
            "outcome": "accepted",
            "reasoning": (
                "Objective: enhance aqueous solubility by mutating only designated residues, "
                "preserving length and fold/function. "
                "Explored 4 rational variants via sequence_edit. "
                "Ranked by sequence_analysis solubility proxy: D lowered hydrophobicity "
                "and raised net charge. "
                "objective_rank is not registered in this build, so ranking was done manually. "
                "Folded WT, B, D with protenix_v2: all pLDDT ~89, no collapse. "
                "tmalign vs WT: B 0.923, D 0.907. "
                "D selected as it maximizes the solubility objective while preserving the fold. "
                "B is the conservative runner-up."
            ),
        },
    )

    html = render_report_html(model)
    rationale = html[html.index('id="agent-rationale"'):]

    assert "decision-text-structured" in rationale
    assert "<b>Objective</b>" in rationale
    assert "<b>Exploration</b>" in rationale
    assert "<b>Ranking</b>" in rationale
    assert "<b>Validation</b>" in rationale
    assert "<b>Selection</b>" in rationale
    assert "<b>Alternative</b>" in rationale
    assert "Objective: enhance aqueous solubility" not in rationale
    assert "<b>Objective</b><p>Enhance aqueous solubility" in rationale


def test_workflow_audit_failed_tool_calls_show_reason() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_fail", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[
                WorkflowNodeSummary(
                    node_id="qc_bad",
                    tool="protein_qc",
                    version="1",
                    state="failed",
                    job_id="job_qc_bad",
                    error_code="E_INVALID_PARAMS",
                    error_message="protein_qc requires at least one of sequence / pdb_path",
                )
            ],
        ),
    )

    html = render_report_html(model)

    assert "Needs review" in html
    assert "has-failure" in html
    assert "tool-call-error" in html
    assert "E_INVALID_PARAMS: protein_qc requires at least one of sequence / pdb_path" in html


def test_agent_rationale_cards_use_scrollable_bodies() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_rationale", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[_node("submit", "submit_candidate")],
        ),
        final_submission={
            "terminal": True,
            "outcome": "accepted",
            "reasoning": " ".join(["selection rationale detail"] * 30) + " terminal-marker",
            "candidates_resolved": [
                {
                    "candidate_id": "cand",
                    "justification": " ".join(["candidate justification detail"] * 30) + " final-marker",
                }
            ],
        },
    )

    html = render_report_html(model)

    assert "agent-rationale-section" in html
    assert html.count('<div class="agent-rationale-card-body">') == 4
    assert "terminal-marker" in html
    assert "final-marker" in html


def test_workflow_audit_groups_repeated_tool_calls() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_grouped", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[
                WorkflowNodeSummary(
                    node_id=f"fold_candidate_{idx}",
                    tool="protenix_v2",
                    version="1",
                    state="completed",
                    job_id=f"job_fold_{idx}",
                )
                for idx in range(6)
            ],
        ),
    )

    html = render_report_html(model)

    assert "Grouped by tool and outcome" in html
    assert "tool-call-group" in html
    assert "6 calls" in html
    assert "Predicted candidate structures and confidence metrics." in html
    assert "Show exact calls" not in html
    assert "fold_candidate_0" not in html
    assert "fold_candidate_5" not in html


def test_workflow_audit_tool_call_titles_wrap_without_ellipsis() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_tool_titles", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[_node("edit", "sequence_edit")],
        ),
    )

    html = render_report_html(model)
    strong_rule_start = html.index(".workflow-tool-call-list li strong")
    strong_rule = html[strong_rule_start : html.index("}", strong_rule_start)]

    assert "<strong>Sequence edit</strong>" in html
    assert "white-space:normal" in strong_rule
    assert "overflow-wrap:anywhere" in strong_rule
    assert "text-overflow:ellipsis" not in strong_rule


def test_workflow_audit_uses_domain_specific_stage_labels() -> None:
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_stages", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[
                _node("edit", "sequence_edit"),
                _node("esm", "esm2"),
                _node("fold", "protenix_v2"),
                _node("align", "tmalign"),
                _node("energy", "pyrosetta"),
                _node("qc", "protein_qc"),
            ],
        ),
    )

    html = render_report_html(model)

    assert "Engineering" in html
    assert "Sequence Generation" in html
    assert "Structure Prediction" in html
    assert "Structure Comparison" in html
    assert "Energy Analysis" in html
    assert "Sequence QC" in html
    assert "Model Call" not in html


def test_rosetta_stability_metrics_render_on_candidate_cards() -> None:
    model = ReportModel(
        summary=RunSummary(kind="job", id="job_pyro", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[_node("pyro", "pyrosetta")],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="cand_stable",
                rank=1,
                score=1.0,
                sequence="ACDEFGHIK",
                scores={"pyrosetta": {"total_score": -705.4, "ddg_reu": -52.99}},
                source={"tool": "pyrosetta", "node_id": "pyro"},
            )
        ],
    )

    html = render_report_html(model)

    assert "Rosetta total score" in html
    assert "ddG" in html
    assert "REU" in html
    assert "-52.99" in html
    assert 'class="snapshot-score is-compact"' in html
    assert "grid-template-columns:96px minmax(0,1fr)" in html
    assert "width:92px" in html
    assert "font-variant-numeric:tabular-nums" in html


def test_protenix_monomer_snapshot_uses_fold_confidence_not_iptm(tmp_path: Path) -> None:
    structure = _pdb(tmp_path / "d6.pdb")
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_serine", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[
                _node("fold", "protenix_v2"),
                _node("submit", "submit_candidate"),
            ],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="d6",
                rank=1,
                sequence="AC",
                structure_path=str(structure),
                scores={"protenix_v2": {"plddt": 90.45, "ptm": 0.912, "iptm": 0.0}},
            )
        ],
        final_candidates=[
            ReportCandidate(
                candidate_id="d6",
                sequence="AC",
                structure_path=str(structure),
                scores={"protenix_v2": {"plddt": 90.45, "ptm": 0.912, "iptm": 0.0}},
            )
        ],
        raw_summary={
            "user_query": "Design a protein sequence with serine-type peptidase activity."
        },
    )

    html = render_report_html(model)

    assert "De novo design review" in html
    assert "<strong>90.5</strong><span>pLDDT</span>" in html
    assert "<strong>0</strong><span>ipTM</span>" not in html
    assert "No objective ranking formula was run." in html


def test_candidate_review_hides_alternative_ranks_without_objective_rank(tmp_path: Path) -> None:
    final_structure = _pdb(tmp_path / "cand_final.pdb")
    alt_structure = _pdb(tmp_path / "cand_alt.pdb")
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_manual_rank", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[_node("fold", "protenix_v2"), _node("submit", "submit_candidate")],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="cand_final",
                rank=1,
                sequence="AC",
                structure_path=str(final_structure),
                scores={"protenix_v2": {"plddt": 91.0, "ptm": 0.88}},
            ),
            ReportCandidate(
                candidate_id="cand_alt",
                rank=2,
                sequence="AD",
                structure_path=str(alt_structure),
                scores={"protenix_v2": {"plddt": 89.0, "ptm": 0.86}},
            ),
        ],
        final_candidates=[ReportCandidate(candidate_id="cand_final", sequence="AC")],
    )

    html = render_report_html(model)
    candidates_section = html[html.index('id="candidates"'):html.index('id="structure-comparison"')]

    assert "No objective ranking formula was run." in candidates_section
    assert "Binding review:" not in candidates_section
    assert "Foldability review:" not in candidates_section
    assert "Open a dossier for full evidence" not in candidates_section
    assert "<span class=\"board-group-title\">Alternatives</span>" in candidates_section
    assert "<span class=\"crow-rank-num\">1</span>" in candidates_section
    assert "<span class=\"crow-rank-num\">2</span>" not in candidates_section
    assert "cand_alt" in candidates_section


def test_evidence_summary_renders_submit_selection_summary(tmp_path: Path) -> None:
    structure = _pdb(tmp_path / "cand_final.pdb")
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_selection_summary", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[_node("fold", "protenix_v2"), _node("submit", "submit_candidate")],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="cand_final",
                rank=1,
                sequence="AC",
                structure_path=str(structure),
                scores={"protenix_v2": {"plddt": 91.0, "ptm": 0.88}},
            )
        ],
        final_candidates=[ReportCandidate(candidate_id="cand_final", sequence="AC")],
        final_submission={
            "outcome": "accepted",
            "selection_summary": {
                "headline": "Best balance of confidence and catalytic geometry.",
                "why_selected": "The final candidate passed the stated gates and had the clearest supporting evidence.",
                "key_points": [
                    "Highest fold confidence in the evaluated set",
                    "Catalytic geometry matched the target checks",
                ],
            },
        },
    )

    html = render_report_html(model)
    evidence_section = html[
        html.index('id="candidate-dossiers"'):html.index('id="candidates"')
    ]

    assert "Why this candidate" in evidence_section
    assert "Best balance of confidence and catalytic geometry." in evidence_section
    assert "The final candidate passed the stated gates" in evidence_section
    assert "Highest fold confidence in the evaluated set" in evidence_section
    assert "selection-summary-card" in evidence_section


def test_binder_evidence_summary_keeps_final_complex_over_monomer_variant(
    tmp_path: Path,
) -> None:
    complex_structure = _binder_complex_pdb(tmp_path / "bg_04_complex.pdb")
    monomer_structure = _pdb(tmp_path / "bg_04_mono.pdb")
    complex_source = {
        "candidate_artifacts": {
            "protenix_v2": {
                "complex_chains": [
                    {
                        "role": "target",
                        "input_order": 0,
                        "sequence_length": 2,
                        "observed_chain_id": "A",
                    },
                    {
                        "role": "binder",
                        "input_order": 1,
                        "sequence_length": 2,
                        "observed_chain_id": "B",
                    },
                ]
            }
        }
    }
    model = ReportModel(
        summary=RunSummary(kind="trajectory", id="traj_binder_variant", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id=None, nodes=[_node("fold", "protenix_v2")]),
        final_candidates=[
            ReportCandidate(
                candidate_id="bg_04",
                sequence="MA",
                structure_path=str(complex_structure),
                scores={
                    "protenix_v2": {"iptm": 0.94, "plddt": 93.6, "ptm": 0.94},
                    "ipsae": {"ipsae": 0.91, "ipsae_score": 0.91},
                },
                source=complex_source,
            )
        ],
        candidates=[
            ReportCandidate(
                candidate_id="bg_04_mono",
                sequence="MA",
                structure_path=str(monomer_structure),
                scores={"protenix_v2": {"plddt": 95.0, "ptm": 0.91}},
            ),
            ReportCandidate(
                candidate_id="bg_04_qc",
                sequence="MA",
                scores={
                    "sequence_analysis": {"gravy": 0.2, "net_charge": -2.0},
                    "protein_qc": {"passes_filters": True, "instability_index": 25.0},
                },
            ),
        ],
        raw_summary={"objective": "Design a binder for the target."},
    )

    html = render_report_html(model)
    evidence_section = html[
        html.index('id="candidate-dossiers"'):html.index('id="candidates"')
    ]
    candidates_section = html[
        html.index('id="candidates"'):html.index('id="structure-comparison"')
    ]

    assert "<h3>bg_04</h3>" in evidence_section
    assert "<h3>bg_04_mono</h3>" not in evidence_section
    assert "<span>ipSAE</span>" in evidence_section
    assert "<span>ipTM</span>" in evidence_section
    assert "pLDDT</span>" in evidence_section
    assert "bg_04_mono" not in candidates_section


def test_candidate_comparison_marks_missing_paired_ddg_compactly() -> None:
    model = ReportModel(
        summary=RunSummary(
            kind="trajectory",
            id="traj_stability_missing_ddg",
            workspace="ws",
            state="completed",
        ),
        graph=WorkflowGraph(
            workflow_id=None,
            nodes=[_node("pyro", "pyrosetta")],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="total_only",
                rank=2,
                sequence="ACDEFGHIK",
                scores={"pyrosetta": {"total_score": -112.0}},
                source={"tool": "pyrosetta", "node_id": "pyro"},
            ),
            ReportCandidate(
                candidate_id="paired_ddg",
                rank=1,
                sequence="ACDEFGHIL",
                scores={
                    "pyrosetta": {
                        "total_score": -128.0,
                        "ddg_reu": -15.5,
                        "mean_ddg": -15.0,
                    }
                },
                source={"tool": "pyrosetta", "node_id": "pyro"},
            ),
        ],
    )

    html = render_report_html(model)

    assert html.count('title="No paired ddG"') == 2
    assert 'title="Metric not recorded"' in html
    assert "absolute Rosetta total score" in html
    assert "-15.50 REU" in html
    assert "-15 REU" in html


def _viewer_payloads(html: str) -> list[dict]:
    start = html.index('<script id="vpa-structures" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def test_report_top_k_limits_candidate_heavy_surfaces_and_viewer_payloads(tmp_path: Path):
    candidates = []
    for idx in range(1, 7):
        structure = _pdb(tmp_path / f"cand_{idx}.pdb")
        candidates.append(
            ReportCandidate(
                candidate_id=f"cand_{idx}",
                rank=idx,
                sequence="A" * (50 + idx),
                structure_path=str(structure),
                scores={
                    "esm3": {"mean_plddt": 0.9 - idx * 0.01, "ptm": 0.7},
                    "ipsae": {"ipsae": 0.8 - idx * 0.02},
                },
            )
        )
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_topk", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_topk",
            nodes=[_node("n1", "pxdesign"), _node("n2", "esm3"), _node("n3", "ipsae")],
        ),
        candidates=candidates,
    )

    html = render_report_html(model, top_k=3)

    assert html.count('<button type="button" class="board-card candidate-board-row') == 3
    assert html.count('<template class="cand-detail-tpl"') == 3
    assert html.count('class="comparison-row') == 3
    assert 'data-display-limit-notice' in html
    assert "Showing top 3 of 6 candidates" in html
    assert (
        '<span class="saas-stat-value">6</span><span class="saas-stat-label">Generated</span>'
        in html
    )
    candidates_section = html[html.index('id="candidates"'):html.index('id="structure-comparison"')]
    for idx in range(1, 4):
        assert f"cand_{idx}" in candidates_section
    for idx in range(4, 7):
        assert f"cand_{idx}" not in candidates_section
    payload_ids = {payload["candidate_id"] for payload in _viewer_payloads(html)}
    assert payload_ids == {"cand_1", "cand_2", "cand_3"}
    assert all(not payload["id"].startswith("viewer-cand_4") for payload in _viewer_payloads(html))


def test_report_top_k_always_includes_final_candidate(tmp_path: Path):
    candidates = []
    for idx in range(1, 4):
        structure = _pdb(tmp_path / f"cand_{idx}.pdb")
        candidates.append(
            ReportCandidate(
                candidate_id=f"cand_{idx}",
                rank=idx,
                sequence="A" * 60,
                structure_path=str(structure),
                scores={"esm3": {"mean_plddt": 0.9, "ptm": 0.7}},
            )
        )
    final_structure = _pdb(tmp_path / "cand_final.pdb")
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_topk_final", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_topk_final",
            nodes=[_node("n1", "pxdesign"), _node("n2", "esm3")],
        ),
        candidates=candidates,
        final_candidates=[
            ReportCandidate(
                candidate_id="cand_final",
                rank=99,
                sequence="A" * 60,
                structure_path=str(final_structure),
                scores={"esm3": {"mean_plddt": 0.88, "ptm": 0.7}},
            )
        ],
    )

    html = render_report_html(model, top_k=2)

    candidates_section = html[html.index('id="candidates"'):html.index('id="structure-comparison"')]
    assert "cand_final" in candidates_section
    assert html.count('<button type="button" class="board-card candidate-board-row') == 1
    assert "cand_2" not in candidates_section
    payload_ids = {payload["candidate_id"] for payload in _viewer_payloads(html)}
    assert payload_ids == {"cand_final"}


def test_cross_candidate_binder_length_filters_long_complex_sequence_and_places_structure_download(
    tmp_path: Path,
):
    structure = _pdb(tmp_path / "binder_short.pdb")
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_length", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_length",
            nodes=[_node("n1", "pxdesign"), _node("n2", "esm3"), _node("n3", "ipsae")],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="binder_short",
                rank=1,
                sequence="A" * 60,
                structure_path=str(structure),
                scores={"esm3": {"mean_plddt": 0.9, "ptm": 0.7}, "ipsae": {"ipsae": 0.8}},
            ),
            ReportCandidate(
                candidate_id="long_complex",
                rank=2,
                sequence="A" * 560,
                scores={"esm3": {"mean_plddt": 0.85, "ptm": 0.65}, "ipsae": {"ipsae": 0.1}},
            ),
        ],
    )

    html = render_report_html(model)
    cross = html[html.index('id="cross-candidate-analysis"'):html.index('id="workflow-audit"')]

    assert "Binder length (aa)" in cross
    assert 'data-metric="Binder length (aa)"' in cross
    assert 'data-default-min="60"' in cross
    assert 'data-default-max="60"' in cross
    assert 'data-default-max="560"' not in cross
    assert 'data-candidate-range-filter' in cross
    assert 'data-filter-results' in cross
    assert 'data-filter-result' in cross
    assert 'data-metrics="' in cross
    assert "&quot;Binder length (aa)&quot;:60" in cross
    result_start = cross.index('<article class="candidate-filter-result"')
    result_end = cross.index("</article>", result_start)
    result = cross[result_start:result_end]
    assert result.index("candidate-filter-result-metrics") < result.index("asset-download-link")
    assert 'href="data:chemical/x-pdb;base64,' in result
    assert 'download="binder_short.pdb"' in result
    assert "long_complex" not in result


def test_non_binder_filter_highlights_fold_confidence_primary_metric(tmp_path: Path):
    structure = _pdb(tmp_path / "fold_candidate.pdb")
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_fold", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_fold",
            nodes=[
                _node("design", "esm3"),
                _node("qc", "protein_qc", upstream=["design"]),
            ],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="fold_1",
                rank=1,
                sequence="ACDEFG",
                structure_path=str(structure),
                scores={
                    "esm3": {"mean_plddt": 0.82, "ptm": 0.66},
                    "sequence_analysis": {"gravy": -0.15, "net_charge": -1.0},
                },
            )
        ],
        final_candidates=[
            ReportCandidate(
                candidate_id="fold_1",
                sequence="ACDEFG",
                structure_path=str(structure),
            )
        ],
    )

    html = render_report_html(model)

    assert "De novo design review" in html
    assert 'class="candidate-filter-primary-metric">pLDDT ' in html
    assert 'class="candidate-filter-primary-metric">ipSAE ' not in html


def test_metric_range_filter_step_preserves_decimal_max_boundary(tmp_path: Path):
    low_structure = _pdb(tmp_path / "low.pdb")
    high_structure = _pdb(tmp_path / "high.pdb")
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_fold", workspace="ws", state="completed"),
        graph=WorkflowGraph(
            workflow_id="wf_fold",
            nodes=[_node("fold", "protenix_v2")],
        ),
        candidates=[
            ReportCandidate(
                candidate_id="low",
                rank=1,
                sequence="ACDEFG",
                structure_path=str(low_structure),
                scores={"protenix_v2": {"plddt": 36.1, "ptm": 0.29}},
            ),
            ReportCandidate(
                candidate_id="high",
                rank=2,
                sequence="ACDEFH",
                structure_path=str(high_structure),
                scores={"protenix_v2": {"plddt": 89.8, "ptm": 0.94}},
            ),
        ],
    )

    html = render_report_html(model)
    cross = html[html.index('id="cross-candidate-analysis"'):html.index('id="workflow-audit"')]
    plddt_card = cross[cross.index('data-metric="pLDDT"'):]
    plddt_card = plddt_card[:plddt_card.index("</div></div>")]

    assert 'min="36.1" max="89.8" step="0.01" value="36.1"' in plddt_card
    assert 'min="36.1" max="89.8" step="0.01" value="89.8"' in plddt_card
    assert 'min="0.29" max="0.94" step="0.001" value="0.29"' in cross
    assert 'min="0.29" max="0.94" step="0.001" value="0.94"' in cross
    assert "&quot;pLDDT&quot;:89.8" in cross


def test_appendix_plots_are_suppressed_for_msa_context() -> None:
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_msa", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id="wf_msa", nodes=[_node("msa", "mmseqs2_gpu")]),
        msa_blocks=[
            MSABlock(
                path="/tmp/query.a3m",
                label="query.a3m",
                sequence_count=4,
                sequences=[
                    ("query", "ACDE"),
                    ("hit1", "ACDE"),
                    ("hit2", "AC-E"),
                    ("hit3", "A--E"),
                ],
                identities=[1.0, 1.0, 0.75, 0.5],
                mean_identity=0.8125,
                column_count=4,
                source_tool="mmseqs2_gpu",
            )
        ],
    )

    html = render_report_html(model)

    assert 'id="appendix"' not in html
    assert '<details id="plots" class="card appendix-panel">' not in html
    assert "<summary><span>MSA Plots</span>" not in html
    assert "Mmseqs2 Gpu identity distribution" not in html
    assert "Per-residue pLDDT ·" not in html
    assert "pLDDT vs ipTM" not in html


def test_top_candidate_evidence_snapshot_renders_matching_msa_context() -> None:
    model = ReportModel(
        summary=RunSummary(kind="workflow", id="wf_msa_top", workspace="ws", state="completed"),
        graph=WorkflowGraph(workflow_id="wf_msa_top", nodes=[_node("msa", "mmseqs2_gpu")]),
        candidates=[
            ReportCandidate(
                candidate_id="cand_msa",
                rank=1,
                sequence="ACDE",
                scores={"sequence_analysis": {"gravy": -0.25}},
            )
        ],
        final_candidates=[ReportCandidate(candidate_id="cand_msa", sequence="ACDE")],
        msa_blocks=[
            MSABlock(
                path="/tmp/query.a3m",
                label="query.a3m",
                sequence_count=3,
                sequences=[
                    ("query", "ACDE"),
                    ("hit1", "AC-E"),
                    ("hit2", "A--E"),
                ],
                identities=[1.0, 0.75, 0.5],
                mean_identity=0.75,
                column_count=4,
                source_tool="mmseqs2_gpu",
            )
        ],
    )

    html = render_report_html(model)

    assert "Top candidate evidence at a glance" in html
    assert "Top candidate MSA context" in html
    assert 'class="evidence-snapshot-msa"' in html
    assert "cand_msa MSA context" in html
    assert "query.a3m" in html
    assert "msa-grid" in html
