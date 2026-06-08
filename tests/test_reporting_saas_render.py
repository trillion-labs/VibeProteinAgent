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
                _node("n4", "rank_designs", upstream=["n3"]),
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
    assert "Compare candidate structures side by side" in html
    assert "Drag candidate panels to reorder them" in html
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
    assert "max-height:150px" in html
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
