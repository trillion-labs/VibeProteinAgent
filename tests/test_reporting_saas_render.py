"""Rendering smoke tests for the SaaS report hierarchy.

These tests intentionally check semantic section presence and protected
structure controls without snapshotting the full HTML.
"""

from __future__ import annotations

from pathlib import Path

from vibeproteinagent.reporting.models import (
    AgentTranscript,
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
    candidate = ReportCandidate(
        candidate_id="cand_1",
        rank=1,
        sequence="AC",
        structure_path=str(structure),
        scores={
            "esm3": {"mean_plddt": 0.91, "ptm": 0.82},
            "boltz2_affinity": {"iptm": 0.7},
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
                structure_path=str(structure),
                scores={
                    "esm3": {"mean_plddt": 0.81, "ptm": 0.72},
                    "boltz2_affinity": {"iptm": 0.55},
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
        agent_transcript=AgentTranscript(
            run_id="agent_1",
            user_query="Design a small binder.",
            plan_text="Generate candidates, predict structures, score interface, rank.",
        ),
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
        'id="candidate-dossiers"',
        'id="cross-candidate-analysis"',
        'id="workflow-audit"',
        'id="agent-rationale"',
        'id="reproducibility"',
        'id="appendix"',
    ):
        assert marker in html
    assert html.index('id="overview"') < html.index('id="candidate-dossiers"')
    assert html.index('id="candidate-dossiers"') < html.index('id="candidates"')
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
    assert '<button class="top-bar-action top-bar-action-primary" type="button" data-export-report>Export Report</button>' in html
    assert "Download Data" not in html
    assert "saas-icon-card" in html
    assert "overview-card-icon" in html
    assert "Design Review Summary" in html
    assert "Evidence Summary" in html
    assert "Top candidate evidence at a glance" in html
    assert "Candidate review" in html
    assert "Open dossier" in html
    assert "candidate-board-row" in html
    assert "candidate-board-card" in html
    assert "Recommended" in html
    assert "Strong Alternatives" in html or "Strong alternative" in html
    assert "Why this candidate?" not in html
    assert "Binder design review" in html
    assert "Which candidates are most likely to bind the target?" in html
    assert "Recommended due to strong recorded interface evidence and high complex structure confidence" in html
    assert "ipTM" in html
    assert "candidate-dossier-primary" in html
    assert '<div class="dossier-summary-panel">' not in html
    assert "dossier-alert-grid" in html
    assert "candidate-comparison-matrix" in html
    assert "Decision matrix" in html
    assert 'id="workflow-trace"' not in html
    assert "Why this workflow?" not in html
    assert '<ol class="workflow-stage-strip' not in html
    assert '<section id="workflow-audit"' in html
    assert "Workflow Execution Graph" in html
    assert "audit-stage-strip" in html
    assert "Advanced node DAG" not in html
    assert html.index('id="workflow-audit"') < html.index('id="appendix"')
    assert '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' in html
    assert "Patterns across candidates" in html
    assert "Diversity clustering is not recorded" in html
    assert "Advanced candidate evidence" not in html
    assert "metric-bar-row" in html
    assert "metric-bar-track" in html
    assert "metric-value-pill" in html
    assert "Full agent transcript" in html
    assert '<details id="details" class="card appendix-panel">' in html
    assert '<details id="workflow" class="card appendix-panel">' not in html
    assert "<span>Workflow DAG</span>" not in html
    assert "Workflow IDs</span><strong>" not in html
    assert "Tool versions</span><strong>" not in html
    assert "Parameter hashes</span><strong>" not in html
    assert "Artifact hashes</span><strong>" not in html
    assert 'data-open-cand="viewer-cand_1"' in html
    assert 'class="asset-pill is-on asset-download-link" href="data:chemical/x-pdb;base64,' in html
    assert '<span class="asset-pill is-missing" aria-disabled="true">No structure</span>' in html
    assert '<span class="asset-pill is-missing" aria-disabled="true">No downloads</span>' in html

    # Protected molecular review controls remain present in the candidate
    # detail template after the SaaS hierarchy changes.
    assert "3D structure" in html
    assert "pLDDT (" in html
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
