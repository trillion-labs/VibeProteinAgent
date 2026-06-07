from __future__ import annotations

from vibeproteinagent.reporting.models import (
    ReportModel,
    RunSummary,
    WorkflowGraph,
    WorkflowNodeSummary,
)
from vibeproteinagent.reporting.render import render_report_html


def test_search_sequence_report_omits_per_tool_appendix_widget() -> None:
    model = ReportModel(
        summary=RunSummary(
            kind="workflow",
            id="wf_search",
            workspace="test",
            state="completed",
            name="search report",
            version="1.0",
        ),
        graph=WorkflowGraph(
            workflow_id="wf_search",
            nodes=[
                WorkflowNodeSummary(
                    node_id="retrieve_sequences",
                    tool="search_sequence",
                    version="1.0.0",
                    state="completed",
                    job_id="job_search",
                    output={
                        "query": "NAD(P)HX repair proteins",
                        "retrieval_status": "partial",
                        "retrieval_sources": ["go"],
                        "source_summaries": {
                            "go": {"status": "completed", "candidate_count": 34, "term_count": 1},
                            "interpro": {"status": "no_terms", "candidate_count": 0, "term_count": 0},
                        },
                        "summary": {
                            "source_candidate_count": 34,
                            "downstream_candidate_count": 20,
                            "max_downstream_candidates": 40,
                        },
                        "empty_facets": [
                            {
                                "event_type": "empty_candidate_event",
                                "label": "InterPro family",
                                "retrieval_status": "no_terms",
                                "stage": "interpro_sequences",
                            }
                        ],
                        "candidates": [],
                        "sequences": [],
                        "artifacts": {},
                        "runtime_ms": 12,
                    },
                )
            ],
        ),
    )

    html = render_report_html(model)

    assert "Workflow execution graph" in html
    assert "search_sequence" in html
    assert "Sequence retrieval" not in html
    assert 'seqret-yield-label">GO' not in html
    assert "Per-tool results" not in html
    assert 'id="appendix"' not in html
