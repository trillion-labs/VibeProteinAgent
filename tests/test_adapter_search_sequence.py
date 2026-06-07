from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from vibeproteinagent.adapters.context import null_context
from vibeproteinagent.adapters.function_term_resolver.adapter import FunctionTermResolverOutput
from vibeproteinagent.adapters.search_sequence.adapter import (
    SearchSequenceAdapter,
    SearchSequenceInput,
    SequenceSourceResult,
)
from vibeproteinagent.candidates.schema import stable_candidate_id


def _candidate(
    tool: str,
    local_id: str,
    sequence: str,
    *,
    score: float,
    term_id: str,
    intent_id: str = "intent_a",
    reviewed: bool = True,
) -> dict[str, Any]:
    source_scores_key = "go_sequences" if tool == "go_sequences" else "interpro_sequences"
    semantic = {
        "sequence_relevance_score": score,
        "term_relevance_score": 0.8,
        "score_breakdown": {"term_match_score": 0.8},
        "intent_id": intent_id,
        "selected_reason": f"selected term {term_id}",
        "intent": {
            "intent_id": intent_id,
            "term_id": term_id,
            "term_relevance_score": 0.8,
            "selected_reason": f"selected term {term_id}",
        },
    }
    return {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(tool, local_id, sequence=sequence),
        "source": {
            "tool": tool,
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": local_id,
            "index": int(local_id.split("_")[-1]),
        },
        "sequence": sequence,
        "capabilities": {"has_sequence": True, "has_candidate_scores": True},
        "scores": {
            "semantic_relevance": semantic,
            source_scores_key: {
                "reviewed": reviewed,
                "sequence_relevance_score": score,
                "term_relevance_score": 0.8,
                "intent": semantic["intent"],
            },
        },
        "artifacts": {},
    }


class _SearchSequenceStub(SearchSequenceAdapter):
    def __init__(self) -> None:
        super().__init__({}, Path("/tmp/vpa_test_weights"), torch.device("cpu"))
        self.calls: list[str] = []
        self.resolver_output = FunctionTermResolverOutput(
            query="query",
            status="no_match",
            matches=[],
            semantic_intents=[],
            empty_facets=[],
            selected_go_terms=[],
            selected_interpro_terms=[],
            warnings=[],
            targets=[],
            ready_for_execution=False,
            ambiguous=False,
            runtime_ms=0,
        )
        self.go_result = SequenceSourceResult("go", "no_terms")
        self.interpro_result = SequenceSourceResult("interpro", "no_terms")

    def _run_resolver(self, params, ctx, query):  # type: ignore[override]
        self.calls.append("resolver")
        return self.resolver_output

    def _run_go(self, params, ctx, terms):  # type: ignore[override]
        self.calls.append("go")
        result = self.go_result
        result.term_count = len(terms)
        return result

    def _run_interpro(self, params, ctx, terms):  # type: ignore[override]
        self.calls.append("interpro")
        result = self.interpro_result
        result.term_count = len(terms)
        return result


def test_search_sequence_explicit_go_runs_go_only(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()
    adapter.go_result = SequenceSourceResult(
        "go",
        "completed",
        candidates=[
            _candidate("go_sequences", "seq_0000", "ACDEFG", score=0.72, term_id="GO:0052855")
        ],
        term_count=1,
    )

    out = adapter.run(
        SearchSequenceInput(go_term="GO:0052855", max_downstream_candidates=5),
        null_context(tmp_path / "job"),
    )

    assert adapter.calls == ["go"]
    assert out.retrieval_status == "completed"
    assert out.retrieval_sources == ["go"]
    assert out.resolved_terms["go_terms"][0]["term_id"] == "GO:0052855"
    assert out.candidates[0]["scores"]["semantic_relevance"]["sequence_relevance_score"] == 0.72


def test_search_sequence_explicit_interpro_runs_interpro_only(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()
    adapter.interpro_result = SequenceSourceResult(
        "interpro",
        "completed",
        candidates=[
            _candidate(
                "interpro_sequences",
                "seq_0000",
                "MNPQRST",
                score=0.61,
                term_id="IPR000001",
            )
        ],
        term_count=1,
    )

    out = adapter.run(
        SearchSequenceInput(interpro_term="IPR000001", max_downstream_candidates=5),
        null_context(tmp_path / "job"),
    )

    assert adapter.calls == ["interpro"]
    assert out.retrieval_status == "completed"
    assert out.retrieval_sources == ["interpro"]
    assert out.resolved_terms["interpro_terms"][0]["term_id"] == "IPR000001"
    assert out.candidates[0]["source"]["tool"] == "interpro_sequences"


def test_search_sequence_semantic_go_only_is_partial_not_failure(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()
    adapter.resolver_output = FunctionTermResolverOutput(
        query="dehydratase",
        status="resolved",
        matches=[],
        semantic_intents=[{"intent_id": "dehydratase", "label": "dehydratase", "query": "dehydratase", "required": True}],
        empty_facets=[],
        selected_go_terms=[
            {
                "intent_id": "dehydratase",
                "intent_label": "dehydratase",
                "term_id": "GO:0052855",
                "label": "ADP-dependent NAD(P)H-hydrate dehydratase activity",
                "term_relevance_score": 0.9,
            }
        ],
        selected_interpro_terms=[],
        warnings=[],
        targets=[
            {
                "intent_id": "dehydratase",
                "label": "dehydratase",
                "query": "dehydratase",
                "selected_go_id": "GO:0052855",
                "selected_interpro_id": None,
            }
        ],
        ready_for_execution=True,
        ambiguous=False,
        runtime_ms=0,
    )
    adapter.go_result = SequenceSourceResult(
        "go",
        "completed",
        candidates=[
            _candidate("go_sequences", "seq_0000", "ACDEFG", score=0.8, term_id="GO:0052855", intent_id="dehydratase")
        ],
        term_count=1,
    )

    out = adapter.run(SearchSequenceInput(context="dehydratase"), null_context(tmp_path / "job"))

    assert adapter.calls == ["resolver", "go"]
    assert out.retrieval_status == "partial"
    assert out.source_summaries["go"]["status"] == "completed"
    assert out.source_summaries["interpro"]["status"] == "no_terms"
    assert out.candidates
    assert any(
        facet["event_type"] == "empty_candidate_event"
        and facet["stage"] == "interpro_sequences"
        and facet["retrieval_status"] == "no_terms"
        for facet in out.empty_facets
    )


def test_search_sequence_both_sources_merge_dedupe_and_top_k(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()
    adapter.resolver_output = FunctionTermResolverOutput(
        query="repair",
        status="resolved",
        matches=[],
        semantic_intents=[],
        empty_facets=[],
        selected_go_terms=[{"intent_id": "repair", "term_id": "GO:0000001"}],
        selected_interpro_terms=[{"intent_id": "repair", "term_id": "IPR000001"}],
        warnings=[],
        targets=[
            {
                "intent_id": "repair",
                "label": "repair",
                "query": "repair",
                "selected_go_id": "GO:0000001",
                "selected_interpro_id": "IPR000001",
            }
        ],
        ready_for_execution=True,
        ambiguous=False,
        runtime_ms=0,
    )
    adapter.go_result = SequenceSourceResult(
        "go",
        "completed",
        candidates=[
            _candidate("go_sequences", "seq_0000", "AAAA", score=0.4, term_id="GO:0000001", intent_id="repair"),
            _candidate("go_sequences", "seq_0001", "CCCC", score=0.9, term_id="GO:0000001", intent_id="repair"),
        ],
        term_count=1,
    )
    adapter.interpro_result = SequenceSourceResult(
        "interpro",
        "completed",
        candidates=[
            _candidate("interpro_sequences", "seq_0000", "AAAA", score=0.7, term_id="IPR000001", intent_id="repair", reviewed=False),
            _candidate("interpro_sequences", "seq_0001", "DDDD", score=0.8, term_id="IPR000001", intent_id="repair"),
        ],
        term_count=1,
    )

    out = adapter.run(
        SearchSequenceInput(context="repair", max_downstream_candidates=2),
        null_context(tmp_path / "job"),
    )

    assert adapter.calls == ["resolver", "go", "interpro"]
    assert out.retrieval_status == "completed"
    assert len(out.candidates) == 2
    assert [candidate["sequence"] for candidate in out.candidates] == ["CCCC", "DDDD"]
    assert out.candidates[0]["source"]["index"] == 0
    assert out.candidates[1]["source"]["index"] == 1
    assert all("semantic_relevance" in candidate["scores"] for candidate in out.candidates)


def test_search_sequence_duplicate_sequence_preserves_source_score_details(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()
    adapter.resolver_output = FunctionTermResolverOutput(
        query="repair",
        status="resolved",
        matches=[],
        semantic_intents=[],
        empty_facets=[],
        selected_go_terms=[{"intent_id": "repair", "term_id": "GO:0000001"}],
        selected_interpro_terms=[{"intent_id": "repair", "term_id": "IPR000001"}],
        warnings=[],
        targets=[
            {
                "intent_id": "repair",
                "label": "repair",
                "query": "repair",
                "selected_go_id": "GO:0000001",
                "selected_interpro_id": "IPR000001",
            }
        ],
        ready_for_execution=True,
        ambiguous=False,
        runtime_ms=0,
    )
    adapter.go_result = SequenceSourceResult(
        "go",
        "completed",
        candidates=[
            _candidate("go_sequences", "seq_0000", "AAAA", score=0.7, term_id="GO:0000001", intent_id="repair")
        ],
        term_count=1,
    )
    adapter.interpro_result = SequenceSourceResult(
        "interpro",
        "completed",
        candidates=[
            _candidate("interpro_sequences", "seq_0000", "AAAA", score=0.4, term_id="IPR000001", intent_id="repair")
        ],
        term_count=1,
    )

    out = adapter.run(SearchSequenceInput(context="repair"), null_context(tmp_path / "job"))

    assert len(out.candidates) == 1
    assert out.candidates[0]["source"]["tool"] == "go_sequences"
    assert "go_sequences" in out.candidates[0]["scores"]
    assert "interpro_sequences" in out.candidates[0]["scores"]


def test_search_sequence_no_hits_normalizes_empty_facets(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()
    adapter.go_result = SequenceSourceResult(
        "go",
        "no_hits",
        candidates=[],
        empty_facets=[
            {
                "event_type": "empty_candidate_event",
                "intent_id": "GO:0052855",
                "label": "dehydratase",
                "selected_term": "GO:0052855",
                "retrieval_status": "empty",
                "candidate_count": 0,
                "stage": "go_sequences",
            }
        ],
        term_count=1,
    )

    out = adapter.run(SearchSequenceInput(go_term="GO:0052855"), null_context(tmp_path / "job"))

    assert out.retrieval_status == "no_hits"
    assert out.candidates == []
    assert out.empty_facets[0]["retrieval_status"] == "no_hits"


def test_search_sequence_one_source_failure_other_success_is_partial(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()
    adapter.go_result = SequenceSourceResult(
        "go",
        "failed",
        term_count=1,
        error={"type": "RuntimeError", "message": "go backend down"},
    )
    adapter.interpro_result = SequenceSourceResult(
        "interpro",
        "completed",
        candidates=[
            _candidate("interpro_sequences", "seq_0000", "MNPQRST", score=0.61, term_id="IPR000001")
        ],
        term_count=1,
    )

    out = adapter.run(
        SearchSequenceInput(go_term="GO:0052855", interpro_term="IPR000001"),
        null_context(tmp_path / "job"),
    )

    assert out.retrieval_status == "partial"
    assert out.candidates
    assert out.source_summaries["go"]["status"] == "failed"
    assert out.source_summaries["go"]["error"]["message"] == "go backend down"


def test_search_sequence_no_terms_is_structured_empty(tmp_path: Path) -> None:
    adapter = _SearchSequenceStub()

    out = adapter.run(SearchSequenceInput(context="unknown facet"), null_context(tmp_path / "job"))

    assert adapter.calls == ["resolver"]
    assert out.retrieval_status == "no_terms"
    assert out.candidates == []
    assert out.source_summaries["go"]["status"] == "no_terms"
    assert out.source_summaries["interpro"]["status"] == "no_terms"
    assert any(facet["event_type"] == "empty_candidate_event" for facet in out.empty_facets)
