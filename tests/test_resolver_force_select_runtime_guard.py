from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from vibeproteinagent.adapters import ToolRegistry, null_context
from vibeproteinagent.adapters.function_term_resolver.adapter import (
    FunctionTermResolverAdapter,
    FunctionTermResolverInput,
)
from vibeproteinagent.adapters.go_term_search.adapter import (
    GOTermSearchAdapter,
    GOTermSearchInput,
)
from vibeproteinagent.adapters.interpro_entry_search.adapter import (
    InterProEntrySearchAdapter,
    InterProEntrySearchInput,
)
from vibeproteinagent.workflows.descriptor import Workflow
from vibeproteinagent.workflows.executor import (
    WorkflowExecutor,
    _unresolved_term_source_error,
)
from vibeproteinagent.workspace import Workspace


def test_go_term_search_selects_exact_label_match_from_equal_confidence_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_search(
        client: Any,
        query: str,
        *,
        aspect: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, aspect, limit
        return (
            [
                {
                    "id": "GO:0052855",
                    "name": "ADP-dependent NAD(P)H-hydrate dehydratase activity",
                    "score": 1.0,
                },
                {
                    "id": "GO:0047453",
                    "name": "ATP-dependent NAD(P)H-hydrate dehydratase activity",
                    "score": 1.0,
                },
            ],
            {},
        )

    monkeypatch.setattr(
        "vibeproteinagent.adapters.go_term_search.adapter.search_go_terms",
        fake_search,
    )

    out = GOTermSearchAdapter({}, tmp_path, torch.device("cpu")).run(
        GOTermSearchInput(query="ADP-dependent NAD(P)H-hydrate dehydratase activity"),
        null_context(tmp_path / "go_exact"),
    )

    assert out.status == "resolved"
    assert out.ready_for_execution is True
    assert out.ambiguous is False
    assert out.selected_go_id == "GO:0052855"
    assert out.selection_reason == "exact_label_match"


def test_go_term_search_force_selects_from_non_empty_close_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_search(
        client: Any,
        query: str,
        *,
        aspect: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, aspect, limit
        return (
            [
                {"id": "GO:0000002", "name": "near term beta", "score": 0.71},
                {"id": "GO:0000001", "name": "near term alpha", "score": 0.71},
            ],
            {},
        )

    monkeypatch.setattr(
        "vibeproteinagent.adapters.go_term_search.adapter.search_go_terms",
        fake_search,
    )

    out = GOTermSearchAdapter({}, tmp_path, torch.device("cpu")).run(
        GOTermSearchInput(query="distant query"),
        null_context(tmp_path / "go_force"),
    )

    assert out.status == "resolved"
    assert out.ready_for_execution is True
    assert out.ambiguous is True
    assert out.selected_go_id == "GO:0000001"
    assert out.warnings


def test_go_term_search_null_only_for_no_match_or_backend_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def no_match(
        client: Any,
        query: str,
        *,
        aspect: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, aspect, limit
        return ([], {})

    monkeypatch.setattr(
        "vibeproteinagent.adapters.go_term_search.adapter.search_go_terms",
        no_match,
    )
    no_match_out = GOTermSearchAdapter({}, tmp_path, torch.device("cpu")).run(
        GOTermSearchInput(query="missing term"),
        null_context(tmp_path / "go_missing"),
    )
    assert no_match_out.status == "no_match"
    assert no_match_out.ready_for_execution is False
    assert no_match_out.selected_go_id is None

    def backend_error(
        client: Any,
        query: str,
        *,
        aspect: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, aspect, limit
        raise RuntimeError("offline")

    monkeypatch.setattr(
        "vibeproteinagent.adapters.go_term_search.adapter.search_go_terms",
        backend_error,
    )
    error_out = GOTermSearchAdapter({}, tmp_path, torch.device("cpu")).run(
        GOTermSearchInput(query="backend failure"),
        null_context(tmp_path / "go_error"),
    )
    assert error_out.status == "backend_error"
    assert error_out.ready_for_execution is False
    assert error_out.selected_go_id is None


def test_interpro_entry_search_force_selects_from_non_empty_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_search(
        client: Any,
        query: str,
        *,
        entry_type: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, entry_type, limit
        return (
            [
                {"accession": "IPR000002", "name": "beta family", "score": 0.8},
                {"accession": "IPR000001", "name": "alpha family", "score": 0.8},
            ],
            {},
        )

    monkeypatch.setattr(
        "vibeproteinagent.adapters.interpro_entry_search.adapter.search_interpro_entries",
        fake_search,
    )

    out = InterProEntrySearchAdapter({}, tmp_path, torch.device("cpu")).run(
        InterProEntrySearchInput(query="family"),
        null_context(tmp_path / "interpro_force"),
    )

    assert out.status == "resolved"
    assert out.ready_for_execution is True
    assert out.ambiguous is True
    assert out.selected_interpro_id == "IPR000001"


def test_function_term_resolver_selects_go_id_from_close_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_go(
        client: Any,
        query: str,
        *,
        aspect: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, aspect, limit
        return (
            [
                {
                    "id": "GO:0052855",
                    "name": "ADP-dependent NAD(P)H-hydrate dehydratase activity",
                    "score": 1.0,
                },
                {
                    "id": "GO:0047453",
                    "name": "ATP-dependent NAD(P)H-hydrate dehydratase activity",
                    "score": 1.0,
                },
            ],
            {},
        )

    def empty_interpro(
        client: Any,
        query: str,
        *,
        limit: int,
        entry_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, limit, entry_type
        return ([], {})

    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_go_terms",
        fake_go,
    )
    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_interpro_entries",
        empty_interpro,
    )

    out = FunctionTermResolverAdapter({}, tmp_path, torch.device("cpu")).run(
        FunctionTermResolverInput(
            query="ADP-dependent NAD(P)H-hydrate dehydratase activity",
            preferred_namespaces=["go", "interpro"],
        ),
        null_context(tmp_path / "function_go"),
    )

    assert out.status == "resolved"
    assert out.ready_for_execution is True
    assert out.selected_go_id == "GO:0052855"
    assert out.selected_interpro_id is None


def test_function_term_resolver_ready_when_one_requested_namespace_resolves(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_go(
        client: Any,
        query: str,
        *,
        aspect: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, aspect, limit
        return ([{"id": "GO:0008150", "name": "biological process", "score": 0.9}], {})

    def empty_interpro(
        client: Any,
        query: str,
        *,
        limit: int,
        entry_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, limit, entry_type
        return ([], {})

    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_go_terms",
        fake_go,
    )
    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_interpro_entries",
        empty_interpro,
    )

    out = FunctionTermResolverAdapter({}, tmp_path, torch.device("cpu")).run(
        FunctionTermResolverInput(
            query="biological process",
            preferred_namespaces=["go", "interpro"],
        ),
        null_context(tmp_path / "function_partial"),
    )

    assert out.status == "resolved"
    assert out.ready_for_execution is True
    assert out.selected_go_id == "GO:0008150"
    assert out.targets[0]["ready_for_execution"] is True


def test_function_term_resolver_preserves_independent_semantic_intents(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_go(
        client: Any,
        query: str,
        *,
        aspect: str | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, aspect, limit
        if "facet a" in query.lower():
            return (
                [
                    {
                        "id": "GO:0000001",
                        "name": "facet A catalytic activity",
                        "definition": "Supports facet A.",
                        "source": "quickgo",
                    }
                ],
                {},
            )
        return ([], {})

    def empty_interpro(
        client: Any,
        query: str,
        *,
        limit: int,
        entry_type: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, limit, entry_type
        return ([], {})

    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_go_terms",
        fake_go,
    )
    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_interpro_entries",
        empty_interpro,
    )

    out = FunctionTermResolverAdapter({}, tmp_path, torch.device("cpu")).run(
        FunctionTermResolverInput(
            query="facet A and facet B",
            semantic_intents=[
                {"intent_id": "facet_a", "label": "facet A", "query": "facet A"},
                {"intent_id": "facet_b", "label": "facet B", "query": "facet B"},
            ],
            preferred_namespaces=["go"],
        ),
        null_context(tmp_path / "function_multi_intent"),
    )

    assert [intent["intent_id"] for intent in out.semantic_intents] == [
        "facet_a",
        "facet_b",
    ]
    assert out.ready_for_execution is True
    assert out.matches[0]["intent_id"] == "facet_a"
    assert "score_breakdown" in out.matches[0]
    assert out.selected_go_terms[0]["intent_id"] == "facet_a"
    assert out.selected_go_terms[0]["term_id"] == "GO:0000001"
    assert len(out.empty_facets) == 1
    empty_facet = out.empty_facets[0]
    assert empty_facet["event_type"] == "empty_candidate_event"
    assert empty_facet["intent_id"] == "facet_b"
    assert empty_facet["label"] == "facet B"
    assert empty_facet["query"] == "facet B"
    assert empty_facet["selected_go_id"] is None
    assert empty_facet["selected_interpro_id"] is None
    assert empty_facet["candidate_count"] == 0
    assert empty_facet["retrieval_status"] == "empty"
    assert empty_facet["selected_reason"]
    assert empty_facet["resolver_evidence"]["intent_id"] == "facet_b"


def test_runtime_guard_reports_unresolved_term_source_before_submission() -> None:
    consumer = SimpleNamespace(
        runtime_id="dehydratase_sequences",
        desc=SimpleNamespace(
            tool="go_sequences",
            inputs={"go_id": "${outputs.search_dehydratase_term.selected_go_id}"},
        ),
    )
    producer = SimpleNamespace(
        output={
            "status": "no_match",
            "selected_go_id": None,
            "matches": [],
        }
    )

    error = _unresolved_term_source_error(
        consumer,
        {"go_id": None},
        {"search_dehydratase_term": producer},
    )

    assert error is not None
    assert error.details["error_kind"] == "unresolved_term_source"
    assert error.details["producer_node"] == "search_dehydratase_term"
    assert error.details["consumer_node"] == "dehydratase_sequences"
    assert error.details["consumer_tool"] == "go_sequences"
    assert error.details["input_field"] == "go_id"
    assert error.details["resolver_status"] == "no_match"


def test_runtime_guard_completes_exact_id_consumer_as_empty_branch(
    tmp_workspace: Workspace,
) -> None:
    workflow = Workflow(name="resolver_guard", version="1.0.0")
    workflow.node(
        "dehydratase_sequences",
        tool="go_sequences",
        inputs={
            "go_id": "${outputs.external_candidates.selected_go_id}",
            "max_sequences": 2,
        },
    )
    executor = WorkflowExecutor(
        tmp_workspace,
        ToolRegistry(),
        supervisor=None,
        virtual_outputs={
            "external_candidates": {
                "status": "no_match",
                "selected_go_id": None,
                "matches": [],
            }
        },
    )

    workflow_id = executor.submit(workflow, {})
    record = executor.run_sync(workflow_id, max_wait_s=2, poll_s=0.01)
    node = executor.store.get_node(workflow_id, "dehydratase_sequences")

    assert record.state == "completed"
    assert node.state == "completed"
    assert node.job_id is None
    assert node.output["status"] == "skipped_unresolved_term_source"
    assert node.output["candidates"] == []
    assert node.output["error_details"]["error_kind"] == "unresolved_term_source"
    assert node.output["error_details"]["consumer_tool"] == "go_sequences"
