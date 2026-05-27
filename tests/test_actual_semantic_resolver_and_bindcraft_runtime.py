from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import ToolRegistry, null_context
from vibeproteinagent.adapters.bindcraft.adapter import (
    BindCraftInput,
    _bindcraft_budget_warnings,
    _bindcraft_failure_summary,
    _build_advanced,
    _count_attempted_bindcraft_trajectories,
)
from vibeproteinagent.adapters.go_term_search.adapter import (
    GOTermSearchAdapter,
    GOTermSearchInput,
)
from vibeproteinagent.adapters.function_term_resolver.adapter import (
    FunctionTermResolverAdapter,
    FunctionTermResolverInput,
)
from vibeproteinagent.adapters.interpro_entry_search.adapter import (
    InterProEntrySearchAdapter,
    InterProEntrySearchInput,
)
from vibeproteinagent.agents.execution.controller import _is_terminal_planner_entry
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.workflow_planner import (
    PlannerIRSchemaViolation,
    WorkflowPlanner,
)
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.core.errors import InvalidParamsError


class _Provider:
    provider_name = "fake"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        self.messages.append(list(messages))
        payload = self.payloads[min(len(self.messages) - 1, len(self.payloads) - 1)]
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(payload))


def _selection(*tools: str) -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "selected_tools": [
            {"name": tool, "reason": "Selected for the requested workflow."}
            for tool in tools
        ],
        "missing_inputs": [],
    }


def _root_go_without_id() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Retrieve sequences for a natural-language GO function.",
        "assumptions": [],
        "missing_inputs": [
            {
                "name": "dehydratase_go_id",
                "reason": "The user did not provide an exact GO accession.",
                "required_for_execution": True,
            }
        ],
        "nodes": [
            {
                "id": "dehydratase_sequences",
                "tool": "go_sequences",
                "intent": "Retrieve sequences for the dehydratase function.",
                "inputs": {"max_sequences": 2},
                "upstream": [],
            }
        ],
        "finalization": {"success_criteria": {"candidates": True}},
    }


def test_empty_tool_selection_does_not_call_planner_ir() -> None:
    provider = _Provider(
        [{"mode": "tool_selection", "selected_tools": [], "missing_inputs": []}]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    try:
        asyncio.run(
            planner._plan_v1(
                "Find protein sequences for a function.",
                available_tools=[{"name": "go_sequences", "description": "Retrieve GO sequences."}],
            )
        )
    except InvalidParamsError as exc:
        assert exc.details["structured_error"]["error_kind"] == "empty_tool_selection"
    else:  # pragma: no cover - defensive
        raise AssertionError("empty tool selection should fail before Planner IR")

    assert len(provider.messages) == 1


def test_unsatisfied_root_required_input_reports_structured_fields() -> None:
    provider = _Provider([_selection("go_sequences"), _root_go_without_id()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    try:
        asyncio.run(
            planner._plan_v1(
                "Find sequences for dehydratase activity.",
                available_tools=[{"name": "go_sequences", "description": "Retrieve GO sequences."}],
                max_schema_retries=0,
            )
        )
    except PlannerIRSchemaViolation as exc:
        violations = exc.details["structured_error"]["violations"]
    else:  # pragma: no cover - defensive
        raise AssertionError("missing root go_id should block executable compile")

    root_issue = next(
        item for item in violations if item["issue"] == "unsatisfied_root_required_input"
    )
    assert root_issue["tool"] == "go_sequences"
    assert root_issue["node_id"] == "dehydratase_sequences"
    assert root_issue["missing_fields"] == ["go_id"]
    assert root_issue["known_inputs"] == []
    assert "go_id" in root_issue["valid_root_inputs"]
    assert root_issue["required_for_execution"] is True


def test_selected_go_resolver_is_inserted_before_exact_id_retriever() -> None:
    provider = _Provider(
        [_selection("go_term_search", "go_sequences"), _root_go_without_id()]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find sequences for NAD(P)HX dehydratase activity.",
            available_tools=[
                {"name": "go_term_search", "description": "Resolve GO terms."},
                {"name": "go_sequences", "description": "Retrieve GO sequences."},
            ],
            max_schema_retries=0,
        )
    )

    nodes = plan["workflow_plan"]["nodes"]
    assert [node["tool"] for node in nodes] == ["go_term_search", "go_sequences"]
    assert nodes[1]["inputs"]["go_id"] == "${outputs.resolve_dehydratase_sequences.selected_go_id}"
    assert "Target term context:" in nodes[0]["inputs"]["query"]
    assert plan["source_kind"] == "planner_compiled"
    assert len(provider.messages) == 2


def test_term_source_compiler_wires_go_and_interpro_identifiers() -> None:
    go_plan = compile_planner_ir(
        {
            "version": "planner_ir_v1",
            "goal": "Resolve and retrieve GO candidates.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "resolve_go",
                    "tool": "go_term_search",
                    "intent": "Resolve GO term.",
                    "inputs": {"query": "dehydratase activity"},
                    "upstream": [],
                },
                {
                    "id": "go_candidates",
                    "tool": "go_sequences",
                    "intent": "Retrieve GO sequences.",
                    "inputs": {"max_sequences": 2},
                    "upstream": [{"node": "resolve_go", "relationship": "term_source"}],
                },
            ],
            "finalization": {"success_criteria": {"candidates": True}},
        }
    )
    assert go_plan["workflow_plan"]["nodes"][1]["inputs"]["go_id"] == (
        "${outputs.resolve_go.selected_go_id}"
    )

    interpro_plan = compile_planner_ir(
        {
            "version": "planner_ir_v1",
            "goal": "Resolve and retrieve InterPro candidates.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "resolve_interpro",
                    "tool": "interpro_entry_search",
                    "intent": "Resolve InterPro entry.",
                    "inputs": {"query": "epimerase family"},
                    "upstream": [],
                },
                {
                    "id": "family_candidates",
                    "tool": "interpro_sequences",
                    "intent": "Retrieve InterPro sequences.",
                    "inputs": {"max_sequences": 2},
                    "upstream": [
                        {"node": "resolve_interpro", "relationship": "term_source"}
                    ],
                },
            ],
            "finalization": {"success_criteria": {"candidates": True}},
        }
    )
    assert interpro_plan["workflow_plan"]["nodes"][1]["inputs"]["interpro_id"] == (
        "${outputs.resolve_interpro.selected_interpro_id}"
    )


def test_resolver_adapters_report_confidence_without_live_network(monkeypatch, tmp_path: Path) -> None:
    def fake_go_search(client: Any, query: str, *, aspect: str | None, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, aspect, limit
        return ([{"id": "GO:1234567", "name": query, "score": 0.99}], {})

    def fake_interpro_search(client: Any, query: str, *, entry_type: str | None, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, entry_type, limit
        return (
            [
                {"accession": "IPR000001", "name": "A", "score": 0.72},
                {"accession": "IPR000002", "name": "B", "score": 0.70},
            ],
            {},
        )

    monkeypatch.setattr(
        "vibeproteinagent.adapters.go_term_search.adapter.search_go_terms",
        fake_go_search,
    )
    monkeypatch.setattr(
        "vibeproteinagent.adapters.interpro_entry_search.adapter.search_interpro_entries",
        fake_interpro_search,
    )

    go_out = GOTermSearchAdapter({}, tmp_path, torch.device("cpu")).run(
        GOTermSearchInput(query="dehydratase activity"),
        null_context(tmp_path / "go_job"),
    )
    assert go_out.ready_for_execution is True
    assert go_out.selected_go_id == "GO:1234567"
    assert go_out.ambiguous is False

    interpro_out = InterProEntrySearchAdapter({}, tmp_path, torch.device("cpu")).run(
        InterProEntrySearchInput(query="epimerase family"),
        null_context(tmp_path / "interpro_job"),
    )
    assert interpro_out.ready_for_execution is True
    assert interpro_out.selected_interpro_id == "IPR000001"
    assert interpro_out.ambiguous is True


def test_function_term_resolver_exposes_only_confident_selected_ids(monkeypatch, tmp_path: Path) -> None:
    def fake_go_search(client: Any, query: str, *, aspect: str | None, limit: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, aspect, limit
        return ([{"id": "GO:7654321", "score": 0.95}], {})

    def fake_interpro_search(client: Any, query: str, *, limit: int, entry_type: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        del client, query, limit, entry_type
        return (
            [
                {"accession": "IPR000001", "score": 0.6},
                {"accession": "IPR000002", "score": 0.58},
            ],
            {},
        )

    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_go_terms",
        fake_go_search,
    )
    monkeypatch.setattr(
        "vibeproteinagent.adapters.function_term_resolver.adapter.search_interpro_entries",
        fake_interpro_search,
    )

    out = FunctionTermResolverAdapter({}, tmp_path, torch.device("cpu")).run(
        FunctionTermResolverInput(query="dehydratase activity", preferred_namespaces=["go"]),
        null_context(tmp_path / "function_job"),
    )

    assert out.ready_for_execution is True
    assert out.ambiguous is False
    assert out.selected_go_id == "GO:7654321"
    assert out.selected_interpro_id is None


def test_bindcraft_attempted_trajectory_summary_counts_rejected_outputs(tmp_path: Path) -> None:
    for rel, names in {
        "Trajectory/Relaxed": ["target_l60_s1.pdb"],
        "Trajectory/Clashing": ["target_l60_s2.pdb", "target_l60_s3.pdb"],
        "Trajectory/LowConfidence": ["target_l60_s3.pdb", "target_l60_s4.pdb"],
        "Accepted/Ranked": ["accepted_1.pdb"],
    }.items():
        folder = tmp_path / rel
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            (folder / name).write_text("END\n")
    (tmp_path / "failure_csv.csv").write_text(
        "Trajectory_Clashes,Trajectory_final_pLDDT\n2,1\n"
    )

    assert _count_attempted_bindcraft_trajectories(tmp_path) == 4
    summary = _bindcraft_failure_summary(tmp_path)
    assert summary == {
        "attempted": 4,
        "accepted": 1,
        "relaxed": 1,
        "clashing": 2,
        "low_confidence": 2,
        "failure_events": 3,
    }


def test_bindcraft_attempted_cap_is_forwarded_and_timeout_is_validated(tmp_path: Path) -> None:
    params = BindCraftInput(
        target_pdb="/tmp/target.pdb",
        max_trajectories=9,
        max_attempted_trajectories=2,
        max_runtime_s=30,
    )

    advanced = _build_advanced(params, tmp_path)
    assert advanced["max_trajectories"] == 2

    with_timeout_only = BindCraftInput(target_pdb="/tmp/target.pdb", max_trajectories=3)
    assert _build_advanced(with_timeout_only, tmp_path)["max_trajectories"] == 3

    try:
        BindCraftInput(target_pdb="/tmp/target.pdb", max_runtime_s=0)
    except ValidationError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("max_runtime_s must be positive")


def test_bindcraft_no_design_warning_includes_failure_summary() -> None:
    warnings = _bindcraft_budget_warnings(
        {
            "attempted": 5,
            "accepted": 0,
            "relaxed": 0,
            "clashing": 4,
            "low_confidence": 1,
            "failure_events": 5,
        },
        [],
    )

    assert warnings == [
        "BindCraft attempted 5 trajectories but produced no collectable design PDBs "
        "(clashing=4, low_confidence=1)."
    ]


def test_terminal_planner_entry_fast_fail_detection() -> None:
    assert _is_terminal_planner_entry(
        {
            "details": {
                "structured_error": {
                    "violations": [
                        {"issue": "unsatisfied_root_required_input"},
                    ]
                }
            }
        }
    )
