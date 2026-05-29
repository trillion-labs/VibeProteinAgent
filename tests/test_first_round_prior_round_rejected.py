from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.execution.controller import AgentController
from vibeproteinagent.agents.execution.round_state import PriorRoundContext
from vibeproteinagent.agents.planning.canonical_tools import (
    build_canonical_wiring_for_planner,
)
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import (
    FIXUP_KINDS,
    StructuredFixup,
    apply_enum_fixup_choice,
)
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.workspace import Workspace


class _PlannerProvider:
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


def _prior_context() -> PriorRoundContext:
    return PriorRoundContext(
        round_number=2,
        surviving_candidates=[{"schema_version": "candidate_v1", "candidate_id": "c1"}],
        prior_decision_reason="Continue with survivors.",
    )


def _go_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Fetch GO candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_candidates",
                "tool": "go_sequences",
                "intent": "Retrieve candidates.",
                "inputs": {"go_id": "GO:0003824", "max_sequences": 1},
                "upstream": [],
            }
        ],
        "finalization": {"success_criteria": {}},
    }


def _prior_rank_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Rank prior candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "rank_prior",
                "tool": "rank_designs",
                "intent": "Rank prior-round candidates.",
                "inputs": {"top_k": 1},
                "upstream": [
                    {"node": "prior_round", "relationship": "candidate_source"}
                ],
            }
        ],
        "finalization": {"success_criteria": {}},
    }


def _mixed_upstream_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Rank old and new candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_candidates",
                "tool": "go_sequences",
                "intent": "Retrieve fresh candidates.",
                "inputs": {"go_id": "GO:0003824"},
                "upstream": [],
            },
            {
                "id": "rank_prior",
                "tool": "rank_designs",
                "intent": "Rank candidate sources.",
                "inputs": {"top_k": 1},
                "upstream": [
                    {"node": "prior_round", "relationship": "candidate_source"},
                    {"node": "go_candidates", "relationship": "candidate_source"},
                ],
            },
            {
                "id": "rank_fresh",
                "tool": "rank_designs",
                "intent": "Rank only fresh candidates.",
                "inputs": {"top_k": 1},
                "upstream": [
                    {"node": "go_candidates", "relationship": "candidate_source"}
                ],
            },
        ],
        "finalization": {"success_criteria": {}},
    }


def _selection(*tools: str) -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "selected_tools": [
            {"name": tool, "reason": "Selected tools for the planner IR fixture."}
            for tool in tools
        ],
        "missing_inputs": [],
    }


def _tools() -> list[dict[str, str]]:
    return [
        {"name": "go_sequences", "description": "Retrieve GO sequences."},
        {"name": "rank_designs", "description": "Rank candidates."},
    ]


def test_canonical_wiring_gates_prior_round_advertisement() -> None:
    first_round = build_canonical_wiring_for_planner(has_prior_round_context=False)
    continuation = build_canonical_wiring_for_planner(has_prior_round_context=True)

    assert "prior_round" not in first_round["candidate_producers"]
    assert "prior_round" in continuation["candidate_producers"]
    assert "merge_candidates" not in first_round["candidate_producers"]


def test_plan_v1_prompt_excludes_prior_round_without_context() -> None:
    provider = _PlannerProvider([_selection("go_sequences"), _go_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    asyncio.run(planner._plan_v1("Fetch GO candidates.", available_tools=_tools()))

    user_message = json.loads(provider.messages[1][1].content)
    assert "prior_round_context" not in user_message
    assert "prior_round" not in user_message["canonical_wiring"]["candidate_producers"]


def test_plan_v1_prompt_includes_prior_round_with_context() -> None:
    provider = _PlannerProvider([_selection("go_sequences"), _go_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    asyncio.run(
        planner._plan_v1(
            "Continue ranking candidates.",
            available_tools=_tools(),
            prior_round_context=_prior_context(),
        )
    )

    user_message = json.loads(provider.messages[1][1].content)
    assert "prior_round_context" in user_message
    assert "prior_round" in user_message["canonical_wiring"]["candidate_producers"]


def test_compiler_rejects_first_round_prior_round_reference() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(_prior_rank_ir())

    payload = exc_info.value.payload
    assert payload["error_kind"] == "prior_round_referenced_without_context"
    assert {option["choice_id"] for option in payload["options"]} == {
        "remove_upstream",
        "remove_node",
    }


def test_compiler_allows_first_round_ir_without_prior_round_reference() -> None:
    plan = compile_planner_ir(_go_ir())

    assert plan["workflow_plan"]["nodes"][0]["id"] == "go_candidates"


def test_compiler_allows_prior_round_when_context_exists() -> None:
    plan = compile_planner_ir(
        _prior_rank_ir(),
        prior_round_context=_prior_context(),
    )

    assert plan["workflow_plan"]["nodes"][0]["inputs"]["candidates"] == (
        "${outputs.prior_round.candidates}"
    )


def test_compiler_uses_actual_prior_round_candidate_capabilities() -> None:
    context = PriorRoundContext(
        round_number=2,
        surviving_candidates=[
            {
                "schema_version": "candidate_v1",
                "candidate_id": "seq_only",
                "sequence": "ACDE",
                "capabilities": {
                    "has_sequence": True,
                    "has_backbone_structure": False,
                    "has_full_structure": False,
                    "has_candidate_scores": False,
                },
            }
        ],
        prior_decision_reason="Continue with sequence-only survivor.",
    )
    ir = {
        "version": "planner_ir_v1",
        "goal": "Fold prior sequence.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "fold_prior",
                "tool": "protenix_v2",
                "intent": "Fold the surviving sequence.",
                "inputs": {},
                "upstream": [
                    {"node": "prior_round", "relationship": "candidate_source"}
                ],
            }
        ],
        "finalization": {"success_criteria": {}},
    }

    plan = compile_planner_ir(ir, prior_round_context=context)
    assert plan["workflow_plan"]["nodes"][0]["inputs"]["candidates"] == (
        "${outputs.prior_round.candidates}"
    )

    ir["nodes"][0]["tool"] = "proteinmpnn"
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(ir, prior_round_context=context)
    assert exc_info.value.payload["error_kind"] == "candidate_capability_mismatch"
    assert "has_backbone_structure" in exc_info.value.payload["missing_capabilities"]


def test_remove_upstream_fixup_preserves_other_edges_and_nodes() -> None:
    fixup_payload = {
        "error_kind": "prior_round_referenced_without_context",
        "node_id": "rank_prior",
        "upstream_node": "prior_round",
    }

    updated = apply_enum_fixup_choice(
        _mixed_upstream_ir(),
        fixup_payload,
        {"choice_id": "remove_upstream"},
    )

    rank_prior = updated["nodes"][1]
    assert rank_prior["upstream"] == [
        {"node": "go_candidates", "relationship": "candidate_source"}
    ]
    assert updated["nodes"][2]["upstream"] == [
        {"node": "go_candidates", "relationship": "candidate_source"}
    ]


def test_remove_node_fixup_drops_offending_node() -> None:
    updated = apply_enum_fixup_choice(
        _mixed_upstream_ir(),
        {
            "error_kind": "prior_round_referenced_without_context",
            "node_id": "rank_prior",
            "upstream_node": "prior_round",
        },
        {"choice_id": "remove_node"},
    )

    assert [node["id"] for node in updated["nodes"]] == [
        "go_candidates",
        "rank_fresh",
    ]


def test_invalid_fixup_choice_raises() -> None:
    with pytest.raises(ValueError):
        apply_enum_fixup_choice(
            _prior_rank_ir(),
            {
                "error_kind": "prior_round_referenced_without_context",
                "node_id": "rank_prior",
                "upstream_node": "prior_round",
            },
            {"choice_id": "invent_new_tool"},
        )


def test_first_round_prior_round_fixup_retries_to_compiled_plan(
    tmp_workspace: Workspace,
) -> None:
    provider = _PlannerProvider([
        _selection("rank_designs"),
        _prior_rank_ir(),
        {"choice_id": "remove_upstream"},
    ])
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    planner = WorkflowPlanner(ToolRegistry(), provider)

    async def compile_with_fixup(planner_ir: dict[str, Any]) -> dict[str, Any]:
        return await controller._compile_with_fixup(planner_ir, provider=provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Rank candidates.",
            available_tools=_tools(),
            compile_with_fixup=compile_with_fixup,
        )
    )

    assert len(provider.messages) == 3
    fixup_message = json.loads(provider.messages[2][1].content)
    assert fixup_message["structured_fixup"]["error_kind"] == (
        "prior_round_referenced_without_context"
    )
    assert plan["workflow_plan"]["nodes"] == [
        {
            "id": "rank_prior",
            "tool": "rank_designs",
            "inputs": {"top_k": 1},
            "input_provenance": {"top_k": "planner_literal"},
        }
    ]


def test_fixup_kinds_exact_set() -> None:
    assert FIXUP_KINDS == {
        "malformed_fixup_response",
        "missing_explicit_mode",
        "missing_canonical_input_slot",
        "node_reference_used_as_literal_path",
        "prior_round_used_as_tool",
        "upstream_does_not_produce_candidates",
        "candidate_capability_mismatch",
        "prior_round_referenced_without_context",
    }
