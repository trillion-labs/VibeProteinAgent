from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.adapters.go_term_search.adapter import GOTermSearchAdapter, GOTermSearchInput
from vibeproteinagent.agents.evaluation import DecisionEngine, validate_decision_v1
from vibeproteinagent.agents.execution.controller import (
    _first_submit_candidate_ref,
    _resolve_decision_branch,
    _safe_call_id,
)
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.fixup import StructuredFixup
from vibeproteinagent.agents.planning.prompts import (
    PLAN_VERIFICATION_SYSTEM_PROMPT,
    PLANNER_IR_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.workflows.executor import _annotate_output_candidate_capabilities


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


def _ir(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "contract test",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": nodes,
        "finalization": {"success_criteria": {}},
    }


def _rfdiffusion_node() -> dict[str, Any]:
    return {
        "id": "backbone",
        "tool": "rfdiffusion",
        "intent": "Generate backbone candidates.",
        "inputs": {"mode": "unconditional"},
        "upstream": [],
    }


def test_decision_schema_accepts_only_intent_values() -> None:
    validate_decision_v1(
        {
            "decision": "next_round_with_no_acceptable_candidate",
            "reasoning": "No current candidate is acceptable.",
            "next_round": {"reason": "Try another local round."},
            "metrics_used": [],
        }
    )
    with pytest.raises(Exception):
        validate_decision_v1(
            {
                "decision": "no_acceptable_candidate",
                "reasoning": "legacy direct branch",
                "no_acceptable_candidate": {"reason": "legacy"},
                "metrics_used": [],
            }
        )


def test_decision_branch_resolver_separates_intent_from_action() -> None:
    assert _resolve_decision_branch(
        {
            "decision": "next_round_with_no_acceptable_candidate",
            "next_round": {"reason": "try again"},
        },
        workflow_state="completed",
        round_index=0,
        max_rounds=1,
    )["action"] == "stop"
    assert _resolve_decision_branch(
        {
            "decision": "next_round_with_surviving_candidate",
            "next_round": {"reason": "keep survivors"},
        },
        workflow_state="completed",
        round_index=0,
        max_rounds=2,
    )["action"] == "next_round"
    invalid_submit = _resolve_decision_branch(
        {"decision": "submit_candidate", "submit_candidate": {"candidates": []}},
        workflow_state="completed",
        round_index=0,
        max_rounds=1,
    )
    assert invalid_submit == {
        "action": "stop",
        "reason": "submit intent did not include a valid candidate reference",
        "failure": True,
    }
    assert ":" not in _safe_call_id("run:submit-candidate:0")


def test_first_submit_candidate_ref_uses_real_candidate_job_not_target_fallback() -> None:
    ref = _first_submit_candidate_ref(
        {
            "nodes": [
                {
                    "node_id": "bindcraft_empty",
                    "tool": "bindcraft",
                    "job_id": "job_empty",
                    "output": {
                        "status": "no_acceptable_candidate",
                        "target_pdb": "/tmp/target.pdb",
                        "candidates": [],
                    },
                },
                {
                    "node_id": "ranked",
                    "tool": "rank_designs",
                    "job_id": "job_ranked",
                    "output": {
                        "candidates": [
                            {
                                "schema_version": "candidate_v1",
                                "candidate_id": "cand_a",
                                "sequence": "ACDE",
                            }
                        ]
                    },
                },
            ]
        }
    )

    assert ref is not None
    assert ref["job_id"] == "job_ranked"
    assert ref["candidate_id"] == "cand_a"


def test_compiler_rejects_backbone_only_candidate_into_sequence_consumer() -> None:
    with pytest.raises(StructuredFixup) as excinfo:
        compile_planner_ir(
            _ir(
                [
                    _rfdiffusion_node(),
                    {
                        "id": "sequence_summary",
                        "tool": "sequence_analysis",
                        "intent": "Analyze sequence.",
                        "inputs": {},
                        "upstream": [
                            {"node": "backbone", "relationship": "candidate_source"}
                        ],
                    },
                ]
            )
        )
    assert excinfo.value.payload["error_kind"] == "candidate_capability_mismatch"
    assert excinfo.value.payload["missing_capabilities"] == ["has_sequence"]


def test_compiler_allows_backbone_to_mpnn_to_sequence_analysis() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                _rfdiffusion_node(),
                {
                    "id": "mpnn",
                    "tool": "proteinmpnn",
                    "intent": "Design sequences for the backbone.",
                    "inputs": {},
                    "upstream": [
                        {"node": "backbone", "relationship": "candidate_source"}
                    ],
                },
                {
                    "id": "sequence_summary",
                    "tool": "sequence_analysis",
                    "intent": "Analyze designed sequences.",
                    "inputs": {},
                    "upstream": [{"node": "mpnn", "relationship": "candidate_source"}],
                },
            ]
        )
    )
    assert [node["tool"] for node in plan["workflow_plan"]["nodes"]] == [
        "rfdiffusion",
        "proteinmpnn",
        "sequence_analysis",
    ]


def test_compiler_treats_boltzgen_default_as_full_candidate_producer() -> None:
    plan = compile_planner_ir(
        _ir(
            [
                {
                    "id": "boltzgen_designs",
                    "tool": "boltzgen",
                    "intent": "Generate full design candidates.",
                    "inputs": {"target_pdb": "/tmp/target.pdb"},
                    "upstream": [],
                },
                {
                    "id": "sequence_summary",
                    "tool": "sequence_analysis",
                    "intent": "Analyze designed sequences.",
                    "inputs": {},
                    "upstream": [
                        {
                            "node": "boltzgen_designs",
                            "relationship": "candidate_source",
                        }
                    ],
                },
            ]
        )
    )
    assert [node["tool"] for node in plan["workflow_plan"]["nodes"]] == [
        "boltzgen",
        "sequence_analysis",
    ]


def test_compiler_rejects_boltzgen_design_only_into_sequence_consumer() -> None:
    with pytest.raises(StructuredFixup) as excinfo:
        compile_planner_ir(
            _ir(
                [
                    {
                        "id": "boltzgen_backbones",
                        "tool": "boltzgen",
                        "intent": "Generate backbone-only candidates.",
                        "inputs": {
                            "target_pdb": "/tmp/target.pdb",
                            "design_only": True,
                        },
                        "upstream": [],
                    },
                    {
                        "id": "sequence_summary",
                        "tool": "sequence_analysis",
                        "intent": "Analyze designed sequences.",
                        "inputs": {},
                        "upstream": [
                            {
                                "node": "boltzgen_backbones",
                                "relationship": "candidate_source",
                            }
                        ],
                    },
                ]
            )
        )
    assert excinfo.value.payload["error_kind"] == "candidate_capability_mismatch"
    assert excinfo.value.payload["missing_capabilities"] == ["has_sequence"]


def test_tool_selection_invalid_single_mode_is_repaired_before_planner_ir() -> None:
    provider = _Provider(
        [
            {
                "mode": "tool_selection",
                "selected_tools": [
                    {"name": "go_sequences", "selected_mode": "easy-search"}
                ],
            },
            {
                "set_selected_modes": [
                    {"name": "go_sequences", "selected_mode": None}
                ],
                "reason": "go_sequences is single-mode.",
            },
            _ir(
                [
                    {
                        "id": "go_fetch",
                        "tool": "go_sequences",
                        "intent": "Fetch GO sequences.",
                        "inputs": {"go_id": "GO:0003824", "max_sequences": 2},
                        "upstream": [],
                    }
                ]
            ),
        ]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)
    plan = asyncio.run(
        planner._plan_v1(
            "Fetch GO sequences.",
            available_tools=[{"name": "go_sequences", "description": "Fetch GO."}],
        )
    )
    assert [message[0].content for message in provider.messages] == [
        TOOL_SELECTION_SYSTEM_PROMPT,
        PLAN_VERIFICATION_SYSTEM_PROMPT,
        PLANNER_IR_SYSTEM_PROMPT,
    ]
    assert plan["source_kind"] == "planner_compiled"


def test_resolver_backend_error_is_structured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("backend unavailable")

    monkeypatch.setattr(
        "vibeproteinagent.adapters.go_term_search.adapter.search_go_terms",
        boom,
    )

    class _Ctx:
        job_dir = tmp_path

        def emit_progress(self, value: float) -> None:
            del value

    adapter = GOTermSearchAdapter({}, tmp_path, torch.device("cpu"))
    out = adapter.run(GOTermSearchInput(query="NAD(P)H hydrate dehydratase"), _Ctx())
    assert out.status == "backend_error"
    assert out.matches == []
    assert out.warnings
    assert out.ready_for_execution is False


def test_runtime_candidate_capability_annotation_is_derived_from_output() -> None:
    output = {
        "candidates": [
            {
                "candidate_id": "c1",
                "sequence": "ACDE",
                "structure_path": "/tmp/c1.pdb",
                "scores": {"rank": 1},
            }
        ]
    }
    annotated = _annotate_output_candidate_capabilities(output)
    capabilities = annotated["candidates"][0]["capabilities"]
    assert capabilities == {
        "has_backbone_structure": True,
        "has_full_structure": True,
        "has_sequence": True,
        "has_candidate_scores": True,
    }


def test_completed_empty_candidates_are_normal_no_acceptable_decision_input() -> None:
    decision = DecisionEngine(None).local_decision(
        {
            "state": "completed",
            "outputs": {
                "bindcraft": {
                    "status": "no_acceptable_candidate",
                    "candidates": [],
                    "num_designs_returned": 0,
                }
            },
            "metrics_used": [],
        }
    )

    assert decision["decision"] == "next_round_with_no_acceptable_candidate"
    assert _resolve_decision_branch(
        decision,
        workflow_state="completed",
        round_index=0,
        max_rounds=1,
    ) == {
        "action": "stop",
        "reason": "no remaining local rounds",
        "failure": False,
    }
