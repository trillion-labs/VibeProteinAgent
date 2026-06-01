from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.planning.canonical_tools import EXPLICIT_MODE_TOOLS
from vibeproteinagent.agents.planning.prompts import (
    PLAN_VERIFICATION_SYSTEM_PROMPT,
    PLANNER_IR_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)
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
            {"name": tool, "reason": "Selected tools are sufficient for the workflow."}
            for tool in tools
        ],
        "missing_inputs": [],
    }


def _mode_selection(tool: str, mode: str) -> dict[str, Any]:
    return {
        "mode": "tool_selection",
        "selected_tools": [
            {
                "name": tool,
                "selected_mode": mode,
                "reason": "The request determines this operating mode.",
            }
        ],
        "missing_inputs": [],
    }


def _go_rank_ir(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Collect and rank GO-linked candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_candidates",
                "tool": "go_sequences",
                "intent": "Retrieve GO-linked protein sequences.",
                "inputs": inputs or {"go_id": "GO:0003824", "max_sequences": 2},
                "upstream": [],
            },
            {
                "id": "rank_candidates",
                "tool": "rank_designs",
                "intent": "Rank retrieved candidates.",
                "inputs": {"top_k": 1},
                "upstream": [
                    {"node": "go_candidates", "relationship": "candidate_source"}
                ],
            },
        ],
        "finalization": {"success_criteria": {"ranked_candidates": True}},
    }


def _go_only_ir(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Collect GO-linked candidates.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "go_candidates",
                "tool": "go_sequences",
                "intent": "Retrieve GO-linked protein sequences.",
                "inputs": inputs or {"go_id": "GO:0003824", "max_sequences": 2},
                "upstream": [],
            },
        ],
        "finalization": {"success_criteria": {"candidates": True}},
    }


def _tools() -> list[dict[str, str]]:
    return [
        {"name": "go_sequences", "description": "Retrieve GO sequences."},
        {"name": "rank_designs", "description": "Rank candidates."},
        {"name": "pxdesign", "description": "Design binders."},
    ]


def _tools_with_submit_candidate() -> list[dict[str, str]]:
    return [
        {"name": "go_sequences", "description": "Retrieve GO sequences."},
        {"name": "rank_designs", "description": "Rank candidates."},
        {"name": "submit_candidate", "description": "Controller terminal submission."},
    ]


def _pxdesign_tools() -> list[dict[str, str]]:
    return [{"name": "pxdesign", "description": "Design compact binders."}]


def _pxdesign_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Design compact binders.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "binders",
                "tool": "pxdesign",
                "intent": "Generate binder candidates against the target.",
                "inputs": {"target_pdb": "/tmp/target.pdb"},
                "upstream": [],
            }
        ],
        "finalization": {"success_criteria": {"binders": True}},
    }


def _many_tools() -> list[dict[str, str]]:
    return [
        {"name": name, "description": f"{name} available for planning."}
        for name in [
            "bindcraft",
            "esm2",
            "go_sequences",
            "protein_qc",
            "proteinmpnn",
            "pxdesign",
            "rank_designs",
            "rfdiffusion",
            "sequence_analysis",
            "structure_analysis",
        ]
    ]


def _rfdiffusion_tools() -> list[dict[str, str]]:
    return [{"name": "rfdiffusion", "description": "Generate protein backbones."}]


def _pairwise_structure_tools() -> list[dict[str, str]]:
    return [
        {"name": "rfdiffusion", "description": "Generate protein backbones."},
        {"name": "tmalign", "description": "Compare candidate structures."},
    ]


def _rfdiffusion_tmalign_ir() -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Generate a candidate stream and compare structures when enough candidates exist.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "backbones",
                "tool": "rfdiffusion",
                "intent": "Generate backbone candidates.",
                "inputs": {"mode": "unconditional"},
                "upstream": [],
            },
            {
                "id": "align",
                "tool": "tmalign",
                "intent": "Compare generated structures if the stream has enough items.",
                "inputs": {},
                "upstream": [
                    {"node": "backbones", "relationship": "candidate_source"}
                ],
            },
        ],
        "finalization": {"success_criteria": {"alignment_attempted": True}},
    }


def _sequence_edit_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "sequence_edit",
            "description": "Edit or compare protein sequences.",
            "input_schema": {
                "type": "object",
                "required": ["operation"],
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["mutate_residues", "check_wt_consistency"],
                    },
                    "sequence": {"type": "string"},
                    "mapping": {"type": "object"},
                    "candidates": {"type": "array", "items": {"type": "object"}},
                },
                "additionalProperties": False,
            },
        }
    ]


def _boltzgen_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "boltzgen",
            "description": "Generate binder design candidates.",
            "input_schema": {
                "type": "object",
                "oneOf": [
                    {"required": ["target_pdb"]},
                    {"required": ["target_yaml"]},
                    {"required": ["candidates"]},
                ],
                "properties": {
                    "target_pdb": {"type": "string"},
                    "target_yaml": {"type": "string"},
                    "candidates": {"type": "array", "items": {"type": "object"}},
                    "num_designs": {"type": "integer", "default": 10},
                },
            },
        },
        {
            "name": "sequence_analysis",
            "description": "Analyze candidate sequences.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "candidates": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
    ]


def _rfdiffusion_ir(mode: str | None) -> dict[str, Any]:
    inputs = {} if mode is None else {"mode": mode}
    return {
        "version": "planner_ir_v1",
        "goal": "Generate a protein backbone.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "backbone",
                "tool": "rfdiffusion",
                "intent": "Generate a backbone with the selected operating mode.",
                "inputs": inputs,
                "upstream": [],
            }
        ],
        "finalization": {"success_criteria": {"backbone_generated": True}},
    }


def _bindcraft_ir(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Design binders against a known target.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "binders",
                "tool": "bindcraft",
                "intent": "Design binders against the supplied target structure.",
                "inputs": inputs or {},
                "upstream": [],
            }
        ],
        "finalization": {"success_criteria": {"binders": True}},
    }


def test_initial_planner_runs_tool_selection_then_planner_ir() -> None:
    provider = _Provider([_selection("go_sequences", "rank_designs"), _go_rank_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1("Find GO candidates and rank them.", available_tools=_tools())
    )

    assert [message[0].content for message in provider.messages] == [
        TOOL_SELECTION_SYSTEM_PROMPT,
        PLANNER_IR_SYSTEM_PROMPT,
    ]
    selection_prompt = json.loads(provider.messages[0][1].content)
    assert selection_prompt["output_contract"]["mode"] == "tool_selection"
    assert "selected_tools" in selection_prompt["output_contract"]
    assert "input_schema_outline" not in selection_prompt["available_tools"][0]
    assert {"role", "root_inputs", "requires_pdb_file"} <= set(
        selection_prompt["available_tools"][0]
    )

    planner_prompt = json.loads(provider.messages[1][1].content)
    selected_names = {tool["name"] for tool in planner_prompt["tool_catalog"]}
    assert selected_names == {"go_sequences", "rank_designs"}
    go_tool = next(
        tool for tool in planner_prompt["tool_catalog"] if tool["name"] == "go_sequences"
    )
    assert "input_schema_outline" in go_tool
    assert "required" in go_tool["input_schema_outline"]
    assert "go_id" in go_tool["input_schema_outline"]["properties"]
    assert {"role", "root_inputs"} <= set(go_tool)
    assert plan["source_kind"] == "planner_compiled"


def test_submit_candidate_selected_tool_is_ignored_for_planner_ir_usage() -> None:
    provider = _Provider(
        [
            _selection("go_sequences", "rank_designs", "submit_candidate"),
            _go_rank_ir(),
        ]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates, rank them, and submit the final choice.",
            available_tools=_tools_with_submit_candidate(),
        )
    )

    assert plan["source_kind"] == "planner_compiled"
    assert "submit_candidate" not in plan["required_tools"]
    planner_prompt = json.loads(provider.messages[1][1].content)
    assert "submit_candidate" not in {tool["name"] for tool in planner_prompt["tool_catalog"]}
    assert plan["planner_audit_warnings"][0]["event_type"] == "terminal_controller_tool_selected"


def test_pxdesign_defaulted_inputs_are_recorded_in_plan_metadata() -> None:
    provider = _Provider([_selection("pxdesign"), _pxdesign_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design a compact binder against target_pdb=/tmp/target.pdb.",
            available_tools=_pxdesign_tools(),
            known_inputs={"target_pdb": "/tmp/target.pdb"},
            deliverable_contract={
                "version": "deliverable_contract_v1",
                "answer_type": "candidate_submission",
                "required_candidate_capabilities": ["has_sequence", "has_backbone_structure"],
                "terminal_submission_allowed": True,
            },
        )
    )

    node = plan["workflow_plan"]["nodes"][0]
    assert node["inputs"]["target_chain"] == "A"
    assert node["inputs"]["binder_length"] == 60
    assert plan["defaulted_inputs"] == [
        {
            "node_id": "binders",
            "tool": "pxdesign",
            "field": "target_chain",
            "value": "A",
            "reason": "user did not specify chain; conventional default",
        },
        {
            "node_id": "binders",
            "tool": "pxdesign",
            "field": "binder_length",
            "value": 60,
            "reason": "user did not specify binder length; bounded compact-binder default",
        },
    ]
    assert plan["metadata"]["deliverable_contract"]["answer_type"] == "candidate_submission"
    planner_prompt = json.loads(provider.messages[1][1].content)
    pxdesign_schema = next(
        tool for tool in planner_prompt["tool_catalog"]
        if tool["name"] == "pxdesign"
    )
    assert pxdesign_schema["candidate_contract"]["produces"] == [
        "has_backbone_structure",
        "has_candidate_scores",
        "has_sequence",
    ]
    profile = pxdesign_schema["candidate_capability_profiles"][0]
    assert "has_full_structure" in profile["consumes"]
    assert "has_full_structure" not in profile["produces"]
    assert "binder backbone coordinates" in profile["description"]


def test_pxdesign_known_inputs_take_precedence_over_defaults() -> None:
    provider = _Provider([_selection("pxdesign"), _pxdesign_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design binders against target_pdb=/tmp/target.pdb using chain B.",
            available_tools=_pxdesign_tools(),
            known_inputs={
                "target_pdb": "/tmp/target.pdb",
                "target_chain": "B",
                "binder_length": 85,
            },
        )
    )

    node = plan["workflow_plan"]["nodes"][0]
    assert node["inputs"]["target_chain"] == "B"
    assert node["inputs"]["binder_length"] == 85
    assert "defaulted_inputs" not in plan


def test_pairwise_structure_tool_single_upstream_stream_is_planner_valid() -> None:
    provider = _Provider(
        [
            {
                "mode": "tool_selection",
                "selected_tools": [
                    {
                        "name": "rfdiffusion",
                        "selected_mode": "unconditional",
                        "reason": "Generate candidate structures.",
                    },
                    {
                        "name": "tmalign",
                        "reason": "Compare candidate structures at runtime.",
                    },
                ],
                "missing_inputs": [],
            },
            _rfdiffusion_tmalign_ir(),
        ]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Generate structures, then compare them if the stream has enough candidates.",
            available_tools=_pairwise_structure_tools(),
        )
    )

    align = plan["workflow_plan"]["nodes"][-1]
    assert align["tool"] == "tmalign"
    assert align["inputs"]["candidates"] == "${outputs.backbones.candidates}"
    assert not any(
        warning.get("issue") == "candidate_item_count_mismatch"
        for warning in plan.get("planner_audit_warnings", [])
    )


def test_tool_selection_prompt_uses_full_catalog_not_lexical_inventory() -> None:
    provider = _Provider([_selection("go_sequences", "rank_designs"), _go_rank_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    asyncio.run(
        planner._plan_v1(
            "Find GO candidates and rank them.",
            available_tools=_many_tools(),
        )
    )

    selection_prompt = json.loads(provider.messages[0][1].content)
    exposed_names = {tool["name"] for tool in selection_prompt["available_tools"]}
    assert exposed_names == {tool["name"] for tool in _many_tools()}
    assert len(selection_prompt["available_tools"]) > 8


def test_tool_selection_prompt_exposes_allowed_modes_without_schema() -> None:
    provider = _Provider([
        _mode_selection("rfdiffusion", "binder_design"),
        _rfdiffusion_ir("binder_design"),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    asyncio.run(
        planner._plan_v1(
            "Use rfdiffusion in binder design mode.",
            available_tools=_rfdiffusion_tools(),
        )
    )

    selection_prompt = json.loads(provider.messages[0][1].content)
    tool_outline = selection_prompt["available_tools"][0]
    assert tool_outline["name"] == "rfdiffusion"
    assert tool_outline["allowed_modes"] == EXPLICIT_MODE_TOOLS["rfdiffusion"]
    assert "input_schema_outline" not in tool_outline
    assert "root_inputs" in tool_outline
    assert "requires_pdb_file" in tool_outline


def test_tool_selection_and_planner_ir_prompts_expose_capability_profiles() -> None:
    provider = _Provider([
        _selection("boltzgen", "sequence_analysis"),
        {
            "version": "planner_ir_v1",
            "goal": "Generate and analyze full design candidates.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "boltzgen_designs",
                    "tool": "boltzgen",
                    "intent": "Generate design candidates.",
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
            ],
            "finalization": {"success_criteria": {"analyzed": True}},
        },
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Generate binder designs and analyze sequences.",
            available_tools=_boltzgen_tools(),
            known_inputs={"target_pdb": "/tmp/target.pdb"},
        )
    )

    selection_prompt = json.loads(provider.messages[0][1].content)
    boltzgen_outline = next(
        tool for tool in selection_prompt["available_tools"]
        if tool["name"] == "boltzgen"
    )
    profiles = boltzgen_outline["candidate_capability_profiles"]
    full_profile = next(
        profile for profile in profiles
        if profile["name"] == "target_conditioned_design_inverse_fold_folding_analyze"
    )
    assert full_profile["default"] is False
    assert full_profile["inputs"] == {"target_pdb": "<path or canonical candidates>"}
    assert "has_full_structure" in full_profile["consumes"]
    assert "has_sequence" in full_profile["produces"]

    planner_prompt = json.loads(provider.messages[1][1].content)
    boltzgen_schema = next(
        tool for tool in planner_prompt["tool_catalog"]
        if tool["name"] == "boltzgen"
    )
    assert boltzgen_schema["candidate_capability_profiles"] == profiles
    assert boltzgen_schema["candidate_contract"]["produces"] == [
        "has_backbone_structure",
        "has_candidate_scores",
        "has_full_structure",
        "has_sequence",
    ]
    assert plan["source_kind"] == "planner_compiled"


def test_tool_selection_selected_mode_is_passed_to_planner_ir_prompt() -> None:
    provider = _Provider([
        _mode_selection("rfdiffusion", "binder_design"),
        _rfdiffusion_ir("binder_design"),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design a binder backbone with rfdiffusion.",
            available_tools=_rfdiffusion_tools(),
        )
    )

    planner_prompt = json.loads(provider.messages[1][1].content)
    selected_tool = planner_prompt["tool_catalog"][0]
    assert selected_tool["name"] == "rfdiffusion"
    assert selected_tool["selected_mode"] == "binder_design"
    assert selected_tool["allowed_modes"] == EXPLICIT_MODE_TOOLS["rfdiffusion"]
    assert plan["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "binder_design"


def test_selected_mode_can_target_non_mode_input_field() -> None:
    provider = _Provider([
        _mode_selection("sequence_edit", "mutate_residues"),
        {
            "version": "planner_ir_v1",
            "goal": "Mutate a sequence.",
            "assumptions": [],
            "missing_inputs": [],
            "nodes": [
                {
                    "id": "edit_sequence",
                    "tool": "sequence_edit",
                    "intent": "Apply the requested point mutation.",
                    "inputs": {
                        "operation": "mutate_residues",
                        "sequence": "ACDE",
                        "mapping": {"2": "G"},
                    },
                    "upstream": [],
                }
            ],
            "finalization": {"success_criteria": {"edited_sequence": True}},
        },
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Mutate residue 2 to glycine.",
            available_tools=_sequence_edit_tools(),
        )
    )

    selection_prompt = json.loads(provider.messages[0][1].content)
    assert selection_prompt["available_tools"][0]["mode_input_field"] == "operation"
    planner_prompt = json.loads(provider.messages[1][1].content)
    assert planner_prompt["tool_catalog"][0]["mode_input_field"] == "operation"
    assert plan["workflow_plan"]["nodes"][0]["inputs"]["operation"] == "mutate_residues"
    assert "mode" not in plan["workflow_plan"]["nodes"][0]["inputs"]


def test_planner_ir_retry_rejects_selected_mode_mismatch() -> None:
    provider = _Provider(
        [
            _mode_selection("rfdiffusion", "binder_design"),
            _rfdiffusion_ir("unconditional"),
            _rfdiffusion_ir("binder_design"),
        ]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design a binder backbone with rfdiffusion.",
            available_tools=_rfdiffusion_tools(),
            max_schema_retries=2,
        )
    )

    verification_prompt = json.loads(provider.messages[2][1].content)
    violation = verification_prompt["structured_error"]["violations"][0]
    assert provider.messages[2][0].content == PLAN_VERIFICATION_SYSTEM_PROMPT
    retry_prompt = json.loads(provider.messages[3][1].content)
    assert "previous_planner_ir_error" in retry_prompt
    assert retry_prompt["previous_planner_ir_error"]["structured_error"]["violations"][0]["issue"] == (
        "selected_mode_mismatch"
    )
    assert violation["issue"] == "selected_mode_mismatch"
    assert "binder_design" in violation["expected"]
    assert plan["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "binder_design"


def test_planner_ir_retry_rejects_missing_selected_mode() -> None:
    provider = _Provider(
        [
            _mode_selection("rfdiffusion", "binder_design"),
            _rfdiffusion_ir(None),
            _rfdiffusion_ir("binder_design"),
        ]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    asyncio.run(
        planner._plan_v1(
            "Design a binder backbone with rfdiffusion.",
            available_tools=_rfdiffusion_tools(),
            max_schema_retries=2,
        )
    )

    verification_prompt = json.loads(provider.messages[2][1].content)
    violation = verification_prompt["structured_error"]["violations"][0]
    assert provider.messages[2][0].content == PLAN_VERIFICATION_SYSTEM_PROMPT
    assert violation["issue"] == "selected_mode_missing_from_planner_ir"


def test_tool_selection_repairs_invalid_selected_mode() -> None:
    provider = _Provider([
        _mode_selection("rfdiffusion", "not_a_mode"),
        {
            "set_selected_modes": [
                {"name": "rfdiffusion", "selected_mode": "binder_design"}
            ],
            "reason": "Use an allowed RFdiffusion mode.",
        },
        _rfdiffusion_ir("binder_design"),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design a binder backbone with rfdiffusion.",
            available_tools=_rfdiffusion_tools(),
        )
    )

    assert [message[0].content for message in provider.messages] == [
        TOOL_SELECTION_SYSTEM_PROMPT,
        PLAN_VERIFICATION_SYSTEM_PROMPT,
        PLANNER_IR_SYSTEM_PROMPT,
    ]
    assert plan["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "binder_design"


def test_tool_selection_repairs_selected_mode_for_single_mode_tool() -> None:
    provider = _Provider([
        _mode_selection("go_sequences", "anything"),
        {
            "set_selected_modes": [
                {"name": "go_sequences", "selected_mode": None}
            ],
            "reason": "go_sequences is single-mode.",
        },
        _go_only_ir(),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Fetch GO sequences.",
            available_tools=[{"name": "go_sequences", "description": "Retrieve."}],
        )
    )

    assert plan["workflow_plan"]["nodes"][0]["tool"] == "go_sequences"


def test_tool_selection_invalid_mode_still_fails_after_failed_repair() -> None:
    provider = _Provider([
        _mode_selection("rank_designs", "anything"),
        _go_rank_ir(),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    with pytest.raises(PlannerIRSchemaViolation) as excinfo:
        asyncio.run(
            planner._plan_v1(
                "Rank candidates.",
                available_tools=[{"name": "rank_designs", "description": "Rank."}],
            )
        )

    violations = excinfo.value.details["structured_error"]["violations"]
    assert violations[0]["issue"] == "invalid_selected_mode"
    assert violations[0]["tool"] == "rank_designs"


def test_tool_selection_drops_none_selected_mode_for_single_mode_tool() -> None:
    provider = _Provider([
        _mode_selection("go_sequences", "none"),
        _go_only_ir(),
    ])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates.",
            available_tools=[{"name": "go_sequences", "description": "Retrieve."}],
        )
    )

    assert plan["workflow_plan"]["nodes"][0]["tool"] == "go_sequences"


def test_known_inputs_are_prompted_and_injected_before_compile() -> None:
    provider = _Provider([_selection("bindcraft"), _bindcraft_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Design binders.",
            available_tools=[{"name": "bindcraft", "description": "Design binders."}],
            known_inputs={"target_pdb": "/tmp/target.pdb", "pdb_path": "/tmp/target.pdb"},
        )
    )

    selection_prompt = json.loads(provider.messages[0][1].content)
    planner_prompt = json.loads(provider.messages[1][1].content)
    assert selection_prompt["known_inputs"]["target_pdb"] == "/tmp/target.pdb"
    assert planner_prompt["known_inputs"]["target_pdb"] == "/tmp/target.pdb"
    assert (
        plan["workflow_plan"]["nodes"][0]["inputs"]["target_pdb"]
        == "/tmp/target.pdb"
    )
    assert (
        plan["workflow_plan"]["nodes"][0]["input_provenance"]["target_pdb"]
        == "known_input"
    )


def test_required_missing_root_input_blocks_executable_compile() -> None:
    provider = _Provider([_selection("bindcraft"), _bindcraft_ir()])
    planner = WorkflowPlanner(ToolRegistry(), provider)

    with pytest.raises(PlannerIRSchemaViolation) as excinfo:
        asyncio.run(
            planner._plan_v1(
                "Design binders.",
                available_tools=[{"name": "bindcraft", "description": "Design binders."}],
                max_schema_retries=0,
            )
    )

    violations = excinfo.value.details["structured_error"]["violations"]
    issues = {item["issue"] for item in violations}
    assert "required_tool_input_missing" in issues
    assert "unsatisfied_root_required_input" in issues


def test_plan_verification_repair_can_add_ir_introduced_tool() -> None:
    provider = _Provider(
        [
            _selection("go_sequences"),
            _go_rank_ir(),
            {
                "add_selected_tools": ["rank_designs"],
                "drop_selected_tools": [],
                "remove_node_ids": [],
                "reason": "Planner IR uses rank_designs as a downstream selected tool.",
            },
        ]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find GO candidates and rank them.",
            available_tools=_tools(),
        )
    )

    assert [message[0].content for message in provider.messages] == [
        TOOL_SELECTION_SYSTEM_PROMPT,
        PLANNER_IR_SYSTEM_PROMPT,
        PLAN_VERIFICATION_SYSTEM_PROMPT,
    ]
    assert plan["source_kind"] == "planner_compiled"
    assert "rank_designs" in plan["required_tools"]


def test_planner_ir_retry_rejects_unknown_selected_tool_input_field() -> None:
    provider = _Provider(
        [
            _selection("go_sequences", "rank_designs"),
            _go_rank_ir({"go_id": "GO:0003824", "reviewed_only": True}),
            _go_rank_ir({"go_id": "GO:0003824", "reviewed": True}),
        ]
    )
    planner = WorkflowPlanner(ToolRegistry(), provider)

    plan = asyncio.run(
        planner._plan_v1(
            "Find reviewed GO candidates and rank them.",
            available_tools=_tools(),
            max_schema_retries=2,
        )
    )

    assert len(provider.messages) == 4
    verification_prompt = json.loads(provider.messages[2][1].content)
    violation = verification_prompt["structured_error"]["violations"][0]
    assert provider.messages[2][0].content == PLAN_VERIFICATION_SYSTEM_PROMPT
    retry_prompt = json.loads(provider.messages[3][1].content)
    assert retry_prompt["previous_planner_ir_error"]["structured_error"]["violations"][0]["issue"] == (
        "input_field_not_in_tool_schema"
    )
    assert violation["issue"] == "input_field_not_in_tool_schema"
    assert "reviewed" in violation["expected"]
    assert plan["workflow_plan"]["nodes"][0]["inputs"]["reviewed"] is True
