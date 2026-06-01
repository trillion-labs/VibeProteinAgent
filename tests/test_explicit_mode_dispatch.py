from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.execution.controller import AgentController
from vibeproteinagent.agents.planning.canonical_tools import EXPLICIT_MODE_TOOLS
from vibeproteinagent.agents.planning.compiler import (
    apply_fixup_choice,
    compile_planner_ir,
)
from vibeproteinagent.agents.planning.fixup import StructuredFixup
from vibeproteinagent.agents.planning.prompts import PLANNER_IR_SYSTEM_PROMPT
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.agents.tools.schemas import compact_tool_schema
from vibeproteinagent.workspace import Workspace

RFDIFFUSION_MODES = EXPLICIT_MODE_TOOLS["rfdiffusion"]


class _FixupProvider:
    provider_name = "fake"

    def __init__(self, choices: list[dict[str, Any]]) -> None:
        self.choices = choices
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        self.messages.append(list(messages))
        choice = self.choices[min(len(self.messages) - 1, len(self.choices) - 1)]
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(choice))


def _ir(tool: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "version": "planner_ir_v1",
        "goal": "Run a focused explicit-mode dispatch test.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "design",
                "tool": tool,
                "intent": "Exercise planner compiler mode handling.",
                "inputs": inputs or {},
                "upstream": [],
            }
        ],
    }


def _rfdiffusion_schema_modes() -> list[str]:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "vibeproteinagent"
        / "adapters"
        / "rfdiffusion"
        / "schemas"
        / "rfdiffusion.input.json"
    )
    schema = json.loads(schema_path.read_text())
    return list(schema["properties"]["mode"]["enum"])


def test_rfdiffusion_modes_match_adapter_schema() -> None:
    assert RFDIFFUSION_MODES == _rfdiffusion_schema_modes()
    assert len(RFDIFFUSION_MODES) == 11


def test_mode_missing_triggers_fixup() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(_ir("rfdiffusion"))

    payload = exc_info.value.payload
    assert payload["error_kind"] == "missing_explicit_mode"
    assert payload["node_id"] == "design"
    assert payload["tool"] == "rfdiffusion"
    assert payload["received_mode"] is None
    assert payload["allowed_modes"] == RFDIFFUSION_MODES
    assert {option["choice_id"] for option in payload["options"]} == {
        "set_mode",
        "remove_node",
    }


def test_invalid_mode_triggers_fixup() -> None:
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(_ir("rfdiffusion", {"mode": "not_a_real_mode"}))

    payload = exc_info.value.payload
    assert payload["error_kind"] == "missing_explicit_mode"
    assert payload["received_mode"] == "not_a_real_mode"
    assert payload["allowed_modes"] == RFDIFFUSION_MODES


def test_valid_mode_compiles_cleanly() -> None:
    plan = compile_planner_ir(_ir("rfdiffusion", {"mode": "unconditional"}))

    assert plan["workflow_plan"]["nodes"] == [
        {
            "id": "design",
            "tool": "rfdiffusion",
            "inputs": {"mode": "unconditional"},
            "input_provenance": {"mode": "planner_literal"},
        }
    ]


def test_apply_set_mode_fixup_rewrites_node_and_recompiles() -> None:
    ir = _ir("rfdiffusion")
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(ir)

    updated = apply_fixup_choice(
        ir,
        exc_info.value.payload,
        {"choice_id": "set_mode", "mode": "binder_design"},
    )

    assert updated["nodes"][0]["inputs"]["mode"] == "binder_design"
    plan = compile_planner_ir(updated)
    assert plan["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "binder_design"


def test_apply_set_mode_with_invalid_mode_raises_value_error() -> None:
    ir = _ir("rfdiffusion")
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(ir)

    with pytest.raises(ValueError, match="mode 'nonsense' is not in allowed_modes"):
        apply_fixup_choice(
            ir,
            exc_info.value.payload,
            {"choice_id": "set_mode", "mode": "nonsense"},
        )


def test_apply_set_mode_with_empty_mode_raises_value_error() -> None:
    ir = _ir("rfdiffusion")
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(ir)

    with pytest.raises(ValueError, match="requires a non-empty mode"):
        apply_fixup_choice(ir, exc_info.value.payload, {"choice_id": "set_mode"})


def test_apply_remove_node_fixup_for_explicit_mode_rejection() -> None:
    ir = {
        "version": "planner_ir_v1",
        "goal": "Drop the explicit-mode node.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "fetch",
                "tool": "go_sequences",
                "intent": "Keep one valid node so the IR remains non-empty.",
                "inputs": {"go_id": "GO:0003824"},
                "upstream": [],
            },
            {
                "id": "design",
                "tool": "rfdiffusion",
                "intent": "Offending missing-mode node.",
                "inputs": {},
                "upstream": [],
            },
        ],
    }
    with pytest.raises(StructuredFixup) as exc_info:
        compile_planner_ir(ir)

    updated = apply_fixup_choice(ir, exc_info.value.payload, {"choice_id": "remove_node"})
    plan = compile_planner_ir(updated)

    assert [node["id"] for node in plan["workflow_plan"]["nodes"]] == ["fetch"]


def test_controller_transport_retries_missing_mode_to_valid_set_mode(
    tmp_workspace: Workspace,
) -> None:
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    provider = _FixupProvider([{"choice_id": "set_mode", "mode": "binder_design"}])

    plan = asyncio.run(controller._compile_with_fixup(_ir("rfdiffusion"), provider=provider))

    assert plan["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "binder_design"
    user_payload = json.loads(provider.messages[0][1].content)
    assert user_payload["structured_fixup"]["error_kind"] == "missing_explicit_mode"


def test_controller_transport_reprompts_malformed_set_mode_choice(
    tmp_workspace: Workspace,
) -> None:
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))
    provider = _FixupProvider(
        [
            {"choice_id": "set_mode"},
            {"choice_id": "set_mode", "mode": "binder_design"},
        ]
    )

    plan = asyncio.run(controller._compile_with_fixup(_ir("rfdiffusion"), provider=provider))

    assert plan["workflow_plan"]["nodes"][0]["inputs"]["mode"] == "binder_design"
    malformed_payload = json.loads(provider.messages[1][1].content)["structured_fixup"]
    assert malformed_payload["error_kind"] == "malformed_fixup_response"
    assert malformed_payload["original_error_kind"] == "missing_explicit_mode"
    assert malformed_payload["parse_error"] == (
        "set_mode fixup choice requires a non-empty mode"
    )


def test_tool_catalog_exposes_allowed_modes_only_for_explicit_mode_tools() -> None:
    compact_rfdiffusion = compact_tool_schema(
        {
            "name": "rfdiffusion",
            "description": "Backbone generation.",
            "input_schema": {"type": "object", "properties": {}},
        }
    )
    compact_structure_analysis = compact_tool_schema(
        {
            "name": "structure_analysis",
            "description": "Structure annotation.",
            "input_schema": {"type": "object", "properties": {}},
        }
    )
    assert compact_rfdiffusion["allowed_modes"] == RFDIFFUSION_MODES
    assert "allowed_modes" not in compact_structure_analysis

    planner = WorkflowPlanner(ToolRegistry())
    catalog = planner._planner_ir_tool_catalog(
        [
            {"name": "rfdiffusion", "description": "Backbone generation."},
            {"name": "structure_analysis", "description": "Structure annotation."},
        ]
    )
    by_name = {entry["name"]: entry for entry in catalog}
    assert by_name["rfdiffusion"]["allowed_modes"] == RFDIFFUSION_MODES
    assert "allowed_modes" not in by_name["structure_analysis"]


def test_planner_prompt_content_sanity_for_multi_mode_guidance() -> None:
    assert "Multi-mode tools" in PLANNER_IR_SYSTEM_PROMPT
    assert "allowed_modes" in PLANNER_IR_SYSTEM_PROMPT
    assert "inputs.mode" in PLANNER_IR_SYSTEM_PROMPT
    assert "binder_design" not in PLANNER_IR_SYSTEM_PROMPT
    assert "symmetric_oligomer_design" not in PLANNER_IR_SYSTEM_PROMPT

    multi_mode_sentence = next(
        sentence
        for sentence in PLANNER_IR_SYSTEM_PROMPT.split(".")
        if "Multi-mode tools" in sentence
    )
    added_words = re.findall(r"\b\w+\b", multi_mode_sentence)
    assert 8 <= len(added_words) <= 20


def test_single_mode_tool_unaffected() -> None:
    plan = compile_planner_ir(_ir("rank_designs", {"top_k": 1}))

    assert plan["workflow_plan"]["nodes"] == [
        {
            "id": "design",
            "tool": "rank_designs",
            "inputs": {"top_k": 1},
            "input_provenance": {"top_k": "planner_literal"},
        }
    ]
