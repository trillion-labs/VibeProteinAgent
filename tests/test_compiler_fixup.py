from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeproteinagent.agents.planning.compiler import (
    apply_fixup_choice,
    compile_planner_ir,
)
from vibeproteinagent.agents.planning.fixup import StructuredFixup

FIXTURES = Path(__file__).parent / "fixtures" / "planner_ir"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_missing_canonical_consumer_slot_raises_structured_fixup() -> None:
    with pytest.raises(StructuredFixup) as excinfo:
        compile_planner_ir(_fixture("missing_consumer_slot.json"))

    payload = excinfo.value.payload
    assert payload["error_kind"] == "missing_canonical_input_slot"
    assert payload["node_id"] == "analyze"
    assert payload["tool"] == "noop"
    assert "rank_designs" in payload["options"][1]["allowed_tools"]


def test_non_canonical_upstream_raises_structured_fixup() -> None:
    with pytest.raises(StructuredFixup) as excinfo:
        compile_planner_ir(_fixture("bad_upstream_producer.json"))

    payload = excinfo.value.payload
    assert payload["error_kind"] == "upstream_does_not_produce_candidates"
    assert payload["upstream_node"] == "noop_step"
    assert payload["upstream_tool"] == "noop"


def test_apply_swap_tool_choice_rewrites_offending_node_and_recompiles() -> None:
    ir = _fixture("missing_consumer_slot.json")
    try:
        compile_planner_ir(ir)
    except StructuredFixup as exc:
        fixed = apply_fixup_choice(
            ir,
            exc.payload,
            {"choice_id": "swap_tool", "tool": "rank_designs"},
        )

    plan = compile_planner_ir(fixed)
    assert plan["workflow_plan"]["nodes"][1]["tool"] == "rank_designs"
    assert plan["workflow_plan"]["nodes"][1]["inputs"]["candidates"] == (
        "${outputs.go_fetch.candidates}"
    )


def test_apply_remove_node_choice_drops_offending_node() -> None:
    ir = _fixture("missing_consumer_slot.json")
    try:
        compile_planner_ir(ir)
    except StructuredFixup as exc:
        fixed = apply_fixup_choice(ir, exc.payload, {"choice_id": "remove_node"})

    plan = compile_planner_ir(fixed)
    assert [node["id"] for node in plan["workflow_plan"]["nodes"]] == ["go_fetch"]


def test_invalid_fixup_choice_id_raises() -> None:
    ir = _fixture("missing_consumer_slot.json")
    with pytest.raises(StructuredFixup) as excinfo:
        compile_planner_ir(ir)

    with pytest.raises(ValueError, match="invalid fixup choice_id"):
        apply_fixup_choice(ir, excinfo.value.payload, {"choice_id": "freeform"})
