from __future__ import annotations

from copy import deepcopy
from typing import Any

from vibeproteinagent.agents.planning.canonical_tools import (
    CANONICAL_CONSUMERS,
    CANONICAL_PRODUCERS,
    CANONICAL_SCALAR_CONSUMERS,
    EXPLICIT_MODE_TOOLS,
)
from vibeproteinagent.agents.planning.fixup import (
    apply_enum_fixup_choice,
    missing_explicit_mode,
    missing_canonical_input_slot,
    prior_round_referenced_without_context,
    upstream_does_not_produce_candidates,
)
from vibeproteinagent.agents.planning.ir import validate_planner_ir_v1
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.descriptor import Workflow

VIRTUAL_CANONICAL_PRODUCERS = {"prior_round"}


def compile_planner_ir(
    llm_ir: dict[str, Any],
    *,
    prior_round_context: Any | None = None,
) -> dict[str, Any]:
    """
    Compile Planner IR v1 into the existing executor agent-plan shape.

    PR #4 only wires canonical candidate producers into canonical-aware
    consumers. Planner prompts and default runtime behavior remain unchanged.
    """

    validate_planner_ir_v1(llm_ir)
    _reject_prior_round_without_context(
        llm_ir,
        prior_round_context=prior_round_context,
    )
    ir_nodes = llm_ir.get("nodes") or []
    node_by_id = {node["id"]: node for node in ir_nodes}
    compiled_nodes: list[dict[str, Any]] = []

    for node in ir_nodes:
        compiled_nodes.extend(_compile_node(node, node_by_id))

    plan = {
        "mode": "agent_plan",
        "source_kind": "planner_ir_compiled",
        "goal": llm_ir.get("goal", ""),
        "assumptions": deepcopy(llm_ir.get("assumptions", [])),
        "missing_inputs": deepcopy(llm_ir.get("missing_inputs", [])),
        "required_tools": _required_tools(compiled_nodes),
        "workflow_plan": {
            "name": "compiled_planner_ir_v1",
            "version": "0.1.0",
            "nodes": compiled_nodes,
        },
        "success_criteria": deepcopy(
            (llm_ir.get("finalization") or {}).get("success_criteria", {})
        ),
    }
    _validate_executor_ir_shape(plan)
    return plan


def apply_fixup_choice(
    llm_ir: dict[str, Any],
    fixup_payload: dict[str, Any],
    choice: dict[str, Any],
) -> dict[str, Any]:
    return apply_enum_fixup_choice(llm_ir, fixup_payload, choice)


def _reject_prior_round_without_context(
    llm_ir: dict[str, Any],
    *,
    prior_round_context: Any | None,
) -> None:
    if prior_round_context is not None:
        return
    for node in llm_ir.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        for upstream in node.get("upstream") or []:
            if isinstance(upstream, dict) and upstream.get("node") == "prior_round":
                raise prior_round_referenced_without_context(
                    consumer_node_id=str(node.get("id") or ""),
                    upstream_node="prior_round",
                )


def _compile_node(
    node: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    node_id = node["id"]
    tool = node["tool"]
    inputs = deepcopy(node.get("inputs") or {})
    upstreams = list(node.get("upstream") or [])

    _validate_explicit_mode(node_id=node_id, tool=tool, inputs=inputs)

    if not upstreams:
        return [_executor_node(node_id, tool, inputs)]

    scalar_slots = CANONICAL_SCALAR_CONSUMERS.get(tool)
    if scalar_slots:
        _validate_canonical_upstreams(
            node_id=node_id,
            tool=tool,
            upstreams=upstreams,
            node_by_id=node_by_id,
        )
        source_ref, merge_node = _candidate_source_ref(node_id=node_id, upstreams=upstreams)
        for input_field, candidate_field in scalar_slots.items():
            inputs[input_field] = f"${{item.{candidate_field}}}"
        consumer = _executor_node(node_id, tool, inputs)
        consumer["foreach"] = source_ref
        return ([merge_node] if merge_node is not None else []) + [consumer]

    slot = CANONICAL_CONSUMERS.get(tool)
    if slot is None:
        raise missing_canonical_input_slot(
            node_id=node_id,
            tool=tool,
            allowed_tools=sorted(CANONICAL_CONSUMERS),
        )

    _validate_canonical_upstreams(
        node_id=node_id,
        tool=tool,
        upstreams=upstreams,
        node_by_id=node_by_id,
    )
    source_ref, merge_node = _candidate_source_ref(node_id=node_id, upstreams=upstreams)
    inputs[slot] = source_ref
    return ([merge_node] if merge_node is not None else []) + [
        _executor_node(node_id, tool, inputs)
    ]


def _validate_explicit_mode(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
) -> None:
    allowed_modes = EXPLICIT_MODE_TOOLS.get(tool)
    if allowed_modes is None:
        return
    received_mode = inputs.get("mode")
    if received_mode is None or received_mode not in allowed_modes:
        raise missing_explicit_mode(
            node_id=node_id,
            tool=tool,
            allowed_modes=allowed_modes,
            received_mode=received_mode,
        )


def _validate_canonical_upstreams(
    *,
    node_id: str,
    tool: str,
    upstreams: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> None:
    for upstream in upstreams:
        upstream_id = upstream["node"]
        if upstream_id in VIRTUAL_CANONICAL_PRODUCERS:
            upstream_tool = upstream_id
        else:
            upstream_tool = node_by_id[upstream_id]["tool"]
        if upstream_tool not in CANONICAL_PRODUCERS:
            raise upstream_does_not_produce_candidates(
                node_id=node_id,
                tool=tool,
                upstream_node=upstream_id,
                upstream_tool=upstream_tool,
            )


def _candidate_source_ref(
    *,
    node_id: str,
    upstreams: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    if len(upstreams) == 1:
        upstream_id = upstreams[0]["node"]
        return f"${{outputs.{upstream_id}.candidates}}", None

    merge_node_id = f"merge_{node_id}_candidates"
    merge_node = _executor_node(
        merge_node_id,
        "merge_candidates",
        {
            "candidate_sets": [
                f"${{outputs.{upstream['node']}.candidates}}" for upstream in upstreams
            ],
            "dedup_policy": "keep_all",
        },
    )
    return f"${{outputs.{merge_node_id}.candidates}}", merge_node


def _executor_node(node_id: str, tool: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {"id": node_id, "tool": tool, "inputs": inputs}


def _required_tools(nodes: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    required: list[str] = []
    for node in nodes:
        tool = node["tool"]
        if tool not in seen:
            seen.add(tool)
            required.append(tool)
    return required


def _validate_executor_ir_shape(plan: dict[str, Any]) -> None:
    try:
        Workflow.model_validate(plan["workflow_plan"]).validate_dag()
    except Exception as exc:
        raise InvalidParamsError(
            "compiled Planner IR did not produce a valid executor workflow shape",
            details={"error": str(exc)},
        ) from exc
