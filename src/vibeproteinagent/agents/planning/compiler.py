from __future__ import annotations

from copy import deepcopy
from typing import Any

from vibeproteinagent.agents.planning.canonical_tools import (
    CANONICAL_CONSUMERS,
    CANONICAL_PRODUCERS,
    CANONICAL_SCALAR_CONSUMERS,
    CANONICAL_TERM_CONSUMERS,
    CANONICAL_TERM_PRODUCERS,
    EXPLICIT_MODE_TOOLS,
    candidate_contract,
    candidate_contract_for_inputs,
    explicit_mode_input_field,
)
from vibeproteinagent.agents.planning.fixup import (
    apply_enum_fixup_choice,
    candidate_capability_mismatch,
    missing_explicit_mode,
    missing_canonical_input_slot,
    prior_round_referenced_without_context,
    upstream_does_not_produce_candidates,
)
from vibeproteinagent.agents.planning.ir import validate_planner_ir_v1
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.descriptor import Workflow

VIRTUAL_CANONICAL_PRODUCERS = {"prior_round"}
PLANNER_COMPILED_SOURCE_KIND = "planner_compiled"


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
    prior_round_capabilities = _prior_round_candidate_capabilities(prior_round_context)
    compiled_nodes: list[dict[str, Any]] = []

    for node in ir_nodes:
        compiled_nodes.extend(
            _compile_node(
                node,
                node_by_id,
                prior_round_capabilities=prior_round_capabilities,
            )
        )

    plan = {
        "mode": "agent_plan",
        "source_kind": PLANNER_COMPILED_SOURCE_KIND,
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
    validate_compiled_planner_plan(plan)
    return plan


def validate_compiled_planner_plan(plan: dict[str, Any]) -> None:
    """Validate the deterministic compiler boundary output."""

    if plan.get("mode") != "agent_plan":
        raise InvalidParamsError(
            "compiled planner plan must use agent_plan mode",
            details={"mode": plan.get("mode")},
        )
    if plan.get("source_kind") != PLANNER_COMPILED_SOURCE_KIND:
        raise InvalidParamsError(
            "compiled planner plan has an invalid source_kind",
            details={
                "source_kind": plan.get("source_kind"),
                "expected": PLANNER_COMPILED_SOURCE_KIND,
            },
        )
    workflow_plan = plan.get("workflow_plan")
    if not isinstance(workflow_plan, dict):
        raise InvalidParamsError(
            "compiled planner plan missing workflow_plan object",
            details={"field": "workflow_plan"},
        )
    try:
        workflow = Workflow.model_validate(workflow_plan)
        workflow.validate_dag()
    except Exception as exc:
        raise InvalidParamsError(
            "compiled planner plan did not produce a valid workflow DAG",
            details={"error": str(exc)},
        ) from exc
    forbidden = [
        node.tool
        for node in workflow.nodes
        if node.tool == "submit_candidate"
    ]
    if forbidden:
        raise InvalidParamsError(
            "compiled planner plan must not contain decision-time submission tools",
            details={"tools": forbidden},
        )


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
    *,
    prior_round_capabilities: set[str] | None,
) -> list[dict[str, Any]]:
    node_id = node["id"]
    tool = node["tool"]
    inputs = deepcopy(node.get("inputs") or {})
    upstreams = list(node.get("upstream") or [])

    _validate_explicit_mode(node_id=node_id, tool=tool, inputs=inputs)

    if not upstreams:
        return [_executor_node(node_id, tool, inputs)]

    term_upstreams = [
        upstream for upstream in upstreams
        if upstream.get("relationship") in {"term_source", "identifier_source"}
    ]
    candidate_upstreams = [
        upstream for upstream in upstreams
        if upstream.get("relationship") == "candidate_source"
    ]
    if term_upstreams:
        _wire_term_upstreams(
            node_id=node_id,
            tool=tool,
            inputs=inputs,
            upstreams=term_upstreams,
            node_by_id=node_by_id,
        )
        upstreams = candidate_upstreams
        if not upstreams:
            return [_executor_node(node_id, tool, inputs)]

    scalar_slots = CANONICAL_SCALAR_CONSUMERS.get(tool)
    if scalar_slots:
        _validate_canonical_upstreams(
            node_id=node_id,
            tool=tool,
            inputs=inputs,
            upstreams=upstreams,
            node_by_id=node_by_id,
            prior_round_capabilities=prior_round_capabilities,
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
        inputs=inputs,
        upstreams=upstreams,
        node_by_id=node_by_id,
        prior_round_capabilities=prior_round_capabilities,
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
    mode_field = explicit_mode_input_field(tool)
    if mode_field != "mode" and "mode" in inputs and mode_field not in inputs:
        inputs[mode_field] = inputs.pop("mode")
    received_mode = inputs.get(mode_field)
    if received_mode is None or received_mode not in allowed_modes:
        raise missing_explicit_mode(
            node_id=node_id,
            tool=tool,
            allowed_modes=allowed_modes,
            received_mode=received_mode,
            mode_field=mode_field,
        )


def _validate_canonical_upstreams(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    upstreams: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    prior_round_capabilities: set[str] | None,
) -> None:
    _validate_candidate_item_requirement(
        node_id=node_id,
        tool=tool,
        inputs=inputs,
        upstreams=upstreams,
        node_by_id=node_by_id,
        prior_round_capabilities=prior_round_capabilities,
    )
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
        missing = _missing_candidate_capabilities_for_node(
            producer_node=upstream_id,
            producer_tool=upstream_tool,
            consumer_tool=tool,
            consumer_inputs=inputs,
            node_by_id=node_by_id,
            prior_round_capabilities=prior_round_capabilities,
        )
        if missing:
            raise candidate_capability_mismatch(
                node_id=node_id,
                tool=tool,
                upstream_node=upstream_id,
                upstream_tool=upstream_tool,
                missing_capabilities=missing,
            )


def _validate_candidate_item_requirement(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    upstreams: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    prior_round_capabilities: set[str] | None,
) -> None:
    required_items = candidate_contract_for_inputs(tool, inputs).min_candidate_items
    if required_items <= 1 or len(upstreams) >= required_items:
        return
    upstream_id = upstreams[0]["node"] if upstreams else ""
    upstream_tool = (
        upstream_id
        if upstream_id in VIRTUAL_CANONICAL_PRODUCERS
        else str((node_by_id.get(upstream_id) or {}).get("tool") or "")
    )
    if upstream_tool == "merge_candidates":
        return
    raise candidate_capability_mismatch(
        node_id=node_id,
        tool=tool,
        upstream_node=upstream_id,
        upstream_tool=upstream_tool,
        missing_capabilities=[f"at_least_{required_items}_candidate_items"],
    )


def _missing_candidate_capabilities_for_node(
    *,
    producer_node: str,
    producer_tool: str,
    consumer_tool: str,
    consumer_inputs: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    prior_round_capabilities: set[str] | None,
) -> list[str]:
    required = candidate_contract_for_inputs(consumer_tool, consumer_inputs).consumes
    if not required:
        return []
    produced = _node_candidate_capabilities(
        producer_node,
        producer_tool=producer_tool,
        node_by_id=node_by_id,
        prior_round_capabilities=prior_round_capabilities,
        seen=set(),
    )
    return sorted(required - produced)


def _node_candidate_capabilities(
    node_id: str,
    *,
    producer_tool: str | None = None,
    node_by_id: dict[str, dict[str, Any]],
    prior_round_capabilities: set[str] | None,
    seen: set[str],
) -> set[str]:
    if node_id in VIRTUAL_CANONICAL_PRODUCERS:
        if node_id == "prior_round" and prior_round_capabilities is not None:
            return set(prior_round_capabilities)
        return set(candidate_contract(node_id).produces)
    if node_id in seen:
        return set()
    seen.add(node_id)
    node = node_by_id.get(node_id)
    if node is None:
        return set()
    tool = producer_tool or str(node.get("tool") or "")
    inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
    contract = candidate_contract_for_inputs(tool, inputs)
    capabilities = set(contract.produces)
    if contract.preserves_upstream:
        for upstream in node.get("upstream") or []:
            if not isinstance(upstream, dict):
                continue
            if upstream.get("relationship") != "candidate_source":
                continue
            upstream_id = str(upstream.get("node") or "")
            upstream_node = node_by_id.get(upstream_id)
            upstream_tool = (
                upstream_id
                if upstream_id in VIRTUAL_CANONICAL_PRODUCERS
                else str((upstream_node or {}).get("tool") or "")
            )
            capabilities.update(
                _node_candidate_capabilities(
                    upstream_id,
                    producer_tool=upstream_tool,
                    node_by_id=node_by_id,
                    prior_round_capabilities=prior_round_capabilities,
                    seen=seen,
                )
            )
    return capabilities


def _prior_round_candidate_capabilities(prior_round_context: Any | None) -> set[str] | None:
    if prior_round_context is None:
        return None
    candidates = getattr(prior_round_context, "surviving_candidates", None)
    if isinstance(prior_round_context, dict):
        candidates = prior_round_context.get("surviving_candidates")
    if not isinstance(candidates, list):
        return set()
    capabilities: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_caps = candidate.get("capabilities")
        if isinstance(candidate_caps, dict):
            capabilities.update(
                str(name)
                for name, value in candidate_caps.items()
                if name.startswith("has_") and value is True
            )
        if candidate.get("sequence"):
            capabilities.add("has_sequence")
        if candidate.get("structure_path") or candidate.get("backbone_path"):
            capabilities.add("has_backbone_structure")
        if candidate.get("full_structure_path") or (
            isinstance(candidate_caps, dict)
            and candidate_caps.get("has_full_structure") is True
        ):
            capabilities.add("has_full_structure")
        scores = candidate.get("scores")
        if isinstance(scores, dict) and scores:
            capabilities.add("has_candidate_scores")
    return capabilities


def _wire_term_upstreams(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    upstreams: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> None:
    input_field = CANONICAL_TERM_CONSUMERS.get(tool)
    if input_field is None:
        raise missing_canonical_input_slot(
            node_id=node_id,
            tool=tool,
            allowed_tools=sorted(CANONICAL_TERM_CONSUMERS),
        )
    if len(upstreams) != 1:
        raise InvalidParamsError(
            "term_source wiring expects exactly one upstream resolver",
            details={
                "node_id": node_id,
                "tool": tool,
                "upstreams": upstreams,
            },
        )
    upstream_id = upstreams[0]["node"]
    upstream_tool = node_by_id[upstream_id]["tool"]
    output_paths = CANONICAL_TERM_PRODUCERS.get(upstream_tool)
    if output_paths is None:
        raise upstream_does_not_produce_candidates(
            node_id=node_id,
            tool=tool,
            upstream_node=upstream_id,
            upstream_tool=upstream_tool,
        )
    output_path = output_paths.get(input_field)
    if output_path is None:
        raise InvalidParamsError(
            "resolver output cannot satisfy requested term input",
            details={
                "node_id": node_id,
                "tool": tool,
                "input_field": input_field,
                "upstream_node": upstream_id,
                "upstream_tool": upstream_tool,
                "available_fields": sorted(output_paths),
            },
        )
    inputs[input_field] = f"${{outputs.{upstream_id}.{output_path}}}"


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
