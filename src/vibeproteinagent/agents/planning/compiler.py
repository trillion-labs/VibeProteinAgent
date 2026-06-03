from __future__ import annotations

from copy import deepcopy
from typing import Any

from vibeproteinagent.agents.planning.canonical_tools import (
    CANONICAL_CONSUMERS,
    CANONICAL_PROFILE_CONSUMERS,
    CANONICAL_PROFILE_PRODUCERS,
    CANONICAL_PRODUCERS,
    CANONICAL_SCALAR_CONSUMERS,
    CANONICAL_TERM_CONSUMERS,
    CANONICAL_TERM_PRODUCERS,
    EXPLICIT_MODE_TOOLS,
    candidate_contract,
    candidate_contract_for_node,
    candidate_contract_for_inputs,
    explicit_mode_input_field,
    profile_guided_default_inputs,
)
from vibeproteinagent.agents.planning.fixup import (
    apply_enum_fixup_choice,
    candidate_capability_mismatch,
    missing_explicit_mode,
    missing_canonical_input_slot,
    missing_profile_input_slot,
    node_reference_used_as_literal_path,
    planner_literal_candidate_stream_reference,
    planner_literal_profile_reference,
    prior_round_used_as_tool,
    prior_round_referenced_without_context,
    upstream_does_not_produce_candidates,
    upstream_does_not_produce_profile,
)
from vibeproteinagent.agents.planning.ir import validate_planner_ir_v1
from vibeproteinagent.agents.planning.parameter_ownership import policy_for
from vibeproteinagent.agents.planning.terminal_tools import TERMINAL_CONTROLLER_TOOLS
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.descriptor import Workflow

VIRTUAL_CANONICAL_PRODUCERS = {"prior_round"}
PLANNER_COMPILED_SOURCE_KIND = "planner_compiled"
_PATH_FIELD_NAMES = frozenset(
    {
        "pdb_path",
        "target_pdb",
        "template_pdb",
        "structure_path",
        "cif_path",
        "predicted_cif",
        "reference_cif",
        "target_structure",
        "input_pdb",
        "input_cif",
        "input_json",
    }
)
_AUX_CANDIDATE_STREAM_FIELDS_BY_TOOL = {
    "rank_designs": frozenset({"predictions", "analyses", "qc_results"}),
}
PXDESIGN_FULL_STRUCTURE_TAIL_TOOLS: tuple[str, ...] = (
    "protenix_v2",
    "protein_qc",
    "rank_designs",
)


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

    _reject_virtual_tool_nodes(llm_ir)
    llm_ir = _pre_normalize_candidate_stream_executor_refs(llm_ir)
    validate_planner_ir_v1(llm_ir)
    _reject_prior_round_without_context(
        llm_ir,
        prior_round_context=prior_round_context,
    )
    llm_ir = normalize_pxdesign_full_structure_ir(llm_ir)
    validate_planner_ir_v1(llm_ir)
    ir_nodes = llm_ir.get("nodes") or []
    node_by_id = {node["id"]: node for node in ir_nodes}
    _reject_node_ids_used_as_literal_paths(llm_ir, node_by_id=node_by_id)
    prior_round_capabilities = _prior_round_candidate_capabilities(prior_round_context)
    compiled_nodes: list[dict[str, Any]] = []

    for node in ir_nodes:
        compiled_nodes.extend(
            _compile_node(
                node,
                node_by_id,
                prior_round_capabilities=prior_round_capabilities,
                prior_round_context_available=prior_round_context is not None,
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


def _pre_normalize_candidate_stream_executor_refs(llm_ir: dict[str, Any]) -> dict[str, Any]:
    """Convert tolerated candidate-stream executor refs before IR validation.

    Planner IR should express topology through upstream entries. This narrow
    pre-pass exists only so LLM-authored `${outputs.<node>.candidates}` in a
    canonical candidate-stream field can reach the compiler's structured
    candidate-source normalization instead of failing as generic syntax.
    """
    updated = deepcopy(llm_ir)
    for node in updated.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        tool = str(node.get("tool") or "")
        main_stream_field = CANONICAL_CONSUMERS.get(tool)
        if main_stream_field is None:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        provenance = node.get("input_provenance")
        if (
            isinstance(provenance, dict)
            and provenance.get(main_stream_field, "planner_literal") != "planner_literal"
        ):
            continue
        value = inputs.get(main_stream_field)
        if not isinstance(value, str):
            continue
        target = _candidate_stream_ref_target(value)
        if target is not None and value.strip().startswith("${outputs."):
            inputs[main_stream_field] = target
    return updated


def normalize_pxdesign_full_structure_ir(llm_ir: dict[str, Any]) -> dict[str, Any]:
    """Compile PXDesign intent into the canonical full-structure binder workflow."""

    nodes = llm_ir.get("nodes")
    if not isinstance(nodes, list) or not any(
        isinstance(node, dict) and node.get("tool") == "pxdesign" for node in nodes
    ):
        return llm_ir

    expanded = deepcopy(llm_ir)
    node_list = expanded.get("nodes")
    if not isinstance(node_list, list):
        return expanded
    existing_ids = {
        str(node.get("id"))
        for node in node_list
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    chain_by_pxdesign: dict[str, dict[str, str]] = {}
    local_tail_nodes: dict[str, dict[str, Any]] = {}
    local_tail_ids: set[str] = set()
    generated_rank_ids: set[str] = set()

    for node in node_list:
        if not isinstance(node, dict) or node.get("tool") != "pxdesign":
            continue
        px_id = str(node.get("id") or "")
        if not px_id:
            continue
        protenix = _existing_single_candidate_consumer(
            node_list,
            upstream_id=px_id,
            tool="protenix_v2",
        )
        protenix_id = (
            str(protenix.get("id"))
            if isinstance(protenix, dict) and isinstance(protenix.get("id"), str)
            else _unique_node_id(f"{px_id}_protenix", existing_ids)
        )
        existing_ids.add(protenix_id)

        qc = _existing_single_candidate_consumer(
            node_list,
            upstream_id=protenix_id,
            tool="protein_qc",
        ) or _existing_single_candidate_consumer(
            node_list,
            upstream_id=px_id,
            tool="protein_qc",
        )
        qc_id = (
            str(qc.get("id"))
            if isinstance(qc, dict) and isinstance(qc.get("id"), str)
            else _unique_node_id(f"{px_id}_qc", existing_ids)
        )
        existing_ids.add(qc_id)

        chain_by_pxdesign[px_id] = {
            "protenix": protenix_id,
            "qc": qc_id,
        }
        reused_nodes = {
            str(reused.get("id")): reused
            for reused in (protenix, qc)
            if isinstance(reused, dict) and isinstance(reused.get("id"), str)
        }
        local_tail_nodes[protenix_id] = _tail_node(
            reused_nodes.get(protenix_id),
            node_id=protenix_id,
            tool="protenix_v2",
            intent="Predict full protein target-binder structures for PXDesign candidates.",
            upstream_id=px_id,
        )
        local_tail_nodes[qc_id] = _tail_node(
            reused_nodes.get(qc_id),
            node_id=qc_id,
            tool="protein_qc",
            intent="Evaluate full-structure binder candidates before ranking.",
            upstream_id=protenix_id,
        )
        local_tail_ids.update({protenix_id, qc_id})

    if not chain_by_pxdesign:
        return expanded

    source_to_px_ids: dict[str, tuple[str, ...]] = {}
    replacement_qc_ids: dict[str, list[str]] = {}
    for px_id, chain in chain_by_pxdesign.items():
        px_tuple = (px_id,)
        source_to_px_ids[px_id] = px_tuple
        source_to_px_ids[chain["protenix"]] = px_tuple
        source_to_px_ids[chain["qc"]] = px_tuple
        replacement_qc_ids[px_id] = [chain["qc"]]
        replacement_qc_ids[chain["protenix"]] = [chain["qc"]]

    obsolete_intermediate_ids: set[str] = set()
    for raw_node in node_list:
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node.get("id") or "")
        if node_id in local_tail_ids:
            continue
        if raw_node.get("tool") not in {"protenix_v2", "protein_qc"}:
            continue
        represented_px_ids = _represented_px_ids_for_node(
            raw_node,
            source_to_px_ids=source_to_px_ids,
        )
        if not represented_px_ids:
            continue
        obsolete_intermediate_ids.add(node_id)
        source_to_px_ids[node_id] = tuple(represented_px_ids)
        replacement_qc_ids[node_id] = [
            chain_by_pxdesign[px_id]["qc"] for px_id in represented_px_ids
        ]

    px_ids_with_rank_downstream = _pxdesign_ids_with_rank_downstream(
        node_list,
        source_to_px_ids=source_to_px_ids,
        skip_ids=local_tail_ids | obsolete_intermediate_ids,
    )
    node_by_id = {
        str(node.get("id")): node
        for node in node_list
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }
    for px_id, chain in chain_by_pxdesign.items():
        if px_id in px_ids_with_rank_downstream:
            continue
        rank_id = _unique_node_id(f"{px_id}_rank", existing_ids)
        existing_ids.add(rank_id)
        generated_rank_ids.add(rank_id)
        local_tail_nodes[rank_id] = _pxdesign_rank_tail_node(
            pxdesign_node=node_by_id.get(px_id, {}),
            rank_id=rank_id,
            upstream_id=chain["qc"],
        )
        local_tail_ids.add(rank_id)
        chain["rank"] = rank_id

    rebuilt: list[dict[str, Any]] = []
    emitted_tail_ids: set[str] = set()
    for raw_node in node_list:
        if not isinstance(raw_node, dict):
            continue
        node_id = str(raw_node.get("id") or "")
        if node_id in local_tail_ids or node_id in obsolete_intermediate_ids:
            continue
        node = _rewire_pxdesign_candidate_upstreams(
            raw_node,
            replacement_qc_ids=replacement_qc_ids,
        )
        rebuilt.append(node)
        if node.get("tool") != "pxdesign" or node_id not in chain_by_pxdesign:
            continue
        chain = chain_by_pxdesign[node_id]
        for tail_id in (chain["protenix"], chain["qc"], chain.get("rank")):
            if not tail_id:
                continue
            if tail_id in emitted_tail_ids or tail_id not in local_tail_nodes:
                continue
            if tail_id == chain.get("rank") and tail_id not in generated_rank_ids:
                continue
            rebuilt.append(deepcopy(local_tail_nodes[tail_id]))
            emitted_tail_ids.add(tail_id)

    expanded["nodes"] = rebuilt
    return expanded


def _existing_single_candidate_consumer(
    nodes: list[Any],
    *,
    upstream_id: str,
    tool: str,
) -> dict[str, Any] | None:
    for node in nodes:
        if not isinstance(node, dict) or node.get("tool") != tool:
            continue
        upstream_ids = _candidate_upstream_ids(node)
        if upstream_ids == [upstream_id]:
            return node
    return None


def _candidate_upstream_ids(node: dict[str, Any]) -> list[str]:
    return [
        str(upstream.get("node"))
        for upstream in node.get("upstream") or []
        if isinstance(upstream, dict)
        and upstream.get("relationship") == "candidate_source"
        and isinstance(upstream.get("node"), str)
    ]


def _represented_px_ids_for_node(
    node: dict[str, Any],
    *,
    source_to_px_ids: dict[str, tuple[str, ...]],
) -> list[str]:
    px_ids: list[str] = []
    upstream_ids = _candidate_upstream_ids(node)
    if not upstream_ids:
        return []
    for upstream_id in upstream_ids:
        represented = source_to_px_ids.get(upstream_id)
        if represented is None:
            return []
        px_ids.extend(represented)
    return _dedupe_preserve_order(px_ids)


def _pxdesign_ids_with_rank_downstream(
    nodes: list[Any],
    *,
    source_to_px_ids: dict[str, tuple[str, ...]],
    skip_ids: set[str],
) -> set[str]:
    out: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or node.get("tool") != "rank_designs":
            continue
        node_id = str(node.get("id") or "")
        if node_id in skip_ids:
            continue
        out.update(
            _represented_px_ids_for_node(
                node,
                source_to_px_ids=source_to_px_ids,
            )
        )
    return out


def _rewire_pxdesign_candidate_upstreams(
    node: dict[str, Any],
    *,
    replacement_qc_ids: dict[str, list[str]],
) -> dict[str, Any]:
    updated = deepcopy(node)
    upstreams = updated.get("upstream")
    if not isinstance(upstreams, list):
        return updated
    rewritten: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for upstream in upstreams:
        if not isinstance(upstream, dict):
            rewritten.append(upstream)
            continue
        if upstream.get("relationship") != "candidate_source":
            rewritten.append(upstream)
            continue
        source = upstream.get("node")
        replacements = (
            replacement_qc_ids.get(str(source))
            if isinstance(source, str)
            else None
        )
        if not replacements:
            key = (str(source), "candidate_source")
            if key not in seen:
                rewritten.append(upstream)
                seen.add(key)
            continue
        for replacement in replacements:
            key = (replacement, "candidate_source")
            if key in seen:
                continue
            rewritten.append({"node": replacement, "relationship": "candidate_source"})
            seen.add(key)
    updated["upstream"] = rewritten
    return updated


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _pxdesign_rank_tail_node(
    *,
    pxdesign_node: dict[str, Any],
    rank_id: str,
    upstream_id: str,
) -> dict[str, Any]:
    rank_inputs: dict[str, Any] = {}
    top_k = _positive_int_or_none(
        (pxdesign_node.get("inputs") if isinstance(pxdesign_node.get("inputs"), dict) else {}).get(
            "top_k_designs"
        )
    )
    if top_k is not None:
        rank_inputs["top_k"] = top_k
    return _tail_node(
        None,
        node_id=rank_id,
        tool="rank_designs",
        intent="Rank full-structure binder candidates for final selection.",
        upstream_id=upstream_id,
        default_inputs=rank_inputs,
    )


def _tail_node(
    existing: dict[str, Any] | None,
    *,
    node_id: str,
    tool: str,
    intent: str,
    upstream_id: str,
    default_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node = deepcopy(existing) if isinstance(existing, dict) else {}
    node["id"] = node_id
    node["tool"] = tool
    node.setdefault("intent", intent)
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        inputs = {}
    for key, value in (default_inputs or {}).items():
        inputs.setdefault(key, value)
    node["inputs"] = inputs
    node["upstream"] = [{"node": upstream_id, "relationship": "candidate_source"}]
    return node


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not stripped.isdigit():
            return None
        parsed = int(stripped)
        return parsed if parsed > 0 else None
    return None


def _unique_node_id(base: str, existing_ids: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


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
        if node.tool in TERMINAL_CONTROLLER_TOOLS
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


def _reject_virtual_tool_nodes(llm_ir: dict[str, Any]) -> None:
    for node in llm_ir.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        tool = str(node.get("tool") or "")
        if node_id in VIRTUAL_CANONICAL_PRODUCERS or tool in VIRTUAL_CANONICAL_PRODUCERS:
            raise prior_round_used_as_tool(node_id=node_id or tool)


def _reject_node_ids_used_as_literal_paths(
    llm_ir: dict[str, Any],
    *,
    node_by_id: dict[str, dict[str, Any]],
) -> None:
    reference_ids = set(node_by_id) | set(VIRTUAL_CANONICAL_PRODUCERS)
    if not reference_ids:
        return
    for node in llm_ir.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        upstream_ids = {
            str(upstream.get("node") or "")
            for upstream in node.get("upstream") or []
            if isinstance(upstream, dict)
        }
        for path, value in _iter_path_like_literals(inputs):
            if value not in reference_ids:
                continue
            if _candidate_upstream_derived_source_artifact(
                tool=str(node.get("tool") or ""),
                field=path,
                node=node,
            ):
                continue
            raise node_reference_used_as_literal_path(
                node_id=str(node.get("id") or ""),
                tool=str(node.get("tool") or ""),
                field=path,
                literal_value=value,
                upstream_node=value if value in upstream_ids else value,
            )


def _iter_path_like_literals(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(value, dict):
        return out
    for key, child in value.items():
        field = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            out.extend(_iter_path_like_literals(child, field))
            continue
        if isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, dict):
                    out.extend(_iter_path_like_literals(item, f"{field}[{index}]"))
                elif isinstance(item, str) and _is_path_like_field(str(key)):
                    out.append((f"{field}[{index}]", item))
            continue
        if isinstance(child, str) and _is_path_like_field(str(key)):
            out.append((field, child))
    return out


def _is_path_like_field(field: str) -> bool:
    lowered = field.lower()
    return lowered in _PATH_FIELD_NAMES or lowered.endswith("_path") or lowered.endswith("_file")


def _candidate_upstream_derived_source_artifact(
    *,
    tool: str,
    field: str,
    node: dict[str, Any],
) -> bool:
    if not any(
        isinstance(upstream, dict)
        and upstream.get("relationship") == "candidate_source"
        for upstream in node.get("upstream") or []
    ):
        return False
    top_level_field = field.split(".", 1)[0].split("[", 1)[0]
    policy = policy_for(tool, top_level_field)
    if policy.ownership != "source_artifact" or not policy.derive_when_upstream:
        return False
    provenance = node.get("input_provenance")
    if not isinstance(provenance, dict):
        return True
    return provenance.get(top_level_field, "planner_literal") == "planner_literal"


def _compile_node(
    node: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
    *,
    prior_round_capabilities: set[str] | None,
    prior_round_context_available: bool,
) -> list[dict[str, Any]]:
    node_id = node["id"]
    tool = node["tool"]
    inputs = deepcopy(node.get("inputs") or {})
    provenance = _planner_input_provenance(node, inputs)
    upstreams = list(node.get("upstream") or [])

    _sync_provenance_fields(provenance, inputs, default_source="planner_literal")
    _reject_planner_literal_profile_inputs(
        node_id=node_id,
        tool=tool,
        inputs=inputs,
        provenance=provenance,
    )
    _apply_profile_guided_default_inputs(tool, inputs, provenance, upstreams)
    _validate_explicit_mode(node_id=node_id, tool=tool, inputs=inputs)
    _normalize_planner_literal_candidate_stream_refs(
        node_id=node_id,
        tool=tool,
        inputs=inputs,
        provenance=provenance,
        upstreams=upstreams,
        node_by_id=node_by_id,
        prior_round_context_available=prior_round_context_available,
    )
    _drop_empty_planner_literal_aux_candidate_streams(tool, inputs, provenance)

    if not upstreams:
        return [_executor_node(node_id, tool, inputs, provenance)]

    term_upstreams = [
        upstream for upstream in upstreams
        if upstream.get("relationship") in {"term_source", "identifier_source"}
    ]
    candidate_upstreams = [
        upstream for upstream in upstreams
        if upstream.get("relationship") == "candidate_source"
    ]
    profile_upstreams = [
        upstream for upstream in upstreams
        if upstream.get("relationship") == "profile_source"
    ]
    if candidate_upstreams:
        _drop_planner_literal_derived_source_artifacts(tool, inputs, provenance)
    if term_upstreams:
        _wire_term_upstreams(
            node_id=node_id,
            tool=tool,
            inputs=inputs,
            upstreams=term_upstreams,
            node_by_id=node_by_id,
        )
        for field in inputs:
            if _contains_executor_ref(inputs[field]):
                provenance[field] = "compiled_wiring"
            else:
                provenance.setdefault(field, "planner_literal")
    if profile_upstreams:
        _wire_profile_upstreams(
            node_id=node_id,
            tool=tool,
            inputs=inputs,
            provenance=provenance,
            upstreams=profile_upstreams,
            node_by_id=node_by_id,
        )
    upstreams = candidate_upstreams
    if not upstreams:
        if term_upstreams or profile_upstreams:
            return [_executor_node(node_id, tool, inputs, provenance)]

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
            provenance[input_field] = "compiled_wiring"
        consumer = _executor_node(node_id, tool, inputs, provenance)
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
    provenance[slot] = "compiled_wiring"
    return ([merge_node] if merge_node is not None else []) + [
        _executor_node(node_id, tool, inputs, provenance)
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


def _apply_profile_guided_default_inputs(
    tool: str,
    inputs: dict[str, Any],
    provenance: dict[str, str],
    upstreams: list[dict[str, Any]],
) -> None:
    for field, value in profile_guided_default_inputs(tool, upstreams).items():
        if field in inputs:
            continue
        inputs[field] = value
        provenance[field] = "compiled_wiring"


def _drop_empty_planner_literal_aux_candidate_streams(
    tool: str,
    inputs: dict[str, Any],
    provenance: dict[str, str],
) -> None:
    fields = _AUX_CANDIDATE_STREAM_FIELDS_BY_TOOL.get(tool)
    if not fields:
        return
    for field in fields:
        if inputs.get(field) != []:
            continue
        if provenance.get(field, "planner_literal") != "planner_literal":
            continue
        inputs.pop(field, None)
        provenance.pop(field, None)


def _normalize_planner_literal_candidate_stream_refs(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    provenance: dict[str, str],
    upstreams: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    prior_round_context_available: bool,
) -> None:
    main_stream_field = CANONICAL_CONSUMERS.get(tool)
    if main_stream_field is None:
        return
    value = inputs.get(main_stream_field)
    if not isinstance(value, str):
        return
    if provenance.get(main_stream_field, "planner_literal") != "planner_literal":
        return
    target = _candidate_stream_ref_target(value)
    if target is None:
        raise planner_literal_candidate_stream_reference(
            node_id=node_id,
            tool=tool,
            field=main_stream_field,
            literal_value=value,
        )
    if target == "prior_round":
        if not prior_round_context_available:
            raise prior_round_referenced_without_context(
                consumer_node_id=node_id,
                upstream_node="prior_round",
            )
    elif target not in node_by_id:
        raise planner_literal_candidate_stream_reference(
            node_id=node_id,
            tool=tool,
            field=main_stream_field,
            literal_value=value,
        )
    _append_candidate_upstream(upstreams, target)
    inputs.pop(main_stream_field, None)
    provenance.pop(main_stream_field, None)


def _candidate_stream_ref_target(value: str) -> str | None:
    text = value.strip()
    if text == "prior_round":
        return "prior_round"
    prefix = "${outputs."
    suffix = ".candidates}"
    if text.startswith(prefix) and text.endswith(suffix):
        target = text[len(prefix):-len(suffix)]
        return target or None
    return text or None


def _append_candidate_upstream(upstreams: list[dict[str, Any]], target: str) -> None:
    for upstream in upstreams:
        if (
            isinstance(upstream, dict)
            and upstream.get("node") == target
            and upstream.get("relationship") == "candidate_source"
        ):
            return
    upstreams.append({"node": target, "relationship": "candidate_source"})


def _reject_planner_literal_profile_inputs(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    provenance: dict[str, str],
) -> None:
    for field, value in inputs.items():
        if provenance.get(field, "planner_literal") != "planner_literal":
            continue
        if policy_for(tool, field).ownership != "profile_evidence":
            continue
        raise planner_literal_profile_reference(
            node_id=node_id,
            tool=tool,
            field=field,
            literal_value=value,
        )


def _drop_planner_literal_derived_source_artifacts(
    tool: str,
    inputs: dict[str, Any],
    provenance: dict[str, str],
) -> None:
    """Let canonical upstream candidates supply runtime-owned source artifacts.

    When a node has candidate_source upstreams, fields such as
    protenix_v2.input_json or proteinmpnn.pdb_path are derived from the wired
    candidates by runtime/adapter code. Planner literals in those fields are
    not authoritative and should not reach executor ownership validation.
    """
    for field in list(inputs):
        policy = policy_for(tool, field)
        if policy.ownership != "source_artifact":
            continue
        if not policy.derive_when_upstream:
            continue
        if provenance.get(field, "planner_literal") != "planner_literal":
            continue
        inputs.pop(field, None)
        provenance.pop(field, None)


def _validate_canonical_upstreams(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    upstreams: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
    prior_round_capabilities: set[str] | None,
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
        upstream_node = node_by_id.get(upstream_id, {})
        upstream_inputs = (
            upstream_node.get("inputs") if isinstance(upstream_node.get("inputs"), dict) else {}
        )
        upstream_relationships = (
            upstream_node.get("upstream") if isinstance(upstream_node.get("upstream"), list) else []
        )
        upstream_contract = candidate_contract_for_node(
            upstream_tool,
            upstream_inputs,
            upstream_relationships,
        )
        if (
            upstream_tool == "family_profile"
            and not upstream_contract.produces
            and not upstream_contract.preserves_upstream
        ):
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
    contract = candidate_contract_for_node(
        tool,
        inputs,
        node.get("upstream") if isinstance(node.get("upstream"), list) else [],
    )
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
    prompt_context = getattr(prior_round_context, "prompt_context", None)
    if isinstance(prior_round_context, dict):
        prompt_context = (
            prior_round_context.get("prompt_context")
            or prior_round_context.get("digest")
        )
    return _prior_round_capabilities_from_prompt_context(prompt_context)


def _prior_round_capabilities_from_prompt_context(value: Any) -> set[str]:
    if not isinstance(value, dict):
        return set()
    latest = value.get("latest_round") if isinstance(value.get("latest_round"), dict) else value
    summary = latest.get("capability_summary") if isinstance(latest, dict) else None
    capabilities: set[str] = set()
    if isinstance(summary, dict):
        for section in ("survivor_pool", "survivor", "current_round"):
            section_value = summary.get(section)
            if not isinstance(section_value, dict):
                continue
            counts = section_value.get("capability_counts")
            if isinstance(counts, dict):
                capabilities.update(str(name) for name, count in counts.items() if count)
        direct = summary.get("capabilities")
        if isinstance(direct, dict):
            capabilities.update(str(name) for name, count in direct.items() if count)
    return capabilities


def _wire_profile_upstreams(
    *,
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    provenance: dict[str, str],
    upstreams: list[dict[str, Any]],
    node_by_id: dict[str, dict[str, Any]],
) -> None:
    input_field = CANONICAL_PROFILE_CONSUMERS.get(tool)
    if input_field is None:
        first_upstream = upstreams[0]["node"] if upstreams else ""
        raise missing_profile_input_slot(
            node_id=node_id,
            tool=tool,
            upstream_node=str(first_upstream),
            allowed_tools=sorted(CANONICAL_PROFILE_CONSUMERS),
        )
    if len(upstreams) != 1:
        raise InvalidParamsError(
            "profile_source wiring expects exactly one upstream profile producer",
            details={
                "node_id": node_id,
                "tool": tool,
                "upstreams": upstreams,
            },
        )
    upstream_id = str(upstreams[0]["node"])
    upstream_node = node_by_id.get(upstream_id)
    upstream_tool = (
        upstream_id
        if upstream_id in VIRTUAL_CANONICAL_PRODUCERS
        else str((upstream_node or {}).get("tool") or "")
    )
    output_field = CANONICAL_PROFILE_PRODUCERS.get(upstream_tool)
    if output_field is None:
        raise upstream_does_not_produce_profile(
            node_id=node_id,
            tool=tool,
            upstream_node=upstream_id,
            upstream_tool=upstream_tool,
        )
    inputs[input_field] = f"${{outputs.{upstream_id}.{output_field}}}"
    provenance[input_field] = "compiled_wiring"


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
    if upstream_tool == "function_term_resolver":
        list_field = {
            "go_sequences": "go_terms",
            "interpro_sequences": "interpro_terms",
        }.get(tool)
        if list_field is not None and output_paths.get(list_field) is not None:
            inputs[input_field] = None
            input_field = list_field
            output_path = output_paths[list_field]
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


def _planner_input_provenance(
    node: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, str]:
    raw = node.get("input_provenance")
    provenance = {
        str(field): str(source)
        for field, source in (raw.items() if isinstance(raw, dict) else [])
        if isinstance(field, str) and isinstance(source, str)
    }
    _sync_provenance_fields(provenance, inputs, default_source="planner_literal")
    return provenance


def _sync_provenance_fields(
    provenance: dict[str, str],
    inputs: dict[str, Any],
    *,
    default_source: str,
) -> None:
    for field in inputs:
        provenance.setdefault(str(field), default_source)
    for field in list(provenance):
        if field not in inputs:
            provenance.pop(field, None)


def _contains_executor_ref(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("${") and value.endswith("}")
    if isinstance(value, list):
        return any(_contains_executor_ref(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_executor_ref(item) for item in value.values())
    return False


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
        {
            "candidate_sets": "compiled_wiring",
            "dedup_policy": "runtime_derived",
        },
    )
    return f"${{outputs.{merge_node_id}.candidates}}", merge_node


def _executor_node(
    node_id: str,
    tool: str,
    inputs: dict[str, Any],
    input_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    node = {"id": node_id, "tool": tool, "inputs": inputs}
    provenance = {
        field: source
        for field, source in (input_provenance or {}).items()
        if field in inputs
    }
    if provenance:
        node["input_provenance"] = provenance
    return node


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
