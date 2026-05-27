from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

from vibeproteinagent.adapters import ToolRegistry, validate_params
from vibeproteinagent.adapters.availability import descriptor_availability
from vibeproteinagent.adapters.descriptor import (
    drop_optional_none_params,
    resolve_input_schema_dict,
    resolve_output_schema_dict,
)
from vibeproteinagent.adapters.visibility import is_agent_visible_tool
from vibeproteinagent.agents.planning.canonical_tools import (
    CANONICAL_PRODUCERS,
    CANONICAL_TERM_CONSUMERS,
    CANONICAL_TERM_PRODUCERS,
    EXPLICIT_MODE_TOOLS,
    build_canonical_wiring_for_planner,
    candidate_capability_profiles,
    candidate_contract,
    candidate_contract_for_inputs,
    explicit_mode_input_field,
)
from vibeproteinagent.agents.planning.compiler import (
    PLANNER_COMPILED_SOURCE_KIND,
    compile_planner_ir,
    validate_compiled_planner_plan,
)
from vibeproteinagent.agents.planning.ir import (
    planner_ir_schema,
    validate_planner_ir_v1,
)
from vibeproteinagent.agents.provider import ChatMessage
from vibeproteinagent.agents.provider.types import ChatProvider
from vibeproteinagent.agents.tools.schemas import (
    compact_input_schema_outline,
    compact_tool_outline,
    compact_tool_schema,
    root_input_fields,
    tool_role,
    tool_names_from_plan,
)
from vibeproteinagent.core.errors import ErrorCode, InvalidParamsError, VPAError
from vibeproteinagent.workflows import Workflow, serialize_workflow
from vibeproteinagent.workflows.planner import params_dict_from_workflow
from vibeproteinagent.workflows.resolver import iter_refs, resolve

from .prompts import (
    PLAN_VERIFICATION_SYSTEM_PROMPT,
    PLANNER_IR_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from vibeproteinagent.agents.execution.round_state import PriorRoundContext

_MAX_SELECTED_TOOLS = 6
_MAX_PLAN_VERIFICATION_REPAIRS = 3
PlannerIRCompiler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class PlannerIRViolation:
    path: str
    issue: str
    expected: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PlannerIRSchemaViolation(VPAError):
    code = ErrorCode.INVALID_PARAMS

    def __init__(
        self,
        violations: list[PlannerIRViolation],
        *,
        payload: dict[str, Any] | None = None,
        raw_output: str | None = None,
    ) -> None:
        self.violations = violations
        self.payload = payload or {}
        self.raw_output = raw_output
        self.structured_error = {
            "error_kind": "planner_ir_schema_violation",
            "violations": [
                {
                    "path": item.path,
                    "issue": item.issue,
                    "expected": item.expected,
                    **item.metadata,
                }
                for item in violations
            ],
            "fix_instruction": (
                "Re-emit the Planner IR JSON. Use version planner_ir_v1, "
                "semantic nodes, literal scalar inputs, and upstream entries "
                "whose relationship is candidate_source or term_source. Do not emit executor "
                "wiring or decision-time submission nodes."
            ),
        }
        super().__init__(
            "planner payload failed planner_ir_v1 schema validation",
            details={
                "structured_error": self.structured_error,
                "payload": self.payload,
                "raw_output": self.raw_output,
            },
        )


@lru_cache(maxsize=1)
def _planner_ir_validator() -> Draft202012Validator:
    return Draft202012Validator(planner_ir_schema())


def _install_readiness(tool_name: str) -> dict[str, Any]:
    try:
        from vibeproteinagent.install.installer import install_readiness

        return install_readiness((tool_name,), runtime_probe=False)
    except Exception as exc:
        return {
            "install_ready": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }


def _descriptor_availability(descriptor: Any, adapter_cls: type[Any]) -> dict[str, Any]:
    try:
        return descriptor_availability(descriptor, adapter_cls)
    except TypeError:
        return descriptor_availability(descriptor)


def _rfdiffusion_mode_readiness(params: dict[str, Any]) -> dict[str, Any]:
    try:
        from vibeproteinagent.adapters.rfdiffusion import mode_readiness

        return dict(mode_readiness(params))
    except VPAError:
        raise
    except Exception as exc:
        raise InvalidParamsError(
            "could not validate RFdiffusion mode readiness",
            details={"error": str(exc), "exception_type": type(exc).__name__},
        ) from exc


def _phase_idempotency_key(idempotency_key: str | None, phase: str) -> str | None:
    if idempotency_key is None:
        return None
    return f"{idempotency_key}:{phase}"


def _candidate_tool_specs(selection: dict[str, Any]) -> list[dict[str, Any]]:
    selected = selection.get("selected_tools")
    if isinstance(selected, list):
        raw_items = selected
    else:
        raw_items = selection.get("candidate_tools")
        if not isinstance(raw_items, list):
            raw_items = tool_names_from_plan(selection)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, str):
            name = item
            selected_mode = None
        elif isinstance(item, dict):
            raw_name = item.get("name") or item.get("tool")
            if not isinstance(raw_name, str):
                continue
            name = raw_name
            selected_mode = item.get("selected_mode")
        else:
            continue
        if not isinstance(name, str) or name in seen:
            continue
        seen.add(name)
        spec: dict[str, Any] = {"name": name}
        normalized_mode = _normalize_selected_mode(selected_mode)
        if normalized_mode is not None:
            spec["selected_mode"] = normalized_mode
        out.append(spec)
    return out[:_MAX_SELECTED_TOOLS]


def _normalize_selected_mode(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.lower() in {"", "none", "null", "n/a"}:
        return None
    return stripped


def _candidate_tool_names(selection: dict[str, Any]) -> list[str]:
    return [spec["name"] for spec in _candidate_tool_specs(selection)]


def _candidate_tool_inventory(
    request: str,
    available_tools: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return a compact, request-relevant public tool inventory for the LLM.

    This is deliberately generic lexical pruning, not task routing. The planner
    still chooses tools; we just avoid sending every public adapter description
    to small/slow providers for every request.
    """

    if len(available_tools) <= limit:
        return list(available_tools)
    request_tokens = _text_tokens(request)
    scored: list[tuple[int, int, str, dict[str, Any]]] = []
    for index, tool in enumerate(available_tools):
        name = str(tool.get("name") or "")
        haystack = f"{name} {tool.get('description') or ''}"
        tokens = _text_tokens(haystack)
        mentioned = 1 if name.lower() in request.lower() else 0
        overlap = len(request_tokens & tokens)
        score = (mentioned * 1000) + (overlap * 10) - _required_input_penalty(
            tool,
            request_tokens,
        )
        scored.append((score, -index, name, tool))
    selected = [tool for score, _neg_index, _name, tool in sorted(scored, reverse=True) if score > 0]
    if len(selected) < limit:
        selected.extend(
            tool
            for _score, _neg_index, _name, tool in sorted(scored, reverse=True)
            if tool not in selected
        )
    return selected[:limit]


_FILLABLE_TEXT_FIELDS = {
    "query",
    "request",
    "prompt",
    "text",
    "description",
    "objective",
}


def _required_input_penalty(tool: dict[str, Any], request_tokens: set[str]) -> int:
    schema = tool.get("input_schema")
    if not isinstance(schema, dict):
        return 0
    required = schema.get("required")
    if not isinstance(required, list):
        return 0
    penalty = 0
    for field in required:
        if not isinstance(field, str):
            continue
        field_tokens = _text_tokens(field)
        if field in _FILLABLE_TEXT_FIELDS:
            continue
        if field_tokens and field_tokens & request_tokens:
            continue
        penalty += 25
    return penalty


def _text_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9_+-]+", text.lower())
        if len(token) > 2
    }


def _load_json_object_response(
    content: str,
    *,
    label: str,
    error_message: str,
) -> dict[str, Any]:
    try:
        obj: Any = json.loads(content)
        for _ in range(2):
            if not isinstance(obj, str):
                break
            obj = json.loads(obj)
    except json.JSONDecodeError as exc:
        raise InvalidParamsError(
            error_message,
            details={"error": str(exc), "content": content[:1000]},
        ) from exc
    if not isinstance(obj, dict):
        raise InvalidParamsError(
            f"{label} must return a JSON object",
            details={"got_type": type(obj).__name__},
        )
    return obj


def _load_planner_ir_json_or_raise(content: str) -> dict[str, Any]:
    try:
        return _load_json_object_response(
            content,
            label="planner_ir",
            error_message="planner returned non-JSON Planner IR content",
        )
    except InvalidParamsError as exc:
        raise PlannerIRSchemaViolation(
            [
                PlannerIRViolation(
                    path="/",
                    issue="invalid_json",
                    expected="Return exactly one valid Planner IR JSON object.",
                )
            ],
            raw_output=content,
        ) from exc


def _validate_planner_ir_v1_or_raise(payload: dict[str, Any]) -> None:
    errors = sorted(
        _planner_ir_validator().iter_errors(payload),
        key=lambda err: list(err.path),
    )
    if errors:
        raise PlannerIRSchemaViolation(
            [_planner_ir_violation_from_error(error) for error in errors],
            payload=payload,
        )
    try:
        validate_planner_ir_v1(payload)
    except JSONSchemaValidationError as exc:
        raise PlannerIRSchemaViolation(
            [_planner_ir_violation_from_exception(exc)],
            payload=payload,
        ) from exc


def _planner_ir_violation_from_error(error: JSONSchemaValidationError) -> PlannerIRViolation:
    path = _json_pointer(list(error.path)) or "/"
    if (
        error.validator in {"const", "enum"}
        and path.endswith("/relationship")
    ):
        return PlannerIRViolation(
            path=path,
            issue="relationship_not_allowed_in_v1",
            expected=(
                'relationship must be one of "candidate_source", "term_source", '
                'or "identifier_source" in planner_ir_v1'
            ),
        )
    if error.validator == "required":
        missing = _missing_required_property(error.message)
        return PlannerIRViolation(
            path=f"{path.rstrip('/')}/{_escape_json_pointer(missing)}" if missing else path,
            issue="required_property_missing",
            expected=f"Required property {missing!r} must be present.",
        )
    if error.validator == "additionalProperties":
        extra = _additional_property(error.message)
        return PlannerIRViolation(
            path=f"{path.rstrip('/')}/{_escape_json_pointer(extra)}" if extra else path,
            issue="additional_property_not_allowed",
            expected="Remove fields not declared by planner_ir_v1.json.",
        )
    if error.validator == "minItems" and path == "/nodes":
        return PlannerIRViolation(
            path=path,
            issue="nodes_empty",
            expected="nodes must contain at least one Planner IR node.",
        )
    if error.validator == "pattern" and path.endswith("/id"):
        return PlannerIRViolation(
            path=path,
            issue="invalid_node_id",
            expected="Node id must match ^[a-z_][a-z0-9_]{0,63}$.",
        )
    return PlannerIRViolation(
        path=path,
        issue=str(error.validator),
        expected=error.message,
    )


def _planner_ir_violation_from_exception(
    error: JSONSchemaValidationError,
) -> PlannerIRViolation:
    message = str(error)
    path = _json_pointer(list(error.path)) or "/"
    if "executor reference syntax" in message:
        return PlannerIRViolation(
            path=path,
            issue="executor_reference_forbidden",
            expected="Remove executor wiring references from Planner IR strings.",
        )
    if "workflow_ref" in message:
        return PlannerIRViolation(
            path=path,
            issue="workflow_ref_forbidden",
            expected="Planner IR must describe semantic nodes, not workflow shortcuts.",
        )
    if "submit_final" in message:
        return PlannerIRViolation(
            path=path,
            issue="submit_final_forbidden",
            expected="Final submission is a decision-time action, not Planner IR.",
        )
    if "submit_candidate" in message:
        return PlannerIRViolation(
            path=path,
            issue="submit_candidate_node_forbidden",
            expected="Do not include submit_candidate as a Planner IR node.",
        )
    if "duplicate node id" in message:
        return PlannerIRViolation(
            path="/nodes",
            issue="duplicate_node_id",
            expected="Each Planner IR node id must be unique.",
        )
    if "unknown upstream node" in message:
        return PlannerIRViolation(
            path="/nodes",
            issue="unknown_upstream_node",
            expected="Each upstream node must refer to an earlier node in the same IR.",
        )
    if "must appear before" in message:
        return PlannerIRViolation(
            path="/nodes",
            issue="upstream_must_be_earlier",
            expected="Each upstream node must appear before its consumer node.",
        )
    return PlannerIRViolation(path=path, issue="planner_ir_invalid", expected=message)


def _json_pointer(parts: list[Any]) -> str:
    if not parts:
        return ""
    return "/" + "/".join(_escape_json_pointer(str(part)) for part in parts)


def _escape_json_pointer(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _missing_required_property(message: str) -> str | None:
    match = re.search(r"'([^']+)' is a required property", message)
    return match.group(1) if match else None


def _additional_property(message: str) -> str | None:
    match = re.search(r"'([^']+)' was unexpected", message)
    return match.group(1) if match else None


def _planner_ir_prompt(
    *,
    user_request: str,
    tool_catalog: list[dict[str, Any]],
    known_inputs: dict[str, Any] | None,
    retry_context: dict[str, Any] | None,
    prior_round_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt: dict[str, Any] = {}
    if prior_round_context is not None:
        prompt["prior_round_context"] = prior_round_context
    prompt.update({
        "request": user_request,
        "known_inputs": known_inputs or {},
        "tool_catalog": tool_catalog,
        "canonical_wiring": build_canonical_wiring_for_planner(
            has_prior_round_context=prior_round_context is not None,
        ),
        "output_contract": {
            "version": "planner_ir_v1",
            "goal": "string",
            "assumptions": ["string"],
            "missing_inputs": [
                {
                    "name": "string",
                    "reason": "string",
                    "required_for_execution": True,
                }
            ],
            "nodes": [
                {
                    "id": "stable_node_id",
                    "tool": "tool_name",
                    "intent": "why this tool is used",
                    "inputs": {"known_scalar_param": "literal value"},
                    "upstream": [
                        {
                            "node": "earlier_node_id",
                            "relationship": "candidate_source or term_source",
                        }
                    ],
                }
            ],
            "finalization": {
                "ranking_preference": "string or null",
                "success_criteria": {},
            },
        },
    })
    if retry_context is not None:
        prompt["previous_planner_ir_error"] = retry_context
    return prompt


def _tool_selection_haystack(
    user_request: str,
    prior_round_context: "PriorRoundContext | Any | None" = None,
) -> str:
    if prior_round_context is None:
        return user_request
    payload = _prior_round_context_payload(prior_round_context)
    if payload is None:
        return user_request
    return f"{user_request}\n{json.dumps(payload, sort_keys=True)}"


def _tool_selection_prompt(
    *,
    user_request: str,
    available_tools: list[dict[str, Any]],
    known_inputs: dict[str, Any] | None = None,
    prior_round_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt: dict[str, Any] = {
        "request": user_request,
        "known_inputs": known_inputs or {},
        "available_tools": [
            compact_tool_outline(tool)
            for tool in available_tools
        ],
        "output_contract": {
            "mode": "tool_selection",
            "selected_tools": [
                {
                    "name": "tool_name",
                    "selected_mode": (
                        "one allowed mode, only when available_tools shows allowed_modes"
                    ),
                    "reason": "why this tool and mode are useful",
                }
            ],
            "max_selected_tools": _MAX_SELECTED_TOOLS,
            "missing_inputs": ["string"],
        },
    }
    if prior_round_context is not None:
        prompt["prior_round_context"] = prior_round_context
    return prompt


def _validate_planner_ir_against_selected_tools(
    planner_ir: dict[str, Any],
    selected_tools: list[dict[str, Any]],
    *,
    known_inputs: dict[str, Any] | None = None,
) -> None:
    by_name = {
        str(tool.get("name")): tool
        for tool in selected_tools
        if isinstance(tool.get("name"), str)
    }
    violations: list[PlannerIRViolation] = []
    used_tools: set[str] = set()
    known = known_inputs or {}
    required_missing_inputs = [
        item
        for item in planner_ir.get("missing_inputs") or []
        if isinstance(item, dict) and item.get("required_for_execution") is True
    ]
    for item in required_missing_inputs:
        name = item.get("name")
        violations.append(
            PlannerIRViolation(
                path="/missing_inputs",
                issue="required_missing_input_blocks_compile",
                expected=(
                    "Planner IR may not compile an executable workflow while "
                    f"required input {name!r} is still missing. Known inputs: "
                    + ", ".join(sorted(known))
                ),
            )
        )
    for index, node in enumerate(planner_ir.get("nodes") or []):
        if not isinstance(node, dict):
            continue
        tool_name = node.get("tool")
        if not isinstance(tool_name, str) or tool_name not in by_name:
            violations.append(
                PlannerIRViolation(
                    path=f"/nodes/{index}/tool",
                    issue="tool_not_selected",
                    expected=(
                        "Use only tool names selected by the tool-selection phase: "
                        + ", ".join(sorted(by_name))
                    ),
                )
            )
            continue
        used_tools.add(tool_name)
        schema = by_name[tool_name].get("input_schema")
        selected_mode = by_name[tool_name].get("selected_mode")
        inputs = node.get("inputs") or {}
        if selected_mode is not None:
            mode_field = explicit_mode_input_field(tool_name)
            received_mode = inputs.get(mode_field) if isinstance(inputs, dict) else None
            if received_mode is None:
                violations.append(
                    PlannerIRViolation(
                        path=f"/nodes/{index}/inputs/{mode_field}",
                        issue="selected_mode_missing_from_planner_ir",
                        expected=(
                            f"Tool {tool_name!r} was selected with mode "
                            f"{selected_mode!r}; set inputs.{mode_field} to that exact value."
                        ),
                    )
                )
            elif received_mode != selected_mode:
                violations.append(
                    PlannerIRViolation(
                        path=f"/nodes/{index}/inputs/{mode_field}",
                        issue="selected_mode_mismatch",
                        expected=(
                            f"Tool {tool_name!r} was selected with mode "
                            f"{selected_mode!r}; received inputs.{mode_field}={received_mode!r}."
                        ),
                    )
                )
        if not isinstance(schema, dict):
            if _node_has_placeholder(node):
                violations.append(
                    PlannerIRViolation(
                        path=f"/nodes/{index}",
                        issue="placeholder_value_not_allowed",
                        expected="Use known literal inputs or mark unknown values as missing_inputs.",
                    )
                )
            continue
        upstream = node.get("upstream")
        upstreams = upstream if isinstance(upstream, list) else []
        node_by_id = _planner_ir_nodes_by_id(planner_ir)
        is_root = not upstreams
        if (
            is_root
            and tool_name not in CANONICAL_PRODUCERS
            and tool_name not in CANONICAL_TERM_PRODUCERS
        ):
            violations.append(
                PlannerIRViolation(
                    path=f"/nodes/{index}/tool",
                    issue="root_tool_not_canonical_producer",
                    expected=(
                        f"Root executable tool {tool_name!r} must be a canonical "
                        "producer or receive candidate_source input from an upstream node."
                    ),
                )
            )
        term_source_fields = _term_source_input_fields(
            node=node,
            tool_name=tool_name,
            node_by_id=node_by_id,
        )
        for term_violation in _term_source_violations(
            node=node,
            node_index=index,
            tool_name=tool_name,
            upstreams=upstreams,
            node_by_id=node_by_id,
        ):
            violations.append(term_violation)
        for capability_violation in _candidate_capability_violations(
            node=node,
            node_index=index,
            tool_name=tool_name,
            upstreams=upstreams,
            node_by_id=node_by_id,
        ):
            violations.append(capability_violation)
        missing_required = _missing_required_inputs_for_node(
            node=node,
            schema=schema,
            known_inputs=known,
            is_root=is_root,
            wired_inputs=term_source_fields,
        )
        if missing_required:
            if is_root:
                violations.append(
                    PlannerIRViolation(
                        path=f"/nodes/{index}/inputs",
                        issue="unsatisfied_root_required_input",
                        expected=(
                            f"Root tool {tool_name!r} node {node.get('id')!r} is missing "
                            f"required executable inputs {missing_required}. Known inputs: "
                            f"{sorted(known)}. Valid root inputs: {root_input_fields(schema)}. "
                            "Use a resolver/lookup producer first when exact identifiers are absent."
                        ),
                        metadata={
                            "tool": tool_name,
                            "node_id": node.get("id"),
                            "missing_fields": missing_required,
                            "known_inputs": sorted(known),
                            "valid_root_inputs": root_input_fields(schema),
                            "required_for_execution": True,
                        },
                    )
                )
            violations.append(
                PlannerIRViolation(
                    path=f"/nodes/{index}/inputs",
                    issue="required_tool_input_missing",
                    expected=(
                        f"Tool {tool_name!r} node {node.get('id')!r} is missing "
                        f"required input fields {missing_required}. Known inputs: "
                        f"{sorted(known)}. Valid root inputs: {root_input_fields(schema)}."
                    ),
                )
            )
        if _node_has_placeholder(node):
            violations.append(
                PlannerIRViolation(
                    path=f"/nodes/{index}",
                    issue="placeholder_value_not_allowed",
                    expected="Use known literal inputs or mark unknown values as missing_inputs.",
                )
            )
        if schema.get("additionalProperties") is not False:
            continue
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            continue
        valid_fields = {str(name) for name in properties}
        if not isinstance(inputs, dict):
            continue
        for field in inputs:
            if str(field) not in valid_fields:
                violations.append(
                    PlannerIRViolation(
                        path=f"/nodes/{index}/inputs/{field}",
                        issue="input_field_not_in_tool_schema",
                        expected=(
                            f"Tool {tool_name!r} inputs may use only fields from "
                            "input_schema_outline.properties: "
                            + ", ".join(sorted(valid_fields))
                        ),
                    )
                )
    unused = sorted(set(by_name) - used_tools)
    for tool_name in unused:
        violations.append(
            PlannerIRViolation(
                path="/nodes",
                issue="selected_tool_unused",
                expected=(
                    f"Tool {tool_name!r} was selected by tool selection but is "
                    "not present in Planner IR. Use it in the topology or drop "
                    "it through the bounded verification repair patch."
                ),
            )
        )
    if violations:
        raise PlannerIRSchemaViolation(violations, payload=planner_ir)


def _prepare_planner_ir_for_verification(
    planner_ir: dict[str, Any],
    selected_tools: list[dict[str, Any]],
    *,
    known_inputs: dict[str, Any] | None,
) -> dict[str, Any]:
    prepared = deepcopy(planner_ir)
    known_inputs = known_inputs or {}
    schema_by_name = {
        str(tool.get("name")): tool.get("input_schema")
        for tool in selected_tools
        if isinstance(tool.get("name"), str)
    }
    selected_modes = {
        str(tool.get("name")): tool.get("selected_mode")
        for tool in selected_tools
        if isinstance(tool.get("name"), str)
        and isinstance(tool.get("selected_mode"), str)
    }
    satisfied_fields = set(known_inputs)
    for node in prepared.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        tool_name = node.get("tool")
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
            node["inputs"] = inputs
        if isinstance(tool_name, str):
            mode_field = explicit_mode_input_field(tool_name)
            if mode_field != "mode" and "mode" in inputs and mode_field not in inputs:
                inputs[mode_field] = inputs.pop("mode")
            selected_mode = selected_modes.get(tool_name)
            if selected_mode is not None and mode_field not in inputs and "mode" in inputs:
                inputs[mode_field] = inputs.pop("mode")
        satisfied_fields.update(str(field) for field in inputs)
        schema = schema_by_name.get(tool_name)
        if not isinstance(schema, dict):
            continue
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            continue
        for field in _required_field_options(schema):
            if field in inputs or field not in properties:
                continue
            if field in known_inputs:
                inputs[field] = known_inputs[field]
                satisfied_fields.add(field)
    prepared["missing_inputs"] = _remaining_missing_inputs(
        prepared.get("missing_inputs"),
        satisfied_fields,
    )
    return prepared


def _remaining_missing_inputs(
    missing_inputs: Any,
    known_inputs: dict[str, Any],
) -> list[Any]:
    if not isinstance(missing_inputs, list):
        return []
    remaining: list[Any] = []
    for item in missing_inputs:
        if isinstance(item, dict) and item.get("name") in known_inputs:
            continue
        remaining.append(item)
    return remaining


def _required_field_options(schema: dict[str, Any]) -> list[str]:
    names: list[str] = []
    required = schema.get("required")
    if isinstance(required, list):
        names.extend(str(item) for item in required if isinstance(item, str))
    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            required_variant = variant.get("required")
            if not isinstance(required_variant, list):
                continue
            names.extend(
                str(item) for item in required_variant if isinstance(item, str)
            )
    return sorted(set(names))


def _missing_required_inputs_for_node(
    *,
    node: dict[str, Any],
    schema: dict[str, Any],
    known_inputs: dict[str, Any],
    is_root: bool,
    wired_inputs: set[str] | None = None,
) -> list[str]:
    if not is_root and not wired_inputs:
        return []
    inputs = node.get("inputs")
    input_names = set(inputs) if isinstance(inputs, dict) else set()
    available = input_names | set(known_inputs) | set(wired_inputs or set())
    missing: list[str] = []
    for field in _required_field_options(schema):
        if field in available:
            continue
        missing.append(field)
    if _schema_branch_satisfied(schema, available):
        return []
    return missing


def _planner_ir_nodes_by_id(planner_ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id")): node
        for node in planner_ir.get("nodes") or []
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }


def _term_source_input_fields(
    *,
    node: dict[str, Any],
    tool_name: str,
    node_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    field = CANONICAL_TERM_CONSUMERS.get(tool_name)
    if field is None:
        return set()
    for upstream in node.get("upstream") or []:
        if not isinstance(upstream, dict):
            continue
        if upstream.get("relationship") not in {"term_source", "identifier_source"}:
            continue
        upstream_id = upstream.get("node")
        upstream_node = node_by_id.get(str(upstream_id))
        upstream_tool = upstream_node.get("tool") if isinstance(upstream_node, dict) else None
        if isinstance(upstream_tool, str) and field in CANONICAL_TERM_PRODUCERS.get(upstream_tool, {}):
            return {field}
    return set()


def _term_source_violations(
    *,
    node: dict[str, Any],
    node_index: int,
    tool_name: str,
    upstreams: list[Any],
    node_by_id: dict[str, dict[str, Any]],
) -> list[PlannerIRViolation]:
    out: list[PlannerIRViolation] = []
    for upstream_index, upstream in enumerate(upstreams):
        if not isinstance(upstream, dict):
            continue
        relationship = upstream.get("relationship")
        if relationship not in {"term_source", "identifier_source"}:
            continue
        input_field = CANONICAL_TERM_CONSUMERS.get(tool_name)
        if input_field is None:
            out.append(
                PlannerIRViolation(
                    path=f"/nodes/{node_index}/upstream/{upstream_index}/relationship",
                    issue="term_source_consumer_not_supported",
                    expected=(
                        f"Tool {tool_name!r} cannot consume term_source. "
                        "Use term_source only into exact-ID retrieval tools such as "
                        f"{sorted(CANONICAL_TERM_CONSUMERS)}."
                    ),
                )
            )
            continue
        upstream_id = upstream.get("node")
        upstream_node = node_by_id.get(str(upstream_id))
        upstream_tool = upstream_node.get("tool") if isinstance(upstream_node, dict) else None
        if not isinstance(upstream_tool, str) or upstream_tool not in CANONICAL_TERM_PRODUCERS:
            out.append(
                PlannerIRViolation(
                    path=f"/nodes/{node_index}/upstream/{upstream_index}/node",
                    issue="term_source_not_produced",
                    expected=(
                        f"Upstream node {upstream_id!r} must use a resolver/lookup tool "
                        f"that produces term identifiers: {sorted(CANONICAL_TERM_PRODUCERS)}."
                    ),
                )
            )
            continue
        if input_field not in CANONICAL_TERM_PRODUCERS[upstream_tool]:
            out.append(
                PlannerIRViolation(
                    path=f"/nodes/{node_index}/upstream/{upstream_index}/node",
                    issue="term_source_wrong_identifier_type",
                    expected=(
                        f"Resolver tool {upstream_tool!r} cannot provide {input_field!r} "
                        f"for {tool_name!r}."
                    ),
                )
            )
    return out


def _candidate_capability_violations(
    *,
    node: dict[str, Any],
    node_index: int,
    tool_name: str,
    upstreams: list[Any],
    node_by_id: dict[str, dict[str, Any]],
) -> list[PlannerIRViolation]:
    required = candidate_contract_for_inputs(
        tool_name,
        node.get("inputs") if isinstance(node.get("inputs"), dict) else {},
    )
    if not required.consumes and required.min_candidate_items <= 1:
        return []
    out: list[PlannerIRViolation] = []
    candidate_upstreams = [
        upstream
        for upstream in upstreams
        if isinstance(upstream, dict) and upstream.get("relationship") == "candidate_source"
    ]
    if (
        required.min_candidate_items > 1
        and 0 < len(candidate_upstreams) < required.min_candidate_items
    ):
        upstream_id = str(candidate_upstreams[0].get("node") or "")
        upstream_node = node_by_id.get(upstream_id)
        upstream_tool = (
            "prior_round"
            if upstream_id == "prior_round"
            else str((upstream_node or {}).get("tool") or "")
        )
        if upstream_tool != "merge_candidates":
            out.append(
                PlannerIRViolation(
                    path=f"/nodes/{node_index}/upstream",
                    issue="candidate_item_count_mismatch",
                    expected=(
                        f"Tool {tool_name!r} expects at least "
                        f"{required.min_candidate_items} candidate items. Provide "
                        "multiple candidate_source edges or use an upstream merged "
                        "candidate set."
                    ),
                    metadata={
                        "consumer_node": node.get("id"),
                        "consumer_tool": tool_name,
                        "min_candidate_items": required.min_candidate_items,
                        "repair_instruction": (
                            "Repair the topology by providing enough candidate "
                            "items, usually via multiple candidate_source edges "
                            "that the compiler can merge."
                        ),
                    },
                )
            )
    for upstream_index, upstream in enumerate(upstreams):
        if not isinstance(upstream, dict):
            continue
        if upstream.get("relationship") != "candidate_source":
            continue
        upstream_id = str(upstream.get("node") or "")
        upstream_node = node_by_id.get(upstream_id)
        upstream_tool = (
            "prior_round"
            if upstream_id == "prior_round"
            else str((upstream_node or {}).get("tool") or "")
        )
        produced = _planner_node_candidate_capabilities(
            upstream_id,
            node_by_id=node_by_id,
            producer_tool=upstream_tool,
            seen=set(),
        )
        missing = sorted(required.consumes - produced)
        if not missing:
            continue
        out.append(
            PlannerIRViolation(
                path=f"/nodes/{node_index}/upstream/{upstream_index}",
                issue="candidate_capability_mismatch",
                expected=(
                    f"Upstream node {upstream_id!r} using tool {upstream_tool!r} "
                    f"must provide candidate capabilities {missing} before "
                    f"{tool_name!r} can consume it. Insert or select an "
                    "intermediate tool that produces the missing capabilities."
                ),
                metadata={
                    "producer_node": upstream_id,
                    "producer_tool": upstream_tool,
                    "consumer_node": node.get("id"),
                    "consumer_tool": tool_name,
                    "missing_capabilities": missing,
                    "repair_instruction": (
                        "Repair the topology by inserting or selecting a tool "
                        "that produces the missing candidate capabilities before "
                        "the consumer node."
                    ),
                },
            )
        )
    return out


def _planner_node_candidate_capabilities(
    node_id: str,
    *,
    node_by_id: dict[str, dict[str, Any]],
    producer_tool: str | None = None,
    seen: set[str],
) -> set[str]:
    if node_id == "prior_round":
        return set(candidate_contract("prior_round").produces)
    if node_id in seen:
        return set()
    seen.add(node_id)
    node = node_by_id.get(node_id)
    if node is None:
        return set()
    tool_name = producer_tool or str(node.get("tool") or "")
    contract = candidate_contract_for_inputs(
        tool_name,
        node.get("inputs") if isinstance(node.get("inputs"), dict) else {},
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
                "prior_round"
                if upstream_id == "prior_round"
                else str((upstream_node or {}).get("tool") or "")
            )
            capabilities.update(
                _planner_node_candidate_capabilities(
                    upstream_id,
                    node_by_id=node_by_id,
                    producer_tool=upstream_tool,
                    seen=seen,
                )
            )
    return capabilities


def _schema_branch_satisfied(schema: dict[str, Any], available: set[str]) -> bool:
    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        branch_requirements = [
            {
                str(item)
                for item in variant.get("required", [])
                if isinstance(item, str)
            }
            for variant in variants
            if isinstance(variant, dict)
        ]
        branch_requirements = [req for req in branch_requirements if req]
        if branch_requirements and any(req <= available for req in branch_requirements):
            return True
        if branch_requirements:
            return False
    required = {
        str(item)
        for item in schema.get("required", [])
        if isinstance(item, str)
    }
    return required <= available


_PLACEHOLDER_RE = re.compile(r"(\{\{[^{}]+\}\}|<[^<>]+>|placeholder)", re.IGNORECASE)


def _node_has_placeholder(node: dict[str, Any]) -> bool:
    return _contains_placeholder(node.get("inputs"))


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        stripped = value.strip()
        if _PLACEHOLDER_RE.search(stripped):
            return True
        if stripped.endswith(("_go_id", "_pdb_path", "_target_pdb", "_input_json")):
            return True
        return False
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _clean_known_inputs(known_inputs: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(known_inputs, dict):
        return {}
    return {
        str(key): value
        for key, value in known_inputs.items()
        if isinstance(key, str) and value is not None
    }


def _planner_ir_violation_signature(exc: PlannerIRSchemaViolation) -> tuple[str, str]:
    violations = exc.structured_error.get("violations")
    if isinstance(violations, list) and violations:
        first = violations[0]
        if isinstance(first, dict):
            return (str(first.get("path") or ""), str(first.get("issue") or ""))
    return ("", "unknown")


def _should_attempt_plan_verification_repair(exc: PlannerIRSchemaViolation) -> bool:
    violations = exc.structured_error.get("violations")
    if not isinstance(violations, list):
        return False
    repairable = {
        "candidate_capability_mismatch",
        "candidate_item_count_mismatch",
        "executor_reference_forbidden",
        "invalid_selected_mode",
        "input_field_not_in_tool_schema",
        "required_missing_input_blocks_compile",
        "required_tool_input_missing",
        "selected_mode_for_single_mode_tool",
        "selected_mode_mismatch",
        "selected_mode_missing_from_planner_ir",
        "selected_tool_unused",
        "tool_not_selected",
        "unsatisfied_root_required_input",
    }
    return any(
        isinstance(item, dict) and item.get("issue") in repairable
        for item in violations
    )


def _repair_unsatisfied_root_with_selected_term_producer(
    *,
    planner_ir: dict[str, Any],
    selected_tools: list[dict[str, Any]],
    user_request: str,
    known_inputs: dict[str, Any],
) -> dict[str, Any] | None:
    """Insert selected resolver roots before exact-ID retrieval roots when unambiguous."""

    selected_by_name = {
        str(tool.get("name")): tool
        for tool in selected_tools
        if isinstance(tool.get("name"), str)
    }
    nodes = planner_ir.get("nodes")
    if not isinstance(nodes, list):
        return None
    repaired_nodes: list[dict[str, Any]] = []
    changed = False
    existing_ids = {
        str(node.get("id"))
        for node in nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str)
    }

    for node in nodes:
        if not isinstance(node, dict):
            repaired_nodes.append(node)
            continue
        tool_name = node.get("tool")
        input_field = CANONICAL_TERM_CONSUMERS.get(str(tool_name))
        upstreams = node.get("upstream") if isinstance(node.get("upstream"), list) else []
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        if (
            input_field is None
            or upstreams
            or input_field in inputs
            or input_field in known_inputs
        ):
            repaired_nodes.append(node)
            continue
        candidates = [
            producer_name
            for producer_name, output_fields in CANONICAL_TERM_PRODUCERS.items()
            if input_field in output_fields
            and producer_name in selected_by_name
            and _tool_can_start_from_query_or_known_inputs(
                selected_by_name[producer_name],
                known_inputs=known_inputs,
            )
        ]
        if not candidates:
            repaired_nodes.append(node)
            continue
        producer = _preferred_term_producer(candidates, input_field=input_field)
        resolver_id = _unique_node_id(
            f"resolve_{node.get('id') or tool_name}",
            existing_ids,
        )
        existing_ids.add(resolver_id)
        repaired_nodes.append(
            {
                "id": resolver_id,
                "tool": producer,
                "intent": (
                    f"Resolve the identifier required by {tool_name} before "
                    "retrieving executable sequence candidates."
                ),
                "inputs": {"query": _resolver_query(user_request, node)},
                "upstream": [],
            }
        )
        repaired_node = deepcopy(node)
        repaired_node["upstream"] = [
            {"node": resolver_id, "relationship": "term_source"}
        ]
        repaired_nodes.append(repaired_node)
        changed = True

    if not changed:
        return None
    out = deepcopy(planner_ir)
    out["nodes"] = repaired_nodes
    out["missing_inputs"] = _drop_resolved_term_missing_inputs(
        out.get("missing_inputs"),
    )
    return out


def _preferred_term_producer(candidates: list[str], *, input_field: str) -> str:
    if "function_term_resolver" in candidates:
        return "function_term_resolver"
    preferred_by_field = {
        "go_id": ["go_term_search", "ontology_lookup"],
        "interpro_id": ["interpro_entry_search", "interpro_entry_lookup"],
    }
    for name in preferred_by_field.get(input_field, []):
        if name in candidates:
            return name
    return sorted(candidates)[0]


def _tool_can_start_from_query_or_known_inputs(
    tool: dict[str, Any],
    *,
    known_inputs: dict[str, Any],
) -> bool:
    schema = tool.get("input_schema")
    if not isinstance(schema, dict):
        return False
    required = set(_required_field_options(schema))
    available = {"query"} | set(known_inputs)
    return bool(required) and required <= available


def _resolver_query(user_request: str, node: dict[str, Any]) -> str:
    hints = [
        str(node.get("intent") or "").strip(),
        str(node.get("id") or "").replace("_", " ").strip(),
    ]
    hint_text = "; ".join(hint for hint in hints if hint)
    if not hint_text:
        return user_request
    return f"{user_request}\nTarget term context: {hint_text}"


def _unique_node_id(base: str, existing_ids: set[str]) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", base.lower()).strip("_")
    if not normalized or not re.match(r"^[a-z_]", normalized):
        normalized = f"resolve_{normalized}" if normalized else "resolve_term"
    if normalized not in existing_ids:
        return normalized
    suffix = 2
    while f"{normalized}_{suffix}" in existing_ids:
        suffix += 1
    return f"{normalized}_{suffix}"


def _drop_resolved_term_missing_inputs(missing_inputs: Any) -> list[Any]:
    if not isinstance(missing_inputs, list):
        return []
    out: list[Any] = []
    term_fields = set(CANONICAL_TERM_CONSUMERS.values())
    suffixes = tuple(f"_{field}" for field in term_fields)
    for item in missing_inputs:
        if isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and (name in term_fields or name.endswith(suffixes)):
                continue
        out.append(item)
    return out


def _apply_plan_verification_patch(
    *,
    planner_ir: dict[str, Any],
    selected_tools: list[dict[str, Any]],
    available_tools: list[dict[str, Any]],
    patch: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    available_by_name = {
        str(tool.get("name")): tool
        for tool in available_tools
        if isinstance(tool.get("name"), str)
    }
    selected_by_name = {
        str(tool.get("name")): tool
        for tool in selected_tools
        if isinstance(tool.get("name"), str)
    }
    selected_names = list(selected_by_name)

    for name in _string_list(patch.get("drop_selected_tools")):
        if name not in selected_by_name:
            raise InvalidParamsError(
                "plan verification patch tried to drop an unselected tool",
                details={"tool": name, "selected_tools": selected_names},
            )
        selected_by_name.pop(name, None)

    for name in _string_list(patch.get("add_selected_tools")):
        if name not in available_by_name:
            raise InvalidParamsError(
                "plan verification patch tried to add an unavailable tool",
                details={"tool": name, "available_tools": sorted(available_by_name)},
            )
        if name not in selected_by_name:
            selected_by_name[name] = available_by_name[name]

    selected_mode_updates = patch.get("set_selected_modes")
    if isinstance(selected_mode_updates, list):
        for item in selected_mode_updates:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("tool")
            if not isinstance(name, str) or name not in selected_by_name:
                continue
            selected_mode = _normalize_selected_mode(item.get("selected_mode"))
            if selected_mode is None:
                selected_by_name[name].pop("selected_mode", None)
            else:
                selected_by_name[name] = {
                    **selected_by_name[name],
                    "selected_mode": selected_mode,
                }

    out_ir = deepcopy(planner_ir)
    remove_ids = set(_string_list(patch.get("remove_node_ids")))
    if remove_ids:
        nodes = out_ir.get("nodes")
        if not isinstance(nodes, list):
            raise InvalidParamsError(
                "plan verification patch cannot remove nodes from malformed Planner IR",
                details={"remove_node_ids": sorted(remove_ids)},
            )
        known_ids = {
            str(node.get("id"))
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        unknown_ids = sorted(remove_ids - known_ids)
        if unknown_ids:
            raise InvalidParamsError(
                "plan verification patch tried to remove unknown Planner IR nodes",
                details={"node_ids": unknown_ids, "known_node_ids": sorted(known_ids)},
            )
        out_ir["nodes"] = [
            node
            for node in nodes
            if not (isinstance(node, dict) and node.get("id") in remove_ids)
        ]

    nodes = out_ir.setdefault("nodes", [])
    if not isinstance(nodes, list):
        raise InvalidParamsError(
            "plan verification patch cannot update malformed Planner IR nodes",
            details={"nodes_type": type(nodes).__name__},
        )

    replace_nodes = patch.get("replace_nodes")
    if isinstance(replace_nodes, list):
        by_id = {
            str(node.get("id")): index
            for index, node in enumerate(nodes)
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        for replacement in replace_nodes:
            if not isinstance(replacement, dict) or not isinstance(replacement.get("id"), str):
                continue
            node_id = str(replacement["id"])
            if node_id not in by_id:
                raise InvalidParamsError(
                    "plan verification patch tried to replace unknown Planner IR node",
                    details={"node_id": node_id, "known_node_ids": sorted(by_id)},
                )
            nodes[by_id[node_id]] = replacement

    add_nodes = patch.get("add_nodes")
    if isinstance(add_nodes, list):
        existing_ids = {
            str(node.get("id"))
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        for new_node in add_nodes:
            if not isinstance(new_node, dict) or not isinstance(new_node.get("id"), str):
                continue
            node_id = str(new_node["id"])
            if node_id in existing_ids:
                raise InvalidParamsError(
                    "plan verification patch tried to add duplicate Planner IR node",
                    details={"node_id": node_id},
                )
            nodes.append(new_node)
            existing_ids.add(node_id)

    input_updates = patch.get("update_node_inputs")
    if isinstance(input_updates, list):
        nodes_by_id = {
            str(node.get("id")): node
            for node in nodes
            if isinstance(node, dict) and isinstance(node.get("id"), str)
        }
        for update in input_updates:
            if not isinstance(update, dict):
                continue
            node_id = update.get("node_id") or update.get("id")
            inputs = update.get("inputs")
            if not isinstance(node_id, str) or not isinstance(inputs, dict):
                continue
            node = nodes_by_id.get(node_id)
            if node is None:
                raise InvalidParamsError(
                    "plan verification patch tried to update unknown Planner IR node inputs",
                    details={"node_id": node_id, "known_node_ids": sorted(nodes_by_id)},
                )
            current_inputs = node.setdefault("inputs", {})
            if not isinstance(current_inputs, dict):
                current_inputs = {}
                node["inputs"] = current_inputs
            for key, value in inputs.items():
                if value is None:
                    current_inputs.pop(key, None)
                else:
                    current_inputs[key] = value

    if "set_missing_inputs" in patch:
        missing_inputs = patch.get("set_missing_inputs")
        if isinstance(missing_inputs, list):
            out_ir["missing_inputs"] = missing_inputs

    return out_ir, list(selected_by_name.values())


def _tool_selection_mode_structured_error(
    selection: dict[str, Any],
    available_tools: list[dict[str, Any]],
) -> dict[str, Any] | None:
    available = {
        str(tool.get("name")): tool
        for tool in available_tools
        if isinstance(tool.get("name"), str)
    }
    violations: list[dict[str, Any]] = []
    for index, spec in enumerate(_candidate_tool_specs(selection)):
        name = spec["name"]
        selected_mode = spec.get("selected_mode")
        if selected_mode is None:
            continue
        allowed_modes = EXPLICIT_MODE_TOOLS.get(name)
        if allowed_modes is None:
            violations.append(
                {
                    "path": f"/selected_tools/{index}/selected_mode",
                    "issue": "invalid_selected_mode",
                    "tool": name,
                    "selected_mode": selected_mode,
                    "allowed_modes": [],
                    "expected": (
                        "Remove selected_mode for this tool because the "
                        "available tool outline does not include allowed_modes."
                    ),
                    "repair_instruction": (
                        "Emit set_selected_modes with selected_mode=null for "
                        f"{name!r}, or drop the tool if it was selected only "
                        "because of the invalid mode."
                    ),
                }
            )
            continue
        if selected_mode not in allowed_modes:
            violations.append(
                {
                    "path": f"/selected_tools/{index}/selected_mode",
                    "issue": "invalid_selected_mode",
                    "tool": name,
                    "selected_mode": selected_mode,
                    "allowed_modes": list(allowed_modes),
                    "expected": (
                        "selected_mode must be one of the tool's allowed_modes."
                    ),
                    "repair_instruction": (
                        f"Set selected_mode for {name!r} to one of "
                        f"{list(allowed_modes)}, or remove selected_mode."
                    ),
                }
            )
    if not violations:
        return None
    return {
        "error_kind": "tool_selection_verification_error",
        "violations": violations,
        "available_tools": sorted(available),
        "fix_instruction": (
            "Repair selected tool modes without drafting workflow inputs. "
            "Use selected_mode only for tools whose outline includes allowed_modes."
        ),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _prior_round_context_payload(
    prior_round_context: "PriorRoundContext | Any | None",
) -> dict[str, Any] | None:
    if prior_round_context is None:
        return None
    if is_dataclass(prior_round_context):
        return asdict(prior_round_context)
    if isinstance(prior_round_context, dict):
        return dict(prior_round_context)
    return {
        "round_number": getattr(prior_round_context, "round_number"),
        "surviving_candidates": list(
            getattr(prior_round_context, "surviving_candidates")
        ),
        "prior_decision_reason": getattr(
            prior_round_context,
            "prior_decision_reason",
        ),
        "summary": getattr(prior_round_context, "summary", None),
    }


def _fallback_tool_selection(
    prompt: dict[str, Any],
    content: str,
) -> dict[str, Any] | None:
    available = prompt.get("available_tools")
    if not isinstance(available, list):
        return None
    allowed = {
        str(item.get("name"))
        for item in available
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    names: list[str] = []
    haystack = f"{prompt.get('request') or ''}\n{content}".lower()
    for name in sorted(allowed):
        if name.lower() in haystack and name not in names:
            names.append(name)
    if not names:
        return None
    return {
        "mode": "tool_selection",
        "selected_tools": [{"name": name} for name in names],
        "reason": (
            "Recovered tool names from a non-JSON provider response using only "
            "the public tool inventory."
        ),
        "missing_inputs": [],
    }


def _invocation_params_from_plan(
    plan: dict[str, Any],
    params: dict[str, Any],
    workflow: Workflow | None = None,
) -> dict[str, Any]:
    plan_params = plan.get("params")
    if not isinstance(plan_params, dict):
        return dict(params)
    if workflow is not None:
        plan_params = {
            name: value
            for name, value in plan_params.items()
            if not _is_leaked_param_schema(name, value, workflow)
        }
    # Explicit CLI / params-file values win over LLM-suggested defaults.
    return {**plan_params, **params}


_WORKFLOW_PARAM_TYPE_ALIASES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}


_WORKFLOW_PLAN_KEYS = {
    "name",
    "version",
    "description",
    "params",
    "nodes",
}


def _sanitize_workflow_plan_shape(workflow_plan: dict[str, Any]) -> dict[str, Any]:
    """Drop agent-plan metadata accidentally nested inside workflow_plan."""

    if not any(key in workflow_plan for key in ("name", "nodes")):
        return workflow_plan
    extras = set(workflow_plan) - _WORKFLOW_PLAN_KEYS
    if not extras:
        return workflow_plan
    return {key: value for key, value in workflow_plan.items() if key in _WORKFLOW_PLAN_KEYS}


def _normalize_workflow_param_types(workflow_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize common JSON-Schema type words in LLM-authored plans.

    The workflow descriptor has a deliberately small type vocabulary
    (``str``, ``int``, ``float``, ``bool``, ``path``, ``list``, ``dict``).
    Some providers naturally emit JSON Schema terms such as ``integer`` or
    ``array``. Accepting those aliases at the planner boundary keeps the LLM
    contract portable while preserving the canonical descriptor format after
    validation.
    """

    params = workflow_plan.get("params")
    if not isinstance(params, dict):
        return workflow_plan
    normalized_params: dict[str, Any] = {}
    changed = False
    for name, spec in params.items():
        if not isinstance(spec, dict):
            normalized_params[name] = {
                "type": _workflow_param_type_for_value(spec),
                "required": False,
                "default": spec,
            }
            changed = True
            continue
        normalized_spec = dict(spec)
        if "value" in normalized_spec and "default" not in normalized_spec:
            normalized_spec["default"] = normalized_spec.pop("value")
            changed = True
        param_type = normalized_spec.get("type")
        if isinstance(param_type, list):
            non_null_types = [item for item in param_type if item != "null"]
            if len(non_null_types) == 1 and isinstance(non_null_types[0], str):
                normalized_spec["type"] = non_null_types[0]
                param_type = non_null_types[0]
                changed = True
        if isinstance(param_type, str):
            alias = _WORKFLOW_PARAM_TYPE_ALIASES.get(param_type.strip().lower())
            if alias is not None:
                normalized_spec["type"] = alias
                changed = True
        normalized_params[name] = normalized_spec
    if not changed:
        return workflow_plan
    return {**workflow_plan, "params": normalized_params}


def _workflow_param_type_for_value(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return "str"


_PARAM_SCHEMA_KEYS = {"type", "required", "default", "description"}


def _is_leaked_param_schema(name: str, value: Any, workflow: Workflow) -> bool:
    """Detect schema objects mistakenly emitted as top-level runtime params."""

    if name not in workflow.params or not isinstance(value, dict):
        return False
    if not value or not set(value).issubset(_PARAM_SCHEMA_KEYS):
        return False
    param_type = value.get("type")
    if not isinstance(param_type, str):
        return False
    canonical_types = {"str", "int", "float", "bool", "path", "list", "dict"}
    normalized = _WORKFLOW_PARAM_TYPE_ALIASES.get(param_type.strip().lower(), param_type)
    return normalized in canonical_types


def _reject_post_round_action_tools(plan: dict[str, Any]) -> None:
    if "submit_candidate" not in tool_names_from_plan(plan):
        return
    raise InvalidParamsError(
        "submit_candidate is a post-round decision action, not a planner workflow node",
        details={"tool": "submit_candidate"},
    )


def _validate_static_output_refs(workflow: Workflow, registry: ToolRegistry) -> None:
    """Reject references to output fields not declared by upstream tools.

    This catches LLM mistakes such as ``${outputs.rfd.sequence}`` before the
    workflow spends GPU time on the upstream node. Nested fields are still
    resolved at runtime because many adapter outputs contain heterogeneous
    candidate rows, but the top-level output key must match the schema when
    the tool declares one.
    """

    nodes_by_id = {node.id: node for node in workflow.nodes}
    for node in workflow.nodes:
        for ref in iter_refs(node.inputs):
            if not ref.startswith("outputs."):
                continue
            path = ref.removeprefix("outputs.")
            upstream_id, sep, rest = path.partition(".")
            upstream_id = upstream_id.split("[", 1)[0]
            if not sep or not upstream_id or upstream_id not in nodes_by_id:
                continue
            field = rest.split(".", 1)[0].split("[", 1)[0]
            if not field:
                continue
            upstream = nodes_by_id[upstream_id]
            descriptor, _adapter_cls = registry.get(upstream.tool, upstream.version)
            schema = resolve_output_schema_dict(descriptor)
            props = schema.get("properties") if isinstance(schema, dict) else None
            if not isinstance(props, dict) or field in props:
                continue
            raise InvalidParamsError(
                "agent workflow references an unknown upstream output field",
                details={
                    "node_id": node.id,
                    "ref": ref,
                    "upstream_node_id": upstream_id,
                    "upstream_tool": upstream.tool,
                    "field": field,
                    "available_fields": sorted(props),
                },
            )


def _validate_static_output_ref_types(workflow: Workflow, registry: ToolRegistry) -> None:
    nodes_by_id = {node.id: node for node in workflow.nodes}
    output_schemas: dict[str, dict[str, Any]] = {}
    for node in workflow.nodes:
        descriptor, _adapter_cls = registry.get(node.tool, node.version)
        schema = resolve_output_schema_dict(descriptor)
        if schema:
            output_schemas[node.id] = schema
    for node in workflow.nodes:
        descriptor, _adapter_cls = registry.get(node.tool, node.version)
        input_schema = resolve_input_schema_dict(descriptor)
        props = input_schema.get("properties") if isinstance(input_schema, dict) else None
        if not isinstance(props, dict) or not isinstance(node.inputs, dict):
            continue
        for name, value in node.inputs.items():
            prop_schema = props.get(name)
            if isinstance(prop_schema, dict):
                _validate_output_refs_against_input_schema(
                    value,
                    prop_schema,
                    input_schema,
                    node_id=node.id,
                    tool=node.tool,
                    field=name,
                    nodes_by_id=nodes_by_id,
                    output_schemas=output_schemas,
                    registry=registry,
                )


def _validate_output_refs_against_input_schema(
    value: Any,
    input_schema: dict[str, Any],
    input_root: dict[str, Any],
    *,
    node_id: str,
    tool: str,
    field: str,
    nodes_by_id: dict[str, Any],
    output_schemas: dict[str, dict[str, Any]],
    registry: ToolRegistry,
) -> None:
    resolved_schema = _resolve_schema_refs(input_schema, input_root)
    if resolved_schema is not None:
        input_schema = resolved_schema
    if isinstance(value, str):
        refs = [ref for ref in iter_refs(value) if ref.startswith("outputs.")]
        if not refs:
            return
        for ref in refs:
            output_schema = _schema_for_output_ref(ref, nodes_by_id, output_schemas)
            if output_schema and not _schema_types_compatible(
                input_schema,
                output_schema,
                input_root,
                output_schema,
            ):
                upstream_id = ref.removeprefix("outputs.").split(".", 1)[0]
                upstream = nodes_by_id.get(upstream_id)
                raise InvalidParamsError(
                    "agent workflow output reference has an incompatible schema type",
                    details={
                        "node_id": node_id,
                        "tool": tool,
                        "field": field,
                        "ref": ref,
                        "upstream_tool": getattr(upstream, "tool", None),
                        "expected_types": sorted(_schema_possible_types(input_schema, input_root)),
                        "actual_types": sorted(_schema_possible_types(output_schema, output_schema)),
                        "hint": (
                            "Use an output field whose schema matches the target input; "
                            "for candidate rows prefer fields such as sequences, designs, "
                            "top_candidates, representative_sequences, or ranked_candidates "
                            "rather than scalar count fields."
                        ),
                    },
                )
        return
    if isinstance(value, list):
        item_schema = _array_item_schema(input_schema, input_root)
        if isinstance(item_schema, dict):
            for item in value:
                _validate_output_refs_against_input_schema(
                    item,
                    item_schema,
                    input_root,
                    node_id=node_id,
                    tool=tool,
                    field=field,
                    nodes_by_id=nodes_by_id,
                    output_schemas=output_schemas,
                    registry=registry,
                )
        return
    if isinstance(value, dict):
        props = input_schema.get("properties")
        if isinstance(props, dict):
            for key, item in value.items():
                child_schema = props.get(key)
                if isinstance(child_schema, dict):
                    _validate_output_refs_against_input_schema(
                        item,
                        child_schema,
                        input_root,
                        node_id=node_id,
                        tool=tool,
                        field=f"{field}.{key}",
                        nodes_by_id=nodes_by_id,
                        output_schemas=output_schemas,
                        registry=registry,
                    )


def _resolve_schema_refs(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any] | None:
    current = schema
    seen: set[str] = set()
    while isinstance(current.get("$ref"), str):
        ref = current["$ref"]
        if ref in seen:
            return current
        seen.add(ref)
        resolved = _resolve_local_schema_ref(root, ref)
        if resolved is None:
            return current
        current = resolved
    return current


def _array_item_schema(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any] | None:
    schema = _resolve_schema_refs(schema, root) or schema
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        return schema["items"]
    for key in ("anyOf", "oneOf", "allOf"):
        options = schema.get(key)
        if not isinstance(options, list):
            continue
        for option in options:
            if not isinstance(option, dict):
                continue
            resolved = _resolve_schema_refs(option, root) or option
            item_schema = _array_item_schema(resolved, root)
            if item_schema is not None:
                return item_schema
    return None


def _exact_output_ref(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.startswith("${") or not stripped.endswith("}"):
        return None
    ref = stripped[2:-1].strip()
    return ref if ref.startswith("outputs.") else None


def _compatible_item_field(
    input_field: str,
    input_schema: dict[str, Any],
    input_root: dict[str, Any],
    item_schema: dict[str, Any],
) -> str | None:
    props = item_schema.get("properties")
    if not isinstance(props, dict):
        return None
    candidates = [input_field, "sequence", "structure_path", "pdb_path", "cif_path"]
    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        prop = props.get(name)
        if isinstance(prop, dict) and _schema_types_compatible(
            input_schema,
            prop,
            input_root,
            item_schema,
        ):
            return name
    return None


def _schema_for_output_ref(
    ref: str,
    nodes_by_id: dict[str, Any],
    output_schemas: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    path = ref.removeprefix("outputs.")
    upstream_id, sep, rest = path.partition(".")
    upstream_id = upstream_id.split("[", 1)[0]
    if not sep or upstream_id not in nodes_by_id:
        return None
    schema = output_schemas.get(upstream_id)
    if not schema:
        return None
    current = schema
    for part in rest.split("."):
        key = part.split("[", 1)[0]
        if not key:
            continue
        props = current.get("properties") if isinstance(current, dict) else None
        if not isinstance(props, dict) or key not in props:
            return None
        current = props[key]
        if "[" in part and isinstance(current, dict):
            items = current.get("items")
            if isinstance(items, dict):
                current = items
    return current if isinstance(current, dict) else None


def _schema_types_compatible(
    expected: dict[str, Any],
    actual: dict[str, Any],
    expected_root: dict[str, Any],
    actual_root: dict[str, Any],
) -> bool:
    expected_types = _schema_possible_types(expected, expected_root)
    actual_types = _schema_possible_types(actual, actual_root)
    if not expected_types or not actual_types:
        return True
    return bool(expected_types & actual_types)


def _schema_possible_types(schema: dict[str, Any], root: dict[str, Any]) -> set[str]:
    if "$ref" in schema and isinstance(schema["$ref"], str):
        resolved = _resolve_local_schema_ref(root, schema["$ref"])
        if resolved is not None and resolved is not schema:
            return _schema_possible_types(resolved, root)
    types: set[str] = set()
    raw_type = schema.get("type")
    if isinstance(raw_type, str):
        types.add(raw_type)
    elif isinstance(raw_type, list):
        types.update(str(item) for item in raw_type if isinstance(item, str))
    for key in ("anyOf", "oneOf", "allOf"):
        options = schema.get(key)
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict):
                    types.update(_schema_possible_types(option, root))
    return {item for item in types if item != "null"}


def _resolve_local_schema_ref(root: dict[str, Any], ref: str) -> dict[str, Any] | None:
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current if isinstance(current, dict) else None


def _validate_static_job_refs(workflow: Workflow, registry: ToolRegistry) -> None:
    for node in workflow.nodes:
        descriptor, _adapter_cls = registry.get(node.tool, node.version)
        schema = resolve_input_schema_dict(descriptor)
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if not isinstance(properties, dict):
            continue
        if not isinstance(node.inputs, dict):
            continue
        for name, value in node.inputs.items():
            prop_schema = properties.get(name)
            if isinstance(prop_schema, dict):
                _validate_job_refs_against_schema(
                    value,
                    prop_schema,
                    node_id=node.id,
                    tool=node.tool,
                    field=name,
                )


def _validate_job_refs_against_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    node_id: str,
    tool: str,
    field: str,
) -> None:
    refs = [ref for ref in iter_refs(value) if ref.startswith("jobs.")]
    if not refs:
        return
    if isinstance(value, str):
        if _schema_accepts_job_ref(schema):
            return
        _raise_invalid_job_ref(node_id=node_id, tool=tool, field=field, refs=refs)
    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for item in value:
                _validate_job_refs_against_schema(
                    item,
                    item_schema,
                    node_id=node_id,
                    tool=tool,
                    field=field,
                )
            return
        if not _schema_accepts_job_ref(schema):
            _raise_invalid_job_ref(node_id=node_id, tool=tool, field=field, refs=refs)
        return
    if isinstance(value, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            for key, item in value.items():
                item_schema = props.get(key)
                if isinstance(item_schema, dict):
                    _validate_job_refs_against_schema(
                        item,
                        item_schema,
                        node_id=node_id,
                        tool=tool,
                        field=f"{field}.{key}",
                    )
            return
        if not _schema_accepts_job_ref(schema):
            _raise_invalid_job_ref(node_id=node_id, tool=tool, field=field, refs=refs)
        return
    if not _schema_accepts_job_ref(schema):
        _raise_invalid_job_ref(node_id=node_id, tool=tool, field=field, refs=refs)


def _raise_invalid_job_ref(
    *,
    node_id: str,
    tool: str,
    field: str,
    refs: list[str],
) -> None:
    raise InvalidParamsError(
        "agent workflow uses a job-id reference where tool outputs are required",
        details={
            "node_id": node_id,
            "tool": tool,
            "field": field,
            "refs": refs,
            "hint": (
                "Use ${outputs.node.field} for result objects/lists; reserve "
                "${jobs.node} for inputs that explicitly expect job IDs."
            ),
        },
    )


def _schema_accepts_job_ref(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "string" or (
        isinstance(schema_type, list) and "string" in schema_type
    ):
        return True
    if schema_type == "array":
        items = schema.get("items")
        return isinstance(items, dict) and _schema_accepts_job_ref(items)
    for key in ("anyOf", "oneOf"):
        options = schema.get(key)
        if isinstance(options, list) and any(
            isinstance(option, dict) and _schema_accepts_job_ref(option)
            for option in options
        ):
            return True
    return False


def _validate_runtime_param_types(
    workflow: Workflow,
    runtime_params: dict[str, Any],
) -> None:
    for name, value in runtime_params.items():
        spec = workflow.params.get(name)
        if spec is None or value is None:
            continue
        expected = spec.type
        ok = (
            (expected in {"str", "path"} and isinstance(value, str))
            or (expected == "int" and isinstance(value, int) and not isinstance(value, bool))
            or (
                expected == "float"
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
            or (expected == "bool" and isinstance(value, bool))
            or (expected == "list" and isinstance(value, list))
            or (expected == "dict" and isinstance(value, dict))
        )
        if not ok:
            raise InvalidParamsError(
                "agent workflow runtime param has the wrong type",
                details={
                    "workflow": workflow.name,
                    "param": name,
                    "expected_type": expected,
                    "actual_type": type(value).__name__,
                    "value": value,
                },
            )


class WorkflowPlanner:
    """LLM-backed workflow plan proposal plus deterministic local validation."""

    def __init__(self, registry: ToolRegistry, provider: ChatProvider | None = None) -> None:
        self.registry = registry
        self.provider = provider
        self.registry.discover()

    async def _plan_v1(
        self,
        user_request: str,
        *,
        available_tools: list[dict[str, Any]] | None = None,
        known_inputs: dict[str, Any] | None = None,
        model: str | None = None,
        idempotency_key: str | None = None,
        max_schema_retries: int = 3,
        compile_with_fixup: PlannerIRCompiler | None = None,
        prior_round_context: "PriorRoundContext | None" = None,
    ) -> dict[str, Any]:
        """
        Flag-on Planner IR path. Returns executor IR ready for validation/execution.
        """
        if self.provider is None:
            raise InvalidParamsError(
                "agent planning requires an LLM provider",
                details={"mode": "agent_plan", "planner_ir": "v1"},
            )
        available_tool_schemas = self._available_planner_tool_schemas(available_tools)
        known = _clean_known_inputs(known_inputs)
        selection_prompt = _tool_selection_prompt(
            user_request=user_request,
            available_tools=available_tool_schemas,
            known_inputs=known,
            prior_round_context=_prior_round_context_payload(prior_round_context),
        )
        selection = await self._select_candidate_tools(
            selection_prompt,
            label="planner tool selection",
            model=model,
            idempotency_key=_phase_idempotency_key(idempotency_key, "tool-selection"),
        )
        selection = await self._repair_tool_selection_modes_if_needed(
            selection=selection,
            available_tools=available_tool_schemas,
            model=model,
            idempotency_key=_phase_idempotency_key(idempotency_key, "tool-selection-mode-repair"),
        )
        selected_tools = self._selected_full_tool_schemas(
            selection,
            available_tool_schemas,
        )
        tool_catalog = self._planner_ir_tool_catalog(selected_tools)
        retry_context: dict[str, Any] | None = None
        last_violation: PlannerIRSchemaViolation | None = None
        total_attempts = max(1, max_schema_retries + 1)
        llm_calls = 1
        max_total_llm_calls = max(3, max_schema_retries + 3)
        verification_repairs = 0
        deterministic_repairs = 0

        for attempt in range(1, total_attempts + 1):
            if llm_calls >= max_total_llm_calls:
                break
            prompt = _planner_ir_prompt(
                user_request=user_request,
                tool_catalog=tool_catalog,
                known_inputs=known,
                retry_context=retry_context,
                prior_round_context=_prior_round_context_payload(prior_round_context),
            )
            response = await self.provider.chat(
                [
                    ChatMessage(role="system", content=PLANNER_IR_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
                ],
                model=model,
                response_format={"type": "json_object"},
                idempotency_key=_phase_idempotency_key(
                    idempotency_key,
                    f"planner-ir-v1-{attempt}",
                ),
            )
            llm_calls += 1
            try:
                planner_ir = _load_planner_ir_json_or_raise(response.content)
                _validate_planner_ir_v1_or_raise(planner_ir)
                planner_ir = _prepare_planner_ir_for_verification(
                    planner_ir,
                    selected_tools,
                    known_inputs=known,
                )
                _validate_planner_ir_against_selected_tools(
                    planner_ir,
                    selected_tools,
                    known_inputs=known,
                )
            except PlannerIRSchemaViolation as exc:
                last_violation = exc
                if deterministic_repairs < 1:
                    repaired_ir = _repair_unsatisfied_root_with_selected_term_producer(
                        planner_ir=exc.payload,
                        selected_tools=selected_tools,
                        user_request=user_request,
                        known_inputs=known,
                    )
                    if repaired_ir is not None:
                        deterministic_repairs += 1
                        try:
                            repaired_ir = _prepare_planner_ir_for_verification(
                                repaired_ir,
                                selected_tools,
                                known_inputs=known,
                            )
                            _validate_planner_ir_v1_or_raise(repaired_ir)
                            _validate_planner_ir_against_selected_tools(
                                repaired_ir,
                                selected_tools,
                                known_inputs=known,
                            )
                        except PlannerIRSchemaViolation as repair_exc:
                            last_violation = repair_exc
                        else:
                            if compile_with_fixup is not None:
                                return await compile_with_fixup(repaired_ir)
                            return compile_planner_ir(
                                repaired_ir,
                                prior_round_context=prior_round_context,
                            )
                if (
                    verification_repairs < _MAX_PLAN_VERIFICATION_REPAIRS
                    and llm_calls < max_total_llm_calls
                    and _should_attempt_plan_verification_repair(exc)
                ):
                    patch = await self._ask_plan_verification_repair(
                        planner_ir=exc.payload,
                        selected_tools=selected_tools,
                        available_tools=available_tool_schemas,
                        structured_error=exc.structured_error,
                        model=model,
                        idempotency_key=_phase_idempotency_key(
                            idempotency_key,
                            f"plan-verification-{attempt}",
                        ),
                    )
                    llm_calls += 1
                    planner_ir, selected_tools = _apply_plan_verification_patch(
                        planner_ir=exc.payload,
                        selected_tools=selected_tools,
                        available_tools=available_tool_schemas,
                        patch=patch,
                    )
                    tool_catalog = self._planner_ir_tool_catalog(selected_tools)
                    verification_repairs += 1
                    try:
                        planner_ir = _prepare_planner_ir_for_verification(
                            planner_ir,
                            selected_tools,
                            known_inputs=known,
                        )
                        _validate_planner_ir_v1_or_raise(planner_ir)
                        _validate_planner_ir_against_selected_tools(
                            planner_ir,
                            selected_tools,
                            known_inputs=known,
                        )
                    except PlannerIRSchemaViolation as repair_exc:
                        last_violation = repair_exc
                    else:
                        if compile_with_fixup is not None:
                            return await compile_with_fixup(planner_ir)
                        return compile_planner_ir(
                            planner_ir,
                            prior_round_context=prior_round_context,
                        )
                retry_context = {
                    "previous_output": response.content,
                    "structured_error": last_violation.structured_error,
                    "instruction": (
                        "Re-emit only corrected Planner IR JSON. Preserve the "
                        "same planning goal unless the structured error says a "
                        "field is invalid."
                    ),
                }
                continue
            if compile_with_fixup is not None:
                return await compile_with_fixup(planner_ir)
            return compile_planner_ir(
                planner_ir,
                prior_round_context=prior_round_context,
            )

        if last_violation is not None:
            raise last_violation
        raise InvalidParamsError(
            "planner_ir_v1 planning exhausted without a response",
            details={
                "max_schema_retries": max_schema_retries,
                "max_total_llm_calls": max_total_llm_calls,
            },
        )

    def _planner_ir_tool_catalog(
        self,
        available_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if available_tools is None:
            available_tools = self._available_planner_tool_schemas(None)
        catalog: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tool in available_tools:
            name = tool.get("name")
            if not isinstance(name, str) or not name or name in seen:
                continue
            seen.add(name)
            entry: dict[str, Any] = {
                "name": name,
                "description": str(tool.get("description") or "")[:180],
                "role": tool_role(name),
                "root_inputs": root_input_fields(tool.get("input_schema")),
                "input_schema_outline": compact_input_schema_outline(
                    tool.get("input_schema")
                ),
                "candidate_contract": compact_tool_outline(tool).get(
                    "candidate_contract",
                    {},
                ),
            }
            allowed_modes = EXPLICIT_MODE_TOOLS.get(name)
            if allowed_modes is not None:
                entry["allowed_modes"] = list(allowed_modes)
                mode_input_field = explicit_mode_input_field(name)
                if mode_input_field != "mode":
                    entry["mode_input_field"] = mode_input_field
            profiles = candidate_capability_profiles(name)
            if profiles:
                entry["candidate_capability_profiles"] = profiles
            selected_mode = tool.get("selected_mode")
            if isinstance(selected_mode, str) and selected_mode:
                entry["selected_mode"] = selected_mode
            catalog.append(entry)
        return sorted(catalog, key=lambda item: item["name"])

    def _available_planner_tool_schemas(
        self,
        available_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if available_tools is None:
            available_tools = [
                {
                    "name": descriptor.name,
                    "description": descriptor.description,
                    "input_schema": resolve_input_schema_dict(descriptor),
                    "sharding": (
                        asdict(descriptor.sharding)
                        if is_dataclass(descriptor.sharding)
                        else (
                            descriptor.sharding.model_dump(mode="json")
                            if descriptor.sharding is not None
                            else None
                        )
                    ),
                }
                for descriptor, adapter_cls in self.registry.entries()
                if is_agent_visible_tool(descriptor)
                and _install_readiness(descriptor.name).get("install_ready") is True
                and _descriptor_availability(descriptor, adapter_cls).get("available")
                is True
            ]
        return [
            self._tool_with_resolved_input_schema(tool)
            for tool in available_tools
        ]

    async def draft_plan(
        self,
        request: str,
        *,
        available_tools: list[dict[str, Any]],
        workflows: list[dict[str, Any]] | None = None,
        known_inputs: dict[str, Any] | None = None,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del workflows
        return await self._plan_v1(
            request,
            available_tools=available_tools,
            known_inputs=known_inputs,
            model=model,
            idempotency_key=idempotency_key,
        )

    async def _select_candidate_tools(
        self,
        prompt: dict[str, Any],
        *,
        label: str,
        model: str | None,
        idempotency_key: str | None,
        system_prompt: str = TOOL_SELECTION_SYSTEM_PROMPT,
    ) -> dict[str, Any]:
        if self.provider is None:  # pragma: no cover - guarded by callers
            raise InvalidParamsError(
                "agent planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=idempotency_key,
        )
        try:
            return _load_json_object_response(
                response.content,
                label=label,
                error_message=f"{label} returned non-JSON content",
            )
        except InvalidParamsError:
            selection = _fallback_tool_selection(prompt, response.content)
            if selection is None:
                raise
            return selection

    async def _ask_plan_verification_repair(
        self,
        *,
        planner_ir: dict[str, Any],
        selected_tools: list[dict[str, Any]],
        available_tools: list[dict[str, Any]],
        structured_error: dict[str, Any],
        model: str | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        if self.provider is None:  # pragma: no cover - guarded by callers
            raise InvalidParamsError(
                "agent planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        prompt = {
            "planner_ir": planner_ir,
            "selected_tools": [
                {"name": tool.get("name")}
                for tool in selected_tools
                if isinstance(tool.get("name"), str)
            ],
            "available_tools": [
                compact_tool_outline(tool)
                for tool in available_tools
            ],
            "structured_error": structured_error,
            "output_contract": {
                "add_selected_tools": ["tool_name"],
                "drop_selected_tools": ["tool_name"],
                "set_selected_modes": [
                    {"name": "tool_name", "selected_mode": "allowed_mode_or_null"}
                ],
                "add_nodes": [{"id": "node_id", "tool": "tool_name", "intent": "...", "inputs": {}, "upstream": []}],
                "replace_nodes": [{"id": "node_id", "tool": "tool_name", "intent": "...", "inputs": {}, "upstream": []}],
                "update_node_inputs": [{"node_id": "node_id", "inputs": {"field": "value"}}],
                "set_missing_inputs": [],
                "remove_node_ids": ["node_id"],
                "reason": "short reason",
            },
        }
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=PLAN_VERIFICATION_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            idempotency_key=idempotency_key,
        )
        return _load_json_object_response(
            response.content,
            label="plan verification repair",
            error_message="plan verification repair returned non-JSON content",
        )

    async def _repair_tool_selection_modes_if_needed(
        self,
        *,
        selection: dict[str, Any],
        available_tools: list[dict[str, Any]],
        model: str | None,
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        structured_error = _tool_selection_mode_structured_error(
            selection,
            available_tools,
        )
        if structured_error is None:
            return selection
        patch = await self._ask_plan_verification_repair(
            planner_ir={},
            selected_tools=[
                {"name": spec["name"], **({"selected_mode": spec["selected_mode"]} if "selected_mode" in spec else {})}
                for spec in _candidate_tool_specs(selection)
            ],
            available_tools=available_tools,
            structured_error=structured_error,
            model=model,
            idempotency_key=idempotency_key,
        )
        _empty_ir, repaired_tools = _apply_plan_verification_patch(
            planner_ir={},
            selected_tools=[
                {"name": spec["name"], **({"selected_mode": spec["selected_mode"]} if "selected_mode" in spec else {})}
                for spec in _candidate_tool_specs(selection)
            ],
            available_tools=available_tools,
            patch=patch,
        )
        repaired = {
            **selection,
            "selected_tools": [
                {
                    "name": str(tool.get("name")),
                    **(
                        {"selected_mode": str(tool.get("selected_mode"))}
                        if isinstance(tool.get("selected_mode"), str)
                        else {}
                    ),
                }
                for tool in repaired_tools
                if isinstance(tool.get("name"), str)
            ],
        }
        remaining_error = _tool_selection_mode_structured_error(
            repaired,
            available_tools,
        )
        if remaining_error is not None:
            raise PlannerIRSchemaViolation(
                [
                    PlannerIRViolation(
                        path=str(item.get("path") or "/selected_tools"),
                        issue=str(item.get("issue") or "invalid_selected_mode"),
                        expected=str(item.get("expected") or item.get("repair_instruction") or ""),
                        metadata={
                            key: value
                            for key, value in item.items()
                            if key not in {"path", "issue", "expected"}
                        },
                    )
                    for item in remaining_error.get("violations", [])
                    if isinstance(item, dict)
                ],
                payload=repaired,
            )
        return repaired

    def _selected_tool_schemas(
        self,
        selection: dict[str, Any],
        available_tools: list[dict[str, Any]],
        *,
        fallback_available_tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            compact_tool_schema(self._tool_with_resolved_input_schema(tool))
            for tool in self._selected_full_tool_schemas(
                selection,
                available_tools,
                fallback_available_tools=fallback_available_tools,
            )
        ]

    def _selected_full_tool_schemas(
        self,
        selection: dict[str, Any],
        available_tools: list[dict[str, Any]],
        *,
        fallback_available_tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        by_name = {
            str(tool.get("name")): tool
            for tool in available_tools
            if isinstance(tool.get("name"), str)
        }
        specs = _candidate_tool_specs(selection)
        if not specs:
            raise InvalidParamsError(
                "planner did not select any candidate tools",
                details={
                    "structured_error": {
                        "error_kind": "empty_tool_selection",
                        "violations": [
                            {
                                "path": "/selected_tools",
                                "issue": "empty_tool_selection",
                                "expected": "Select at least one available tool before Planner IR is called.",
                            }
                        ],
                    },
                    "selection": selection,
                },
            )
        names = [spec["name"] for spec in specs]
        unknown = [name for name in names if name not in by_name]
        if unknown and fallback_available_tools is not None:
            fallback_by_name = {
                str(tool.get("name")): tool
                for tool in fallback_available_tools
                if isinstance(tool.get("name"), str)
            }
            for name in list(unknown):
                if name in fallback_by_name:
                    by_name[name] = fallback_by_name[name]
                    unknown.remove(name)
        if unknown:
            raise InvalidParamsError(
                "planner selected unknown tools",
                details={"tools": unknown, "available_tools": sorted(by_name)},
            )
        selected: list[dict[str, Any]] = []
        for spec in specs:
            name = spec["name"]
            tool = self._tool_with_resolved_input_schema(by_name[name])
            selected_mode = spec.get("selected_mode")
            allowed_modes = EXPLICIT_MODE_TOOLS.get(name)
            if selected_mode is not None:
                if allowed_modes is None:
                    raise InvalidParamsError(
                        "planner selected a mode for a single-mode tool",
                        details={
                            "tool": name,
                            "selected_mode": selected_mode,
                            "hint": (
                                "Use selected_mode only when the available tool "
                                "outline includes allowed_modes."
                            ),
                        },
                    )
                if selected_mode not in allowed_modes:
                    raise InvalidParamsError(
                        "planner selected an invalid tool mode",
                        details={
                            "tool": name,
                            "selected_mode": selected_mode,
                            "allowed_modes": list(allowed_modes),
                        },
                    )
                tool = {**tool, "selected_mode": selected_mode}
            selected.append(tool)
        return selected

    def _tool_with_resolved_input_schema(self, tool: dict[str, Any]) -> dict[str, Any]:
        schema = tool.get("input_schema")
        name = tool.get("name")
        if not isinstance(name, str):
            return {**tool, "input_schema": {}}
        descriptor, _adapter_cls = self.registry.get(name, "*")
        input_schema = schema if isinstance(schema, dict) else resolve_input_schema_dict(descriptor)
        return {
            **tool,
            "input_schema": input_schema,
            "output_schema": resolve_output_schema_dict(descriptor),
        }

    def validate_plan(
        self,
        plan: dict[str, Any],
        *,
        allow_internal_tools: bool = False,
    ) -> dict[str, Any]:
        allow_internal_tools = allow_internal_tools or plan.get("source_kind") == "workflow_yaml"
        if not isinstance(plan, dict):
            raise InvalidParamsError(
                "agent workflow plan must be an object",
                details={"got_type": type(plan).__name__},
            )
        if "workflow_ref" in plan or "workflow" in plan:
            raise InvalidParamsError(
                "LLM agent plans must include an explicit workflow_plan, not workflow_ref",
                details={"fields": [key for key in ("workflow_ref", "workflow") if key in plan]},
            )
        if plan.get("mode") not in (None, "agent_plan"):
            raise InvalidParamsError(
                "planner returned an unsupported mode for workflow planning",
                details={"mode": plan.get("mode")},
            )
        workflow_plan = plan.get("workflow_plan")
        if not isinstance(workflow_plan, dict):
            raise InvalidParamsError(
                "agent plan missing workflow_plan object",
                details={"field": "workflow_plan"},
            )
        nodes = workflow_plan.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise InvalidParamsError(
                "agent workflow_plan.nodes must contain at least one executable node",
                details={"field": "workflow_plan.nodes"},
            )

        for tool_name in tool_names_from_plan(plan):
            descriptor, adapter_cls = self.registry.get(tool_name, "*")
            if not allow_internal_tools and not is_agent_visible_tool(descriptor):
                raise InvalidParamsError(
                    "agent plan requested an internal tool",
                    details={
                        "tool": tool_name,
                        "reason": (
                            "internal/glue tools are not part of the LLM agent "
                            "first-class tool surface"
                        ),
                    },
                )
            availability = _descriptor_availability(descriptor, adapter_cls)
            install_status = _install_readiness(tool_name)
            if (
                availability.get("available") is not True
                or install_status.get("install_ready") is not True
            ):
                raise InvalidParamsError(
                    "agent plan requested an unavailable tool",
                    details={
                        "tool": tool_name,
                        "availability": availability,
                        "install_readiness": install_status,
                    },
                )

        workflow = self.compile_workflow(plan)
        workflow.validate_dag()
        if plan.get("source_kind") == PLANNER_COMPILED_SOURCE_KIND:
            validate_compiled_planner_plan(plan)
        else:
            _validate_static_output_refs(workflow, self.registry)
            _validate_static_output_ref_types(workflow, self.registry)
            _validate_static_job_refs(workflow, self.registry)
        for node in workflow.nodes:
            if node.tool != "rfdiffusion":
                continue
            has_runtime_ref = any(
                ref.startswith(("params.", "outputs.", "item"))
                for ref in iter_refs(node.inputs)
            )
            if has_runtime_ref:
                continue
            mode_status = _rfdiffusion_mode_readiness(node.inputs)
            if mode_status.get("ready") is not True:
                raise InvalidParamsError(
                    "agent plan requested an unavailable RFdiffusion mode",
                    details={"tool": node.tool, "mode_readiness": mode_status},
                )
        plan["workflow_plan"] = workflow.model_dump(mode="json", exclude_none=True)
        return plan

    def validate_round_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        valid = self.validate_plan(plan)
        _reject_post_round_action_tools(valid)
        return valid

    def validate_invocation(
        self,
        plan: dict[str, Any],
        params: dict[str, Any],
        *,
        allow_internal_tools: bool = False,
    ) -> None:
        workflow = self.compile_workflow(
            self.validate_plan(dict(plan), allow_internal_tools=allow_internal_tools)
        )
        runtime_params = self.invocation_params(plan, params, workflow=workflow)
        _validate_runtime_param_types(workflow, runtime_params)
        merged = params_dict_from_workflow(workflow, runtime_params)
        scope = {"params": merged, "outputs": {}}
        for node in workflow.nodes:
            refs = iter_refs(node.inputs)
            if any(ref.startswith(("outputs.", "jobs.", "item")) for ref in refs):
                continue
            try:
                resolved = resolve(node.inputs, scope)
            except VPAError:
                raise
            if not isinstance(resolved, dict):
                raise InvalidParamsError(
                    "agent workflow node inputs resolved to a non-object",
                    details={"node_id": node.id, "type": type(resolved).__name__},
                )
            descriptor, _ = self.registry.get(node.tool, node.version)
            resolved = drop_optional_none_params(descriptor, resolved)
            validate_params(descriptor, resolved)

    def invocation_params(
        self,
        plan: dict[str, Any],
        params: dict[str, Any],
        *,
        workflow: Workflow | None = None,
    ) -> dict[str, Any]:
        if workflow is None:
            try:
                workflow = self.compile_workflow(plan)
            except VPAError:
                workflow = None
        return _invocation_params_from_plan(plan, params, workflow)

    def compile_workflow(self, plan: dict[str, Any]) -> Workflow:
        workflow_plan = plan.get("workflow_plan")
        if not isinstance(workflow_plan, dict):
            raise InvalidParamsError(
                "agent plan missing workflow_plan object",
                details={"field": "workflow_plan"},
            )
        workflow_plan = _sanitize_workflow_plan_shape(workflow_plan)
        workflow_plan = _normalize_workflow_param_types(workflow_plan)
        try:
            return Workflow.model_validate(workflow_plan)
        except ValidationError as exc:
            raise InvalidParamsError(
                "agent workflow plan failed local validation",
                details={"errors": exc.errors()},
            ) from exc

    def compile_workflow_yaml(self, plan: dict[str, Any]) -> str:
        return serialize_workflow(self.compile_workflow(plan))
