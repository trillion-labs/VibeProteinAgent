from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, is_dataclass
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
    EXPLICIT_MODE_TOOLS,
    build_canonical_wiring_for_planner,
)
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.planning.ir import (
    planner_ir_schema,
    validate_planner_ir_v1,
)
from vibeproteinagent.agents.provider import ChatMessage
from vibeproteinagent.agents.provider.types import ChatProvider
from vibeproteinagent.agents.tools.schemas import (
    compact_tool_outline,
    compact_tool_schema,
    tool_names_from_plan,
)
from vibeproteinagent.core.errors import ErrorCode, InvalidParamsError, VPAError
from vibeproteinagent.workflows import Workflow, serialize_workflow
from vibeproteinagent.workflows.planner import params_dict_from_workflow
from vibeproteinagent.workflows.resolver import iter_refs, resolve

from .prompts import (
    NEXT_ROUND_SYSTEM_PROMPT,
    PLANNER_IR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    TOOL_SELECTION_SYSTEM_PROMPT,
)

if TYPE_CHECKING:
    from vibeproteinagent.agents.execution.round_state import PriorRoundContext

_MAX_SELECTED_TOOLS = 6
PlannerIRCompiler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class PlannerIRViolation:
    path: str
    issue: str
    expected: str


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
                }
                for item in violations
            ],
            "fix_instruction": (
                "Re-emit the Planner IR JSON. Use version planner_ir_v1, "
                "semantic nodes, literal scalar inputs, and upstream entries "
                "whose relationship is candidate_source. Do not emit executor "
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


def _candidate_tool_names(selection: dict[str, Any]) -> list[str]:
    names = selection.get("candidate_tools")
    if not isinstance(names, list):
        names = tool_names_from_plan(selection)
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not isinstance(name, str) or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out[:_MAX_SELECTED_TOOLS]


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
        error.validator == "const"
        and path.endswith("/relationship")
    ):
        return PlannerIRViolation(
            path=path,
            issue="relationship_not_allowed_in_v1",
            expected='relationship must equal "candidate_source" in planner_ir_v1',
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
    tool_catalog: list[dict[str, str]],
    retry_context: dict[str, Any] | None,
    prior_round_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt: dict[str, Any] = {}
    if prior_round_context is not None:
        prompt["prior_round_context"] = prior_round_context
    prompt.update({
        "request": user_request,
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
                            "relationship": "candidate_source",
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
        "candidate_tools": names,
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
        tool_catalog = self._planner_ir_tool_catalog(available_tools)
        retry_context: dict[str, Any] | None = None
        last_violation: PlannerIRSchemaViolation | None = None
        total_attempts = max(1, max_schema_retries + 1)

        for attempt in range(1, total_attempts + 1):
            prompt = _planner_ir_prompt(
                user_request=user_request,
                tool_catalog=tool_catalog,
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
            try:
                planner_ir = _load_planner_ir_json_or_raise(response.content)
                _validate_planner_ir_v1_or_raise(planner_ir)
            except PlannerIRSchemaViolation as exc:
                last_violation = exc
                retry_context = {
                    "previous_output": response.content,
                    "structured_error": exc.structured_error,
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
            details={"max_schema_retries": max_schema_retries},
        )

    def _planner_ir_tool_catalog(
        self,
        available_tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if available_tools is None:
            available_tools = [
                {
                    "name": descriptor.name,
                    "description": descriptor.description,
                }
                for descriptor, adapter_cls in self.registry.entries()
                if is_agent_visible_tool(descriptor)
                and _install_readiness(descriptor.name).get("install_ready") is True
                and _descriptor_availability(descriptor, adapter_cls).get("available")
                is True
            ]
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
            }
            allowed_modes = EXPLICIT_MODE_TOOLS.get(name)
            if allowed_modes is not None:
                entry["allowed_modes"] = list(allowed_modes)
            catalog.append(entry)
        return sorted(catalog, key=lambda item: item["name"])

    async def draft_plan(
        self,
        request: str,
        *,
        available_tools: list[dict[str, Any]],
        workflows: list[dict[str, Any]] | None = None,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del workflows
        if self.provider is None:
            raise InvalidParamsError(
                "agent planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        available_tool_schemas = [
            self._tool_with_resolved_input_schema(tool)
            for tool in available_tools
        ]
        candidate_tools = _candidate_tool_inventory(request, available_tool_schemas)
        selection = await self._select_candidate_tools(
            {
                "request": request,
                "available_tools": [compact_tool_outline(t) for t in candidate_tools],
                "output_contract": {
                    "mode": "tool_selection",
                    "candidate_tools": ["tool_name"],
                    "max_candidate_tools": _MAX_SELECTED_TOOLS,
                    "reason": "string",
                    "missing_inputs": ["string"],
                },
            },
            label="planner tool selection",
            model=model,
            idempotency_key=_phase_idempotency_key(idempotency_key, "tool-selection"),
            system_prompt=TOOL_SELECTION_SYSTEM_PROMPT,
        )
        selected_tools = self._selected_tool_schemas(
            selection,
            candidate_tools,
        )
        prompt = {
            "request": request,
            "tool_selection": selection,
            "selected_tools": selected_tools,
            "output_contract": {
                "mode": "agent_plan",
                "goal": "string",
                "assumptions": ["string"],
                "missing_inputs": ["string"],
                "required_tools": ["tool_name"],
                "params": {"param_name": "runtime value"},
                "workflow_plan": {
                    "name": "string",
                    "version": "string",
                    "params": {"param_name": {"type": "str|int|float|bool|path|list|dict"}},
                    "nodes": [
                        {
                            "id": "node_id",
                            "tool": "tool_name",
                            "inputs": {
                                "param": "runtime value or ${params.x}/${outputs.node.field}"
                            },
                        }
                    ],
                },
                "success_criteria": {},
                "warm_tools": ["tool_name"],
                "estimated_llm_calls": 2,
            },
        }
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=_phase_idempotency_key(idempotency_key, "workflow-plan"),
        )
        plan = _load_json_object_response(
            response.content,
            label="planner",
            error_message="planner returned non-JSON content",
        )
        return self.validate_round_plan(plan)

    async def draft_next_round_plan(
        self,
        *,
        previous_plan: dict[str, Any],
        summary: dict[str, Any],
        decision: dict[str, Any],
        available_tools: list[dict[str, Any]],
        workflows: list[dict[str, Any]] | None = None,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        del workflows
        if self.provider is None:
            raise InvalidParamsError(
                "next-round planning requires an LLM provider",
                details={"mode": "agent_plan"},
            )
        available_tool_schemas = [
            self._tool_with_resolved_input_schema(tool)
            for tool in available_tools
        ]
        candidate_tools = _candidate_tool_inventory(
            " ".join(
                str(part)
                for part in (
                    previous_plan.get("goal"),
                    decision.get("decision"),
                    (decision.get("next_round") or {}).get("reason")
                    if isinstance(decision.get("next_round"), dict)
                    else None,
                    summary.get("name") if isinstance(summary, dict) else None,
                )
                if part
            ),
            available_tool_schemas,
        )
        selection = await self._select_candidate_tools(
            {
                "previous_plan": previous_plan,
                "compact_summary": summary,
                "decision": decision,
                "available_tools": [compact_tool_outline(t) for t in candidate_tools],
                "output_contract": {
                    "mode": "tool_selection",
                    "candidate_tools": ["tool_name"],
                    "max_candidate_tools": _MAX_SELECTED_TOOLS,
                    "reason": "string",
                    "missing_inputs": [],
                },
            },
            label="next-round tool selection",
            model=model,
            idempotency_key=_phase_idempotency_key(idempotency_key, "tool-selection"),
            system_prompt=TOOL_SELECTION_SYSTEM_PROMPT,
        )
        selected_tools = self._selected_tool_schemas(
            selection,
            candidate_tools,
        )
        prompt = {
            "previous_plan": previous_plan,
            "compact_summary": summary,
            "decision": decision,
            "tool_selection": selection,
            "selected_tools": selected_tools,
            "output_contract": {
                "mode": "agent_plan",
                "goal": "string",
                "assumptions": ["include parameter changes and reasons"],
                "missing_inputs": [],
                "required_tools": ["tool_name"],
                "params": {"param_name": "runtime value"},
                "workflow_plan": {
                    "name": "string",
                    "version": "string",
                    "nodes": [{"id": "node_id", "tool": "tool_name", "inputs": {}}],
                },
                "success_criteria": previous_plan.get("success_criteria", {}),
                "warm_tools": ["tool_name"],
                "estimated_llm_calls": 1,
            },
        }
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=NEXT_ROUND_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(prompt, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=_phase_idempotency_key(idempotency_key, "workflow-plan"),
        )
        plan = _load_json_object_response(
            response.content,
            label="next-round planner",
            error_message="next-round planner returned non-JSON content",
        )
        return self.validate_round_plan(plan)

    async def _select_candidate_tools(
        self,
        prompt: dict[str, Any],
        *,
        label: str,
        model: str | None,
        idempotency_key: str | None,
        system_prompt: str = PLANNER_SYSTEM_PROMPT,
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
        names = _candidate_tool_names(selection)
        if not names:
            raise InvalidParamsError(
                "planner did not select any candidate tools",
                details={"selection": selection},
            )
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
        return [
            self._tool_with_resolved_input_schema(by_name[name])
            for name in names
        ]

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
        # Compiler-emitted Planner IR v1 plans are valid by construction; keep
        # legacy validators demoted and off the v1 execution route.
        if plan.get("source_kind") != "planner_ir_compiled":
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
