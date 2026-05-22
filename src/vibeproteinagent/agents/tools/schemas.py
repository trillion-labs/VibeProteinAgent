from __future__ import annotations

from typing import Any

AGENT_TOOL_SCHEMA_VERSION = "vpa-agent-tools-v2"


def compact_tool_outline(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool.get("name"),
        "description": tool.get("description"),
    }


def compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool.get("name"),
        "description": tool.get("description"),
        "parameters": _compact_parameters(tool.get("input_schema")),
    }


def _compact_parameters(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"required": [], "properties": {}}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    return {
        "required": required,
        "properties": {
            str(name): _compact_parameter_schema(prop)
            for name, prop in properties.items()
            if isinstance(prop, dict)
        },
    }


def _compact_parameter_schema(prop: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "type",
        "description",
        "enum",
        "default",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "oneOf",
        "anyOf",
    )
    out = {key: prop[key] for key in keep if key in prop}
    items = prop.get("items")
    if isinstance(items, dict):
        out["items"] = _compact_parameter_schema(items)
        nested = items.get("properties")
        if isinstance(nested, dict):
            out["items"]["properties"] = {
                str(name): _compact_parameter_schema(value)
                for name, value in nested.items()
                if isinstance(value, dict)
            }
    return out


def tool_names_from_plan(plan: dict[str, Any]) -> list[str]:
    workflow_plan = plan.get("workflow_plan") or {}
    nodes = workflow_plan.get("nodes") or []
    names = [n.get("tool") for n in nodes if isinstance(n, dict) and n.get("tool")]
    names.extend(plan.get("warm_tools") or [])
    names.extend(plan.get("required_tools") or [])
    return sorted(set(str(n) for n in names if n))
