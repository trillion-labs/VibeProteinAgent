from __future__ import annotations

from typing import Any

from vibeproteinagent.agents.planning.canonical_tools import EXPLICIT_MODE_TOOLS

AGENT_TOOL_SCHEMA_VERSION = "vpa-agent-tools-v6"

_OUTLINE_DESCRIPTION_LIMIT = 120
_PARAMETER_DESCRIPTION_LIMIT = 80
_MAX_COMPACT_PARAMETERS = 24


def compact_tool_outline(tool: dict[str, Any]) -> dict[str, Any]:
    out = {
        "name": tool.get("name"),
        "description": _short_text(tool.get("description"), _OUTLINE_DESCRIPTION_LIMIT),
    }
    sharding = _compact_sharding(tool.get("sharding"))
    if sharding:
        out["sharding"] = sharding
    return out


def compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    out = {
        "name": tool.get("name"),
        "description": _short_text(tool.get("description"), _OUTLINE_DESCRIPTION_LIMIT),
        "parameters": _compact_parameters(tool.get("input_schema")),
    }
    allowed_modes = EXPLICIT_MODE_TOOLS.get(str(tool.get("name") or ""))
    if allowed_modes is not None:
        out["allowed_modes"] = list(allowed_modes)
    sharding = _compact_sharding(tool.get("sharding"))
    if sharding:
        out["sharding"] = sharding
    return out


def _compact_sharding(sharding: Any) -> dict[str, Any]:
    if not isinstance(sharding, dict):
        return {}
    return {
        key: sharding[key]
        for key in (
            "strategy",
            "shardable_fields",
            "recommended_shard_size",
            "default_max_parallel",
        )
        if key in sharding
    }


def _compact_parameters(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"required": [], "properties": {}}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    ordered_names = [
        str(name)
        for name in required
        if isinstance(name, str) and isinstance(properties.get(name), dict)
    ]
    ordered_names.extend(
        str(name)
        for name, prop in properties.items()
        if isinstance(prop, dict) and str(name) not in ordered_names
    )
    compact_names = ordered_names[:_MAX_COMPACT_PARAMETERS]
    return {
        "required": required,
        "properties": {
            name: _compact_parameter_schema(properties[name])
            for name in compact_names
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
    if "description" in out:
        out["description"] = _short_text(out["description"], _PARAMETER_DESCRIPTION_LIMIT)
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


def _short_text(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def tool_names_from_plan(plan: dict[str, Any]) -> list[str]:
    workflow_plan = plan.get("workflow_plan") or {}
    nodes = workflow_plan.get("nodes") or []
    names = [n.get("tool") for n in nodes if isinstance(n, dict) and n.get("tool")]
    names.extend(plan.get("warm_tools") or [])
    names.extend(plan.get("required_tools") or [])
    return sorted(set(str(n) for n in names if n))
