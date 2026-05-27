from __future__ import annotations

from typing import Any

from vibeproteinagent.agents.planning.canonical_tools import (
    CANONICAL_CONSUMERS,
    CANONICAL_PRODUCERS,
    CANONICAL_TERM_CONSUMERS,
    CANONICAL_TERM_PRODUCERS,
    EXPLICIT_MODE_TOOLS,
    candidate_capability_profiles,
    candidate_contract,
    explicit_mode_input_field,
)

AGENT_TOOL_SCHEMA_VERSION = "vpa-agent-tools-v7"

_OUTLINE_DESCRIPTION_LIMIT = 120
_PARAMETER_DESCRIPTION_LIMIT = 80
_MAX_COMPACT_PARAMETERS = 24


def compact_tool_outline(tool: dict[str, Any]) -> dict[str, Any]:
    name = str(tool.get("name") or "")
    out = {
        "name": tool.get("name"),
        "description": _short_text(tool.get("description"), _OUTLINE_DESCRIPTION_LIMIT),
        "role": tool_role(name),
        "candidate_contract": _compact_candidate_contract(name),
        "root_inputs": root_input_fields(tool.get("input_schema")),
        "requires_pdb_file": requires_pdb_file(tool.get("input_schema")),
    }
    allowed_modes = EXPLICIT_MODE_TOOLS.get(name)
    if allowed_modes is not None:
        out["allowed_modes"] = list(allowed_modes)
        mode_input_field = explicit_mode_input_field(name)
        if mode_input_field != "mode":
            out["mode_input_field"] = mode_input_field
    profiles = candidate_capability_profiles(name)
    if profiles:
        out["candidate_capability_profiles"] = profiles
    sharding = _compact_sharding(tool.get("sharding"))
    if sharding:
        out["sharding"] = sharding
    return out


def compact_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    name = str(tool.get("name") or "")
    out = {
        "name": tool.get("name"),
        "description": _short_text(tool.get("description"), _OUTLINE_DESCRIPTION_LIMIT),
        "parameters": _compact_parameters(tool.get("input_schema")),
        "role": tool_role(name),
        "candidate_contract": _compact_candidate_contract(name),
        "root_inputs": root_input_fields(tool.get("input_schema")),
        "requires_pdb_file": requires_pdb_file(tool.get("input_schema")),
    }
    allowed_modes = EXPLICIT_MODE_TOOLS.get(name)
    if allowed_modes is not None:
        out["allowed_modes"] = list(allowed_modes)
        mode_input_field = explicit_mode_input_field(name)
        if mode_input_field != "mode":
            out["mode_input_field"] = mode_input_field
    profiles = candidate_capability_profiles(name)
    if profiles:
        out["candidate_capability_profiles"] = profiles
    sharding = _compact_sharding(tool.get("sharding"))
    if sharding:
        out["sharding"] = sharding
    return out


def compact_input_schema_outline(schema: Any) -> dict[str, Any]:
    """Return the planner-facing top-level input contract for one tool.

    This intentionally preserves field names while trimming verbose JSON Schema
    metadata. The Planner IR prompt needs to know which inputs exist; it does
    not need a full recursive schema dump for every selected tool.
    """

    return _compact_parameters(schema)


def tool_role(tool_name: str) -> str:
    is_producer = tool_name in CANONICAL_PRODUCERS or tool_name in CANONICAL_TERM_PRODUCERS
    is_consumer = tool_name in CANONICAL_CONSUMERS or tool_name in CANONICAL_TERM_CONSUMERS
    if is_producer and is_consumer:
        return "both"
    if is_producer:
        return "producer"
    if is_consumer:
        return "consumer"
    return "tool"


def _compact_candidate_contract(tool_name: str) -> dict[str, Any]:
    contract = candidate_contract(tool_name)
    out: dict[str, Any] = {}
    if contract.consumes:
        out["consumes"] = sorted(contract.consumes)
    if contract.produces:
        out["produces"] = sorted(contract.produces)
    if contract.consumes or contract.produces:
        out["preserves_upstream"] = contract.preserves_upstream
    if contract.min_candidate_items > 1:
        out["min_candidate_items"] = contract.min_candidate_items
    return out


_CANONICAL_CANDIDATE_FIELDS = {
    "candidates",
    "predictions",
    "analyses",
    "qc_results",
}

_NON_ROOT_INPUT_FIELDS = {"mode"}

_ROOT_INPUT_FIELD_NAMES = {
    "go_id",
    "interpro_id",
    "query",
    "sequence",
    "sequences",
    "fasta_path",
    "target_pdb",
    "pdb_path",
    "structure_path",
    "target_yaml",
    "input_json",
    "target_sequence",
    "target_fasta_path",
    "motif_pdb",
    "motif_sequence",
    "motif_fasta_path",
    "ligand_smiles",
}

_PDB_FIELD_MARKERS = ("pdb", "structure_path", "input_json")


def root_input_fields(schema: Any) -> list[str]:
    """Return compact external inputs that can start a workflow node.

    This is intentionally schema-derived and generic: required fields are
    included, plus well-known top-level literal inputs such as `target_pdb`,
    `sequence`, and `go_id`. Canonical candidate arrays are excluded because
    those are compiler-wired from upstream nodes rather than user-provided
    root inputs.
    """

    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    names: list[str] = []
    for field in _required_field_options(schema):
        if (
            field in properties
            and field not in _CANONICAL_CANDIDATE_FIELDS
            and field not in _NON_ROOT_INPUT_FIELDS
        ):
            names.append(field)
    for field in properties:
        field_name = str(field)
        if (
            field_name in _ROOT_INPUT_FIELD_NAMES
            and field_name not in _CANONICAL_CANDIDATE_FIELDS
            and field_name not in _NON_ROOT_INPUT_FIELDS
            and field_name not in names
        ):
            names.append(field_name)
    return names[:12]


def requires_pdb_file(schema: Any) -> bool:
    for field in root_input_fields(schema):
        lowered = field.lower()
        if any(marker in lowered for marker in _PDB_FIELD_MARKERS):
            return True
    return False


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
            variant_required = variant.get("required")
            if not isinstance(variant_required, list):
                continue
            names.extend(
                str(item) for item in variant_required if isinstance(item, str)
            )
    return sorted(set(names))


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
