from __future__ import annotations

from typing import Any


def conditional_required_summary(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize common JSON-Schema conditional required sets for agents."""
    if not isinstance(schema, dict):
        return []
    out: list[dict[str, Any]] = []
    for clause in schema.get("allOf") or []:
        if not isinstance(clause, dict):
            continue
        condition = _const_condition(clause.get("if"))
        then = clause.get("then")
        if not condition or not isinstance(then, dict):
            continue
        entry: dict[str, Any] = {"when": condition}
        required = _required_list(then)
        if required:
            entry["required"] = required
        alternatives = _required_alternatives(then)
        if alternatives:
            entry["one_of_required"] = alternatives
        if "required" in entry or "one_of_required" in entry:
            out.append(entry)
    return out


def matching_conditional_required_hints(
    schema: dict[str, Any],
    params: dict[str, Any],
) -> list[str]:
    hints: list[str] = []
    for entry in conditional_required_summary(schema):
        when = entry.get("when")
        if not isinstance(when, dict):
            continue
        if not all(params.get(k) == v for k, v in when.items()):
            continue
        pieces: list[str] = []
        required = entry.get("required")
        if isinstance(required, list) and required:
            pieces.append("required: " + ", ".join(str(v) for v in required))
        alternatives = entry.get("one_of_required")
        if isinstance(alternatives, list) and alternatives:
            alt_text = [
                " + ".join(str(v) for v in alt)
                for alt in alternatives
                if isinstance(alt, list)
            ]
            if alt_text:
                pieces.append("one of: " + " OR ".join(alt_text))
        if pieces:
            cond = ", ".join(f"{k}={v!r}" for k, v in when.items())
            hints.append(f"When {cond}, " + "; ".join(pieces) + ".")
    return hints


def _const_condition(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in props.items():
        if isinstance(value, dict) and "const" in value:
            out[str(key)] = value["const"]
    return out


def _required_list(schema: dict[str, Any]) -> list[str]:
    required = schema.get("required")
    if not isinstance(required, list):
        return []
    return [str(v) for v in required if isinstance(v, str)]


def _required_alternatives(schema: dict[str, Any]) -> list[list[str]]:
    out: list[list[str]] = []
    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            required = _required_list(variant)
            if required:
                out.append(required)
    return out
