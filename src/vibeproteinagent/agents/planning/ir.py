from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import (
    Draft202012Validator,
    ValidationError,
)

from vibeproteinagent.agents.planning.terminal_tools import TERMINAL_CONTROLLER_TOOLS

_EXECUTOR_REF_PREFIXES = ("${outputs.", "${jobs.", "${item.")
_FORBIDDEN_TOKENS = ("workflow_ref", "submit_final")
_VIRTUAL_UPSTREAM_NODES = frozenset({"prior_round"})


def planner_ir_schema() -> dict[str, Any]:
    """Load the Planner IR v1 JSON Schema."""

    schema_path = (
        Path(__file__).resolve().parents[2] / "schemas" / "planner_ir_v1.json"
    )
    return json.loads(schema_path.read_text())


def validate_planner_ir_v1(ir: dict[str, Any]) -> None:
    """Validate Planner IR v1 structure plus cross-node contract rules.

    JSON Schema covers local shape. This helper enforces rules that depend on
    the whole document, such as unique node ids and upstream ordering.
    """

    Draft202012Validator(planner_ir_schema()).validate(ir)
    _reject_forbidden_tokens(ir)
    _validate_nodes(ir)


def _validate_nodes(ir: dict[str, Any]) -> None:
    seen: set[str] = set()
    for index, node in enumerate(ir.get("nodes", [])):
        node_id = node["id"]
        tool = node["tool"]
        if node_id in seen:
            raise ValidationError(f"duplicate node id: {node_id}")
        if node_id in TERMINAL_CONTROLLER_TOOLS:
            raise ValidationError(f"{node_id} is not a valid Planner IR node id")
        if tool in TERMINAL_CONTROLLER_TOOLS:
            raise ValidationError(f"{tool} is a decision action, not a Planner IR node")
        if node_id in _VIRTUAL_UPSTREAM_NODES:
            raise ValidationError(f"{node_id} is an internal upstream source, not a Planner IR node id")
        if tool in _VIRTUAL_UPSTREAM_NODES:
            raise ValidationError(f"{tool} is an internal upstream source, not a Planner IR tool")

        for upstream in node.get("upstream", []):
            upstream_id = upstream["node"]
            if upstream_id in _VIRTUAL_UPSTREAM_NODES:
                continue
            if upstream_id not in seen:
                known_later = any(
                    later.get("id") == upstream_id
                    for later in ir.get("nodes", [])[index + 1 :]
                    if isinstance(later, dict)
                )
                if known_later:
                    raise ValidationError(
                        f"upstream node {upstream_id!r} must appear before {node_id!r}"
                    )
                raise ValidationError(f"unknown upstream node: {upstream_id}")

        seen.add(node_id)


def _reject_forbidden_tokens(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_string(key, f"{path}.{key}")
            _reject_forbidden_tokens(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_tokens(child, f"{path}[{index}]")
    elif isinstance(value, str):
        _reject_string(value, path)


def _reject_string(value: str, path: str) -> None:
    for prefix in _EXECUTOR_REF_PREFIXES:
        if prefix in value:
            raise ValidationError(f"executor reference syntax {prefix!r} is not allowed at {path}")
    for token in _FORBIDDEN_TOKENS:
        if token in value:
            raise ValidationError(f"{token!r} is not allowed in Planner IR v1 at {path}")
