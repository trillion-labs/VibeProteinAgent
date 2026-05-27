from __future__ import annotations

from copy import deepcopy
from typing import Any

FIXUP_KINDS = {
    "malformed_fixup_response",
    "missing_explicit_mode",
    "missing_canonical_input_slot",
    "prior_round_referenced_without_context",
    "upstream_does_not_produce_candidates",
    "candidate_capability_mismatch",
}


class StructuredFixup(Exception):
    """Raised when the compiler needs an enum-style planner choice."""

    def __init__(self, payload: dict[str, Any]) -> None:
        kind = payload.get("error_kind")
        if kind not in FIXUP_KINDS:
            raise ValueError(f"unknown structured fixup kind: {kind!r}")
        self._payload = payload
        super().__init__(str(kind))

    @property
    def payload(self) -> dict[str, Any]:
        return self._payload


def missing_canonical_input_slot(
    *,
    node_id: str,
    tool: str,
    allowed_tools: list[str],
) -> StructuredFixup:
    return StructuredFixup(
        {
            "error_kind": "missing_canonical_input_slot",
            "node_id": node_id,
            "tool": tool,
            "field": "candidates",
            "explanation": (
                f"Tool {tool!r} does not accept canonical candidates as input. "
                f"Available canonical consumers in this workflow: {', '.join(allowed_tools)}."
            ),
            "options": [
                {
                    "choice_id": "remove_node",
                    "description": "Drop this node from the plan.",
                },
                {
                    "choice_id": "swap_tool",
                    "description": (
                        "Replace this node's tool with one of the canonical consumers."
                    ),
                    "allowed_tools": allowed_tools,
                },
            ],
        }
    )


def upstream_does_not_produce_candidates(
    *,
    node_id: str,
    tool: str,
    upstream_node: str,
    upstream_tool: str,
) -> StructuredFixup:
    return StructuredFixup(
        {
            "error_kind": "upstream_does_not_produce_candidates",
            "node_id": node_id,
            "tool": tool,
            "field": "candidates",
            "upstream_node": upstream_node,
            "upstream_tool": upstream_tool,
            "explanation": (
                f"Upstream node {upstream_node!r} uses tool {upstream_tool!r}, "
                "which does not emit canonical candidates."
            ),
            "options": [
                {
                    "choice_id": "remove_node",
                    "description": "Drop the consumer node that cannot be wired.",
                },
                {
                    "choice_id": "swap_tool",
                    "description": (
                        "Replace the upstream node's tool with one of the canonical producers."
                    ),
                    "allowed_tools": ["go_sequences", "sequence_analysis"],
                    "target_node_id": upstream_node,
                },
            ],
        }
    )


def candidate_capability_mismatch(
    *,
    node_id: str,
    tool: str,
    upstream_node: str,
    upstream_tool: str,
    missing_capabilities: list[str],
) -> StructuredFixup:
    return StructuredFixup(
        {
            "error_kind": "candidate_capability_mismatch",
            "node_id": node_id,
            "tool": tool,
            "field": "candidates",
            "upstream_node": upstream_node,
            "upstream_tool": upstream_tool,
            "missing_capabilities": list(missing_capabilities),
            "explanation": (
                f"Upstream node {upstream_node!r} using tool {upstream_tool!r} "
                f"does not provide capabilities required by {tool!r}: "
                + ", ".join(missing_capabilities)
            ),
            "options": [
                {
                    "choice_id": "remove_node",
                    "description": "Drop the incompatible consumer node.",
                },
                {
                    "choice_id": "swap_tool",
                    "description": (
                        "Replace the consumer node with a tool compatible "
                        "with the upstream candidate capabilities."
                    ),
                    "target_node_id": node_id,
                    "allowed_tools": [],
                },
            ],
        }
    )


def prior_round_referenced_without_context(
    *,
    consumer_node_id: str,
    upstream_node: str = "prior_round",
) -> StructuredFixup:
    return StructuredFixup(
        {
            "error_kind": "prior_round_referenced_without_context",
            "node_id": consumer_node_id,
            "upstream_node": upstream_node,
            "explanation": (
                f"Node {consumer_node_id!r} references {upstream_node!r} as a "
                "candidate_source upstream, but this is a first-round plan "
                "with no prior_round_context. prior_round is only available "
                "in continuation rounds."
            ),
            "options": [
                {
                    "choice_id": "remove_upstream",
                    "description": (
                        "Remove the prior_round upstream entry from this node."
                    ),
                    "target_node_id": consumer_node_id,
                    "target_upstream": upstream_node,
                },
                {
                    "choice_id": "remove_node",
                    "description": (
                        "Drop the consumer node entirely if it cannot work "
                        "without prior_round."
                    ),
                    "target_node_id": consumer_node_id,
                },
            ],
        }
    )


def missing_explicit_mode(
    *,
    node_id: str,
    tool: str,
    allowed_modes: list[str],
    received_mode: Any = None,
    mode_field: str = "mode",
) -> StructuredFixup:
    if received_mode is None:
        problem = f"missing `inputs.{mode_field}`"
    else:
        problem = f"invalid `inputs.{mode_field}`={received_mode!r}"
    return StructuredFixup(
        {
            "error_kind": "missing_explicit_mode",
            "node_id": node_id,
            "tool": tool,
            "allowed_modes": list(allowed_modes),
            "received_mode": received_mode,
            "mode_field": mode_field,
            "explanation": (
                f"Tool {tool!r} requires an explicit mode; {problem}. "
                f"Set inputs.{mode_field} to one of: {', '.join(allowed_modes)}."
            ),
            "options": [
                {
                    "choice_id": "set_mode",
                    "description": (
                        "Specify the mode for this node by selecting one "
                        "of the allowed_modes."
                    ),
                    "allowed_modes": list(allowed_modes),
                },
                {
                    "choice_id": "remove_node",
                    "description": (
                        "Drop this node if no appropriate mode applies."
                    ),
                },
            ],
        }
    )


def malformed_fixup_response(
    *,
    original_payload: dict[str, Any],
    received_choice: dict[str, Any],
    parse_error: str,
) -> StructuredFixup:
    return StructuredFixup(
        {
            "error_kind": "malformed_fixup_response",
            "original_error_kind": original_payload.get("error_kind"),
            "original_options": original_payload.get("options"),
            "received_choice": received_choice,
            "parse_error": parse_error,
            "explanation": (
                f"Your previous fixup choice was malformed: {parse_error}. "
                "Re-choose from the original options below. Include all "
                "required fields per the original options."
            ),
            "options": original_payload.get("options", []),
        }
    )


def apply_enum_fixup_choice(
    llm_ir: dict[str, Any],
    fixup_payload: dict[str, Any],
    choice: dict[str, Any],
) -> dict[str, Any]:
    choice_id = choice.get("choice_id")
    if choice_id == "remove_node":
        node_id = fixup_payload["node_id"]
        updated = deepcopy(llm_ir)
        updated["nodes"] = [
            node for node in updated.get("nodes", []) if node.get("id") != node_id
        ]
        return updated
    if choice_id == "remove_upstream":
        target_node_id = fixup_payload["node_id"]
        target_upstream = fixup_payload["upstream_node"]
        updated = deepcopy(llm_ir)
        for node in updated.get("nodes", []):
            if node.get("id") == target_node_id:
                node["upstream"] = [
                    upstream
                    for upstream in node.get("upstream", [])
                    if upstream.get("node") != target_upstream
                ]
                return updated
        raise ValueError(f"fixup target node {target_node_id!r} was not found")
    if choice_id == "swap_tool":
        tool = choice.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError("swap_tool fixup choice requires a non-empty tool")
        allowed = _allowed_tools(fixup_payload)
        if allowed and tool not in allowed:
            raise ValueError(f"tool {tool!r} is not allowed for this fixup")
        target_node_id = choice.get("target_node_id") or fixup_payload["node_id"]
        updated = deepcopy(llm_ir)
        for node in updated.get("nodes", []):
            if node.get("id") == target_node_id:
                node["tool"] = tool
                return updated
        raise ValueError(f"fixup target node {target_node_id!r} was not found")
    if choice_id == "set_mode":
        mode = choice.get("mode")
        if not isinstance(mode, str) or not mode:
            raise ValueError("set_mode fixup choice requires a non-empty mode")
        allowed = fixup_payload.get("allowed_modes") or []
        if allowed and mode not in allowed:
            raise ValueError(f"mode {mode!r} is not in allowed_modes")
        target_node_id = fixup_payload["node_id"]
        updated = deepcopy(llm_ir)
        for node in updated.get("nodes", []):
            if node.get("id") == target_node_id:
                inputs = node.setdefault("inputs", {})
                mode_field = str(fixup_payload.get("mode_field") or "mode")
                inputs[mode_field] = mode
                if mode_field != "mode":
                    inputs.pop("mode", None)
                return updated
        raise ValueError(f"fixup target node {target_node_id!r} was not found")
    raise ValueError(f"invalid fixup choice_id: {choice_id!r}")


def _allowed_tools(fixup_payload: dict[str, Any]) -> list[str]:
    for option in fixup_payload.get("options", []):
        if isinstance(option, dict) and option.get("choice_id") == "swap_tool":
            tools = option.get("allowed_tools")
            if isinstance(tools, list):
                return [str(tool) for tool in tools]
    return []
