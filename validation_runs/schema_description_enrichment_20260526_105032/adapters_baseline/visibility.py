from __future__ import annotations

from vibeproteinagent.adapters.descriptor import ToolDescriptor

_INTERNAL_TOOL_NAMES = frozenset(
    {
        "mpnn_to_protenix_input",
        "merge_candidates",
        "noop",
        "pxdesign_pipeline_multi_gpu",
        "sequence_to_protenix_input",
    }
)

def is_retired_registry_tool_name(name: str) -> bool:
    """Whether a legacy built-in adapter must stay out of the public registry."""
    del name
    return False


def is_internal_tool_name(name: str) -> bool:
    """Whether a registry tool is an internal runtime/glue adapter."""
    return name in _INTERNAL_TOOL_NAMES


def is_agent_visible_tool(descriptor: ToolDescriptor) -> bool:
    """Whether a tool belongs in the LLM/MCP dynamic first-class tool surface."""
    return (
        bool(getattr(descriptor, "planner_visible", True))
        and not is_internal_tool_name(descriptor.name)
        and not is_retired_registry_tool_name(descriptor.name)
    )


def is_public_tool(descriptor: ToolDescriptor) -> bool:
    """Whether a tool should appear in user-facing first-class tool catalogs."""
    return is_agent_visible_tool(descriptor)


__all__ = [
    "is_agent_visible_tool",
    "is_internal_tool_name",
    "is_public_tool",
    "is_retired_registry_tool_name",
]
