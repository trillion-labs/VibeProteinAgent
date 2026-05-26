from __future__ import annotations

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.agents.planning.workflow_planner import WorkflowPlanner


def _all_ready(registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Force both readiness gates to True and return the resulting catalog names."""
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": True},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._descriptor_availability",
        lambda descriptor, adapter_cls: {"available": True},
    )
    planner = WorkflowPlanner(registry)
    return [entry["name"] for entry in planner._planner_ir_tool_catalog(None)]


def test_catalog_filters_out_tools_failing_install_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    registry.discover()

    baseline = _all_ready(registry, monkeypatch)
    assert baseline, "registry should expose at least one tool"
    blocked = baseline[0]

    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: (
            {"install_ready": False, "reason": "test-manifest-blocked"}
            if tool_name == blocked
            else {"install_ready": True}
        ),
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._descriptor_availability",
        lambda descriptor, adapter_cls: {"available": True},
    )

    planner = WorkflowPlanner(registry)
    names = {entry["name"] for entry in planner._planner_ir_tool_catalog(None)}

    assert blocked not in names, (
        f"unavailable tool {blocked!r} leaked into LLM-facing catalog "
        "(install_readiness gate failed to filter)"
    )
    assert names, "catalog should still contain other ready tools"


def test_catalog_filters_out_tools_failing_descriptor_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ToolRegistry()
    registry.discover()

    baseline = _all_ready(registry, monkeypatch)
    assert baseline, "registry should expose at least one tool"
    blocked = baseline[0]

    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": True},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._descriptor_availability",
        lambda descriptor, adapter_cls: (
            {
                "available": False,
                "availability_reason": "test-probe-blocked",
            }
            if descriptor.name == blocked
            else {"available": True}
        ),
    )

    planner = WorkflowPlanner(registry)
    names = {entry["name"] for entry in planner._planner_ir_tool_catalog(None)}

    assert blocked not in names, (
        f"unavailable tool {blocked!r} leaked into LLM-facing catalog "
        "(descriptor_availability gate failed to filter)"
    )
    assert names, "catalog should still contain other ready tools"


def test_catalog_respects_explicit_available_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._install_readiness",
        lambda tool_name: {"install_ready": False},
    )
    monkeypatch.setattr(
        "vibeproteinagent.agents.planning.workflow_planner._descriptor_availability",
        lambda descriptor, adapter_cls: {"available": False},
    )

    planner = WorkflowPlanner(ToolRegistry())
    explicit = [
        {"name": "go_sequences", "description": "Retrieve GO sequences."},
        {"name": "rank_designs", "description": "Rank candidates."},
    ]

    catalog = planner._planner_ir_tool_catalog(explicit)

    names = {entry["name"] for entry in catalog}
    assert names == {"go_sequences", "rank_designs"}
