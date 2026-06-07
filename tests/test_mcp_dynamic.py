"""Tests for the dynamic per-tool MCP-tool generation.

For every first-class agent-visible ToolDescriptor in the registry,
register_dynamic_tools must register a single MCP tool named
`run_<descriptor.name>` whose `inputSchema` is the descriptor's resolved input
schema (not the FastMCP auto-derived one).
"""
from __future__ import annotations

from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.adapters.visibility import is_agent_visible_tool
from vibeproteinagent.core.errors import DaemonWorkspaceMismatchError
from vibeproteinagent.mcp.dynamic import register_dynamic_tools
from vibeproteinagent.mcp.server import build_server
from vibeproteinagent.mcp.tools import SessionCtx
from vibeproteinagent.mcp.tools import list_tools as mcp_list_tools
from vibeproteinagent.mcp.util import resolve_input_schema
from vibeproteinagent.workspace import Workspace


def test_dynamic_tools_named_run_prefix(tmp_workspace: Workspace) -> None:
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = SessionCtx(workspace=tmp_workspace, registry=ToolRegistry())
    names = register_dynamic_tools(app, ctx)
    assert "run_noop" not in names
    assert "run_merge_candidates" in names
    assert "run_proteina_complexa" in names
    assert "run_proteinmpnn" in names
    assert "run_protenix_v2" in names
    for n in names:
        assert n.startswith("run_")


def test_mcp_list_tools_hides_internal_by_default(tmp_workspace: Workspace) -> None:
    ctx = SessionCtx(workspace=tmp_workspace, registry=ToolRegistry())

    public_names = {tool["name"] for tool in mcp_list_tools(ctx)}
    internal_names = {
        "noop",
        "mpnn_to_protenix_input",
        "sequence_to_protenix_input",
    }

    assert internal_names.isdisjoint(public_names)
    assert "merge_candidates" in public_names
    assert "proteina_complexa" in public_names
    assert internal_names.issubset(
        {tool["name"] for tool in mcp_list_tools(ctx, include_internal=True)}
    )


def test_dynamic_tool_schema_matches_descriptor(tmp_workspace: Workspace) -> None:
    """The advertised inputSchema equals the descriptor's resolved schema
    augmented with VPA runtime-control knobs."""
    from mcp.server.fastmcp import FastMCP

    from vibeproteinagent.mcp.dynamic import _augment_schema_with_runtime_controls

    app = FastMCP("test")
    registry = ToolRegistry()
    ctx = SessionCtx(workspace=tmp_workspace, registry=registry)
    register_dynamic_tools(app, ctx)

    descriptors = [d for d in registry.list() if is_agent_visible_tool(d)]
    for descriptor in descriptors:
        tool = app._tool_manager.get_tool(f"run_{descriptor.name}")
        assert tool is not None, f"missing run_{descriptor.name}"
        expected = _augment_schema_with_runtime_controls(resolve_input_schema(descriptor))
        assert tool.parameters == expected, descriptor.name
        # The descriptor's own schema must not have been mutated in place.
        assert "force_new" not in resolve_input_schema(descriptor).get("properties", {})
        assert "max_retries" not in resolve_input_schema(descriptor).get("properties", {})


def test_build_server_registers_all_tools(
    tmp_workspace: Workspace,
) -> None:
    """Sanity check that build_server() wires both orchestration + dynamic tools."""
    import asyncio

    app = build_server(tmp_workspace.name)
    tools = asyncio.run(app.list_tools())
    names = {t.name for t in tools}

    # Orchestration tools.
    for n in (
        "list_tools",
        "tool_info",
        "submit_job",
        "get_status",
        "wait_job",
        "get_results",
        "get_logs",
        "cancel_job",
        "list_artifacts",
        "daemon_status",
        "warm_tool",
        "list_workspaces",
        "current_workspace",
        "list_workflows",
        "get_workflow_status",
        "run_workflow",
    ):
        assert n in names, f"missing orchestration tool {n}"

    # Dynamic tools (first-class agent-visible tools).
    for n in ("run_proteina_complexa", "run_proteinmpnn", "run_protenix_v2"):
        assert n in names, f"missing dynamic tool {n}"
    assert "run_noop" not in names


def test_dynamic_tool_returns_error_for_invalid_params(
    tmp_workspace: Workspace,
) -> None:
    """Calling the dynamic tool with bad params should return E_INVALID_PARAMS."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = SessionCtx(workspace=tmp_workspace, registry=ToolRegistry())
    register_dynamic_tools(app, ctx)
    tool = app._tool_manager.get_tool("run_protein_qc")
    assert tool is not None
    result: Any = asyncio.run(tool.run({"not_payload": 1}))
    assert result["ok"] is False
    assert result["code"] == "E_INVALID_PARAMS"


def test_dynamic_tool_reports_daemon_unavailable_without_daemon(
    tmp_workspace: Workspace,
) -> None:
    """Dynamic run_<tool> is submit-and-wait, so it requires a live daemon."""
    import asyncio

    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = SessionCtx(workspace=tmp_workspace, registry=ToolRegistry())
    register_dynamic_tools(app, ctx)
    tool = app._tool_manager.get_tool("run_protein_qc")
    assert tool is not None
    result: Any = asyncio.run(tool.run({"sequence": "ACDEFGHIK"}))
    assert result["ok"] is False
    assert result["code"] == "E_DAEMON_UNAVAILABLE"
    assert result["details"]["requested_workspace"] == tmp_workspace.name


def test_dynamic_tool_reports_daemon_workspace_mismatch(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    def _mismatch(self, workspace_name: str):
        raise DaemonWorkspaceMismatchError(
            "running supervisor is attached to a different workspace",
            details={
                "requested_workspace": workspace_name,
                "supervisor_workspace": "other",
            },
        )

    monkeypatch.setattr(
        "vibeproteinagent.workers.supervisor_client.SupervisorClient.require_workspace",
        _mismatch,
    )

    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = SessionCtx(workspace=tmp_workspace, registry=ToolRegistry())
    register_dynamic_tools(app, ctx)
    tool = app._tool_manager.get_tool("run_protein_qc")
    assert tool is not None
    result: Any = asyncio.run(tool.run({"sequence": "ACDEFGHIK"}))
    assert result["ok"] is False
    assert result["code"] == "E_DAEMON_WORKSPACE_MISMATCH"
    assert result["details"]["supervisor_workspace"] == "other"
