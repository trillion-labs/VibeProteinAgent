"""Phase 11 / Codex round-5 RC1: the MCP surface must expose the same
``force_new`` escape hatch the CLI added in Phase 10 (Codex round-4 RC1).

We exercise three paths in-process:

1. ``mcp.tools.submit_job`` — twice with identical params → same ``job_id``.
2. ``mcp.tools.submit_job(..., force_new=True)`` → a *different* ``job_id``.
3. The dynamic ``run_protein_qc`` tool (synthesised in ``mcp.dynamic``) — calling
   it with ``force_new=True`` likewise produces a fresh ``job_id``.

Together these prove the MCP entry points stay in lockstep with
``vpa run --force-new``. The submission path is the same shared
``JobStore.submit`` call used by the CLI; the only thing we're testing is
that the new optional argument is wired through cleanly.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.mcp.dynamic import register_dynamic_tools
from vibeproteinagent.mcp.tools import SessionCtx, submit_job
from vibeproteinagent.workspace import Workspace


def _ctx(workspace: Workspace) -> SessionCtx:
    return SessionCtx(workspace=workspace, registry=ToolRegistry())


def test_submit_job_deterministic_repeat_returns_same_id(
    tmp_workspace: Workspace,
) -> None:
    ctx = _ctx(tmp_workspace)
    first = submit_job(ctx, tool="noop", params={"payload": "x"})
    second = submit_job(ctx, tool="noop", params={"payload": "x"})
    assert first.get("ok") is not False, first
    assert second.get("ok") is not False, second
    assert first["job_id"] == second["job_id"], (
        "identical submission should collapse to one deterministic job_id"
    )
    assert first["idempotent_hit"] is False
    assert second["idempotent_hit"] is True


def test_submit_job_force_new_yields_distinct_id(
    tmp_workspace: Workspace,
) -> None:
    ctx = _ctx(tmp_workspace)
    base = submit_job(ctx, tool="noop", params={"payload": "x"})
    fresh = submit_job(
        ctx, tool="noop", params={"payload": "x"}, force_new=True,
    )
    assert base.get("ok") is not False, base
    assert fresh.get("ok") is not False, fresh
    assert base["job_id"] != fresh["job_id"], (
        "force_new=True must bypass the deterministic-id hit"
    )
    assert base["idempotent_hit"] is False
    assert fresh["idempotent_hit"] is False


def test_submit_job_force_new_requires_bool(tmp_workspace: Workspace) -> None:
    ctx = _ctx(tmp_workspace)
    out = submit_job(ctx, tool="noop", params={"payload": "x"}, force_new="false")  # type: ignore[arg-type]
    assert out["ok"] is False
    assert out["code"] == "E_INVALID_PARAMS"
    assert out["details"]["field"] == "force_new"


def test_submit_job_max_retries_persists(tmp_workspace: Workspace) -> None:
    ctx = _ctx(tmp_workspace)
    out = submit_job(
        ctx,
        tool="noop",
        params={"payload": "retry"},
        max_retries=2,
    )
    assert out.get("ok") is not False, out
    assert out["max_retries"] == 2


def test_submit_job_max_retries_rejects_bool(tmp_workspace: Workspace) -> None:
    ctx = _ctx(tmp_workspace)
    out = submit_job(  # type: ignore[arg-type]
        ctx,
        tool="noop",
        params={"payload": "retry"},
        max_retries=True,
    )
    assert out["ok"] is False
    assert out["code"] == "E_INVALID_PARAMS"
    assert out["details"]["field"] == "max_retries"


def test_dynamic_run_protein_qc_force_new_yields_distinct_id(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a supervisor the runner waits the full per-tool budget
    # before returning E_TIMEOUT. Compress that so the
    # test completes in seconds while still exercising the submit path.
    import vibeproteinagent.mcp.dynamic as dyn
    monkeypatch.setattr(dyn, "_wait_timeout_s_for", lambda _d: 0.2)

    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = _ctx(tmp_workspace)
    register_dynamic_tools(app, ctx)
    tool = app._tool_manager.get_tool("run_protein_qc")
    assert tool is not None

    # No supervisor in unit tests; the dynamic runner will time out waiting
    # for the job to leave the queued state. That's fine — we only need the
    # submitted job_ids to compare.
    args = {"sequence": "ACDEFGHIK"}
    a: Any = asyncio.run(tool.run(args))
    b: Any = asyncio.run(tool.run(args))
    c: Any = asyncio.run(tool.run({**args, "force_new": True}))

    assert a.get("job_id"), a
    assert b.get("job_id"), b
    assert c.get("job_id"), c

    assert a["job_id"] == b["job_id"], (
        "dynamic run_<tool> must hit the same deterministic id on repeat"
    )
    assert a["job_id"] != c["job_id"], (
        "dynamic run_<tool> with force_new=True must produce a fresh id"
    )
    assert a["idempotent_hit"] is False
    assert b["idempotent_hit"] is True
    assert c["idempotent_hit"] is False


def test_dynamic_force_new_string_false_is_rejected(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeproteinagent.mcp.dynamic as dyn
    monkeypatch.setattr(dyn, "_wait_timeout_s_for", lambda _d: 0.2)

    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = _ctx(tmp_workspace)
    register_dynamic_tools(app, ctx)
    tool = app._tool_manager.get_tool("run_protein_qc")
    assert tool is not None

    out: Any = asyncio.run(tool.run({"sequence": "ACDEFGHIK", "force_new": "false"}))
    assert out["ok"] is False
    assert out["code"] == "E_INVALID_PARAMS"
    assert out["details"]["field"] == "force_new"


def test_dynamic_max_retries_is_wired_to_submission(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeproteinagent.mcp.dynamic as dyn
    monkeypatch.setattr(dyn, "_wait_timeout_s_for", lambda _d: 0.01)

    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = _ctx(tmp_workspace)
    register_dynamic_tools(app, ctx)
    tool = app._tool_manager.get_tool("run_protein_qc")
    assert tool is not None

    out: Any = asyncio.run(tool.run({"sequence": "ACDEFGHIK", "max_retries": 2}))
    assert out["job_id"], out
    rec = ctx.store().get(out["job_id"])
    assert rec.max_retries == 2


def test_dynamic_runner_reports_idempotent_hit_without_fastmcp(
    tmp_workspace: Workspace,
) -> None:
    import asyncio

    from vibeproteinagent.mcp.dynamic import _make_runner

    ctx = _ctx(tmp_workspace)
    ctx.registry.discover()
    descriptor, _ = ctx.registry.get("protein_qc", "*")
    runner = _make_runner(ctx, descriptor, wait_s=0.01)

    first: Any = asyncio.run(runner({"sequence": "ACDEFGHIK"}))
    second: Any = asyncio.run(runner({"sequence": "ACDEFGHIK"}))

    assert first["job_id"] == second["job_id"]
    assert first["idempotent_hit"] is False
    assert second["idempotent_hit"] is True


def test_dynamic_run_tool_advertises_force_new_in_schema(
    tmp_workspace: Workspace,
) -> None:
    """Agents discover the knob via tools/list, so the parameter MUST appear
    in the advertised input schema for every dynamic tool."""
    from mcp.server.fastmcp import FastMCP

    app = FastMCP("test")
    ctx = _ctx(tmp_workspace)
    names = register_dynamic_tools(app, ctx)
    for name in names:
        tool = app._tool_manager.get_tool(name)
        assert tool is not None, name
        props = (tool.parameters or {}).get("properties", {})
        assert "force_new" in props, f"{name} missing force_new"
        assert props["force_new"]["type"] == "boolean", name
        assert "max_retries" in props, f"{name} missing max_retries"
        assert props["max_retries"]["type"] == "integer", name
