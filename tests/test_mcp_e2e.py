"""End-to-end MCP server test: spawn `vpa mcp serve --transport stdio`, drive it
through the official ``mcp`` SDK client, and run a noop job to completion.

Marked slow: it spawns the supervisor in a child process, the MCP server as a
subprocess, and waits for a real job to finish.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _supervisor_target(
    workspace_name: str,
    runtime_dir_s: str,
    vpa_home_s: str,
) -> None:
    os.environ["VPA_RUNTIME_DIR"] = runtime_dir_s
    os.environ["VPA_HOME"] = vpa_home_s
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from vibeproteinagent.workers.supervisor import (
        supervisor_main,
    )
    supervisor_main(workspace_name, 1, [-1])


def _wait_socket(socket_path: Path, *, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if socket_path.exists():
            return True
        time.sleep(0.1)
    return False


async def _drive(
    vpa_home: Path,
    runtime: Path,
) -> dict:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    env = {
        **os.environ,
        "VPA_HOME": str(vpa_home),
        "VPA_RUNTIME_DIR": str(runtime),
        "VPA_WORKSPACE": "default",
        "PYTHONNOUSERSITE": "1",
    }
    env.pop("PYTHONPATH", None)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "vibeproteinagent.cli.main", "mcp", "serve", "--transport", "stdio"],
        env=env,
    )

    async with stdio_client(params) as (read_s, write_s):
        async with ClientSession(read_s, write_s) as session:
            init_result = await session.initialize()
            assert init_result.serverInfo.name

            listing = await session.list_tools()
            names = {t.name for t in listing.tools}
            assert "list_tools" in names
            assert "submit_job" in names
            assert "wait_job" in names
            assert "get_results" in names
            assert "run_noop" not in names
            assert "run_protein_qc" in names

            # Submit a noop job through the low-level orchestration surface.
            submit = await session.call_tool(
                "submit_job",
                {"tool": "noop", "params": {"payload": "hello mcp"}},
            )
            assert not submit.isError, submit
            # structured content carries the dict
            submit_data = submit.structuredContent or {}
            # FastMCP wraps non-BaseModel returns in {"result": ...}
            if "result" in submit_data and "job_id" not in submit_data:
                submit_data = submit_data["result"]
            assert "job_id" in submit_data, submit_data
            job_id = submit_data["job_id"]

            # Wait for the supervisor to drive it to completed.
            wait_res = await session.call_tool(
                "wait_job",
                {"job_id": job_id, "timeout_ms": 30_000},
            )
            assert not wait_res.isError, wait_res
            wait_data = wait_res.structuredContent or {}
            if "result" in wait_data and "state" not in wait_data:
                wait_data = wait_data["result"]
            assert wait_data["state"] == "completed", wait_data

            # Read back the results.
            results = await session.call_tool(
                "get_results",
                {"job_id": job_id},
            )
            assert not results.isError
            results_data = results.structuredContent or {}
            if "result" in results_data and "output" not in results_data:
                results_data = results_data["result"]
            return results_data


def test_mcp_e2e_noop(tmp_path: Path) -> None:
    home = tmp_path / "vpa_home"
    home.mkdir()
    runtime = tmp_path / "vpa_run"
    runtime.mkdir()
    os.environ["VPA_HOME"] = str(home)
    os.environ["VPA_RUNTIME_DIR"] = str(runtime)

    from vibeproteinagent.workspace import Workspace
    Workspace.create("default")

    socket_path = runtime / "supervisor.sock"
    ctx = mp.get_context("spawn")
    sup = ctx.Process(
        target=_supervisor_target,
        args=("default", str(runtime), str(home)),
        daemon=False,
    )
    sup.start()
    try:
        assert _wait_socket(socket_path, timeout_s=20.0), "supervisor never bound socket"
        results = asyncio.run(_drive(home, runtime))
        assert results["output"]["payload"] == "hello mcp"
    finally:
        try:
            sup.terminate()
            sup.join(timeout=10.0)
        finally:
            if sup.is_alive():
                sup.kill()
                sup.join(timeout=5.0)
