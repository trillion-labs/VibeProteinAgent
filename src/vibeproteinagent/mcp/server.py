from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.workspace import Workspace

from .dynamic import register_dynamic_tools
from .tools import SessionCtx, register_orchestration_tools

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


_LOG_NAME = "vibeproteinagent.mcp"


def _log_path() -> Path:
    home = Path(os.environ.get("VPA_RUNTIME_DIR") or (Path.home() / ".vibeproteinagent" / "run"))
    home.mkdir(parents=True, exist_ok=True)
    return home / "mcp.log"


def _configure_logging() -> logging.Logger:
    """Route MCP server logging to stderr + ~/.vibeproteinagent/run/mcp.log.

    stdio servers exchange JSON-RPC on stdout, so anything else MUST go to stderr.
    """
    logger = logging.getLogger(_LOG_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stderr_h = logging.StreamHandler(stream=sys.stderr)
    stderr_h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(stderr_h)

    try:
        file_h = logging.FileHandler(_log_path(), mode="a", encoding="utf-8")
        file_h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(file_h)
    except OSError:  # pragma: no cover - logfile fallback
        pass
    return logger


def build_server(workspace_name: str | None = None) -> FastMCP:
    """Construct a fresh FastMCP app pinned to one workspace.

    No globals, no shared state — each call returns a new app + ctx, so test
    fixtures can spawn many independent servers under tmp $VPA_HOME.
    """
    from mcp.server.fastmcp import FastMCP  # local import — keeps test collection cheap

    workspace = Workspace.resolve(workspace_name)
    if not workspace.db_path.exists():
        workspace = Workspace.create(workspace.name)

    registry = ToolRegistry()
    registry.discover()

    ctx = SessionCtx(workspace=workspace, registry=registry)
    app = FastMCP("vibeproteinagent")
    orchestration = register_orchestration_tools(app, ctx)
    dynamic = register_dynamic_tools(app, ctx)

    logger = _configure_logging()
    logger.info(
        "MCP server ready; workspace=%s root=%s orch=%d dynamic=%d",
        workspace.name,
        workspace.root,
        len(orchestration),
        len(dynamic),
    )
    return app


def serve_stdio(workspace_name: str | None = None) -> None:
    """Start the MCP server on stdin/stdout (blocks)."""
    app = build_server(workspace_name)
    app.run(transport="stdio")


__all__ = ["build_server", "serve_stdio"]
