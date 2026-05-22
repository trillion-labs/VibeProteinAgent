from __future__ import annotations

import typer

from vibeproteinagent.cli.output import fail_from_exception, print_envelope_err
from vibeproteinagent.core.errors import ErrorCode

mcp_app = typer.Typer(no_args_is_help=True, help="MCP server facade (Phase 4.3).")


@mcp_app.command("serve")
def mcp_serve(
    transport: str = typer.Option("stdio", "--transport", help="stdio | ws."),
    workspace: str | None = typer.Option(
        None, "--workspace", help="Workspace to pin the session to."
    ),
    port: int = typer.Option(
        8765, "--port", help="WebSocket port (ws transport only; not yet wired)."
    ),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Start the MCP server.

    `stdio` is the only transport implemented in v1; `ws` returns E_DEPENDENCY.
    A stdio MCP server speaks JSON-RPC on stdin/stdout, so nothing else writes
    to stdout — logs go to stderr and ``~/.vibeproteinagent/run/mcp.log``.
    """
    if transport != "stdio":
        print_envelope_err(
            ErrorCode.DEPENDENCY,
            f"transport {transport!r} not implemented; only 'stdio' is supported.",
            {"transport": transport, "implemented": ["stdio"], "phase": "4.3"},
            json_mode=json_,
            exit_code=2,
        )
    try:
        from vibeproteinagent.mcp.server import serve_stdio

        serve_stdio(workspace)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


__all__ = ["mcp_app"]
