from __future__ import annotations

import typer

from vibeproteinagent.cli.output import fail_from_exception, print_envelope_ok
from vibeproteinagent.cli.render import render_workspace_list, render_workspace_status
from vibeproteinagent.core import config as cfg
from vibeproteinagent.workspace import Workspace

workspace_app = typer.Typer(no_args_is_help=True, help="Workspace management.")


@workspace_app.command("create")
def workspace_create(
    name: str = typer.Argument(..., help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Create a workspace (idempotent)."""
    try:
        ws = Workspace.create(name)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok({"workspace": ws.name, "root": str(ws.root)}, json_mode=json_)


@workspace_app.command("use")
def workspace_use(
    name: str = typer.Argument(..., help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Mark a workspace as active (writes ~/.vibeproteinagent/active)."""
    try:
        Workspace.use(name)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok({"workspace": name}, json_mode=json_)


@workspace_app.command("status")
def workspace_status(
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show the active workspace and how it was resolved."""
    try:
        via, name = _resolution_source()
        root = cfg.workspace_dir(name)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"active": name, "via": via, "root": str(root)},
        json_mode=json_,
        render=render_workspace_status,
    )


@workspace_app.command("list")
def workspace_list(
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List all workspaces under $VPA_HOME/workspaces."""
    try:
        names = Workspace.list()
        _, active = _resolution_source()
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(
        {"workspaces": names, "active": active},
        json_mode=json_,
        render=render_workspace_list,
    )


@workspace_app.command("delete")
def workspace_delete(
    name: str = typer.Argument(..., help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Delete a workspace (idempotent on missing directory)."""
    try:
        try:
            Workspace.delete(name)
        except Exception as exc:
            from vibeproteinagent.core.errors import NotFoundError
            if not isinstance(exc, NotFoundError):
                raise
            # idempotent: missing workspace is a no-op success
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok({"workspace": name, "deleted": True}, json_mode=json_)


def _resolution_source() -> tuple[str, str]:
    """Return (via, name) describing where the active workspace was resolved from."""
    import os

    if os.environ.get(cfg.VPA_WORKSPACE_ENV):
        return "env", os.environ[cfg.VPA_WORKSPACE_ENV]
    active = cfg.active_workspace_file()
    if active.exists():
        text = active.read_text().strip()
        if text:
            return "file", text
    return "default", cfg.DEFAULT_WORKSPACE


__all__ = ["workspace_app"]
