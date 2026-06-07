from __future__ import annotations

import typer

from vibeproteinagent.cli.output import fail_from_exception, print_envelope_ok
from vibeproteinagent.workspace import TrajectoryStore, Workspace

trajectory_app = typer.Typer(no_args_is_help=True, help="Track one user-query trajectory.")


def _ensure_workspace(name: str | None) -> Workspace:
    ws = Workspace.resolve(name)
    if not ws.db_path.exists():
        ws = Workspace.create(ws.name)
    return ws


@trajectory_app.command("start")
def trajectory_start(
    user_query: str = typer.Option(..., "--user-query", help="Original user query."),
    objective: str | None = typer.Option(None, "--objective", help="Optional objective summary."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Create a trajectory for one user query."""
    try:
        ws = _ensure_workspace(workspace)
        rec = TrajectoryStore(ws).create(user_query=user_query, objective=objective)
        print_envelope_ok(
            {"trajectory": rec.model_dump(), "trajectory_id": rec.trajectory_id},
            json_mode=json_,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@trajectory_app.command("status")
def trajectory_status(
    trajectory_id: str = typer.Argument(..., help="Trajectory id (traj_*)."),
    workspace: str | None = typer.Option(None, "--workspace", help="Workspace name."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show completion accounting for one trajectory."""
    try:
        ws = _ensure_workspace(workspace)
        status = TrajectoryStore(ws).status(trajectory_id)
        print_envelope_ok({"trajectory": status}, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


__all__ = ["trajectory_app"]
