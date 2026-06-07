from __future__ import annotations

import os
import sys

import typer

from vibeproteinagent._version import __version__
from vibeproteinagent.cli.commands.artifacts import artifacts, download, export
from vibeproteinagent.cli.commands.daemon import daemon_app
from vibeproteinagent.cli.commands.doctor import doctor
from vibeproteinagent.cli.commands.jobs import (
    cancel,
    jobs_app,
    logs,
    results,
    retry,
    status,
    wait,
)
from vibeproteinagent.cli.commands.mcp import mcp_app
from vibeproteinagent.cli.commands.report import report
from vibeproteinagent.cli.commands.run import run
from vibeproteinagent.cli.commands.tools import tool_app, tools_app
from vibeproteinagent.cli.commands.trajectory import trajectory_app
from vibeproteinagent.cli.commands.visualize import visualize
from vibeproteinagent.cli.commands.workflow import workflow_app
from vibeproteinagent.cli.commands.workspace import workspace_app
from vibeproteinagent.cli.output import print_envelope_ok
from vibeproteinagent.cli.render import render_version
from vibeproteinagent.core.config import load_vpa_env_file
from vibeproteinagent.core.log import Verbosity, detect_style, set_cli_state
from vibeproteinagent.install.cli import install_app

app = typer.Typer(
    name="vpa",
    help="vibeproteinagent — local-first GPU-ready protein design tool substrate.",
    no_args_is_help=True,
    add_completion=False,
)


# -- root callback: global -v / -q / --no-color ------------------------------


@app.callback()
def _root(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose output."),
    quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress informational output."),
    no_color: bool = typer.Option(
        False, "--no-color", help="Disable ANSI colour (also honours $NO_COLOR)."
    ),
) -> None:
    """Configure global output mode for the invocation."""
    load_vpa_env_file()
    verbosity = Verbosity.NORMAL
    if quiet:
        verbosity = Verbosity.QUIET
    elif verbose:
        verbosity = Verbosity.VERBOSE
    nc = no_color or bool(os.environ.get("NO_COLOR"))
    try:
        isatty = sys.stdout.isatty()
    except (AttributeError, ValueError):
        isatty = False
    # Default to non-JSON; commands that take --json will flip the style themselves
    # via set_cli_state before they touch stdout.
    style = detect_style(json_mode=False, isatty=isatty, no_color=nc)
    set_cli_state(style, verbosity)
    # Stash so commands can read without re-importing.
    ctx.obj = {"style": style, "verbosity": verbosity, "no_color": nc}


# -- vpa version -------------------------------------------------------------


@app.command("version")
def version(json_: bool = typer.Option(False, "--json", help="Emit JSON envelope.")) -> None:
    """Print the installed vpa version."""
    py = sys.version.split()[0]
    torch_v: str | None = None
    cuda_v: str | None = None
    try:
        import torch

        torch_v = torch.__version__
        cuda_v = torch.version.cuda
    except ImportError:
        pass
    print_envelope_ok(
        {"version": __version__, "python": py, "torch": torch_v, "cuda": cuda_v},
        json_mode=json_,
        render=render_version,
    )


# -- subcommand groups -------------------------------------------------------

app.add_typer(tools_app, name="tools")
app.add_typer(tool_app, name="tool")
app.add_typer(workspace_app, name="workspace")
app.add_typer(workflow_app, name="workflow")
app.add_typer(daemon_app, name="daemon")
app.add_typer(mcp_app, name="mcp")
app.add_typer(install_app, name="install")
app.add_typer(jobs_app, name="jobs")
app.add_typer(trajectory_app, name="trajectory")


# -- top-level commands ------------------------------------------------------

app.command("run")(run)
app.command("status")(status)
app.command("wait")(wait)
app.command("logs")(logs)
app.command("results")(results)
app.command("cancel")(cancel)
app.command("retry")(retry)
app.command("artifacts")(artifacts)
app.command("download")(download)
app.command("export")(export)
app.command("doctor")(doctor)
app.command("visualize")(visualize)
app.command("report")(report)


if __name__ == "__main__":
    app()
