from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_ok,
    protect_json_stdout,
)
from vibeproteinagent.cli.render import render_install_result
from vibeproteinagent.install.installer import InstallOptions, ManifestInstaller, install_readiness
from vibeproteinagent.install.manifest import load_manifest

install_app = typer.Typer(
    name="install",
    help="Install VPA shared envs, selected sidecars, and tool weights.",
    no_args_is_help=False,
    add_completion=False,
)


@install_app.callback(invoke_without_command=True)
def install_default(
    ctx: typer.Context,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    tools: Annotated[str | None, typer.Option("--tools")] = None,
    install_root: Annotated[Path | None, typer.Option("--install-root")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    repair: Annotated[bool, typer.Option("--repair")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    json_: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Install the selected profile or tool set."""
    if ctx.invoked_subcommand is not None:
        return
    _run_install(
        profile=profile,
        tools=tools,
        install_root=install_root,
        cache_dir=cache_dir,
        source_root=source_root,
        dry_run=dry_run,
        yes=yes,
        force=force,
        repair=repair,
        offline=offline,
        json_=json_,
    )


@install_app.command("plan")
def plan(
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    tools: Annotated[str | None, typer.Option("--tools")] = None,
    install_root: Annotated[Path | None, typer.Option("--install-root")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    json_: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show the resolved install plan without mutating the machine."""
    try:
        with protect_json_stdout(json_):
            data = ManifestInstaller(load_manifest()).plan(
                InstallOptions(
                    profile=profile,
                    tools=((tools,) if tools else ()),
                    install_root=install_root,
                    cache_dir=cache_dir,
                    dry_run=True,
                )
            )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(data, json_mode=json_, render=render_install_result)


@install_app.command("run")
def run(
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    tools: Annotated[str | None, typer.Option("--tools")] = None,
    install_root: Annotated[Path | None, typer.Option("--install-root")] = None,
    cache_dir: Annotated[Path | None, typer.Option("--cache-dir")] = None,
    source_root: Annotated[Path | None, typer.Option("--source-root")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    repair: Annotated[bool, typer.Option("--repair")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    json_: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Install the selected profile or tool set."""
    _run_install(
        profile=profile,
        tools=tools,
        install_root=install_root,
        cache_dir=cache_dir,
        source_root=source_root,
        dry_run=dry_run,
        yes=yes,
        force=force,
        repair=repair,
        offline=offline,
        json_=json_,
    )


def _run_install(
    *,
    profile: str | None,
    tools: str | None,
    install_root: Path | None,
    cache_dir: Path | None,
    source_root: Path | None,
    dry_run: bool,
    yes: bool,
    force: bool,
    repair: bool,
    offline: bool,
    json_: bool,
    ) -> None:
    try:
        with protect_json_stdout(json_):
            data = ManifestInstaller(load_manifest()).install(
                InstallOptions(
                    profile=profile,
                    tools=((tools,) if tools else ()),
                    install_root=install_root,
                    cache_dir=cache_dir,
                    source_root=source_root,
                    dry_run=dry_run,
                    yes=yes,
                    force=force,
                    repair=repair,
                    offline=offline,
                )
            )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(data, json_mode=json_, render=render_install_result)


@install_app.command("doctor")
def doctor(
    tools: Annotated[str | None, typer.Option("--tools")] = None,
    json_: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report installer-oriented readiness for all or selected tools."""
    try:
        with protect_json_stdout(json_):
            tool_tuple = ((tools,) if tools else ())
            data = install_readiness(tool_tuple)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)
    print_envelope_ok(data, json_mode=json_, render=render_install_result)


def main() -> None:
    install_app()


if __name__ == "__main__":
    main()
