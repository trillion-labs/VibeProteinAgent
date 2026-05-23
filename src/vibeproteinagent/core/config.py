from __future__ import annotations

import os
import shlex
from pathlib import Path

VPA_HOME_ENV = "VPA_HOME"
VPA_WORKSPACE_ENV = "VPA_WORKSPACE"
VPA_MODEL_CACHE_ENV = "VPA_MODEL_CACHE"
DEFAULT_WORKSPACE = "default"


def vpa_home() -> Path:
    """Root config dir. Override with $VPA_HOME; default ~/.vibeproteinagent/."""
    override = os.environ.get(VPA_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".vibeproteinagent"


def load_vpa_env_file(*, override: bool = False) -> None:
    """Load simple ``export KEY=value`` entries from ``$VPA_HOME/env.sh``.

    This mirrors the shell snippet printed by the installer for users who invoke
    ``vpa`` through the npm shim or an absolute binary path without sourcing the
    activation file first. Existing environment variables win by default so a
    caller can still override a cache/source location for one command.
    """
    env_file = vpa_home() / "env.sh"
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("export "):
            continue
        try:
            parts = shlex.split(line[len("export "):], posix=True)
        except ValueError:
            continue
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            if not key:
                continue
            if key in os.environ and not override and key != "PATH":
                continue
            os.environ[key] = os.path.expandvars(value)


def workspaces_root() -> Path:
    return vpa_home() / "workspaces"


def workspace_dir(name: str) -> Path:
    return workspaces_root() / name


def active_workspace_file() -> Path:
    return vpa_home() / "active"


def resolve_workspace(cli_arg: str | None = None) -> str:
    """Resolution order: --workspace > $VPA_WORKSPACE > ~/.vibeproteinagent/active > 'default'."""
    if cli_arg:
        return cli_arg
    env = os.environ.get(VPA_WORKSPACE_ENV)
    if env:
        return env
    active = active_workspace_file()
    if active.exists():
        name = active.read_text().strip()
        if name:
            return name
    return DEFAULT_WORKSPACE


def model_cache_root() -> Path:
    override = os.environ.get(VPA_MODEL_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return vpa_home() / "models"
