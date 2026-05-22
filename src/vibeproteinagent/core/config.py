from __future__ import annotations

import os
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
