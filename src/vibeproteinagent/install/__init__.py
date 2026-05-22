"""Manifest-driven external installer support for VibeProteinAgent."""

from .manifest import InstallManifest, load_manifest
from .planner import InstallSelection, resolve_selection

__all__ = [
    "InstallManifest",
    "InstallSelection",
    "load_manifest",
    "resolve_selection",
]
