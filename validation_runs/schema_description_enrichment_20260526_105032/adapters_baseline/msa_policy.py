"""Shared MSA policy helpers for structure-prediction adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml

MSABackend = Literal["auto", "none", "remote"]

_MSA_KEYS = frozenset({"msa", "msa_path", "msa_file", "paired_msa", "unpaired_msa"})


def should_use_remote_msa(
    *, use_msa: bool, msa_backend: MSABackend, has_prepared_msa: bool
) -> bool:
    """Resolve whether a tool should call its remote MSA server.

    ``none`` is an explicit off-switch. ``auto`` trusts already prepared MSA
    fields and otherwise falls back to the remote backend when MSA is requested.
    """
    if not use_msa or msa_backend == "none":
        return False
    if msa_backend == "remote":
        return True
    return not has_prepared_msa


def has_prepared_msa(value: Any) -> bool:
    """Return true when nested tool-native data declares a non-empty MSA field."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _MSA_KEYS and _non_empty_msa_value(item):
                return True
            if has_prepared_msa(item):
                return True
    elif isinstance(value, list):
        return any(has_prepared_msa(item) for item in value)
    return False


def strip_prepared_msa(value: Any) -> Any:
    """Return a copy of nested tool-native data with prepared MSA fields removed."""
    if isinstance(value, dict):
        return {
            key: strip_prepared_msa(item)
            for key, item in value.items()
            if key not in _MSA_KEYS
        }
    if isinstance(value, list):
        return [strip_prepared_msa(item) for item in value]
    return value


def json_file_has_prepared_msa(path: Path) -> bool:
    with path.open("r") as fh:
        return has_prepared_msa(json.load(fh))


def yaml_file_has_prepared_msa(path: Path) -> bool:
    with path.open("r") as fh:
        return has_prepared_msa(yaml.safe_load(fh))


def _non_empty_msa_value(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


__all__ = [
    "MSABackend",
    "has_prepared_msa",
    "json_file_has_prepared_msa",
    "should_use_remote_msa",
    "strip_prepared_msa",
    "yaml_file_has_prepared_msa",
]
