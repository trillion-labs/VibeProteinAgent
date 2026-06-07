from __future__ import annotations

from pathlib import Path
from typing import Any

PDB_SUFFIXES = frozenset({".pdb", ".ent"})
CIF_SUFFIXES = frozenset({".cif", ".mmcif"})


def is_pdb_path(value: Any) -> bool:
    return isinstance(value, str) and Path(value).suffix.lower() in PDB_SUFFIXES


def is_cif_path(value: Any) -> bool:
    return isinstance(value, str) and Path(value).suffix.lower() in CIF_SUFFIXES


def candidate_pdb_path(candidate: dict[str, Any]) -> str | None:
    """Return a concrete PDB/ENT path from a canonical candidate, if present."""
    for key in (
        "structure_path",
        "pdb_path",
        "predicted_pdb",
        "full_structure_path",
        "backbone_path",
    ):
        value = candidate.get(key)
        if is_pdb_path(value):
            return str(value)
    return (
        _artifact_path(candidate, "pdb_path", pdb_only=True)
        or _artifact_path(candidate, "predicted_pdb", pdb_only=True)
    )


def _artifact_path(
    candidate: dict[str, Any],
    key: str,
    *,
    pdb_only: bool = False,
) -> str | None:
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    for payload in artifacts.values():
        if not isinstance(payload, dict):
            continue
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            continue
        if pdb_only and not is_pdb_path(value):
            continue
        return value
    return None


__all__ = [
    "CIF_SUFFIXES",
    "PDB_SUFFIXES",
    "candidate_pdb_path",
    "is_cif_path",
    "is_pdb_path",
]
