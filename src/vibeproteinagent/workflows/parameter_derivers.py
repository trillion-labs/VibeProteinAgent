from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from vibeproteinagent.candidates.paths import candidate_pdb_path


def is_canonical_candidate(value: Any) -> bool:
    return isinstance(value, dict) and value.get("schema_version") == "candidate_v1"


def candidate_structure_path(
    candidate: dict[str, Any],
    *,
    prefer_pdb: bool = True,
    require_pdb: bool = False,
) -> str | None:
    if prefer_pdb:
        pdb_path = candidate_pdb_path(candidate)
        if pdb_path:
            return pdb_path
        if require_pdb:
            return None
    for key in ("structure_path", "full_structure_path", "backbone_path", "pdb_path", "cif_path"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return value
    artifact_path = _artifact_path(candidate, "pdb_path") or _artifact_path(candidate, "cif_path")
    return artifact_path


def candidate_sequence(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("sequence")
    return str(value) if value else None


def candidate_input_json(candidate: dict[str, Any]) -> str | None:
    value = _artifact_path(candidate, "input_json") or candidate.get("input_json")
    return str(value) if value else None


def candidate_input_yaml(candidate: dict[str, Any]) -> str | None:
    value = _artifact_path(candidate, "input_yaml") or candidate.get("input_yaml")
    return str(value) if value else None


def candidate_score_namespaces(candidate: dict[str, Any]) -> list[str]:
    scores = candidate.get("scores")
    return sorted(str(key) for key in scores) if isinstance(scores, dict) else []


def protein_chain_ids_from_structure(path: str | Path) -> list[str]:
    path = Path(path).expanduser()
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return _cif_chain_ids(path, groups={"ATOM"})
    return _pdb_chain_ids(path, records={"ATOM"})


def protein_chain_ids_excluding_ligands(path: str | Path) -> list[str]:
    return protein_chain_ids_from_structure(path)


def all_atom_chain_ids(path: str | Path) -> list[str]:
    path = Path(path).expanduser()
    if path.suffix.lower() in {".cif", ".mmcif"}:
        return _cif_chain_ids(path, groups={"ATOM", "HETATM"})
    return _pdb_chain_ids(path, records={"ATOM", "HETATM"})


def binder_chain_ids_from_candidate(candidate: dict[str, Any]) -> list[str]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    for key in ("binder_chains", "binder_chain_ids", "designed_chains"):
        chains = _string_list(metadata.get(key)) or _string_list(candidate.get(key))
        if chains:
            return sorted(chains)
    path = candidate_structure_path(candidate, prefer_pdb=True)
    return protein_chain_ids_excluding_ligands(path) if path else []


def target_chain_ids_from_candidate(candidate: dict[str, Any]) -> list[str]:
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    for key in ("target_chains", "target_chain_ids"):
        chains = _string_list(metadata.get(key)) or _string_list(candidate.get(key))
        if chains:
            return sorted(chains)
    path = candidate_structure_path(candidate, prefer_pdb=True)
    return protein_chain_ids_from_structure(path) if path else []


def fixed_length_from_range_for_symmetric(
    low: int | None,
    high: int | None,
    total: int | None = None,
) -> int | None:
    if total is not None:
        return int(total)
    if low is None and high is None:
        return None
    if low is None:
        return int(high) if high is not None else None
    if high is None:
        return int(low)
    lo, hi = sorted((int(low), int(high)))
    return round((lo + hi) / 2)


def ordered_length_range(
    low: int | None,
    high: int | None,
    total: int | None = None,
) -> tuple[int, int] | None:
    if total is not None:
        return int(total), int(total)
    if low is None and high is None:
        return None
    if low is None:
        assert high is not None
        return int(high), int(high)
    if high is None:
        return int(low), int(low)
    lo, hi = sorted((int(low), int(high)))
    return lo, hi


def _artifact_path(candidate: dict[str, Any], key: str) -> str | None:
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    source_tool = str(source.get("tool") or "")
    namespaces = [source_tool] if source_tool else []
    namespaces.extend(str(name) for name in artifacts if str(name) not in namespaces)
    for namespace in namespaces:
        payload = artifacts.get(namespace)
        if isinstance(payload, dict):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _pdb_chain_ids(path: Path, *, records: set[str]) -> list[str]:
    chains: set[str] = set()
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []
    for line in lines:
        if not any(line.startswith(record) for record in records):
            continue
        chain = line[21:22].strip()
        if chain:
            chains.add(chain)
    return sorted(chains)


def _cif_chain_ids(path: Path, *, groups: set[str]) -> list[str]:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []
    columns: list[str] = []
    chains: set[str] = set()
    in_atom_loop = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            columns = []
            in_atom_loop = False
            continue
        if stripped.startswith("_atom_site."):
            in_atom_loop = True
            columns.append(stripped.removeprefix("_atom_site."))
            continue
        if not in_atom_loop:
            continue
        parts = shlex.split(stripped)
        if not parts or parts[0].startswith("#"):
            in_atom_loop = False
            continue
        if len(parts) < len(columns):
            continue
        row = {column: parts[index] for index, column in enumerate(columns)}
        if str(row.get("group_PDB") or "") not in groups:
            continue
        chain = row.get("auth_asym_id") or row.get("label_asym_id")
        if chain and chain not in {".", "?"}:
            chains.add(str(chain))
    return sorted(chains)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list):
        return sorted(str(item) for item in value if str(item))
    return []


__all__ = [
    "all_atom_chain_ids",
    "binder_chain_ids_from_candidate",
    "candidate_input_json",
    "candidate_input_yaml",
    "candidate_score_namespaces",
    "candidate_sequence",
    "candidate_structure_path",
    "fixed_length_from_range_for_symmetric",
    "is_canonical_candidate",
    "ordered_length_range",
    "protein_chain_ids_excluding_ligands",
    "protein_chain_ids_from_structure",
    "target_chain_ids_from_candidate",
]
