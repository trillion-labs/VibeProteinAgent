from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

_MATERIAL_FIELDS = ("sequence", "structure_path", "backbone_path", "full_structure_path")
_STRUCTURE_FIELDS = ("structure_path", "full_structure_path", "backbone_path")


@dataclass(frozen=True)
class CandidateMergeResult:
    candidates: list[dict[str, Any]]
    audit: dict[str, Any]


@dataclass(frozen=True)
class _Entry:
    candidate: dict[str, Any]
    position: int


def merge_candidate_records(
    candidate_sets: Any,
    *,
    strip_private_keys: bool = True,
    include_private_positions: bool = False,
) -> CandidateMergeResult:
    """Merge duplicate enrichment versions without adding fields to Candidate[] items."""

    entries = [
        _Entry(candidate=deepcopy(candidate), position=position)
        for position, candidate in enumerate(_flatten(candidate_sets))
        if isinstance(candidate, dict)
    ]
    groups = _identity_groups(entries)
    merged_groups: list[tuple[int, dict[str, Any], dict[str, Any] | None]] = []
    duplicate_groups: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for group in groups:
        first_position = min(entry.position for entry in group)
        merged, group_audit = _merge_group(group)
        if include_private_positions:
            representative = max(group, key=_representative_key)
            merged["__candidate_merge_positions"] = [entry.position for entry in group]
            merged["__candidate_merge_representative_position"] = representative.position
        if group_audit is not None:
            duplicate_groups.append(group_audit)
            conflicts.extend(group_audit.get("conflicts") or [])
        merged_groups.append((first_position, merged, group_audit))

    merged_groups.sort(key=lambda item: item[0])
    candidates = [
        _strip_private_keys(candidate) if strip_private_keys else candidate
        for _position, candidate, _audit in merged_groups
    ]
    duplicates_removed = len(entries) - len(candidates)
    return CandidateMergeResult(
        candidates=candidates,
        audit={
            "input_count": len(entries),
            "output_count": len(candidates),
            "duplicates_removed": duplicates_removed,
            "duplicate_groups": duplicate_groups,
            "conflicts": conflicts,
        },
    )


def summarize_candidate_merge_audit(audit: dict[str, Any]) -> dict[str, int]:
    duplicate_groups = audit.get("duplicate_groups") if isinstance(audit, dict) else []
    return {
        "duplicates_merged_count": int(audit.get("duplicates_removed") or 0)
        if isinstance(audit, dict)
        else 0,
        "duplicate_group_count": len(duplicate_groups) if isinstance(duplicate_groups, list) else 0,
        "enriched_candidate_count": (
            len(duplicate_groups) if isinstance(duplicate_groups, list) else 0
        ),
    }


def _flatten(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value:
        rows.extend(_flatten(item))
    return rows


def _identity_groups(entries: list[_Entry]) -> list[list[_Entry]]:
    remaining = list(entries)
    groups: list[list[_Entry]] = []

    for key_fn in (_candidate_id_key, _sequence_only_key, _structure_key):
        buckets: dict[str, list[_Entry]] = {}
        for entry in remaining:
            key = key_fn(entry.candidate)
            if key:
                buckets.setdefault(key, []).append(entry)
        mergeable_buckets: list[list[_Entry]] = []
        for bucket in buckets.values():
            if len(bucket) <= 1:
                continue
            mergeable_buckets.extend(
                group for group in _split_material_compatible_groups(bucket) if len(group) > 1
            )
        grouped_positions = {
            entry.position
            for bucket in mergeable_buckets
            for entry in bucket
        }
        groups.extend(mergeable_buckets)
        remaining = [entry for entry in remaining if entry.position not in grouped_positions]

    groups.extend([entry] for entry in remaining)
    groups.sort(key=lambda group: min(entry.position for entry in group))
    return groups


def _candidate_id_key(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("candidate_id")
    return f"candidate_id:{value}" if value else None


def _sequence_only_key(candidate: dict[str, Any]) -> str | None:
    sequence = candidate.get("sequence")
    if not sequence or any(candidate.get(field) for field in _STRUCTURE_FIELDS):
        return None
    return f"sequence:{sequence}"


def _structure_key(candidate: dict[str, Any]) -> str | None:
    for field in _STRUCTURE_FIELDS:
        value = candidate.get(field)
        if value:
            return f"structure:{value}"
    return None


def _split_material_compatible_groups(entries: list[_Entry]) -> list[list[_Entry]]:
    groups: list[list[_Entry]] = []
    for entry in entries:
        for group in groups:
            if _material_compatible([*group, entry]):
                group.append(entry)
                break
        else:
            groups.append([entry])
    return groups


def _material_compatible(entries: list[_Entry]) -> bool:
    for field in _MATERIAL_FIELDS:
        values = {
            str(entry.candidate.get(field))
            for entry in entries
            if entry.candidate.get(field) not in (None, "")
        }
        if len(values) > 1:
            return False
    return True


def _merge_group(entries: list[_Entry]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    representative = max(entries, key=_representative_key)
    merged = deepcopy(representative.candidate)
    group_conflicts: list[dict[str, Any]] = []
    for entry in entries:
        if entry.position == representative.position:
            continue
        _merge_scores(merged, entry.candidate, group_conflicts)
        _merge_artifacts(merged, entry.candidate, group_conflicts)
        _merge_capabilities(merged, entry.candidate)

    if len(entries) == 1:
        return merged, None

    audit = {
        "identity_key": _group_identity_key(entries),
        "identity_type": _group_identity_type(entries),
        "retained_candidate_id": merged.get("candidate_id"),
        "merged_count": len(entries),
        "versions": [_version_audit(entry) for entry in entries],
        "conflicts": group_conflicts,
    }
    return merged, audit


def _representative_key(entry: _Entry) -> tuple[int, int, str]:
    candidate = entry.candidate
    richness = (
        _score_namespace_count(candidate)
        + _artifact_namespace_count(candidate)
        + _true_capability_count(candidate)
        + sum(1 for field in _MATERIAL_FIELDS if candidate.get(field))
        + _source_presence_count(candidate)
    )
    return (richness, entry.position, str(candidate.get("candidate_id") or ""))


def _score_namespace_count(candidate: dict[str, Any]) -> int:
    scores = candidate.get("scores")
    return len(scores) if isinstance(scores, dict) else 0


def _artifact_namespace_count(candidate: dict[str, Any]) -> int:
    artifacts = candidate.get("artifacts")
    return len(artifacts) if isinstance(artifacts, dict) else 0


def _true_capability_count(candidate: dict[str, Any]) -> int:
    capabilities = candidate.get("capabilities")
    if not isinstance(capabilities, dict):
        return 0
    return sum(1 for value in capabilities.values() if value is True)


def _source_presence_count(candidate: dict[str, Any]) -> int:
    source = candidate.get("source")
    if not isinstance(source, dict):
        return 0
    return sum(1 for field in ("job_id", "node_id", "native_id") if source.get(field))


def _merge_scores(
    merged: dict[str, Any],
    other: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> None:
    target = merged.setdefault("scores", {})
    incoming = other.get("scores")
    if not isinstance(target, dict) or not isinstance(incoming, dict):
        return
    for namespace, value in incoming.items():
        if namespace not in target:
            target[namespace] = deepcopy(value)
        elif target[namespace] != value:
            conflicts.append(
                {
                    "field": f"scores.{namespace}",
                    "retained_candidate_id": merged.get("candidate_id"),
                    "discarded_candidate_id": other.get("candidate_id"),
                }
            )


def _merge_artifacts(
    merged: dict[str, Any],
    other: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> None:
    target = merged.setdefault("artifacts", {})
    incoming = other.get("artifacts")
    if not isinstance(target, dict) or not isinstance(incoming, dict):
        return
    for namespace, value in incoming.items():
        if namespace not in target:
            target[namespace] = deepcopy(value)
        elif target[namespace] != value:
            conflicts.append(
                {
                    "field": f"artifacts.{namespace}",
                    "retained_candidate_id": merged.get("candidate_id"),
                    "discarded_candidate_id": other.get("candidate_id"),
                }
            )


def _merge_capabilities(merged: dict[str, Any], other: dict[str, Any]) -> None:
    incoming = other.get("capabilities")
    if not isinstance(incoming, dict):
        return
    target = merged.setdefault("capabilities", {})
    if not isinstance(target, dict):
        return
    for key, value in incoming.items():
        if value is True or key not in target:
            target[key] = bool(target.get(key)) or bool(value)


def _group_identity_key(entries: list[_Entry]) -> str:
    for key_fn in (_candidate_id_key, _sequence_only_key, _structure_key):
        keys = {key_fn(entry.candidate) for entry in entries}
        keys.discard(None)
        if len(keys) == 1:
            return str(next(iter(keys)))
    return "unknown"


def _group_identity_type(entries: list[_Entry]) -> str:
    key = _group_identity_key(entries)
    return key.split(":", 1)[0] if ":" in key else "unknown"


def _version_audit(entry: _Entry) -> dict[str, Any]:
    candidate = entry.candidate
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    artifacts = candidate.get("artifacts") if isinstance(candidate.get("artifacts"), dict) else {}
    capabilities = (
        candidate.get("capabilities") if isinstance(candidate.get("capabilities"), dict) else {}
    )
    return {
        "position": entry.position,
        "candidate_id": candidate.get("candidate_id"),
        "source": deepcopy(source),
        "score_namespaces": sorted(str(key) for key in scores),
        "artifact_namespaces": sorted(str(key) for key in artifacts),
        "true_capabilities": sorted(
            str(key) for key, value in capabilities.items() if value is True
        ),
    }


def _strip_private_keys(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not str(key).startswith("__")}


__all__ = [
    "CandidateMergeResult",
    "merge_candidate_records",
    "summarize_candidate_merge_audit",
]
