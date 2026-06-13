from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def candidate_schema() -> dict[str, Any]:
    """Load the canonical Candidate v1 JSON Schema."""

    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / "candidate_v1.json"
    )
    return json.loads(schema_path.read_text())


def validate_candidate_v1(candidate: dict[str, Any]) -> None:
    """Validate a canonical Candidate v1 object."""

    Draft202012Validator(candidate_schema()).validate(candidate)


def stable_candidate_id(
    producer_tool: str,
    local_id: str,
    *,
    producer_node: str | None = None,
    native_id: str | None = None,
    source_candidate_id: str | None = None,
    source_job_id: str | None = None,
    sequence: str | None = None,
    structure_path: str | None = None,
    index: int | None = None,
) -> str:
    """Return a deterministic producer-rooted candidate identifier."""

    if not producer_tool:
        raise ValueError("producer_tool is required")
    if not local_id:
        raise ValueError("local_id is required")

    components = [
        f"producer_tool={producer_tool}",
        f"producer_node={producer_node or ''}",
        f"local_id={local_id}",
        f"native_id={native_id or ''}",
        f"source_candidate_id={source_candidate_id or ''}",
        f"source_job_id={source_job_id or ''}",
        f"index={'' if index is None else index}",
        f"sequence_digest={_digest(sequence) if sequence else ''}",
        f"structure_digest={_digest(Path(structure_path).name) if structure_path else ''}",
    ]
    payload = "\n".join(components).encode("utf-8")
    return "cand_" + hashlib.sha256(payload).hexdigest()[:12]


def source_job_ids(candidate: dict[str, Any] | None) -> list[str]:
    """Return sorted source job ids recorded on a candidate-like row."""

    if not isinstance(candidate, dict):
        return []
    values: list[str] = []
    for key in ("source_job_id", "source_job_ids", "evidence_job_ids"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            values.append(value)
        elif isinstance(value, list):
            values.extend(str(item) for item in value if item)
    source = candidate.get("source")
    if isinstance(source, dict) and source.get("job_id"):
        values.append(str(source["job_id"]))
    return sorted(dict.fromkeys(values))


def candidate_material_key(candidate: dict[str, Any] | None) -> str:
    """Return a deterministic material identity key for merge/rank/debug use."""

    if not isinstance(candidate, dict):
        return "{}"
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    payload = {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "sequence": str(candidate.get("sequence") or ""),
        "structure_path": str(candidate.get("structure_path") or ""),
        "backbone_path": str(candidate.get("backbone_path") or ""),
        "full_structure_path": str(candidate.get("full_structure_path") or ""),
        "source_tool": str(source.get("tool") or candidate.get("source_tool") or ""),
        "source_local_id": str(source.get("local_id") or ""),
        "source_native_id": str(source.get("native_id") or ""),
        "source_job_ids": source_job_ids(candidate),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
