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
        f"index={'' if index is None else index}",
        f"sequence_digest={_digest(sequence) if sequence else ''}",
        f"structure_digest={_digest(Path(structure_path).name) if structure_path else ''}",
    ]
    payload = "\n".join(components).encode("utf-8")
    return "cand_" + hashlib.sha256(payload).hexdigest()[:12]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
