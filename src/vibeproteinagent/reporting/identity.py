from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import ReportCandidate


def degapped_sequence(value: Any) -> str:
    return re.sub(r"[-.\s]", "", str(value or "")).upper()


def candidate_sequence_identity(candidate: ReportCandidate) -> str | None:
    value = degapped_sequence(candidate.sequence)
    return value or None


def structure_identity(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return Path(raw).name or raw


def candidate_structure_identity(candidate: ReportCandidate) -> str | None:
    return structure_identity(candidate.structure_path)


def row_ranked_candidate_id(row: dict[str, Any]) -> str | None:
    for container in (
        row,
        row.get("candidate_ref") if isinstance(row.get("candidate_ref"), dict) else {},
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("ranked_candidate_id")
        if value not in (None, ""):
            return str(value)
    scores = row.get("scores")
    if isinstance(scores, dict):
        objective_rank = scores.get("objective_rank")
        if isinstance(objective_rank, dict):
            value = objective_rank.get("ranked_candidate_id")
            if value not in (None, ""):
                return str(value)
    return None


def row_material_key(row: dict[str, Any]) -> str | None:
    for container in (
        row,
        row.get("candidate_ref") if isinstance(row.get("candidate_ref"), dict) else {},
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("material_key")
        if value not in (None, ""):
            return str(value)
    scores = row.get("scores")
    if isinstance(scores, dict):
        objective_rank = scores.get("objective_rank")
        if isinstance(objective_rank, dict):
            value = objective_rank.get("material_key")
            if value not in (None, ""):
                return str(value)
    return None


def row_sequence_identity(row: dict[str, Any]) -> str | None:
    value = row.get("sequence")
    if value in (None, ""):
        return None
    sequence = degapped_sequence(value)
    return sequence or None


def candidate_ranked_candidate_id(
    candidate: ReportCandidate,
    *,
    include_selection: bool = False,
) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    containers: list[dict[str, Any]] = [source]
    objective_evidence = source.get("objective_rank_evidence")
    if isinstance(objective_evidence, dict):
        containers.append(objective_evidence)
    if include_selection:
        selection = source.get("selection_evidence")
        if isinstance(selection, dict):
            containers.append(selection)
    for container in containers:
        value = container.get("ranked_candidate_id")
        if value not in (None, ""):
            return str(value)
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    objective_rank = scores.get("objective_rank")
    if isinstance(objective_rank, dict):
        value = objective_rank.get("ranked_candidate_id")
        if value not in (None, ""):
            return str(value)
    return None


def candidate_material_key(
    candidate: ReportCandidate,
    *,
    include_selection: bool = False,
) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    containers: list[dict[str, Any]] = [source]
    objective_evidence = source.get("objective_rank_evidence")
    if isinstance(objective_evidence, dict):
        containers.append(objective_evidence)
    if include_selection:
        selection = source.get("selection_evidence")
        if isinstance(selection, dict):
            containers.append(selection)
    for container in containers:
        value = container.get("material_key")
        if value not in (None, ""):
            return str(value)
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    objective_rank = scores.get("objective_rank")
    if isinstance(objective_rank, dict):
        value = objective_rank.get("material_key")
        if value not in (None, ""):
            return str(value)
    return None


def submitted_ranked_candidate_id(candidate: ReportCandidate) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    for container in (
        source,
        source.get("selection_evidence")
        if isinstance(source.get("selection_evidence"), dict)
        else {},
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("ranked_candidate_id")
        if value not in (None, ""):
            return str(value)
    return None


def submitted_material_key(candidate: ReportCandidate) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    for container in (
        source,
        source.get("selection_evidence")
        if isinstance(source.get("selection_evidence"), dict)
        else {},
    ):
        if not isinstance(container, dict):
            continue
        value = container.get("material_key")
        if value not in (None, ""):
            return str(value)
    return None


def candidate_manifest_identity(candidate: ReportCandidate) -> str:
    ranked_id = candidate_ranked_candidate_id(candidate)
    if ranked_id:
        return f"ranked:{ranked_id}"
    material_key = candidate_material_key(candidate)
    if material_key:
        return f"material:{material_key}"
    structure = candidate_structure_identity(candidate)
    if structure:
        return f"structure:{structure}"
    if candidate.candidate_id:
        sequence = candidate_sequence_identity(candidate) or ""
        return f"candidate:{candidate.candidate_id}:{sequence}"
    return ""


def strong_candidate_manifest_identity(candidate: ReportCandidate) -> str:
    ranked_id = candidate_ranked_candidate_id(candidate)
    if ranked_id:
        return f"ranked:{ranked_id}"
    material_key = candidate_material_key(candidate)
    if material_key:
        return f"material:{material_key}"
    structure = candidate_structure_identity(candidate)
    if structure:
        return f"structure:{structure}"
    return ""
