from __future__ import annotations

from typing import Any

from vibeproteinagent.core.errors import InvalidParamsError

from .identity import (
    candidate_material_key as _material_key,
    candidate_ranked_candidate_id as _ranked_candidate_id,
    candidate_sequence_identity as _candidate_sequence_identity,
    row_material_key as _row_material_key,
    row_ranked_candidate_id as _row_ranked_candidate_id,
    structure_identity as _structure_identity,
    submitted_material_key as _submitted_material_key,
    submitted_ranked_candidate_id as _submitted_ranked_candidate_id,
)
from .models import ReportCandidate, ReportModel


def validate_final_report_contract(model: ReportModel) -> None:
    """Reject final reports whose submitted candidate is detached from ranking evidence."""

    final_submission = model.final_submission if isinstance(model.final_submission, dict) else {}
    if final_submission.get("outcome") != "accepted":
        return
    rank_rows = _objective_rank_rows(model)
    if not rank_rows:
        _validate_unranked_submission_contract(model)
        return

    unresolved = [
        {
            "candidate": candidate,
            "matches": _objective_rank_matches(candidate, rank_rows),
        }
        for candidate in model.final_candidates
    ]
    unresolved = [item for item in unresolved if len(item["matches"]) != 1]
    if not unresolved:
        _validate_review_manifest_contract(model, rank_rows)
        return

    raise InvalidParamsError(
        "final report contract failed: accepted candidate is not resolved from objective_rank",
        details={
            "source_id": model.summary.id,
            "unresolved_candidates": [
                {
                    "candidate_id": item["candidate"].candidate_id,
                    "match_count": len(item["matches"]),
                    "source": _candidate_source_summary(item["candidate"]),
                }
                for item in unresolved
            ],
            "objective_rank_rows": _rank_row_summaries(rank_rows),
            "fix": (
                "Run objective_rank on the enriched canonical candidate rows, then "
                "submit_candidate using rank or ranked_candidate_id from that objective_rank job. "
                "Do not submit a separate evaluator-local row when ranking evidence exists."
            ),
        },
    )


def _validate_unranked_submission_contract(model: ReportModel) -> None:
    """Validate accepted submissions when no objective_rank manifest exists."""

    if not model.final_candidates:
        raise InvalidParamsError(
            "final report contract failed: accepted submission has no final candidates",
            details={
                "source_id": model.summary.id,
                "fix": (
                    "Record at least one accepted candidate in submit_candidate output, "
                    "or report no_acceptable_candidate instead of accepted."
                ),
            },
        )

    ambiguous: list[dict[str, Any]] = []
    for candidate in model.final_candidates:
        matches = _unranked_candidate_matches(candidate, model.candidates)
        if len(matches) > 1:
            ambiguous.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "match_count": len(matches),
                    "source": _candidate_source_summary(candidate),
                    "matched_candidates": [
                        _candidate_source_summary(match) for match in matches[:10]
                    ],
                }
            )
    if not ambiguous:
        return

    raise InvalidParamsError(
        "final report contract failed: accepted candidate is ambiguous without objective_rank",
        details={
            "source_id": model.summary.id,
            "ambiguous_candidates": ambiguous,
            "fix": (
                "Run objective_rank before submit_candidate, or submit a stronger "
                "identifier such as ranked_candidate_id, material_key, structure_path, "
                "or candidate_id plus sequence."
            ),
        },
    )


def _unranked_candidate_matches(
    final_candidate: ReportCandidate,
    candidates: list[ReportCandidate],
) -> list[ReportCandidate]:
    submitted_ranked = _submitted_ranked_candidate_id(final_candidate)
    if submitted_ranked:
        matches = [c for c in candidates if _ranked_candidate_id(c) == submitted_ranked]
        if matches:
            return matches

    submitted_material = _submitted_material_key(final_candidate)
    if submitted_material:
        matches = [c for c in candidates if _material_key(c) == submitted_material]
        if matches:
            return matches

    structure = _structure_identity(final_candidate.structure_path)
    if structure:
        matches = [c for c in candidates if _structure_identity(c.structure_path) == structure]
        if matches:
            return matches

    candidate_id = str(final_candidate.candidate_id or "")
    sequence = _candidate_sequence_identity(final_candidate)
    if candidate_id and sequence:
        matches = [
            c
            for c in candidates
            if str(c.candidate_id or "") == candidate_id
            and _candidate_sequence_identity(c) == sequence
        ]
        if matches:
            return matches

    if candidate_id:
        return [c for c in candidates if str(c.candidate_id or "") == candidate_id]
    return []


def _validate_review_manifest_contract(
    model: ReportModel,
    rank_rows: list[dict[str, Any]],
) -> None:
    from .models import CandidateStatus
    from .review import build_review_model

    review = build_review_model(model)
    bundle_identities = [
        identity
        for identity in (
            _report_candidate_identity(bundle.candidate)
            for bundle in review.bundles
        )
        if identity
    ]
    missing_ranked: list[dict[str, Any]] = []
    duplicate_ranked: list[dict[str, Any]] = []
    for row in rank_rows:
        identity = _rank_row_identity(row)
        if not identity:
            continue
        count = bundle_identities.count(identity)
        if count == 0:
            missing_ranked.append(_rank_row_summary(row))
        elif count > 1:
            duplicate_ranked.append({**_rank_row_summary(row), "bundle_count": count})

    final_bundle_identities = {
        _report_candidate_identity(bundle.candidate)
        for bundle in review.bundles
        if bundle.status == CandidateStatus.FINAL
    } - {""}
    detached_final: list[dict[str, Any]] = []
    for candidate in model.final_candidates:
        identity = _submitted_candidate_identity(candidate)
        if identity and identity not in final_bundle_identities:
            detached_final.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "submitted_identity": identity,
                    "source": _candidate_source_summary(candidate),
                }
            )

    if not missing_ranked and not duplicate_ranked and not detached_final:
        return

    raise InvalidParamsError(
        "final report contract failed: objective_rank manifest is inconsistent with review board",
        details={
            "source_id": model.summary.id,
            "missing_ranked_candidates": missing_ranked,
            "duplicate_ranked_candidates": duplicate_ranked,
            "detached_final_candidates": detached_final,
            "fix": (
                "Preserve ranked_candidate_id/material_key through candidate collection, "
                "build the report board from objective_rank manifest rows, and only merge "
                "raw generator/evaluator rows as support evidence."
            ),
        },
    )


def _objective_rank_rows(model: ReportModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in model.graph.nodes:
        if (node.tool or "").strip().lower() != "objective_rank":
            continue
        output = node.output if isinstance(node.output, dict) else {}
        ranked = output.get("ranked_candidates")
        if not isinstance(ranked, list):
            continue
        for index, row in enumerate(ranked):
            if not isinstance(row, dict):
                continue
            candidate_id = str(row.get("candidate_id") or row.get("name") or "")
            if not candidate_id:
                continue
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "ranked_candidate_id": _row_ranked_candidate_id(row),
                    "material_key": _row_material_key(row),
                    "sequence": _row_sequence_identity(row),
                    "structure_path": _first_string(
                        row,
                        "structure_path",
                        "pdb_path",
                        "cif_path",
                        "predicted_pdb",
                        "predicted_cif",
                    ),
                    "rank": row.get("rank") or index + 1,
                    "node_id": node.node_id,
                    "job_id": node.job_id,
                    "row_index": index,
                }
            )
    return rows


def _objective_rank_matches(
    candidate: ReportCandidate,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked_candidate_id = _submitted_ranked_candidate_id(candidate)
    if ranked_candidate_id:
        matches = [
            row for row in rows if row.get("ranked_candidate_id") == ranked_candidate_id
        ]
        if matches:
            return matches

    material_key = _submitted_material_key(candidate)
    if material_key:
        matches = [row for row in rows if row.get("material_key") == material_key]
        if matches:
            return matches

    structure_path = candidate.structure_path.strip() if candidate.structure_path else ""
    if structure_path:
        matches = [row for row in rows if row.get("structure_path") == structure_path]
        if matches:
            return matches

    candidate_id = str(candidate.candidate_id or "")
    sequence = _candidate_sequence_identity(candidate)
    if candidate_id and sequence:
        matches = [
            row
            for row in rows
            if row.get("candidate_id") == candidate_id
            and row.get("sequence") == sequence
        ]
        if matches:
            return matches

    if candidate_id:
        return [row for row in rows if row.get("candidate_id") == candidate_id]
    return []


def _candidate_source_summary(candidate: ReportCandidate) -> dict[str, Any]:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    selection = source.get("selection_evidence")
    if not isinstance(selection, dict):
        selection = {}
    return {
        "tool": source.get("tool"),
        "job_id": source.get("job_id"),
        "selection_status": selection.get("status"),
        "submitted_ranked_candidate_id": _submitted_ranked_candidate_id(candidate),
        "submitted_material_key": _submitted_material_key(candidate),
        "resolved_ranked_candidate_id": _ranked_candidate_id(candidate),
        "resolved_material_key": _material_key(candidate),
        "structure_path": candidate.structure_path,
        "has_objective_rank_scores": _has_objective_rank_scores(candidate),
    }


def _rank_row_identity(row: dict[str, Any]) -> str | None:
    ranked_candidate_id = row.get("ranked_candidate_id")
    if ranked_candidate_id:
        return f"ranked:{ranked_candidate_id}"
    material_key = row.get("material_key")
    if material_key:
        return f"material:{material_key}"
    structure = _structure_identity(row.get("structure_path"))
    if structure:
        return f"structure:{structure}"
    sequence = row.get("sequence")
    candidate_id = row.get("candidate_id")
    if candidate_id and sequence:
        return f"candidate:{candidate_id}:{sequence}"
    return None


def _report_candidate_identity(candidate: ReportCandidate) -> str | None:
    ranked_candidate_id = _ranked_candidate_id(candidate)
    if ranked_candidate_id:
        return f"ranked:{ranked_candidate_id}"
    material_key = _material_key(candidate)
    if material_key:
        return f"material:{material_key}"
    structure = _structure_identity(candidate.structure_path)
    if structure:
        return f"structure:{structure}"
    sequence = _candidate_sequence_identity(candidate)
    if candidate.candidate_id and sequence:
        return f"candidate:{candidate.candidate_id}:{sequence}"
    return None


def _submitted_candidate_identity(candidate: ReportCandidate) -> str | None:
    ranked_candidate_id = _submitted_ranked_candidate_id(candidate)
    if ranked_candidate_id:
        return f"ranked:{ranked_candidate_id}"
    material_key = _submitted_material_key(candidate)
    if material_key:
        return f"material:{material_key}"
    return _report_candidate_identity(candidate)


def _has_objective_rank_scores(candidate: ReportCandidate) -> bool:
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    return isinstance(scores.get("objective_rank"), dict)


def _row_sequence_identity(row: dict[str, Any]) -> str | None:
    value = row.get("sequence")
    if value in (None, ""):
        return None
    return "".join(str(value).split()).upper()


def _first_string(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _rank_row_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_rank_row_summary(row) for row in rows]


def _rank_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "ranked_candidate_id": row.get("ranked_candidate_id"),
        "material_key": row.get("material_key"),
        "rank": row.get("rank"),
        "node_id": row.get("node_id"),
        "job_id": row.get("job_id"),
    }


__all__ = ["validate_final_report_contract"]
