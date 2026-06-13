from __future__ import annotations

from typing import Any

from vibeproteinagent.core.errors import InvalidParamsError

from .models import ReportCandidate, ReportModel


def validate_final_report_contract(model: ReportModel) -> None:
    """Reject final reports whose submitted candidate is detached from ranking evidence."""

    final_submission = model.final_submission if isinstance(model.final_submission, dict) else {}
    if final_submission.get("outcome") != "accepted":
        return
    rank_rows = _objective_rank_rows(model)
    if not rank_rows:
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


def _submitted_ranked_candidate_id(candidate: ReportCandidate) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    value = source.get("ranked_candidate_id")
    if value:
        return str(value)
    selection = source.get("selection_evidence")
    if isinstance(selection, dict) and selection.get("ranked_candidate_id"):
        return str(selection["ranked_candidate_id"])
    return None


def _submitted_material_key(candidate: ReportCandidate) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    value = source.get("material_key")
    if value:
        return str(value)
    selection = source.get("selection_evidence")
    if isinstance(selection, dict) and selection.get("material_key"):
        return str(selection["material_key"])
    return None


def _ranked_candidate_id(candidate: ReportCandidate) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    for container in (
        source,
        source.get("objective_rank_evidence")
        if isinstance(source.get("objective_rank_evidence"), dict)
        else {},
    ):
        value = container.get("ranked_candidate_id") if isinstance(container, dict) else None
        if value:
            return str(value)
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    objective = scores.get("objective_rank")
    if isinstance(objective, dict) and objective.get("ranked_candidate_id"):
        return str(objective["ranked_candidate_id"])
    return None


def _material_key(candidate: ReportCandidate) -> str | None:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    for container in (
        source,
        source.get("objective_rank_evidence")
        if isinstance(source.get("objective_rank_evidence"), dict)
        else {},
    ):
        value = container.get("material_key") if isinstance(container, dict) else None
        if value:
            return str(value)
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    objective = scores.get("objective_rank")
    if isinstance(objective, dict) and objective.get("material_key"):
        return str(objective["material_key"])
    return None


def _has_objective_rank_scores(candidate: ReportCandidate) -> bool:
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    return isinstance(scores.get("objective_rank"), dict)


def _candidate_sequence_identity(candidate: ReportCandidate) -> str | None:
    value = candidate.sequence
    if value in (None, ""):
        return None
    return "".join(str(value).split()).upper()


def _row_ranked_candidate_id(row: dict[str, Any]) -> str | None:
    for container in (
        row,
        row.get("candidate_ref") if isinstance(row.get("candidate_ref"), dict) else {},
    ):
        value = container.get("ranked_candidate_id") if isinstance(container, dict) else None
        if value not in (None, ""):
            return str(value)
    scores = row.get("scores")
    if isinstance(scores, dict):
        objective = scores.get("objective_rank")
        if isinstance(objective, dict):
            value = objective.get("ranked_candidate_id")
            if value not in (None, ""):
                return str(value)
    return None


def _row_material_key(row: dict[str, Any]) -> str | None:
    for container in (
        row,
        row.get("candidate_ref") if isinstance(row.get("candidate_ref"), dict) else {},
    ):
        value = container.get("material_key") if isinstance(container, dict) else None
        if value not in (None, ""):
            return str(value)
    scores = row.get("scores")
    if isinstance(scores, dict):
        objective = scores.get("objective_rank")
        if isinstance(objective, dict):
            value = objective.get("material_key")
            if value not in (None, ""):
                return str(value)
    return None


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
    return [
        {
            "candidate_id": row.get("candidate_id"),
            "ranked_candidate_id": row.get("ranked_candidate_id"),
            "material_key": row.get("material_key"),
            "rank": row.get("rank"),
            "node_id": row.get("node_id"),
            "job_id": row.get("job_id"),
        }
        for row in rows
    ]


__all__ = ["validate_final_report_contract"]
