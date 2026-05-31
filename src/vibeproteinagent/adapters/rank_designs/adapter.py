from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from vibeproteinagent.adapters._confidence import normalize_plddt, plddt_unit_score
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext


class RankDesignsInput(BaseModel):
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    analyses: list[dict[str, Any]] = Field(default_factory=list)
    qc_results: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=1000)
    output_prefix: str = Field("rank_designs", pattern=r"^[A-Za-z0-9_.-]+$")

    @field_validator("predictions", "analyses", "qc_results", "candidates", mode="before")
    @classmethod
    def _flatten_nested_rows(cls, value: Any, info: ValidationInfo) -> list[dict[str, Any]]:
        rows = _as_list(value)
        if info.field_name != "candidates" or not rows:
            return rows
        canonical = [_is_canonical_candidate(row) for row in rows]
        if all(canonical):
            return [_row_from_canonical_candidate(row) for row in rows]
        if any(canonical):
            raise ValueError(
                "Mixed canonical and legacy candidate items are not supported. "
                "Pass either all canonical Candidate[] or all legacy row objects."
            )
        return rows


class RankDesignsOutput(BaseModel):
    candidate_count: int
    ranked_candidates: list[dict[str, Any]]
    ranked_json: str
    ranked_tsv: str
    runtime_ms: int


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_as_list(item))
        return rows
    return []


def _is_canonical_candidate(row: dict[str, Any]) -> bool:
    return row.get("schema_version") == "candidate_v1"


def _row_from_canonical_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    producer_tool = str(source.get("tool") or "")
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    producer_scores = (
        scores.get(producer_tool)
        if producer_tool and isinstance(scores.get(producer_tool), dict)
        else {}
    )
    row: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id"),
        "sequence": candidate.get("sequence"),
        "structure_path": candidate.get("structure_path"),
        "source": source,
    }
    if isinstance(producer_scores, dict):
        row.update(producer_scores)
    if scores:
        row["scores"] = dict(scores)
    return {key: value for key, value in row.items() if value is not None}


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None and isinstance(row.get("metrics"), dict):
        value = row["metrics"].get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _aliased_metric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _metric(row, key)
        if value is not None:
            return value
    return None


def _path(row: dict[str, Any]) -> str | None:
    for key in ("predicted_cif", "predicted_pdb", "structure_path", "pdb_path", "cif_path"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _first_value(*rows: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None:
                return value
    return None


def _candidate_id(
    idx: int,
    prediction: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    value = _first_value(
        candidate,
        prediction,
        keys=("candidate_id", "name", "id", "seed", "header"),
    )
    if value:
        return str(value)
    return str(idx + 1)


def _sequence(candidate: dict[str, Any], prediction: dict[str, Any]) -> str | None:
    value = _first_value(
        candidate,
        prediction,
        keys=("sequence", "protein_sequence", "designed_sequence"),
    )
    return str(value) if value else None


def _first_metric(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _score_namespace(row: dict[str, Any], namespace: str) -> dict[str, Any]:
    scores = row.get("scores")
    if not isinstance(scores, dict):
        return {}
    payload = scores.get(namespace)
    return payload if isinstance(payload, dict) else {}


def _score(
    prediction: dict[str, Any],
    analysis: dict[str, Any],
    qc: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    candidate_score = _metric(candidate, "score")
    ranking_score = _first_metric(
        _metric(prediction, "ranking_score"),
        _metric(candidate, "ranking_score"),
        candidate_score,
    )
    plddt = _first_metric(
        _aliased_metric(prediction, "plddt", "mean_plddt"),
        _aliased_metric(candidate, "plddt", "mean_plddt"),
    )
    iptm = _first_metric(_metric(prediction, "iptm"), _metric(candidate, "iptm"))
    ptm = _first_metric(_metric(prediction, "ptm"), _metric(candidate, "ptm"))
    candidate_qc = _score_namespace(candidate, "protein_qc")
    qc_metrics = qc.get("metrics") if isinstance(qc.get("metrics"), dict) else {}
    qc_plddt = (
        _aliased_metric(qc_metrics, "plddt", "mean_plddt")
        if isinstance(qc_metrics, dict)
        else None
    )
    candidate_qc_plddt = _aliased_metric(candidate_qc, "plddt", "mean_plddt")
    liabilities = (
        analysis.get("liabilities")
        if isinstance(analysis.get("liabilities"), list)
        else []
    )
    qc_flags = qc.get("flags") if isinstance(qc.get("flags"), list) else None
    if qc_flags is None:
        qc_flags = (
            candidate_qc.get("flags") if isinstance(candidate_qc.get("flags"), list) else []
        )

    score = 0.0
    if ranking_score is not None:
        score += ranking_score
    plddt_score = plddt_unit_score(_first_metric(plddt, qc_plddt, candidate_qc_plddt))
    if plddt_score is not None:
        score += plddt_score
    if iptm is not None:
        score += 0.5 * iptm
    if ptm is not None:
        score += 0.25 * ptm
    score -= 0.05 * len(liabilities)
    score -= 0.10 * len(qc_flags)

    reported_plddt = normalize_plddt(_first_metric(plddt, qc_plddt, candidate_qc_plddt))
    details = {
        "candidate_score": candidate_score,
        "ranking_score": ranking_score,
        "plddt": reported_plddt,
        "iptm": iptm,
        "ptm": ptm,
        "liability_count": len(liabilities),
        "qc_flag_count": len(qc_flags),
        "qc_flags": qc_flags,
        "structure_path": _path(prediction) or _path(candidate),
        "sequence": _sequence(candidate, prediction),
    }
    if isinstance(candidate.get("scores"), dict):
        details["scores"] = dict(candidate["scores"])
    return score, details


@register_adapter("descriptor.yaml")
class RankDesignsAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: RankDesignsInput, ctx: AdapterRunContext) -> RankDesignsOutput:
        t0 = time.time()
        predictions = _as_list(params.predictions)
        analyses = _as_list(params.analyses)
        qc_results = _as_list(params.qc_results)
        candidates = _as_list(params.candidates)
        count = max(len(predictions), len(analyses), len(qc_results), len(candidates))
        if count == 0:
            raise ValueError(
                "rank_designs requires at least one prediction, analysis, "
                "qc result, or candidate"
            )

        ranked: list[dict[str, Any]] = []
        for idx in range(count):
            prediction = predictions[idx] if idx < len(predictions) else {}
            analysis = analyses[idx] if idx < len(analyses) else {}
            qc = qc_results[idx] if idx < len(qc_results) else {}
            candidate = candidates[idx] if idx < len(candidates) else {}
            score, details = _score(prediction, analysis, qc, candidate)
            ranked.append(
                {
                    "rank": 0,
                    "candidate_id": _candidate_id(idx, prediction, candidate),
                    "score": round(score, 6),
                    **details,
                }
            )

        ranked.sort(key=lambda row: (-row["score"], str(row["candidate_id"])))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        selected = ranked[: params.top_k]

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{params.output_prefix}.json"
        tsv_path = out_dir / f"{params.output_prefix}.tsv"
        json_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        with tsv_path.open("w") as fh:
            fh.write(
                "rank\tcandidate_id\tscore\tranking_score\tplddt\tiptm\tptm\t"
                "liability_count\tqc_flag_count\tstructure_path\n"
            )
            for row in selected:
                fh.write(
                    f"{row['rank']}\t{row['candidate_id']}\t{row['score']}\t"
                    f"{row.get('ranking_score') or ''}\t{row.get('plddt') or ''}\t"
                    f"{row.get('iptm') or ''}\t{row.get('ptm') or ''}\t"
                    f"{row['liability_count']}\t{row['qc_flag_count']}\t"
                    f"{row.get('structure_path') or ''}\n"
                )
        ctx.emit_progress(1.0)
        return RankDesignsOutput(
            candidate_count=count,
            ranked_candidates=selected,
            ranked_json=str(json_path),
            ranked_tsv=str(tsv_path),
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: RankDesignsInput) -> int:
        return 0


__all__ = ["RankDesignsAdapter", "RankDesignsInput", "RankDesignsOutput"]
