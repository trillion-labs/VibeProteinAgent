from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext


class RankDesignsInput(BaseModel):
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    analyses: list[dict[str, Any]] = Field(default_factory=list)
    qc_results: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=1000)
    output_prefix: str = Field("rank_designs", pattern=r"^[A-Za-z0-9_.-]+$")


class RankDesignsOutput(BaseModel):
    candidate_count: int
    ranked_candidates: list[dict[str, Any]]
    ranked_json: str
    ranked_tsv: str
    runtime_ms: int


def _as_list(value: list[dict[str, Any]] | dict[str, Any] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return [row for row in value if isinstance(row, dict)]


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None and isinstance(row.get("metrics"), dict):
        value = row["metrics"].get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _path(row: dict[str, Any]) -> str | None:
    for key in ("predicted_cif", "predicted_pdb", "structure_path", "pdb_path"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _score(
    prediction: dict[str, Any],
    analysis: dict[str, Any],
    qc: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    ranking_score = _metric(prediction, "ranking_score")
    plddt = _metric(prediction, "plddt")
    iptm = _metric(prediction, "iptm")
    ptm = _metric(prediction, "ptm")
    qc_metrics = qc.get("metrics") if isinstance(qc.get("metrics"), dict) else {}
    qc_plddt = _metric(qc_metrics, "plddt") if isinstance(qc_metrics, dict) else None
    liabilities = (
        analysis.get("liabilities")
        if isinstance(analysis.get("liabilities"), list)
        else []
    )
    qc_flags = qc.get("flags") if isinstance(qc.get("flags"), list) else []

    score = 0.0
    if ranking_score is not None:
        score += ranking_score
    if plddt is not None:
        score += plddt / 100.0
    elif qc_plddt is not None:
        score += qc_plddt / 100.0
    if iptm is not None:
        score += 0.5 * iptm
    if ptm is not None:
        score += 0.25 * ptm
    score -= 0.05 * len(liabilities)
    score -= 0.10 * len(qc_flags)

    reported_plddt = plddt if plddt is not None else qc_plddt
    details = {
        "ranking_score": ranking_score,
        "plddt": reported_plddt,
        "iptm": iptm,
        "ptm": ptm,
        "liability_count": len(liabilities),
        "qc_flag_count": len(qc_flags),
        "qc_flags": qc_flags,
        "structure_path": _path(prediction) or _path(candidate),
    }
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
                    "candidate_id": str(
                        candidate.get("candidate_id")
                        or prediction.get("name")
                        or prediction.get("seed")
                        or idx + 1
                    ),
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
