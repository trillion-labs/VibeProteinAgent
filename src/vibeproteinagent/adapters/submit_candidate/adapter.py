from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import ArtifactStore, JobStore, Workspace

Outcome = Literal["accepted", "no_acceptable_candidate"]

_CANDIDATE_LIST_KEYS = (
    "ranked_candidates",
    "top_candidates",
    "top_designs",
    "designs",
    "predictions",
    "candidates",
    "representative_sequences",
)
_STRUCTURE_KEYS = (
    "structure_path",
    "pdb_path",
    "cif_path",
    "model_path",
    "predicted_pdb",
    "predicted_cif",
)
_SCORE_KEYS = (
    "score",
    "ranking_score",
    "ranking_confidence",
    "confidence",
    "model_confidence",
    "plddt",
    "mean_plddt",
    "iptm",
    "ptm",
    "pae",
    "mean_pae",
    "dockq",
    "ipsae",
    "affinity",
    "affinity_pred_value",
    "confidence_score",
    "seq_recovery",
    "recovery",
    "gravy",
    "pI",
    "pi",
    "instability_index",
    "length",
)
_NESTED_SCORE_KEYS = ("metrics", "scores", "confidence_scores")


class CandidateRef(BaseModel):
    job_id: str
    candidate_id: str | None = None
    index: int | None = Field(default=None, ge=0)
    justification: str = Field(min_length=1)
    scores_to_highlight: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_selector(self) -> CandidateRef:
        if not self.candidate_id and self.index is None:
            raise ValueError("candidate ref requires candidate_id or zero-based index")
        return self


class SubmitCandidateInput(BaseModel):
    outcome: Outcome
    candidates: list[CandidateRef] = Field(default_factory=list)
    reasoning: str = Field(min_length=1)
    supersedes: str | None = None

    @model_validator(mode="after")
    def _check_outcome(self) -> SubmitCandidateInput:
        if self.outcome == "accepted" and not self.candidates:
            raise ValueError("accepted submission requires at least one candidate")
        if self.outcome == "no_acceptable_candidate" and self.candidates:
            raise ValueError("no_acceptable_candidate submission must not include candidates")
        return self


class SubmitCandidateOutput(BaseModel):
    submission_id: str
    outcome: Outcome
    terminal: bool
    candidates_resolved: list[dict[str, Any]]
    reasoning: str
    supersedes: str | None = None
    auto_selected: bool = False
    runtime_ms: int


@register_adapter("descriptor.yaml")
class SubmitCandidateAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: SubmitCandidateInput,
        ctx: AdapterRunContext,
    ) -> SubmitCandidateOutput:
        t0 = time.time()
        workspace = Workspace(ctx.workspace_root.name, ctx.workspace_root)
        resolved = [
            _resolve_candidate(workspace, ref)
            for ref in params.candidates
        ]
        submission_id = _submission_id(ctx.job_id, params, resolved)
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "submission_id": submission_id,
            "outcome": params.outcome,
            "terminal": True,
            "candidates_resolved": resolved,
            "reasoning": params.reasoning,
            "supersedes": params.supersedes,
            "auto_selected": False,
        }
        (out_dir / "submission.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        ctx.emit_progress(1.0)
        return SubmitCandidateOutput(
            **payload,
            runtime_ms=int((time.time() - t0) * 1000),
        )


def _submission_id(
    job_id: str,
    params: SubmitCandidateInput,
    resolved: list[dict[str, Any]],
) -> str:
    seed = json.dumps(
        {
            "job_id": job_id,
            "outcome": params.outcome,
            "candidates": resolved,
            "reasoning": params.reasoning,
            "supersedes": params.supersedes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sub_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def _resolve_candidate(workspace: Workspace, ref: CandidateRef) -> dict[str, Any]:
    job = JobStore(workspace).get(ref.job_id)
    output = job.output if isinstance(job.output, dict) else {}
    rows = _candidate_rows(output)
    if not rows:
        raise InvalidParamsError(
            "referenced job has no resolvable candidates",
            details={"job_id": ref.job_id, "tool": job.tool_name},
        )
    index, row = _select_row(rows, ref)
    candidate_id = _candidate_id(row, index)
    scores = _scores(row)
    artifacts = _artifact_refs(workspace, ref.job_id, row)
    return {
        "source_job_id": ref.job_id,
        "source_tool": job.tool_name,
        "source_workflow_id": job.workflow_id,
        "source_workflow_node_id": job.workflow_node_id,
        "selector": {
            "candidate_id": ref.candidate_id,
            "index": ref.index,
        },
        "candidate_id": candidate_id,
        "index": index,
        "rank": _as_int(row.get("rank")),
        "sequence": _first_str(row, "sequence", "protein_sequence", "designed_sequence"),
        "structure_path": _first_str(row, *_STRUCTURE_KEYS),
        "full_structure_path": _first_str(row, "full_structure_path"),
        "backbone_path": _first_str(row, "backbone_path"),
        "capabilities": row.get("capabilities") if isinstance(row.get("capabilities"), dict) else {},
        "score": _as_float(row.get("score")),
        "scores": scores,
        "scores_highlighted": {
            key: scores.get(key)
            for key in ref.scores_to_highlight
            if key in scores
        },
        "justification": ref.justification,
        "artifacts": artifacts,
    }


def _candidate_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    if output.get("status") == "no_acceptable_candidate":
        return []
    canonical = _canonical_candidate_rows(output)
    if canonical:
        return canonical
    rows: list[dict[str, Any]] = []
    for key in _CANDIDATE_LIST_KEYS:
        value = output.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    if not rows and (
        any(key in output for key in _STRUCTURE_KEYS)
        or isinstance(output.get("sequence"), str)
    ):
        rows.append(output)
    return rows


def _canonical_candidate_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    value = output.get("candidates")
    if not isinstance(value, list) or not value:
        return []
    rows = [item for item in value if isinstance(item, dict)]
    if len(rows) != len(value) or not all(_is_canonical_candidate(row) for row in rows):
        return []
    return [_row_from_canonical_candidate(row) for row in rows]


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
        "index": source.get("index"),
        "sequence": candidate.get("sequence"),
        "structure_path": candidate.get("structure_path"),
        "full_structure_path": candidate.get("full_structure_path"),
        "backbone_path": candidate.get("backbone_path"),
        "capabilities": candidate.get("capabilities"),
        "source": source,
    }
    if isinstance(producer_scores, dict):
        row.update(producer_scores)
        row["scores"] = dict(producer_scores)
    return {key: value for key, value in row.items() if value is not None}


def _select_row(
    rows: list[dict[str, Any]],
    ref: CandidateRef,
) -> tuple[int, dict[str, Any]]:
    if ref.candidate_id:
        wanted = str(ref.candidate_id)
        for idx, row in enumerate(rows):
            if _candidate_id(row, idx) == wanted:
                return idx, row
        raise InvalidParamsError(
            "candidate_id not found in referenced job output",
            details={"job_id": ref.job_id, "candidate_id": wanted},
        )
    if ref.index is None or ref.index >= len(rows):
        raise InvalidParamsError(
            "candidate index is outside referenced job output",
            details={"job_id": ref.job_id, "index": ref.index, "candidate_count": len(rows)},
        )
    return ref.index, rows[ref.index]


def _candidate_id(row: dict[str, Any], index: int) -> str:
    for key in ("candidate_id", "name", "id", "seed", "header"):
        value = row.get(key)
        if value:
            return str(value)
    rank = _as_int(row.get("rank"))
    if rank is not None:
        return f"rank-{rank}"
    return str(index)


def _scores(row: dict[str, Any]) -> dict[str, float | int | str | None]:
    scores: dict[str, float | int | str | None] = {}
    for key in _NESTED_SCORE_KEYS:
        value = row.get(key)
        if isinstance(value, dict):
            scores.update(_scores(value))
    for key in _SCORE_KEYS:
        value = row.get(key)
        if isinstance(value, (int, float)):
            scores[key] = value
        elif isinstance(value, str):
            scores[key] = value
    if "plddt" not in scores and "mean_plddt" in scores:
        scores["plddt"] = scores["mean_plddt"]
    if "pae" not in scores and "mean_pae" in scores:
        scores["pae"] = scores["mean_pae"]
    return scores


def _artifact_refs(
    workspace: Workspace,
    job_id: str,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    records = ArtifactStore(workspace).for_job(job_id)
    if not records:
        return []
    path = _first_str(row, *_STRUCTURE_KEYS)
    refs: list[dict[str, Any]] = []
    for field, artifact in sorted(records.items()):
        if path and artifact.pool_path != path and Path(artifact.pool_path).name != Path(path).name:
            continue
        refs.append(
            {
                "field": field,
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "mime": artifact.mime,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
                "path": artifact.pool_path,
            }
        )
    if refs or path:
        return refs
    artifact = next(iter(records.values()))
    return [
        {
            "field": next(iter(records.keys())),
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "mime": artifact.mime,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "path": artifact.pool_path,
        }
    ]


def _first_str(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
