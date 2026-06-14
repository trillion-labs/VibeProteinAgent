from __future__ import annotations

import contextlib
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from vibeproteinagent.adapters import ToolRegistry, validate_params
from vibeproteinagent.adapters.base import torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.sequence_analysis.adapter import SequenceAnalysisInput
from vibeproteinagent.reporting.models import ReportCandidate, ReportModel
from vibeproteinagent.workspace import ArtifactStore, JobState, JobStore, Workspace

REPORT_ENRICHMENT_CALL_ID_PREFIX = "report_enrich_sequence_analysis"
REPORT_ENRICHMENT_CONTRACT_VERSION = 3


def ensure_sequence_analysis_enrichment(
    workspace: Workspace,
    source_id: str,
    model: ReportModel,
) -> list[str]:
    """Run sequence_analysis for sequence-bearing candidates not already analyzed."""

    candidates = _sequence_analysis_candidates(model)
    if not candidates:
        return []

    params = {
        "candidates": candidates,
        "include_regions": True,
        "include_residue_table": True,
        "output_prefix": "report_sequence_analysis",
    }
    call_id = _enrichment_call_id(source_id, params)

    registry = ToolRegistry()
    registry.discover()
    descriptor, adapter_cls = registry.get("sequence_analysis", "*")
    validate_params(descriptor, params)

    store = JobStore(workspace)
    rec = store.submit(
        descriptor.name,
        descriptor.version,
        params,
        call_id=call_id,
        workspace_name=workspace.name,
    )
    if rec.state is JobState.COMPLETED:
        return [rec.job_id]
    if rec.state is JobState.SUBMITTED:
        rec = store.transition(rec.job_id, JobState.QUEUED)
    if rec.state is JobState.QUEUED:
        rec = store.transition(rec.job_id, JobState.RUNNING)
    if rec.state is not JobState.RUNNING:
        return [rec.job_id]

    job_dir = workspace.job_dir(rec.job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "stdout.log"

    def _log(line: str) -> None:
        with contextlib.suppress(Exception):
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line if line.endswith("\n") else line + "\n")

    ctx = AdapterRunContext(
        job_id=rec.job_id,
        workspace_root=workspace.root,
        job_dir=job_dir,
        is_cancelled=lambda: False,
        log=_log,
        emit_progress=lambda _pct: None,
        fusion_inputs=None,
    )
    try:
        adapter = adapter_cls({}, Path(), torch.device("cpu"))
        output_model = adapter.run(SequenceAnalysisInput.model_validate(params), ctx)
        output = output_model.model_dump(mode="json")
        _register_output_artifacts(workspace, rec.job_id, output, job_dir)
        store.transition(rec.job_id, JobState.COMPLETED, output=output)
    except Exception as exc:
        store.transition(
            rec.job_id,
            JobState.FAILED,
            error_code=getattr(exc, "code", "E_INTERNAL"),
            error_message=str(exc),
            error_details=getattr(exc, "details", None),
        )
    return [rec.job_id]


def _sequence_analysis_candidates(model: ReportModel) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    all_candidates = [*model.final_candidates, *model.candidates]
    analyzed_ids: set[str] = set()
    for candidate in all_candidates:
        if not _has_sequence_analysis(candidate):
            continue
        if candidate.candidate_id:
            analyzed_ids.add(candidate.candidate_id)
    for candidate in all_candidates:
        if not candidate.sequence or _has_sequence_analysis(candidate):
            continue
        if candidate.candidate_id in analyzed_ids:
            continue
        key = candidate.candidate_id or f"seq:{candidate.sequence}"
        if key in seen:
            continue
        seen.add(key)
        out.append(_candidate_to_canonical(candidate))
    return out


def _has_sequence_analysis(candidate: ReportCandidate) -> bool:
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    if isinstance(scores.get("sequence_analysis"), dict):
        return True
    return any(str(key).startswith("sequence_analysis:") for key in scores)


def _candidate_to_canonical(candidate: ReportCandidate) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "candidate_v1",
        "candidate_id": candidate.candidate_id,
        "sequence": candidate.sequence,
        "source": _canonical_source(candidate),
        "scores": _canonical_score_namespaces(candidate.scores),
        "artifacts": {
            f"{artifact.field}:{idx}": {
                "field": artifact.field,
                "artifact_id": artifact.artifact_id,
                "kind": artifact.kind,
                "mime": artifact.mime,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
                "path": artifact.path,
                "job_id": artifact.job_id,
            }
            for idx, artifact in enumerate(candidate.artifacts)
        },
    }
    if candidate.structure_path:
        payload["structure_path"] = candidate.structure_path
    return payload


def _canonical_source(candidate: ReportCandidate) -> dict[str, Any]:
    source = candidate.source if isinstance(candidate.source, dict) else {}
    local_id = (
        source.get("local_id")
        or source.get("native_id")
        or candidate.candidate_id
        or "candidate"
    )
    out: dict[str, Any] = {
        "tool": str(source.get("tool") or "report_enrichment"),
        "local_id": str(local_id),
    }
    for key in ("node_id", "job_id", "native_id"):
        value = source.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    index = source.get("index")
    if isinstance(index, int) and index >= 0:
        out["index"] = index
    return out


def _canonical_score_namespaces(scores: Any) -> dict[str, Any]:
    if not isinstance(scores, dict):
        return {}
    return {
        str(key): dict(value)
        for key, value in scores.items()
        if isinstance(value, dict)
    }


def _enrichment_call_id(source_id: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            {
                "contract_version": REPORT_ENRICHMENT_CONTRACT_VERSION,
                "params": params,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    safe_source = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in source_id)
    return f"{REPORT_ENRICHMENT_CALL_ID_PREFIX}_{safe_source}_{digest}"


def _register_output_artifacts(
    workspace: Workspace,
    job_id: str,
    output: dict[str, Any],
    job_dir: Path,
) -> None:
    artifact_store = ArtifactStore(workspace)
    for field, path in _iter_output_paths(output):
        p = Path(path)
        with contextlib.suppress(Exception):
            resolved = p.resolve()
            if not resolved.is_file() or not resolved.is_relative_to(job_dir.resolve()):
                continue
            mime, _ = mimetypes.guess_type(resolved.name)
            artifact_store.register(
                job_id,
                field,
                resolved,
                kind=resolved.suffix.lower().lstrip(".") or "file",
                mime=mime,
            )


def _iter_output_paths(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(_iter_output_paths(item, child_prefix))
        return out
    if isinstance(value, list):
        out: list[tuple[str, str]] = []
        for idx, item in enumerate(value):
            out.extend(_iter_output_paths(item, f"{prefix}[{idx}]"))
        return out
    if isinstance(value, str) and value:
        return [(prefix or "artifact", value)]
    return []
