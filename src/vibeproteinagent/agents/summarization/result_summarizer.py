from __future__ import annotations

from typing import Any, cast

from vibeproteinagent.workflows import WorkflowStore
from vibeproteinagent.workspace import ArtifactStore, JobStore, Workspace

from .artifact_summarizer import ArtifactSummarizer


class ResultSummarizer:
    """Build display/local-summary packets from persisted jobs/workflows.

    These summaries are not candidate materialization sources. Full candidate
    payloads belong in raw workflow storage and CandidateArchive records.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.artifacts = ArtifactSummarizer()

    def summarize_job(self, job_id: str) -> dict[str, Any]:
        job = JobStore(self.workspace).get(job_id)
        artifacts = _artifact_refs(ArtifactStore(self.workspace).for_job(job.job_id))
        return {
            "job_id": job.job_id,
            "state": job.state.value,
            "tool": job.tool_name,
            "version": job.tool_version,
            "params_hash": job.params_hash,
            "workflow_id": job.workflow_id,
            "workflow_node_id": job.workflow_node_id,
            "output": decision_safe_summary(_compact(job.output)),
            "artifacts": artifacts,
            "error": (
                {
                    "code": job.error_code,
                    "message": job.error_message,
                    "details": job.error_details,
                }
                if job.error_code or job.error_message
                else None
            ),
        }

    def summarize_workflow(self, workflow_id: str) -> dict[str, Any]:
        store = WorkflowStore(self.workspace)
        wf = store.get(workflow_id)
        nodes = cast(list[Any], store.list_nodes(workflow_id))
        job_ids = [n.job_id for n in nodes if n.job_id]
        artifact_store = ArtifactStore(self.workspace)
        artifacts_by_job = {
            job_id: _artifact_refs(artifact_store.for_job(job_id))
            for job_id in job_ids
        }
        return {
            "workflow_id": wf.workflow_id,
            "name": wf.name,
            "version": wf.version,
            "state": wf.state,
            "job_ids": job_ids,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "tool": n.tool_name,
                    "state": n.state,
                    "job_id": n.job_id,
                    "output": decision_safe_summary(_compact(n.output)),
                    "artifacts": artifacts_by_job.get(n.job_id or "", {}),
                    "error_code": n.error_code,
                    "error_message": n.error_message,
                    "error_details": n.error_details,
                }
                for n in nodes
            ],
            "top_designs": _extract_top_designs(nodes),
            "submissions": _extract_submissions(nodes),
            "final_submission": _latest_final_submission(nodes),
            "metrics_used": _extract_metric_names(nodes),
            "artifacts": artifacts_by_job,
            "notes": [],
        }


def decision_safe_summary(value: Any) -> Any:
    """Remove audit-only duplicate lineage from display/local decision summaries.

    This is not the source for LLM decision prompts; those receive the lossy
    PromptContext contract instead.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [decision_safe_summary(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("schema_version") == "candidate_v1":
        return _candidate_display_card(value)
    out: dict[str, Any] = {}
    for key, item in value.items():
        key_str = str(key)
        if key_str == "merge_audit" and isinstance(item, dict):
            out[key_str] = _count_only_merge_audit(item)
            continue
        if key_str.startswith("__candidate_merge") or key_str in {
            "duplicate_groups",
            "versions",
        }:
            continue
        out[key_str] = decision_safe_summary(item)
    return out


def _count_only_merge_audit(audit: dict[str, Any]) -> dict[str, Any]:
    duplicate_groups = audit.get("duplicate_groups")
    conflicts = audit.get("conflicts")
    summary: dict[str, Any] = {
        "duplicates_removed": _int_or_zero(audit.get("duplicates_removed")),
        "duplicate_group_count": (
            len(duplicate_groups)
            if isinstance(duplicate_groups, list)
            else _int_or_zero(audit.get("duplicate_group_count"))
        ),
        "conflict_count": (
            len(conflicts)
            if isinstance(conflicts, list)
            else _int_or_zero(audit.get("conflict_count"))
        ),
    }
    if "output_count" in audit:
        summary["output_count"] = _int_or_zero(audit.get("output_count"))
    if audit.get("dedup_policy") is not None:
        summary["dedup_policy"] = str(audit.get("dedup_policy"))
    return summary


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compact(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict) and value.get("schema_version") == "candidate_v1":
        return _candidate_display_card(value)
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, list):
        return [_compact(v, depth=depth + 1) for v in value[:20]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= 50:
                out["..."] = f"{len(value) - 50} more keys"
                break
            out[str(key)] = _compact(val, depth=depth + 1)
        return out
    return str(value)


def _candidate_display_card(candidate: dict[str, Any]) -> dict[str, Any]:
    card: dict[str, Any] = {
        "schema_version": "candidate_v1",
        "candidate_id": candidate.get("candidate_id"),
    }
    source = _candidate_source_card(candidate.get("source"))
    if source:
        card["source"] = source
    capabilities = _candidate_capability_flags(candidate.get("capabilities"))
    if capabilities:
        card["capabilities"] = capabilities
    modalities = _candidate_available_modalities(candidate, capabilities)
    if modalities:
        card["available_modalities"] = modalities
    sequence = candidate.get("sequence")
    if isinstance(sequence, str):
        card["sequence_length"] = len(sequence)
    scores = candidate.get("scores")
    if isinstance(scores, dict):
        card["score_keys"] = sorted(str(key) for key in scores)
    artifacts = candidate.get("artifacts")
    if isinstance(artifacts, dict):
        card["artifact_keys"] = sorted(str(key) for key in artifacts)
    for field, output_field in (
        ("structure_path", "has_structure_path"),
        ("backbone_path", "has_backbone_path"),
        ("full_structure_path", "has_full_structure_path"),
    ):
        if isinstance(candidate.get(field), str) and candidate.get(field).strip():
            card[output_field] = True
    status = candidate.get("status")
    if isinstance(status, (str, int, float, bool)):
        card["status"] = status
    return {key: value for key, value in card.items() if value not in (None, {}, [])}


def _candidate_source_card(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    allowed = {
        "tool",
        "node_id",
        "job_id",
        "workflow_id",
        "adapter",
        "source_kind",
        "local_id",
    }
    return {
        str(key): value
        for key, value in source.items()
        if key in allowed and isinstance(value, (str, int, float, bool))
    }


def _candidate_capability_flags(capabilities: Any) -> dict[str, bool]:
    if not isinstance(capabilities, dict):
        return {}
    return {
        str(key): bool(value)
        for key, value in capabilities.items()
        if isinstance(value, bool) or str(key).startswith("has_")
    }


def _candidate_available_modalities(
    candidate: dict[str, Any],
    capabilities: dict[str, bool],
) -> list[str]:
    modalities = [key.removeprefix("has_") for key, value in capabilities.items() if value]
    if isinstance(candidate.get("sequence"), str):
        modalities.append("sequence")
    for field, modality in (
        ("structure_path", "structure"),
        ("backbone_path", "backbone_structure"),
        ("full_structure_path", "full_structure"),
    ):
        if isinstance(candidate.get(field), str) and candidate.get(field).strip():
            modalities.append(modality)
    return sorted(set(modalities))


def _extract_metric_names(nodes: list[Any]) -> list[str]:
    names: set[str] = set()
    for node in nodes:
        if isinstance(node.output, dict):
            for key, value in node.output.items():
                if isinstance(value, (int, float)):
                    names.add(str(key))
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if isinstance(subval, (int, float)):
                            names.add(f"{key}.{subkey}")
    return sorted(names)


def _extract_top_designs(nodes: list[Any]) -> list[dict[str, Any]]:
    designs: list[dict[str, Any]] = []
    for node in nodes:
        output = node.output
        if not isinstance(output, dict):
            continue
        candidates = (
            output.get("top_designs")
            or output.get("designs")
            or output.get("ranked")
            or output.get("ranked_candidates")
            or output.get("representative_sequences")
        )
        if isinstance(candidates, list):
            for idx, item in enumerate(candidates[:10]):
                designs.append({"node_id": node.node_id, "rank": idx + 1, "design": _compact(item)})
    return designs[:10]


def _extract_submissions(nodes: list[Any]) -> list[dict[str, Any]]:
    submissions: list[dict[str, Any]] = []
    for node in nodes:
        if node.tool_name != "submit_candidate" or node.state != "completed":
            continue
        output = node.output
        if not isinstance(output, dict) or output.get("terminal") is not True:
            continue
        submission = _compact(output)
        if isinstance(submission, dict):
            submission["node_id"] = node.node_id
            submission["job_id"] = node.job_id
            submissions.append(submission)
    return submissions


def _latest_final_submission(nodes: list[Any]) -> dict[str, Any] | None:
    submissions = _extract_submissions(nodes)
    if not submissions:
        return None
    superseded = {
        str(submission.get("supersedes"))
        for submission in submissions
        if submission.get("supersedes")
    }
    for submission in reversed(submissions):
        if str(submission.get("submission_id")) not in superseded:
            return submission
    return submissions[-1]


def _artifact_refs(records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        field: {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "mime": artifact.mime,
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "path": artifact.pool_path,
        }
        for field, artifact in records.items()
    }
