from __future__ import annotations

from typing import Any, cast

from vibeproteinagent.workflows import WorkflowStore
from vibeproteinagent.workspace import ArtifactStore, JobStore, Workspace

from .artifact_summarizer import ArtifactSummarizer


class ResultSummarizer:
    """Build compact result packets from persisted jobs/workflows."""

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
            "output": _compact(job.output),
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
                    "output": _compact(n.output),
                    "artifacts": artifacts_by_job.get(n.job_id or "", {}),
                    "error_code": n.error_code,
                    "error_message": n.error_message,
                    "error_details": n.error_details,
                }
                for n in nodes
            ],
            "top_designs": _extract_top_designs(nodes),
            "metrics_used": _extract_metric_names(nodes),
            "artifacts": artifacts_by_job,
            "notes": [],
        }


def _compact(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
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
        )
        if isinstance(candidates, list):
            for idx, item in enumerate(candidates[:10]):
                designs.append({"node_id": node.node_id, "rank": idx + 1, "design": _compact(item)})
    return designs[:10]


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
