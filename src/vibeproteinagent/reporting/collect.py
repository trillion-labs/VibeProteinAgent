from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from vibeproteinagent.agents.planning.candidate_merge import merge_candidate_records
from vibeproteinagent.core.errors import InvalidParamsError, NotFoundError, WorkspaceError
from vibeproteinagent.workspace import Workspace
from vibeproteinagent.workspace.types import (
    ArtifactRecord,
    JobRecord,
    JobState,
    Priority,
    WorkflowNodeRecord,
    WorkflowRecord,
)

from .models import (
    AgentDecision,
    AgentMessage,
    AgentStepSummary,
    AgentToolCall,
    AgentTranscript,
    MSABlock,
    ReportArtifact,
    ReportCandidate,
    ReportModel,
    ReportWarning,
    ReproducibilityBlock,
    RunSummary,
    WorkflowGraph,
    WorkflowNodeSummary,
)

_STRUCTURE_SUFFIXES = {".pdb", ".ent", ".cif", ".mmcif"}
_FASTA_SUFFIXES = {".fa", ".fasta", ".faa", ".a3m", ".sto", ".aln"}

# Tools whose output is (or naturally carries) a multiple-sequence alignment.
# A workflow must include at least one of these for the MSA section to render —
# otherwise a stray single-sequence FASTA gets misclassified as a "1 sequence"
# MSA. Names match descriptor.yaml ``name`` values, lowercased.
_MSA_GENERATING_TOOLS = frozenset({
    "boltz2_affinity",
    "boltzgen",
    "foldseek",
    "mmseqs2_gpu",
    "protenix_v2",
    "template_select",
})

# Top-K alignment rows to keep after sorting by % identity vs the query.
_MSA_DISPLAY_TOP_K = 10
_MSA_HARD_CAP = 200  # safety: do not parse arbitrarily large alignment files
_AVAILABILITY_EVENT_LIMIT = 12
_AVAILABILITY_FILTERED_CANDIDATE_LIMIT = 5
_SCORE_KEYS = (
    # Generic / ranking
    "score",
    "ranking_score",
    "ranking_confidence",
    "confidence",
    "model_confidence",
    # Structural confidence
    "plddt",
    "mean_plddt",
    "iptm",
    "ptm",
    "pae",
    "mean_pae",
    "dockq",
    "ipsae",
    # Binding / affinity (Boltz-2, RFdiffusion, etc.). Includes the
    # specific key names emitted by the boltz2_affinity adapter
    # (``affinity_pred_value``, ``affinity_probability_binary``,
    # ``confidence_score``) so those metrics flow through to candidate
    # cards instead of being silently filtered out.
    "dg",
    "delta_g",
    "affinity",
    "affinity_value",
    "affinity_pred_value",
    "affinity_probability",
    "affinity_probability_binary",
    "binding_probability",
    "binder_probability",
    "confidence_score",
    "ipsae_score",
    "iplddt",
    # Sequence design
    "recovery",
    "seq_recovery",
    # Biophysical / QC (protein_qc)
    "gravy",
    "pi",
    "instability_index",
    "polybasic_hits",
    "aromatic_fraction",
    # Context (kept for filtering, never bar-rendered)
    "num_residues",
    "length",
    "runtime_ms",
)

# Keys whose value is itself a numeric metric dict that should be flattened
# into the candidate's score map (e.g. esm3.output.confidence_scores =
# {mean_plddt: ..., ptm: ...}). Without this, those metrics never reach
# the report's candidate cards or table.
_NESTED_SCORE_KEYS = ("confidence_scores", "metrics", "scores")


def collect_report_model(workspace: Workspace, source_id: str, *, top_k: int = 5) -> ReportModel:
    del top_k  # top-k is a rendering concern; collection keeps the complete model.
    if source_id.startswith("wf_"):
        return _collect_workflow(workspace, source_id)
    if source_id.startswith("job_"):
        return _collect_job(workspace, source_id)
    raise InvalidParamsError(
        "report id must start with job_ or wf_",
        details={"id": source_id, "allowed_prefixes": ["job_", "wf_"]},
    )


def _collect_job(workspace: Workspace, job_id: str) -> ReportModel:
    job = _read_job_readonly(workspace, job_id)
    artifacts = _artifact_list(
        _read_artifacts_for_job_readonly(workspace, job.job_id),
        job_id=job.job_id,
    )
    node = _node_from_job(job, artifacts)
    warnings = _warnings_from_job(job)
    candidates = _candidate_archive_candidates(
        workspace,
        workflow_id=job.workflow_id,
        job_ids=[job.job_id],
        node_ids=[node.node_id],
        artifacts=artifacts,
    )
    if not candidates:
        candidates = _normalize_candidates(
            [node],
            artifacts,
            node_params={node.node_id: job.params if isinstance(job.params, dict) else {}},
        )
    submissions = _submission_outputs([node])
    final_submission = _latest_final_submission(submissions)
    final_candidates = _candidates_from_submission(final_submission)
    if not candidates:
        candidates = _structure_candidates(
            artifacts,
            source={"kind": "job", "job_id": job.job_id},
        )
    msa_blocks = _collect_msa_blocks(
        artifacts,
        workflow_tools={job.tool_name.lower()},
        tool_by_job={job.job_id: job.tool_name.lower()},
    )
    return ReportModel(
        summary=RunSummary(
            kind="job",
            id=job.job_id,
            workspace=workspace.name,
            state=job.state.value,
            name=job.tool_name,
            version=job.tool_version,
            submitted_at=job.submitted_at,
            finished_at=job.finished_at,
            runtime_ms=_duration_ms(job.submitted_at, job.finished_at),
        ),
        graph=WorkflowGraph(workflow_id=job.workflow_id, nodes=[node]),
        candidates=candidates,
        final_candidates=final_candidates,
        final_submission=final_submission,
        submissions=submissions,
        artifacts=artifacts,
        msa_blocks=msa_blocks,
        warnings=warnings,
        reproducibility=_reproducibility(
            source_id=job.job_id,
            nodes=[node],
            artifacts=artifacts,
            workspace=workspace.name,
        ),
        agent_transcript=_collect_agent_transcript(
            workspace, workflow_id=job.workflow_id, job_id=job.job_id,
        ),
        satisfied_semantic_facets=_satisfied_semantic_facets_from_nodes([node]),
        empty_candidate_events=_empty_candidate_events_from_nodes([node]),
        candidate_capability_filter_events=(
            _candidate_capability_filter_events_from_nodes([node])
        ),
        candidate_merge_events=_candidate_merge_events_from_nodes([node]),
        parameter_resolution_events=_parameter_resolution_events_from_nodes([node]),
        raw_summary=_raw_job_summary(job, artifacts),
    )


def _collect_workflow(workspace: Workspace, workflow_id: str) -> ReportModel:
    wf = _read_workflow_readonly(workspace, workflow_id)
    nodes = _read_workflow_nodes_readonly(workspace, workflow_id)
    report_nodes: list[WorkflowNodeSummary] = []
    all_artifacts: list[ReportArtifact] = []
    warnings: list[ReportWarning] = []

    params_by_node: dict[str, dict[str, Any]] = {}
    for node in nodes:
        artifacts: list[ReportArtifact] = []
        job: JobRecord | None = None
        if node.job_id:
            try:
                job = _read_job_readonly(workspace, node.job_id)
                artifacts = _artifact_list(
                    _read_artifacts_for_job_readonly(workspace, job.job_id),
                    job_id=job.job_id,
                )
                warnings.extend(_warnings_from_job(job, source=node.node_id))
                if isinstance(job.params, dict):
                    params_by_node[node.node_id] = job.params
            except Exception as exc:
                warnings.append(
                    ReportWarning(
                        level="warning",
                        message=f"Could not load job {node.job_id}: {exc}",
                        source=node.node_id,
                    )
                )
        report_node = WorkflowNodeSummary(
            node_id=node.node_id,
            tool=node.tool_name,
            version=node.tool_version,
            state=node.state,
            job_id=node.job_id,
            upstream=node.upstream,
            params_hash=node.params_hash,
            params=params_by_node.get(node.node_id, {}),
            cache_key=node.cache_key,
            output=node.output,
            parameter_resolution_events=_parameter_resolution_events_from_node_payload(
                node.output,
                node.error_details,
            ),
            parameter_resolution_summary=_parameter_resolution_summary_from_node_payload(
                node.output,
                node.error_details,
            ),
            artifacts=artifacts,
            error_code=node.error_code,
            error_message=node.error_message,
            error_details=node.error_details,
        )
        report_nodes.append(report_node)
        all_artifacts.extend(artifacts)
        if node.state in {"failed", "skipped"}:
            warnings.append(
                ReportWarning(
                    level="error" if node.state == "failed" else "warning",
                    message=f"Workflow node {node.node_id} is {node.state}",
                    source=node.node_id,
                )
            )

    candidates = _candidate_archive_candidates(
        workspace,
        workflow_id=wf.workflow_id,
        job_ids=[node.job_id for node in report_nodes if node.job_id],
        node_ids=[node.node_id for node in report_nodes],
        artifacts=all_artifacts,
    )
    if not candidates:
        candidates = _normalize_candidates(
            report_nodes, all_artifacts, node_params=params_by_node,
        )
    submissions = [
        *_submission_outputs(report_nodes),
        *_agent_submission_outputs(workspace, workflow_id=wf.workflow_id, job_id=None),
    ]
    final_submission = _latest_final_submission(submissions)
    final_candidates = _candidates_from_submission(final_submission)
    if not candidates:
        candidates = _structure_candidates(
            all_artifacts, source={"kind": "workflow", "workflow_id": wf.workflow_id}
        )
    if (
        final_submission is None
        and candidates
        and _agent_allows_auto_selection(workspace, wf.workflow_id)
    ):
        final_submission, final_candidates = _auto_submission_from_candidates(candidates)
        warnings.append(
            ReportWarning(
                level="warning",
                message=(
                    "No explicit submit_candidate output was found; the report "
                    "auto-selected the highest ranked available candidate."
                ),
                source="submit_candidate",
            )
        )
    workflow_tools = {(n.tool or "").lower() for n in report_nodes}
    tool_by_job = {n.job_id: (n.tool or "").lower() for n in report_nodes if n.job_id}
    return ReportModel(
        summary=RunSummary(
            kind="workflow",
            id=wf.workflow_id,
            workspace=workspace.name,
            state=wf.state,
            name=wf.name,
            version=wf.version,
            submitted_at=wf.submitted_at,
            finished_at=wf.finished_at,
            runtime_ms=_duration_ms(wf.submitted_at, wf.finished_at),
        ),
        graph=WorkflowGraph(workflow_id=wf.workflow_id, nodes=report_nodes),
        candidates=candidates,
        final_candidates=final_candidates,
        final_submission=final_submission,
        submissions=submissions,
        artifacts=all_artifacts,
        msa_blocks=_collect_msa_blocks(
            all_artifacts, workflow_tools=workflow_tools, tool_by_job=tool_by_job,
        ),
        warnings=warnings,
        reproducibility=_reproducibility(
            source_id=wf.workflow_id,
            nodes=report_nodes,
            artifacts=all_artifacts,
            workspace=workspace.name,
        ),
        agent_transcript=_collect_agent_transcript(
            workspace, workflow_id=wf.workflow_id,
        ),
        satisfied_semantic_facets=_satisfied_semantic_facets_from_nodes(report_nodes),
        empty_candidate_events=_empty_candidate_events_from_nodes(report_nodes),
        candidate_capability_filter_events=(
            _candidate_capability_filter_events_from_nodes(report_nodes)
        ),
        candidate_merge_events=_candidate_merge_events_from_nodes(report_nodes),
        parameter_resolution_events=_parameter_resolution_events_from_nodes(report_nodes),
        raw_summary=_raw_workflow_summary(
            workflow_id=wf.workflow_id,
            name=wf.name,
            version=wf.version,
            state=wf.state,
            nodes=report_nodes,
            artifacts=all_artifacts,
        ),
    )


def _node_from_job(job: JobRecord, artifacts: list[ReportArtifact]) -> WorkflowNodeSummary:
    return WorkflowNodeSummary(
        node_id=job.workflow_node_id or job.tool_name,
        tool=job.tool_name,
        version=job.tool_version,
        state=job.state.value,
        job_id=job.job_id,
        params_hash=job.params_hash,
        params=job.params if isinstance(job.params, dict) else {},
        output=job.output,
        parameter_resolution_events=_parameter_resolution_events_from_node_payload(
            job.output,
            job.error_details,
        ),
        parameter_resolution_summary=_parameter_resolution_summary_from_node_payload(
            job.output,
            job.error_details,
        ),
        artifacts=artifacts,
        error_code=job.error_code,
        error_message=job.error_message,
        error_details=job.error_details,
    )


def _empty_candidate_events_from_nodes(
    nodes: list[WorkflowNodeSummary],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        for key in ("empty_candidate_event", "upstream_empty_candidate_event"):
            event = output.get(key)
            if isinstance(event, dict):
                events.append(_availability_empty_event(node, event))
        facets = output.get("empty_facets")
        if isinstance(facets, list):
            for facet in facets:
                if isinstance(facet, dict):
                    events.append(_availability_empty_event(node, facet))
        if len(events) >= _AVAILABILITY_EVENT_LIMIT:
            break
    return events[:_AVAILABILITY_EVENT_LIMIT]


def _satisfied_semantic_facets_from_nodes(
    nodes: list[WorkflowNodeSummary],
) -> list[dict[str, Any]]:
    facets: dict[str, dict[str, Any]] = {}
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        for row in _candidate_rows_for_availability(output):
            semantic = _semantic_relevance_for_row(row)
            if not semantic:
                continue
            intent = semantic.get("intent") if isinstance(semantic.get("intent"), dict) else {}
            row_intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
            intent_id = (
                semantic.get("intent_id")
                or intent.get("intent_id")
                or row_intent.get("intent_id")
            )
            label = semantic.get("label") or intent.get("label")
            query = semantic.get("query") or intent.get("query")
            selected_term = (
                semantic.get("selected_term")
                or intent.get("selected_term")
                or intent.get("term_id")
                or intent.get("go_id")
                or intent.get("interpro_id")
            )
            key = str(intent_id or label or query or selected_term or "").strip()
            if not key:
                continue
            entry = facets.setdefault(
                key,
                {
                    "intent_id": intent_id,
                    "label": label,
                    "query": query,
                    "selected_terms": [],
                    "candidate_count": 0,
                    "candidate_ids": [],
                    "tools": [],
                    "nodes": [],
                    "best_sequence_relevance_score": None,
                    "selected_reason": semantic.get("selected_reason")
                    or intent.get("selected_reason"),
                },
            )
            entry["candidate_count"] += 1
            _append_bounded_unique(entry["candidate_ids"], row.get("candidate_id"), limit=3)
            _append_bounded_unique(entry["tools"], node.tool, limit=3)
            _append_bounded_unique(entry["nodes"], node.node_id, limit=3)
            _append_bounded_unique(entry["selected_terms"], selected_term, limit=3)
            score = _as_float(semantic.get("sequence_relevance_score"))
            current = entry.get("best_sequence_relevance_score")
            if score is not None and (current is None or score > current):
                entry["best_sequence_relevance_score"] = score
        if len(facets) >= _AVAILABILITY_EVENT_LIMIT:
            break
    return list(facets.values())[:_AVAILABILITY_EVENT_LIMIT]


def _candidate_rows_for_availability(output: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "ranked_candidates",
        "top_candidates",
        "top_designs",
        "designs",
        "predictions",
        "candidates",
        "representative_sequences",
        "sequences",
    ):
        value = output.get(key)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    return rows


def _semantic_relevance_for_row(row: dict[str, Any]) -> dict[str, Any] | None:
    scores = row.get("scores")
    if isinstance(scores, dict):
        semantic = scores.get("semantic_relevance")
        if isinstance(semantic, dict):
            return semantic
        for nested in scores.values():
            if not isinstance(nested, dict):
                continue
            semantic = nested.get("semantic_relevance")
            if isinstance(semantic, dict):
                payload = dict(semantic)
                facets = semantic.get("facets")
                if isinstance(facets, list) and facets and isinstance(facets[0], dict):
                    payload.setdefault("intent", facets[0])
                return payload
    semantic = row.get("semantic_relevance")
    if isinstance(semantic, dict):
        return semantic
    return None


def _append_bounded_unique(target: list[Any], value: Any, *, limit: int) -> None:
    if value in (None, "") or len(target) >= limit:
        return
    if value not in target:
        target.append(value)


def _candidate_capability_filter_events_from_nodes(
    nodes: list[WorkflowNodeSummary],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        event = output.get("candidate_capability_filter_event")
        if isinstance(event, dict):
            events.append(_availability_filter_event(node, event))
        if len(events) >= _AVAILABILITY_EVENT_LIMIT:
            break
    return events[:_AVAILABILITY_EVENT_LIMIT]


def _candidate_merge_events_from_nodes(
    nodes: list[WorkflowNodeSummary],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        audit = output.get("merge_audit")
        if not isinstance(audit, dict):
            continue
        groups = audit.get("duplicate_groups")
        if not isinstance(groups, list) or not groups:
            continue
        conflicts = audit.get("conflicts")
        if not isinstance(conflicts, list):
            conflicts = []
        compact_groups = [
            _compact_candidate_merge_group(group)
            for group in groups[:_AVAILABILITY_EVENT_LIMIT]
            if isinstance(group, dict)
        ]
        if not compact_groups:
            continue
        events.append(
            {
                "node_id": node.node_id,
                "tool": node.tool,
                "duplicates_removed": _as_int(audit.get("duplicates_removed")) or 0,
                "duplicate_group_count": len(groups),
                "conflict_count": len(conflicts),
                "groups": compact_groups,
                "truncated": len(groups) > len(compact_groups),
            }
        )
        if len(events) >= _AVAILABILITY_EVENT_LIMIT:
            break
    return events[:_AVAILABILITY_EVENT_LIMIT]


def _compact_candidate_merge_group(group: dict[str, Any]) -> dict[str, Any]:
    versions = group.get("versions")
    if not isinstance(versions, list):
        versions = []
    tools: list[str] = []
    native_ids: list[str] = []
    score_namespaces: list[str] = []
    for version in versions:
        if not isinstance(version, dict):
            continue
        source = version.get("source") if isinstance(version.get("source"), dict) else {}
        _append_bounded_unique(tools, source.get("tool"), limit=6)
        _append_bounded_unique(native_ids, source.get("native_id"), limit=6)
        for namespace in version.get("score_namespaces") or []:
            _append_bounded_unique(score_namespaces, namespace, limit=8)
    return {
        "identity_key": group.get("identity_key"),
        "identity_type": group.get("identity_type"),
        "retained_candidate_id": group.get("retained_candidate_id"),
        "merged_count": _as_int(group.get("merged_count")) or len(versions),
        "source_tools": tools,
        "native_ids": native_ids,
        "score_namespaces": score_namespaces,
    }


def _parameter_resolution_events_from_output(output: Any) -> list[dict[str, Any]]:
    payload = output if isinstance(output, dict) else {}
    raw_events = payload.get("parameter_resolution_events")
    if not isinstance(raw_events, list):
        return []
    events: list[dict[str, Any]] = []
    for event in raw_events:
        if isinstance(event, dict):
            events.append(dict(event))
    return events


def _parameter_resolution_events_from_node_payload(
    output: Any,
    error_details: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    events = _parameter_resolution_events_from_output(output)
    details = error_details if isinstance(error_details, dict) else {}
    conflict = details.get("parameter_conflict_event")
    if isinstance(conflict, dict):
        events.append(dict(conflict))
    return events


def _parameter_resolution_summary_from_node_payload(
    output: Any,
    error_details: dict[str, Any] | None,
) -> dict[str, int]:
    summary = {
        "derived_count": 0,
        "planner_override_count": 0,
        "user_validation_count": 0,
        "adapter_default_count": 0,
        "conflict_count": 0,
    }
    for event in _parameter_resolution_events_from_node_payload(output, error_details):
        source_before = str(event.get("source_before") or event.get("source") or "")
        source_after = str(event.get("source_after") or "")
        status = str(event.get("status") or "")
        if source_after in {"runtime_derived", "upstream_candidate"}:
            summary["derived_count"] += 1
        if source_before == "planner_literal" and source_after != "planner_literal":
            summary["planner_override_count"] += 1
        if source_before in {"user_explicit", "known_input"} and source_after == source_before:
            summary["user_validation_count"] += 1
        if source_after == "adapter_default":
            summary["adapter_default_count"] += 1
        if status.startswith("failed_") or "conflict" in status:
            summary["conflict_count"] += 1
    return summary


def _parameter_resolution_events_from_nodes(
    nodes: list[WorkflowNodeSummary],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node in nodes:
        for event in node.parameter_resolution_events:
            payload = {
                "node_id": node.node_id,
                "tool": node.tool,
                **event,
            }
            events.append(payload)
            if len(events) >= _AVAILABILITY_EVENT_LIMIT:
                return events
    return events


def _availability_empty_event(
    node: WorkflowNodeSummary,
    event: dict[str, Any],
) -> dict[str, Any]:
    selected_term = event.get("selected_term")
    if selected_term is None and event.get("selected_go_id"):
        selected_term = event.get("selected_go_id")
    if selected_term is None and event.get("selected_interpro_id"):
        selected_term = event.get("selected_interpro_id")
    return {
        "event_type": str(event.get("event_type") or "empty_candidate_event"),
        "node_id": node.node_id,
        "tool": node.tool,
        "intent_id": event.get("intent_id"),
        "label": event.get("label") or event.get("facet_label"),
        "query": event.get("query") or event.get("facet_query"),
        "selected_term": selected_term,
        "selected_reason": event.get("selected_reason"),
        "candidate_count": _as_int(event.get("candidate_count")),
        "retrieval_status": event.get("retrieval_status"),
    }


def _availability_filter_event(
    node: WorkflowNodeSummary,
    event: dict[str, Any],
) -> dict[str, Any]:
    filtered = event.get("filtered_candidates")
    if not isinstance(filtered, list):
        filtered = []
    return {
        "node_id": node.node_id,
        "tool": node.tool,
        "status": event.get("status"),
        "reason": event.get("reason"),
        "input_field": event.get("input_field"),
        "candidate_count": _as_int(event.get("candidate_count")),
        "compatible_candidate_count": _as_int(
            event.get("compatible_candidate_count")
        ),
        "skipped_candidate_count": _as_int(event.get("skipped_candidate_count")),
        "required_capabilities": [
            str(item) for item in event.get("required_capabilities") or []
        ],
        "filtered_candidates": [
            _availability_filtered_candidate(item)
            for item in filtered[:_AVAILABILITY_FILTERED_CANDIDATE_LIMIT]
            if isinstance(item, dict)
        ],
    }


def _availability_filtered_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "source": item.get("source"),
        "capabilities": item.get("capabilities"),
        "missing_capabilities": [
            str(cap) for cap in item.get("missing_capabilities") or []
        ],
    }


def _artifact_list(
    records: dict[str, ArtifactRecord],
    *,
    job_id: str | None,
) -> list[ReportArtifact]:
    return [
        ReportArtifact(
            field=field,
            artifact_id=artifact.artifact_id,
            kind=artifact.kind,
            mime=artifact.mime,
            size_bytes=artifact.size_bytes,
            sha256=artifact.sha256,
            path=artifact.pool_path,
            job_id=job_id,
        )
        for field, artifact in sorted(records.items())
    ]


def _readonly_connection(workspace: Workspace) -> sqlite3.Connection:
    if not workspace.db_path.exists():
        raise WorkspaceError(
            f"workspace database missing at {workspace.db_path}",
            details={"workspace": workspace.name, "db_path": str(workspace.db_path)},
        )
    uri_path = quote(str(workspace.db_path.resolve()), safe="/:")
    conn = sqlite3.connect(f"file:{uri_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_workflow_readonly(workspace: Workspace, workflow_id: str) -> WorkflowRecord:
    conn = _readonly_connection(workspace)
    try:
        row = conn.execute(
            "SELECT * FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise NotFoundError(
            f"workflow {workflow_id!r} not found",
            details={"workflow_id": workflow_id},
        )
    return WorkflowRecord(
        workflow_id=row["workflow_id"],
        name=row["name"],
        version=row["version"],
        descriptor=row["descriptor"],
        params=_json_load(row["params_json"], default={}),
        state=row["state"],
        submitted_at=int(row["submitted_at"]),
        finished_at=row["finished_at"],
    )


def _read_workflow_nodes_readonly(
    workspace: Workspace,
    workflow_id: str,
) -> list[WorkflowNodeRecord]:
    conn = _readonly_connection(workspace)
    try:
        rows = conn.execute(
            "SELECT * FROM workflow_nodes WHERE workflow_id = ?",
            (workflow_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_workflow_node_from_row(row) for row in rows]


def _read_job_readonly(workspace: Workspace, job_id: str) -> JobRecord:
    conn = _readonly_connection(workspace)
    try:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise NotFoundError(f"job {job_id!r} not found", details={"job_id": job_id})
    keys = set(row.keys())

    def get(key: str, default: Any = None) -> Any:
        return row[key] if key in keys else default

    return JobRecord(
        job_id=row["job_id"],
        call_id=get("call_id"),
        state=JobState(row["state"]),
        tool_name=row["tool_name"],
        tool_version=row["tool_version"],
        params=_json_load(row["params_json"], default={}),
        params_hash=row["params_hash"],
        output=_json_load(get("output_json"), default=None),
        error_code=get("error_code"),
        error_message=get("error_message"),
        error_details=_json_load(get("error_details_json"), default=None),
        log_path=row["log_path"],
        workspace=row["workspace"],
        workflow_id=get("workflow_id"),
        workflow_node_id=get("workflow_node_id"),
        worker_id=get("worker_id"),
        gpu_ids=_gpu_ids_from_str(get("gpu_ids")),
        priority=Priority(get("priority") or Priority.NORMAL.value),
        retry_count=int(get("retry_count", 0)),
        max_retries=int(get("max_retries", 0)),
        submitted_at=int(row["submitted_at"]),
        queued_at=get("queued_at"),
        started_at=get("started_at"),
        finished_at=get("finished_at"),
        cache_hit=bool(get("cache_hit", False)),
    )


def _read_artifacts_for_job_readonly(
    workspace: Workspace,
    job_id: str,
) -> dict[str, ArtifactRecord]:
    conn = _readonly_connection(workspace)
    try:
        rows = conn.execute(
            """
            SELECT ja.field AS field, a.*
              FROM job_artifacts ja
              JOIN artifacts a ON a.artifact_id = ja.artifact_id
             WHERE ja.job_id = ?
            """,
            (job_id,),
        ).fetchall()
    finally:
        conn.close()
    return {row["field"]: _artifact_record_from_row(row) for row in rows}


def _artifact_record_from_row(row: sqlite3.Row) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=row["artifact_id"],
        sha256=row["sha256"],
        kind=row["kind"],
        mime=row["mime"],
        size_bytes=int(row["size_bytes"]),
        pool_path=row["pool_path"],
        producer_job=row["producer_job"],
        created_at=int(row["created_at"]),
        last_used_at=int(row["last_used_at"]),
        pinned=bool(row["pinned"]),
    )


def _workflow_node_from_row(row: sqlite3.Row) -> WorkflowNodeRecord:
    keys = set(row.keys())

    def get(key: str, default: Any = None) -> Any:
        return row[key] if key in keys else default

    return WorkflowNodeRecord(
        workflow_id=row["workflow_id"],
        node_id=row["node_id"],
        tool_name=row["tool_name"],
        tool_version=row["tool_version"],
        upstream=_json_load(row["upstream"], default=[]),
        params_hash=row["params_hash"],
        input_hash=row["input_hash"],
        cache_key=row["cache_key"],
        state=row["state"],
        job_id=get("job_id"),
        output=_json_load(get("output_json"), default=None),
        error_code=get("error_code"),
        error_message=get("error_message"),
        error_details=_json_load(get("error_details_json"), default=None),
    )


def _gpu_ids_from_str(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(item) for item in value.split(",") if item]


def _json_load(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


_ARCHIVE_RECORD_TABLES = (
    "agent_candidate_records",
    "candidate_archive_records",
    "candidate_archive",
    "candidate_records",
)
_ARCHIVE_SNAPSHOT_TABLES = (
    "agent_candidate_snapshots",
    "candidate_archive_snapshots",
    "candidate_archive_snapshot",
    "candidate_snapshots",
    "candidate_snapshot",
)
_ARCHIVE_RECORD_JSON_COLUMNS = (
    "candidate_json",
    "record_json",
    "payload_json",
    "data_json",
)
_ARCHIVE_SNAPSHOT_JSON_COLUMNS = (
    "snapshot_json",
    "candidates_json",
    "records_json",
    "payload_json",
    "data_json",
)


def _candidate_archive_candidates(
    workspace: Workspace,
    *,
    workflow_id: str | None,
    job_ids: list[str],
    node_ids: list[str],
    artifacts: list[ReportArtifact],
) -> list[ReportCandidate]:
    survivor_snapshot_records = _read_candidate_archive_snapshot(
        workspace,
        workflow_id=workflow_id,
        job_ids=job_ids,
        node_ids=node_ids,
        kind="survivor",
    )
    if survivor_snapshot_records:
        return _report_candidates_from_candidate_records(
            survivor_snapshot_records,
            artifacts,
            source_kind="candidate_archive_snapshot",
        )

    current_snapshot_records = _read_candidate_archive_snapshot(
        workspace,
        workflow_id=workflow_id,
        job_ids=job_ids,
        node_ids=node_ids,
        kind="current_round",
    )
    if current_snapshot_records:
        return _report_candidates_from_candidate_records(
            current_snapshot_records,
            artifacts,
            source_kind="candidate_archive_snapshot",
        )

    records = _read_candidate_archive_records(
        workspace,
        workflow_id=workflow_id,
        job_ids=job_ids,
        node_ids=node_ids,
    )
    if records:
        return _report_candidates_from_candidate_records(
            records,
            artifacts,
            source_kind="candidate_archive_record",
        )
    return []


def _read_candidate_archive_records(
    workspace: Workspace,
    *,
    workflow_id: str | None,
    job_ids: list[str],
    node_ids: list[str],
) -> list[dict[str, Any]]:
    conn = _readonly_connection(workspace)
    try:
        tables = _existing_tables(conn, _ARCHIVE_RECORD_TABLES)
        records: list[dict[str, Any]] = []
        for table in tables:
            columns = _table_columns(conn, table)
            if not columns:
                continue
            predicates, args = _archive_predicates(
                columns,
                workflow_id=workflow_id,
                job_ids=job_ids,
                node_ids=node_ids,
            )
            if not predicates:
                continue
            order = _archive_order_by(columns)
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE {' OR '.join(predicates)}{order}",
                args,
            ).fetchall()
            records.extend(
                record
                for row in rows
                if (record := _candidate_record_from_archive_row(row, columns)) is not None
            )
        return [record for record in records if not _archive_record_is_rejected(record)]
    finally:
        conn.close()


def _read_candidate_archive_snapshot(
    workspace: Workspace,
    *,
    workflow_id: str | None,
    job_ids: list[str],
    node_ids: list[str],
    kind: str | None = None,
) -> list[dict[str, Any]]:
    conn = _readonly_connection(workspace)
    try:
        tables = _existing_tables(conn, _ARCHIVE_SNAPSHOT_TABLES)
        for table in tables:
            columns = _table_columns(conn, table)
            if not columns:
                continue
            predicates, args = _archive_predicates(
                columns,
                workflow_id=workflow_id,
                job_ids=job_ids,
                node_ids=node_ids,
            )
            clauses: list[str] = []
            query_args: list[Any] = []
            if kind and "kind" in columns:
                clauses.append("kind = ?")
                query_args.append(kind)
            if predicates:
                clauses.append(f"({' OR '.join(predicates)})")
                query_args.extend(args)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            order = _snapshot_order_by(columns)
            rows = conn.execute(
                f"SELECT * FROM {table}{where}{order}",
                query_args,
            ).fetchall()
            for row in rows:
                if kind and "kind" in columns and str(row["kind"]) != kind:
                    continue
                records = _candidate_records_from_snapshot_row(conn, row, columns)
                records = [
                    record for record in records
                    if not _archive_record_is_rejected(record)
                ]
                records = _archive_records_in_scope(
                    records,
                    workflow_id=workflow_id,
                    job_ids=job_ids,
                    node_ids=node_ids,
                )
                if records:
                    return records
        return []
    finally:
        conn.close()


def _existing_tables(conn: sqlite3.Connection, names: tuple[str, ...]) -> list[str]:
    placeholders = ",".join("?" for _ in names)
    rows = conn.execute(
        f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
        names,
    ).fetchall()
    found = {str(row["name"]) for row in rows}
    return [name for name in names if name in found]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _archive_predicates(
    columns: set[str],
    *,
    workflow_id: str | None,
    job_ids: list[str],
    node_ids: list[str],
) -> tuple[list[str], list[Any]]:
    predicates: list[str] = []
    args: list[Any] = []
    if workflow_id and "workflow_id" in columns:
        predicates.append("workflow_id = ?")
        args.append(workflow_id)
    if job_ids and "job_id" in columns:
        placeholders = ",".join("?" for _ in job_ids)
        predicates.append(f"job_id IN ({placeholders})")
        args.extend(job_ids)
    node_column = "node_id" if "node_id" in columns else "workflow_node_id"
    if node_ids and node_column in columns:
        placeholders = ",".join("?" for _ in node_ids)
        predicates.append(f"{node_column} IN ({placeholders})")
        args.extend(node_ids)
    return predicates, args


def _archive_order_by(columns: set[str], *, descending: bool = False) -> str:
    direction = " DESC" if descending else ""
    for column in ("rank", "created_at", "updated_at", "id", "candidate_id"):
        if column in columns:
            return f" ORDER BY {column}{direction}"
    return ""


def _snapshot_order_by(columns: set[str]) -> str:
    parts = [
        f"{column} DESC"
        for column in ("round_index", "created_at", "updated_at", "id", "snapshot_id")
        if column in columns
    ]
    return f" ORDER BY {', '.join(parts)}" if parts else ""


def _candidate_record_from_archive_row(
    row: sqlite3.Row,
    columns: set[str],
) -> dict[str, Any] | None:
    for column in _ARCHIVE_RECORD_JSON_COLUMNS:
        if column not in columns:
            continue
        payload = _json_load(row[column], default=None)
        if isinstance(payload, dict):
            record = None
            for key in ("candidate", "record"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    record = dict(nested)
                    break
            if record is None:
                record = dict(payload)
            return _with_archive_row_metadata(record, row, columns)

    record: dict[str, Any] = {}
    for column in columns:
        value = row[column]
        if value is None:
            continue
        if column.endswith("_json"):
            key = column.removesuffix("_json")
            record[key] = _json_load(value, default=value)
        else:
            record[column] = value
    return record if record.get("candidate_id") else None


def _with_archive_row_metadata(
    record: dict[str, Any],
    row: sqlite3.Row,
    columns: set[str],
) -> dict[str, Any]:
    for column in (
        "round_index",
        "workflow_id",
        "node_id",
        "workflow_node_id",
        "job_id",
        "source_node_id",
        "source_job_id",
        "status",
        "snapshot_id",
        "archive_snapshot_id",
        "version_id",
        "run_id",
    ):
        if column in columns and row[column] is not None:
            record.setdefault(column, row[column])
    if "raw_source_json" in columns and row["raw_source_json"] is not None:
        raw_source = _json_load(row["raw_source_json"], default={})
        if isinstance(raw_source, dict):
            record.setdefault("raw_source", raw_source)
    if "validation_errors_json" in columns and row["validation_errors_json"] is not None:
        validation_errors = _json_load(row["validation_errors_json"], default=[])
        if isinstance(validation_errors, list):
            record.setdefault("validation_errors", validation_errors)
    return record


def _candidate_records_from_snapshot_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    columns: set[str],
) -> list[dict[str, Any]]:
    records = _candidate_records_from_snapshot_refs(conn, row, columns)
    if records:
        return [_with_snapshot_row_metadata(record, row, columns) for record in records]
    for column in _ARCHIVE_SNAPSHOT_JSON_COLUMNS:
        if column not in columns:
            continue
        payload = _json_load(row[column], default=None)
        records = _legacy_archive_snapshot_payload_records(payload)
        if records:
            return [_with_snapshot_row_metadata(record, row, columns) for record in records]
    return []


def _candidate_records_from_snapshot_refs(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    columns: set[str],
) -> list[dict[str, Any]]:
    refs: list[Any] = []
    exact_version_refs = False
    if "candidate_refs_json" in columns:
        parsed = _json_load(row["candidate_refs_json"], default=[])
        if isinstance(parsed, list):
            refs = parsed
            exact_version_refs = bool(refs)
    if not refs and "candidate_ids_json" in columns:
        parsed = _json_load(row["candidate_ids_json"], default=[])
        if isinstance(parsed, list):
            refs = parsed
    if not refs or "agent_candidate_records" not in _existing_tables(conn, _ARCHIVE_RECORD_TABLES):
        return []

    record_columns = _table_columns(conn, "agent_candidate_records")
    if not exact_version_refs:
        return _hydrate_legacy_candidate_ids_for_report_only(conn, refs, record_columns)

    records: list[dict[str, Any]] = []
    for ref in refs:
        row_record = _candidate_record_row_for_snapshot_ref(
            conn,
            ref,
            record_columns,
            exact_version_only=True,
        )
        if row_record is None:
            continue
        record = _candidate_record_from_archive_row(row_record, record_columns)
        if record is not None:
            records.append(record)
    return records


def _hydrate_legacy_candidate_ids_for_report_only(
    conn: sqlite3.Connection,
    refs: list[Any],
    columns: set[str],
) -> list[dict[str, Any]]:
    """Hydrate old ``candidate_ids_json`` snapshots for reports only.

    Executor materialization must use CandidateArchive exact ``candidate_refs``.
    """
    records: list[dict[str, Any]] = []
    for ref in refs:
        row_record = _candidate_record_row_for_snapshot_ref(
            conn,
            ref,
            columns,
            exact_version_only=False,
        )
        if row_record is None:
            continue
        record = _candidate_record_from_archive_row(row_record, columns)
        if record is not None:
            records.append(record)
    return records


def _candidate_record_row_for_snapshot_ref(
    conn: sqlite3.Connection,
    ref: Any,
    columns: set[str],
    *,
    exact_version_only: bool = False,
) -> sqlite3.Row | None:
    if isinstance(ref, dict):
        version_id = _clean_ref_value(
            ref.get("version_id")
            or ref.get("candidate_version_id")
            or ref.get("ref")
        )
        if version_id and "version_id" in columns:
            row = conn.execute(
                "SELECT * FROM agent_candidate_records WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            return row
        if exact_version_only:
            return None
        candidate_id = _clean_ref_value(ref.get("candidate_id") or ref.get("id"))
    else:
        if exact_version_only:
            return None
        candidate_id = _clean_ref_value(ref)
    if not candidate_id or "candidate_id" not in columns:
        return None
    return conn.execute(
        """
        SELECT * FROM agent_candidate_records
        WHERE candidate_id = ? AND lower(status) != 'rejected'
        ORDER BY round_index DESC, created_at DESC
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()


def _clean_ref_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _with_snapshot_row_metadata(
    record: dict[str, Any],
    row: sqlite3.Row,
    columns: set[str],
) -> dict[str, Any]:
    out = dict(record)
    snapshot_id = row["snapshot_id"] if "snapshot_id" in columns else None
    if snapshot_id is not None:
        out.setdefault("archive_snapshot_id", snapshot_id)
        out.setdefault("snapshot_id", snapshot_id)
    if "kind" in columns and row["kind"] is not None:
        kind = str(row["kind"])
        out.setdefault("archive_snapshot_kind", kind)
        out.setdefault("status", "survivor" if kind == "survivor" else "current")
    if "round_index" in columns and row["round_index"] is not None:
        out.setdefault("snapshot_round_index", row["round_index"])
    for column in ("workflow_id", "node_id", "workflow_node_id", "job_id"):
        if column in columns and row[column] is not None:
            out.setdefault(column, row[column])
    if "run_id" in columns and row["run_id"] is not None:
        out.setdefault("archive_snapshot_run_id", row["run_id"])
    if "workflow_ids_json" in columns and row["workflow_ids_json"] is not None:
        workflow_ids = _json_load(row["workflow_ids_json"], default=[])
        if isinstance(workflow_ids, list):
            out.setdefault("archive_snapshot_workflow_ids", workflow_ids)
    return out


def _legacy_archive_snapshot_payload_records(payload: Any) -> list[dict[str, Any]]:
    """Report-only compatibility for old archive snapshot JSON payloads.

    This helper is intentionally reached only from archive snapshot tables. It
    must not be used for ``agent_summaries`` or executor materialization.
    """
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (
        "candidates",
        "candidate_records",
        "records",
        "survivor_pool",
        "current_round_candidates",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    for key in ("snapshot", "candidate_snapshot", "archive_snapshot"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            records = _legacy_archive_snapshot_payload_records(nested)
            if records:
                return records
    return []


def _archive_record_is_rejected(record: dict[str, Any]) -> bool:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    status = record.get("archive_status") or record.get("status") or source.get("archive_status")
    return str(status or "").lower() == "rejected"


def _archive_records_in_scope(
    records: list[dict[str, Any]],
    *,
    workflow_id: str | None,
    job_ids: list[str],
    node_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if _archive_record_matches_scope(
            record,
            workflow_id=workflow_id,
            job_ids=job_ids,
            node_ids=node_ids,
        )
    ]


def _archive_record_matches_scope(
    record: dict[str, Any],
    *,
    workflow_id: str | None,
    job_ids: list[str],
    node_ids: list[str],
) -> bool:
    if not workflow_id and not job_ids and not node_ids:
        return True
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    raw_source = record.get("raw_source") if isinstance(record.get("raw_source"), dict) else {}
    if workflow_id:
        workflow_values = {
            str(value)
            for value in (
                record.get("workflow_id"),
                record.get("source_workflow_id"),
                source.get("workflow_id"),
                raw_source.get("workflow_id"),
            )
            if value not in (None, "")
        }
        if workflow_values:
            return workflow_id in workflow_values
    job_values = {
        str(value)
        for value in (
            record.get("source_job_id"),
            record.get("job_id"),
            source.get("source_job_id"),
            source.get("job_id"),
            raw_source.get("job_id"),
        )
        if value not in (None, "")
    }
    if job_values.intersection(str(job_id) for job_id in job_ids):
        return True
    node_values = {
        str(value)
        for value in (
            record.get("source_node_id"),
            record.get("node_id"),
            record.get("workflow_node_id"),
            source.get("source_node_id"),
            source.get("node_id"),
            raw_source.get("node_id"),
        )
        if value not in (None, "")
    }
    if node_values.intersection(str(node_id) for node_id in node_ids):
        return True
    return False


def _report_candidates_from_candidate_records(
    records: list[dict[str, Any]],
    artifacts: list[ReportArtifact],
    *,
    source_kind: str,
) -> list[ReportCandidate]:
    candidates: list[ReportCandidate] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        if _archive_record_is_rejected(record):
            continue
        candidate = _report_candidate_from_candidate_record(
            record,
            idx,
            artifacts,
            source_kind=source_kind,
        )
        if candidate is not None:
            candidates.append(candidate)
    _attach_qc_and_analysis(candidates, [], merge_global_metrics=False)
    return _dedupe_candidates(candidates)


def _report_candidate_from_candidate_record(
    record: dict[str, Any],
    index: int,
    artifacts: list[ReportArtifact],
    *,
    source_kind: str,
) -> ReportCandidate | None:
    candidate_id = str(record.get("candidate_id") or record.get("id") or "").strip()
    if not candidate_id:
        return None
    structure_path = _first_str(
        record,
        *_STRUCTURE_PATH_KEYS,
        "full_structure_path",
        "backbone_path",
    )
    stable_structure_path = _stable_structure_path(structure_path, artifacts)
    source = deepcopy(record.get("source")) if isinstance(record.get("source"), dict) else {}
    raw_source = (
        record.get("raw_source") if isinstance(record.get("raw_source"), dict) else {}
    )
    source_node_id = (
        record.get("source_node_id")
        or record.get("node_id")
        or record.get("workflow_node_id")
        or source.get("node_id")
        or raw_source.get("node_id")
    )
    source_job_id = (
        record.get("source_job_id")
        or record.get("job_id")
        or source.get("job_id")
        or raw_source.get("job_id")
    )
    source.setdefault("kind", source_kind)
    source.setdefault("round_index", record.get("round_index"))
    source.setdefault("archive_status", record.get("archive_status") or record.get("status"))
    source.setdefault(
        "archive_snapshot_id",
        record.get("archive_snapshot_id") or record.get("snapshot_id"),
    )
    source.setdefault("source_node_id", source_node_id)
    source.setdefault("source_job_id", source_job_id)
    source.setdefault("node_id", source_node_id)
    source.setdefault("job_id", source_job_id or source.get("source_job_id"))
    source.setdefault("tool", record.get("tool") or record.get("source_tool"))
    return ReportCandidate(
        candidate_id=candidate_id,
        rank=_as_int(record.get("rank")) or index + 1,
        score=_as_float(record.get("score") or record.get("ranking_score")),
        sequence=_first_str(record, "sequence", "designed_sequence"),
        structure_path=stable_structure_path,
        scores=_candidate_score_map(record),
        qc_flags=_string_list(record.get("qc_flags") or record.get("flags")),
        liabilities=_list(record.get("liabilities")),
        source=source,
        artifacts=_candidate_record_artifacts(record, stable_structure_path, artifacts),
    )


def _candidate_score_map(record: dict[str, Any]) -> dict[str, Any]:
    scores = record.get("scores")
    if isinstance(scores, dict):
        out = deepcopy(scores)
        for key, value in _score_map(record).items():
            out.setdefault(key, value)
        return out
    return _score_map(record)


def _candidate_record_artifacts(
    record: dict[str, Any],
    structure_path: str | None,
    artifacts: list[ReportArtifact],
) -> list[ReportArtifact]:
    out = list(_matching_artifacts(structure_path, artifacts))
    seen = {artifact.path for artifact in out}
    raw = record.get("artifacts")
    for artifact in _candidate_artifacts_from_payload(
        raw,
        job_id=_archive_source_job_id(record),
    ):
        if artifact.path in seen:
            continue
        seen.add(artifact.path)
        out.append(artifact)
    return out


def _archive_source_job_id(record: dict[str, Any]) -> str | None:
    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    raw_source = record.get("raw_source") if isinstance(record.get("raw_source"), dict) else {}
    value = (
        record.get("source_job_id")
        or record.get("job_id")
        or source.get("job_id")
        or raw_source.get("job_id")
    )
    return str(value) if value else None


def _candidate_artifacts_from_payload(
    payload: Any,
    *,
    job_id: str | None,
    field_prefix: str = "candidate_artifact",
) -> list[ReportArtifact]:
    if isinstance(payload, list):
        out: list[ReportArtifact] = []
        for idx, item in enumerate(payload):
            out.extend(
                _candidate_artifacts_from_payload(
                    item,
                    job_id=job_id,
                    field_prefix=f"{field_prefix}_{idx + 1}",
                )
            )
        return out
    if isinstance(payload, dict):
        direct = _candidate_artifact_from_dict(payload, job_id=job_id, field=field_prefix)
        if direct is not None:
            return [direct]
        out: list[ReportArtifact] = []
        for key, value in payload.items():
            out.extend(
                _candidate_artifacts_from_payload(
                    value,
                    job_id=job_id,
                    field_prefix=str(key),
                )
            )
        return out
    if isinstance(payload, str) and payload:
        return [
            ReportArtifact(
                field=field_prefix,
                artifact_id=None,
                kind=Path(payload).suffix.lstrip(".") or "artifact",
                mime=None,
                size_bytes=None,
                sha256=None,
                path=payload,
                job_id=job_id,
            )
        ]
    return []


def _candidate_artifact_from_dict(
    payload: dict[str, Any],
    *,
    job_id: str | None,
    field: str,
) -> ReportArtifact | None:
    path = _first_str(payload, "path", "pool_path", "structure_path", "pdb_path", "cif_path")
    if not path:
        return None
    return ReportArtifact(
        field=str(payload.get("field") or field),
        artifact_id=payload.get("artifact_id"),
        kind=str(payload.get("kind") or Path(path).suffix.lstrip(".") or "artifact"),
        mime=payload.get("mime"),
        size_bytes=_as_int(payload.get("size_bytes")),
        sha256=payload.get("sha256"),
        path=path,
        job_id=str(payload.get("job_id") or job_id) if payload.get("job_id") or job_id else None,
    )


def _normalize_candidates(
    nodes: list[WorkflowNodeSummary],
    artifacts: list[ReportArtifact],
    *,
    node_params: dict[str, dict[str, Any]] | None = None,
) -> list[ReportCandidate]:
    ranked = _ranked_candidates(nodes, artifacts)
    if ranked:
        return ranked
    candidates: list[ReportCandidate] = []
    params_by_node = node_params or {}
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        candidates.extend(
            _candidates_from_output(
                node, output, artifacts,
                node_params=params_by_node.get(node.node_id),
            )
        )
    _attach_qc_and_analysis(candidates, nodes, merge_global_metrics=False)
    return _dedupe_candidates(candidates)


def _submission_outputs(nodes: list[WorkflowNodeSummary]) -> list[dict[str, Any]]:
    submissions: list[dict[str, Any]] = []
    for node in nodes:
        if node.tool != "submit_candidate" or node.state != "completed":
            continue
        output = node.output if isinstance(node.output, dict) else {}
        if output.get("terminal") is not True:
            continue
        submission = dict(output)
        submission["node_id"] = node.node_id
        submission["job_id"] = node.job_id
        submissions.append(submission)
    return submissions


def _latest_final_submission(submissions: list[dict[str, Any]]) -> dict[str, Any] | None:
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


def _candidates_from_submission(
    submission: dict[str, Any] | None,
) -> list[ReportCandidate]:
    if not submission:
        return []
    resolved = submission.get("candidates_resolved")
    if not isinstance(resolved, list):
        return []
    out: list[ReportCandidate] = []
    for idx, row in enumerate(resolved):
        if not isinstance(row, dict):
            continue
        artifacts = [
            ReportArtifact(
                field=str(artifact.get("field") or ""),
                artifact_id=artifact.get("artifact_id"),
                kind=str(artifact.get("kind") or "artifact"),
                mime=artifact.get("mime"),
                size_bytes=_as_int(artifact.get("size_bytes")),
                sha256=artifact.get("sha256"),
                path=str(artifact.get("path") or ""),
                job_id=row.get("source_job_id"),
            )
            for artifact in row.get("artifacts") or []
            if isinstance(artifact, dict) and artifact.get("path")
        ]
        out.append(
            ReportCandidate(
                candidate_id=str(row.get("candidate_id") or idx + 1),
                rank=_as_int(row.get("rank")) or idx + 1,
                score=_as_float(row.get("score")),
                sequence=row.get("sequence") if isinstance(row.get("sequence"), str) else None,
                structure_path=(
                    row.get("structure_path")
                    if isinstance(row.get("structure_path"), str)
                    else None
                ),
                scores=_candidate_score_map(row),
                source={
                    "kind": "final_submission",
                    "submission_id": submission.get("submission_id"),
                    "outcome": submission.get("outcome"),
                    "reasoning": submission.get("reasoning"),
                    "justification": row.get("justification"),
                    "job_id": row.get("source_job_id"),
                    "tool": row.get("source_tool"),
                    "node_id": row.get("source_workflow_node_id"),
                    "auto_selected": bool(submission.get("auto_selected", False)),
                },
                artifacts=artifacts,
            )
        )
    return out


def _auto_submission_from_candidates(
    candidates: list[ReportCandidate],
) -> tuple[dict[str, Any], list[ReportCandidate]]:
    candidate = candidates[0]
    submission = {
        "submission_id": "auto_selected",
        "outcome": "accepted",
        "terminal": True,
        "reasoning": (
            "Auto-selected highest ranked available candidate because no explicit "
            "submit_candidate output was produced."
        ),
        "supersedes": None,
        "auto_selected": True,
    }
    final = replace(
        candidate,
        source={
            **candidate.source,
            "kind": "final_submission",
            "submission_id": "auto_selected",
            "reasoning": submission["reasoning"],
            "auto_selected": True,
        },
    )
    return submission, [final]


# Field names adapters use to surface a structure artifact path on a
# candidate / output dict. Boltzgen writes ``cif_path``; Boltz-2 / Protenix
# write ``predicted_cif`` / ``predicted_pdb``; everyone else writes
# ``pdb_path`` or the generic ``structure_path``. Keep all three lookup
# sites in this module in sync by reading from this single tuple.
_STRUCTURE_PATH_KEYS: tuple[str, ...] = (
    "structure_path",
    "pdb_path",
    "cif_path",
    "predicted_pdb",
    "predicted_cif",
)


def _ranked_candidates(
    nodes: list[WorkflowNodeSummary],
    artifacts: list[ReportArtifact],
) -> list[ReportCandidate]:
    out: list[ReportCandidate] = []
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        rows = output.get("ranked_candidates")
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            structure_path = _first_str(row, *_STRUCTURE_PATH_KEYS)
            stable_structure_path = _stable_structure_path(structure_path, artifacts)
            candidate_id = str(row.get("candidate_id") or row.get("name") or idx + 1)
            out.append(
                ReportCandidate(
                    candidate_id=candidate_id,
                    rank=_as_int(row.get("rank")) or idx + 1,
                    score=_as_float(row.get("score")),
                    sequence=_first_str(row, "sequence"),
                    structure_path=stable_structure_path,
                    scores=_candidate_score_map(row),
                    qc_flags=_string_list(row.get("qc_flags")),
                    liabilities=_list(row.get("liabilities")),
                    source={
                        "node_id": node.node_id,
                        "tool": node.tool,
                        "kind": "ranked_candidates",
                    },
                    artifacts=_matching_artifacts(stable_structure_path, artifacts),
                )
            )
    _attach_qc_and_analysis(out, nodes, merge_global_metrics=True)
    return _dedupe_candidates(sorted(out, key=lambda c: (c.rank or 999999, -(c.score or 0.0))))


def _candidates_from_output(
    node: WorkflowNodeSummary,
    output: dict[str, Any],
    artifacts: list[ReportArtifact],
    *,
    node_params: dict[str, Any] | None = None,
) -> list[ReportCandidate]:
    rows: list[dict[str, Any]] = []
    for key in ("top_candidates", "top_designs", "designs", "predictions", "candidates"):
        value = output.get(key)
        if isinstance(value, list):
            rows.extend(v for v in value if isinstance(v, dict))
    if not rows and any(key in output for key in _STRUCTURE_PATH_KEYS):
        rows.append(output)

    # Fallback: an adapter run with no structure output but real metrics
    # (e.g. a scoring-only tool) still deserves a one-line candidate so the
    # human report shows the numbers instead of "n/a".
    if not rows and _score_map(output):
        rows.append(output)

    # Input sequence inferred from job params (ESM3, structure prediction
    # tools accept the sequence as input — useful even when the output
    # doesn't echo it).
    inferred_sequence = None
    if isinstance(node_params, dict):
        inferred_sequence = _first_str(node_params, "sequence", "wild_type_sequence", "query")

    out: list[ReportCandidate] = []
    for idx, row in enumerate(rows):
        structure_path = _first_str(row, *_STRUCTURE_PATH_KEYS)
        stable_structure_path = _stable_structure_path(structure_path, artifacts)
        candidate_id = str(
            row.get("candidate_id")
            or row.get("name")
            or row.get("id")
            or row.get("seed")
            or f"{node.node_id}-{idx + 1}"
        )
        out.append(
            ReportCandidate(
                candidate_id=candidate_id,
                rank=_as_int(row.get("rank")) or idx + 1,
                score=_as_float(row.get("score") or row.get("ranking_score")),
                sequence=_first_str(row, "sequence", "designed_sequence") or inferred_sequence,
                structure_path=stable_structure_path,
                scores=_candidate_score_map(row),
                qc_flags=_string_list(row.get("qc_flags") or row.get("flags")),
                liabilities=_list(row.get("liabilities")),
                source={"node_id": node.node_id, "tool": node.tool, "kind": "output"},
                artifacts=_matching_artifacts(stable_structure_path, artifacts),
            )
        )
    return out


def _structure_candidates(
    artifacts: list[ReportArtifact],
    *,
    source: dict[str, Any],
) -> list[ReportCandidate]:
    out: list[ReportCandidate] = []
    for artifact in artifacts:
        if _is_structure_path(artifact.path, artifact.kind):
            out.append(
                ReportCandidate(
                    candidate_id=artifact.field,
                    rank=len(out) + 1,
                    structure_path=artifact.path,
                    source={**source, "artifact_field": artifact.field},
                    artifacts=[artifact],
                )
            )
    return out


def _attach_qc_and_analysis(
    candidates: list[ReportCandidate],
    nodes: list[WorkflowNodeSummary],
    *,
    merge_global_metrics: bool,
) -> None:
    """Merge tool-level signals into candidate score maps.

    Ranked candidates are already an aggregate product of the workflow, so they
    can receive global QC/analysis context. Output-derived candidates are more
    often independent catalog entries; for those, attach metrics only from the
    candidate's source node to avoid cross-contaminating flat runs such as
    Boltz2 + BoltzGen + Protenix smoke catalogs.
    """
    qc_flags: list[str] = []
    liabilities: list[Any] = []
    # (node_id, tool, scores) tuples in node order so provenance disambiguation is
    # deterministic across runs.
    per_tool_metrics: list[tuple[str, str, dict[str, Any]]] = []
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        if not isinstance(output, dict):
            continue
        node_id = node.node_id
        tool = (node.tool or "").lower()
        if tool == "protein_qc":
            qc_flags.extend(_string_list(output.get("flags")))
            metrics = output.get("metrics")
            if isinstance(metrics, dict):
                per_tool_metrics.append((node_id, tool, _score_map(metrics)))
        elif tool == "structure_analysis":
            liabilities.extend(_list(output.get("liabilities")))
            # structure_analysis also publishes scalars (helix_pct, sheet_pct,
            # buried_polar_count, etc.) — surface those too.
            per_tool_metrics.append((node_id, tool, _score_map(output)))
        else:
            # Any other tool: extract its top-level numeric scalars (filtered
            # by ``_SCORE_KEYS``), plus anything inside nested metric dicts.
            tool_scores = _score_map(output)
            if tool_scores:
                per_tool_metrics.append((node_id, tool, tool_scores))

    if not qc_flags and not liabilities and not per_tool_metrics:
        return

    # Count how many tools emit each canonical metric key so we can decide
    # which keys need a ``tool:`` prefix to keep them distinguishable.
    emitter_counts: dict[str, int] = {}
    for _node_id, _tool, scores in per_tool_metrics:
        for key in scores:
            emitter_counts[key] = emitter_counts.get(key, 0) + 1

    # Specialised predictor tools whose metrics always carry a provenance
    # tag — readers asked "is this protein pLDDT or complex pLDDT?", so
    # we never let a boltz2 / protenix / esm3 metric render unlabelled,
    # even when no other tool emits the same key.
    _ALWAYS_PREFIX_TOOLS = frozenset({
        "boltz2_affinity", "boltzgen",
        "protenix_v2", "esm3", "esm2",
        "ipsae", "dockq", "prodigy", "pyrosetta", "tmalign",
        "rfdiffusion", "rfdiffusion3", "rfpeptides",
        "bindcraft", "pxdesign", "pxmeter",
    })

    for idx, candidate in enumerate(list(candidates)):
        source = candidate.source if isinstance(candidate.source, dict) else {}
        candidate_node_id = str(source.get("node_id") or "")
        candidate_tool = str(source.get("tool") or "").lower()
        # Pre-existing candidate.scores (from ranked_candidates row) are
        # treated as ground truth — they win when their key is "clean"
        # (i.e. no tool prefix needed). Tool-emitted metrics fill in
        # everything else; conflicting keys get a ``tool:`` prefix so the
        # candidate ends up with both ``plddt`` (whatever the ranker
        # recorded) and e.g. ``boltz2_affinity:plddt`` (Boltz-2's own).
        merged_scores: dict[str, Any] = {}
        for node_id, tool, scores in per_tool_metrics:
            if not merge_global_metrics:
                same_node = node_id == candidate_node_id
                fallback_same_tool = not candidate_node_id and tool == candidate_tool
                if not same_node and not fallback_same_tool:
                    continue
            for key, value in scores.items():
                already_in_candidate = key in candidate.scores
                shared = emitter_counts.get(key, 0) > 1
                always_tag = tool in _ALWAYS_PREFIX_TOOLS
                if shared or already_in_candidate or always_tag:
                    chosen_key = f"{tool}:{key}"
                else:
                    chosen_key = key
                # Later entries from the same tool don't overwrite earlier ones.
                merged_scores.setdefault(chosen_key, value)
        # Candidate's own scores override only their bare-key counterparts.
        merged_scores.update(candidate.scores)

        # Dedup qc_flags — rank_designs.ranked_candidates already carries the
        # flags it counted, and protein_qc.flags is the same list source-of-
        # truth. Without dedup the user sees "low_plddt low_plddt
        # high_instability_index high_instability_index".
        merged_flags: list[str] = []
        seen_flags: set[str] = set()
        for flag in (*candidate.qc_flags, *qc_flags):
            key = str(flag).strip().lower()
            if not key or key in seen_flags:
                continue
            seen_flags.add(key)
            merged_flags.append(str(flag))

        # Dedup liabilities by (type, chain, position-tuple) so structure_
        # analysis emitting "buried_polar_candidate" 9 times (once per
        # residue) doesn't collapse a useful signal into noise — the
        # renderer then groups by ``type`` to show "buried_polar_candidate
        # x9 at A:15,18,20,..." instead of nine identical chips.
        merged_liabilities: list[Any] = []
        seen_liab: set[tuple[Any, ...]] = set()
        for entry in (*candidate.liabilities, *liabilities):
            if isinstance(entry, dict):
                positions = entry.get("positions")
                pos_key: Any
                if isinstance(positions, list):
                    pos_key = tuple(positions)
                else:
                    pos_key = positions
                key = (entry.get("type"), entry.get("chain_id"), pos_key)
            else:
                key = (str(entry),)
            if key in seen_liab:
                continue
            seen_liab.add(key)
            merged_liabilities.append(entry)

        candidates[idx] = ReportCandidate(
            candidate_id=candidate.candidate_id,
            rank=candidate.rank,
            score=candidate.score,
            sequence=candidate.sequence,
            structure_path=candidate.structure_path,
            scores=merged_scores,
            qc_flags=merged_flags,
            liabilities=merged_liabilities,
            source=candidate.source,
            artifacts=candidate.artifacts,
        )


def _classify_msa_source(rows: list[tuple[str, str]]) -> str | None:
    """Return a short human label for the alignment's apparent DB source.

    Detects the DB by inspecting sequence-ID prefixes from rows 1..N (the
    query at row 0 is skipped). Examples:

      * ``UniRef100_xxx``        → ``"UniRef100"``
      * ``MGYP000…`` / ``uvig_`` → ``"Metagenomic + environmental"``
      * Plain UniProt accessions → no firm signal, return ``None``
      * Mixed                    → ``"Mixed UniRef / metagenomic"``

    Used by the report renderer to upgrade the otherwise-uninformative
    artifact field name (e.g. ``remote_msa[0]``) into something the human
    reader actually understands.
    """
    if len(rows) < 2:
        return None
    sample = [str(lbl).strip() for lbl, _ in rows[1:31] if isinstance(lbl, str)]
    sample = [lbl for lbl in sample if lbl]
    if not sample:
        return None
    n_uniref = sum(1 for lbl in sample if lbl.upper().startswith("UNIREF"))
    n_meta = sum(
        1 for lbl in sample
        if lbl.upper().startswith(("MGYP", "MGNIFY", "UVIG_", "UVIG-"))
    )
    n_pdb = sum(1 for lbl in sample if lbl.upper().startswith(("PDB_", "PDB|")))
    n_af = sum(1 for lbl in sample if lbl.upper().startswith(("AF-", "AF_")))
    total = len(sample)
    # Strong single-source signals first
    if n_uniref / total >= 0.7:
        return "UniRef100"
    if n_pdb / total >= 0.5:
        return "PDB"
    if n_af / total >= 0.5:
        return "AlphaFold DB"
    # MGnify/environmental: ColabFold's env_db mixes MGYP/UViG with bare
    # UniProt-style hits, so an absolute UniRef prefix is rare. Any MGYP or
    # UViG presence at ~20% in a low-UniRef block is a confident signal that
    # this is the environmental MSA file rather than the UniRef one.
    if n_meta / total >= 0.5:
        return "Metagenomic + environmental (MGnify)"
    if n_meta / total >= 0.2 and n_uniref / total < 0.2:
        return "Metagenomic + environmental (MGnify)"
    # Mixed but with meaningful contribution from both
    if n_uniref / total >= 0.3 and n_meta / total >= 0.2:
        return "Mixed UniRef / metagenomic"
    return None


def _collect_msa_blocks(
    artifacts: list[ReportArtifact],
    *,
    workflow_tools: set[str] | None = None,
    tool_by_job: dict[str, str] | None = None,
) -> list[MSABlock]:
    """Collect MSA blocks — gated on (a) MSA-generating tool present and
    (b) artifact contains ≥2 aligned sequences.

    A single-sequence FASTA (like the ``mutant_fasta`` of a mutation-repair
    workflow) is *not* an MSA — it gets skipped here so the report doesn't
    show a misleading "1 sequence" alignment block.

    ``workflow_tools`` is the set of tool names used by the run, lowercased.
    ``tool_by_job`` maps job_id → tool name and gates per-artifact (so a
    workflow that has *both* boltz2_affinity and a stray sequence_edit
    artifact only promotes the boltz2 alignment).
    """
    workflow_tools = workflow_tools or set()
    tool_by_job = tool_by_job or {}
    has_msa_tool = bool(workflow_tools & _MSA_GENERATING_TOOLS)
    if not has_msa_tool:
        return []

    blocks: list[MSABlock] = []
    for artifact in artifacts:
        path = Path(artifact.path)
        if path.suffix.lower() not in _FASTA_SUFFIXES or not path.is_file():
            continue
        producer_tool = (tool_by_job.get(artifact.job_id or "") or "").lower()
        if producer_tool and producer_tool not in _MSA_GENERATING_TOOLS:
            # Artifact came from a non-MSA tool inside the same workflow
            # (e.g. sequence_edit's mutant_fasta). Skip it even though the
            # workflow as a whole has an MSA tool.
            continue
        parsed = _parse_alignment(path)
        if len(parsed) < 2:
            continue
        sequences, mean_identity, identities, column_count = _rank_alignment(parsed)
        # Auto-classify the alignment's apparent DB source from sequence-id
        # patterns. When detected, prepend it to the artifact field so the
        # render label becomes e.g. "UniRef100 · remote_msa[0]" instead of
        # an inscrutable indexed slug.
        db_label = _classify_msa_source(parsed)
        label = (
            f"{db_label} · {artifact.field}" if db_label
            else artifact.field
        )
        blocks.append(
            MSABlock(
                path=str(path),
                label=label,
                sequence_count=len(parsed),
                sequences=sequences,
                identities=identities,
                mean_identity=mean_identity,
                column_count=column_count,
                truncated=len(parsed) > len(sequences),
                source_tool=producer_tool or None,
            )
        )
    return blocks[:5]


def _parse_alignment(path: Path) -> list[tuple[str, str]]:
    """Parse FASTA/a3m/Stockholm into (label, aligned_seq) rows.

    a3m specifics: lower-case residues are insertions relative to the query
    column space and are stripped so every row has the same column count as
    the query (uppercase + ``-``). The first record is treated as the query.
    """
    text = path.read_text(errors="replace")
    suffix = path.suffix.lower()

    if suffix == ".sto":
        out: list[tuple[str, str]] = []
        for line in text.splitlines():
            if not line or line.startswith(("#", "//")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
        return out

    rows: list[tuple[str, list[str]]] = []
    label: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if label is not None:
                rows.append((label, chunks))
            # Header line in colabfold a3m can have tab-separated metadata; keep
            # only the leading token (the sequence ID) for display.
            header = line[1:].strip()
            label = header.split()[0] if header else f"sequence_{len(rows) + 1}"
            chunks = []
        elif label is not None:
            chunks.append(line.rstrip())
    if label is not None:
        rows.append((label, chunks))

    if not rows:
        return []

    # a3m: drop lower-case insertions so column space matches query.
    if suffix == ".a3m":
        cleaned: list[tuple[str, str]] = []
        for lbl, parts in rows:
            seq = "".join(parts)
            seq = "".join(ch for ch in seq if ch.isupper() or ch == "-")
            cleaned.append((lbl, seq))
        return cleaned

    return [(lbl, "".join(parts)) for lbl, parts in rows]


def _rank_alignment(
    rows: list[tuple[str, str]],
) -> tuple[list[tuple[str, str]], float, list[float], int]:
    """Sort alignment rows by % identity vs query desc, return display subset.

    Returns ``(display_rows, mean_identity, identities, column_count)`` where
    ``display_rows`` always begins with the query row (sequences[0] of the
    source file) followed by up to ``_MSA_DISPLAY_TOP_K`` highest-similarity
    rows. ``identities`` is parallel to ``display_rows``.
    ``mean_identity`` and the broader stats are computed over *all* rows.
    """
    if not rows:
        return [], 0.0, [], 0
    query_label, query_seq = rows[0]
    column_count = sum(1 for ch in query_seq if ch != "-")
    others: list[tuple[float, tuple[str, str]]] = []
    all_identities: list[float] = []
    for lbl, seq in rows[1:]:
        identity = _row_identity(query_seq, seq)
        all_identities.append(identity)
        others.append((identity, (lbl, seq)))
    # Sort by identity desc, then by label for stability.
    others.sort(key=lambda kv: (-kv[0], kv[1][0]))
    top = others[:_MSA_DISPLAY_TOP_K]
    display_rows = [(query_label, query_seq), *(row for _, row in top)]
    display_identities = [1.0, *(identity for identity, _ in top)]
    mean_identity = (
        sum(all_identities) / len(all_identities) if all_identities else 0.0
    )
    return display_rows, mean_identity, display_identities, column_count


def _row_identity(query: str, row: str) -> float:
    """Fraction of query non-gap positions matched exactly by ``row``.

    The denominator is the query's non-gap column count so insertions in
    the query (which a3m strips anyway) do not inflate or deflate identity.
    """
    if not query:
        return 0.0
    matches = 0
    denom = 0
    n = min(len(query), len(row))
    for idx in range(n):
        q = query[idx]
        if q == "-" or q == ".":
            continue
        denom += 1
        if row[idx] == q:
            matches += 1
    return matches / denom if denom else 0.0


def _warnings_from_job(job: JobRecord, *, source: str | None = None) -> list[ReportWarning]:
    warnings: list[ReportWarning] = []
    warning_source = source or job.job_id
    if job.error_code or job.error_message:
        warnings.append(
            ReportWarning(
                level="error",
                message=job.error_message or job.error_code or "job failed",
                source=warning_source,
            )
        )
    output = job.output if isinstance(job.output, dict) else {}
    for message in _string_list(output.get("warnings")):
        if not message:
            continue
        warnings.append(
            ReportWarning(level="warning", message=message, source=warning_source)
        )
    return warnings


def _raw_job_summary(job: JobRecord, artifacts: list[ReportArtifact]) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "state": job.state.value,
        "tool": job.tool_name,
        "version": job.tool_version,
        "params_hash": job.params_hash,
        "workflow_id": job.workflow_id,
        "workflow_node_id": job.workflow_node_id,
        "output": _compact(job.output),
        "artifacts": [_artifact_summary(a) for a in artifacts],
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


def _raw_workflow_summary(
    *,
    workflow_id: str,
    name: str,
    version: str,
    state: str,
    nodes: list[WorkflowNodeSummary],
    artifacts: list[ReportArtifact],
) -> dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "name": name,
        "version": version,
        "state": state,
        "nodes": [
            {
                "node_id": node.node_id,
                "tool": node.tool,
                "state": node.state,
                "job_id": node.job_id,
                "output": _compact(node.output),
                "artifacts": [_artifact_summary(a) for a in node.artifacts],
                "error_code": node.error_code,
                "error_message": node.error_message,
            }
            for node in nodes
        ],
        "artifacts": [_artifact_summary(a) for a in artifacts],
    }


def _artifact_summary(artifact: ReportArtifact) -> dict[str, Any]:
    return {
        "field": artifact.field,
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind,
        "size_bytes": artifact.size_bytes,
        "sha256": artifact.sha256,
        "path": artifact.path,
        "job_id": artifact.job_id,
    }


def _reproducibility(
    *,
    source_id: str,
    nodes: list[WorkflowNodeSummary],
    artifacts: list[ReportArtifact],
    workspace: str,
) -> ReproducibilityBlock:
    # Dedupe IDs: when a job report is generated, the source_id is also the
    # single node's job_id — listing it twice is noise.
    ids: list[str] = []
    seen_ids: set[str] = set()
    for candidate in [source_id, *[n.job_id for n in nodes if n.job_id]]:
        if candidate and candidate not in seen_ids:
            ids.append(candidate)
            seen_ids.add(candidate)
    return ReproducibilityBlock(
        ids=ids,
        tool_versions={n.node_id: f"{n.tool}@{n.version}" for n in nodes},
        params_hashes={n.node_id: n.params_hash for n in nodes if n.params_hash},
        artifact_sha256={
            f"{a.job_id or 'artifact'}:{a.field}": a.sha256
            for a in artifacts
            if a.sha256
        },
        replay_commands=[f"vpa report {source_id} --workspace {workspace} --open"],
    )


def _duration_ms(start_us: int | None, end_us: int | None) -> int | None:
    if start_us is None or end_us is None:
        return None
    return max(0, int((end_us - start_us) / 1000))


def _score_map(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # Recurse into nested metric dicts so adapter outputs like
    # ``confidence_scores: {mean_plddt: 0.76, ptm: 0.28}`` surface to the
    # candidate's score table instead of disappearing.
    for nested_key in _NESTED_SCORE_KEYS:
        nested = row.get(nested_key)
        if isinstance(nested, dict):
            out.update(_score_map(nested))
    for key in _SCORE_KEYS:
        if key not in row:
            continue
        value = row.get(key)
        if value is None or not isinstance(value, (int, float, str)):
            continue
        parsed = _as_float(value)
        out[key] = parsed if parsed is not None else str(value)
    # Normalize aliases so downstream renderers can look up a single name.
    if "plddt" not in out and "mean_plddt" in out:
        out["plddt"] = out["mean_plddt"]
    if "pae" not in out and "mean_pae" in out:
        out["pae"] = out["mean_pae"]
    return out


def _dedupe_candidates(candidates: list[ReportCandidate]) -> list[ReportCandidate]:
    if not candidates:
        return []
    records = [
        _report_candidate_to_merge_record(candidate, index)
        for index, candidate in enumerate(candidates)
    ]
    result = merge_candidate_records(
        records,
        strip_private_keys=False,
        include_private_positions=True,
    )
    return [
        _report_candidate_from_merged_record(record, candidates)
        for record in result.candidates
    ]


def _report_candidate_to_merge_record(candidate: ReportCandidate, index: int) -> dict[str, Any]:
    source = dict(candidate.source) if isinstance(candidate.source, dict) else {}
    source.setdefault("tool", "report")
    source.setdefault("local_id", candidate.candidate_id)
    record: dict[str, Any] = {
        "__report_index": index,
        "schema_version": "candidate_v1",
        "candidate_id": candidate.candidate_id,
        "source": source,
        "scores": dict(candidate.scores),
        "artifacts": {
            f"{artifact.field}:{idx}": artifact
            for idx, artifact in enumerate(candidate.artifacts)
        },
    }
    if candidate.sequence:
        record["sequence"] = candidate.sequence
    if candidate.structure_path:
        record["structure_path"] = candidate.structure_path
    return record


def _report_candidate_from_merged_record(
    record: dict[str, Any],
    candidates: list[ReportCandidate],
) -> ReportCandidate:
    representative_index = _as_int(
        record.get("__candidate_merge_representative_position")
        or record.get("__report_index")
    )
    representative = (
        candidates[representative_index]
        if representative_index is not None and 0 <= representative_index < len(candidates)
        else candidates[0]
    )
    artifact_values = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    artifacts = [
        artifact
        for artifact in artifact_values.values()
        if isinstance(artifact, ReportArtifact)
    ]
    positions = _report_candidate_merge_positions(record, len(candidates))
    return replace(
        representative,
        scores=dict(record.get("scores") or {}),
        qc_flags=_merged_report_qc_flags(positions, candidates),
        liabilities=_merged_report_liabilities(positions, candidates),
        artifacts=artifacts or representative.artifacts,
    )


def _report_candidate_merge_positions(
    record: dict[str, Any],
    candidate_count: int,
) -> list[int]:
    raw_positions = record.get("__candidate_merge_positions")
    positions: list[int] = []
    if isinstance(raw_positions, list):
        for value in raw_positions:
            position = _as_int(value)
            if position is None or position < 0 or position >= candidate_count:
                continue
            if position not in positions:
                positions.append(position)
    representative_index = _as_int(record.get("__report_index"))
    if (
        not positions
        and representative_index is not None
        and 0 <= representative_index < candidate_count
    ):
        positions.append(representative_index)
    return positions or [0]


def _merged_report_qc_flags(
    positions: list[int],
    candidates: list[ReportCandidate],
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for position in positions:
        if position < 0 or position >= len(candidates):
            continue
        for flag in candidates[position].qc_flags:
            key = str(flag).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(str(flag))
    return merged


def _merged_report_liabilities(
    positions: list[int],
    candidates: list[ReportCandidate],
) -> list[Any]:
    merged: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for position in positions:
        if position < 0 or position >= len(candidates):
            continue
        for liability in candidates[position].liabilities:
            key = _liability_key(liability)
            if key in seen:
                continue
            seen.add(key)
            merged.append(liability)
    return merged


def _liability_key(entry: Any) -> tuple[Any, ...]:
    if isinstance(entry, dict):
        positions = entry.get("positions")
        pos_key: Any
        if isinstance(positions, list):
            pos_key = tuple(positions)
        else:
            pos_key = positions
        return (entry.get("type"), entry.get("chain_id"), pos_key)
    return (str(entry),)


def _matching_artifacts(path: str | None, artifacts: list[ReportArtifact]) -> list[ReportArtifact]:
    if not path:
        return []
    query = Path(path).expanduser()
    resolved = str(query)
    exact = [
        artifact for artifact in artifacts
        if artifact.path == path or str(Path(artifact.path).expanduser()) == resolved
    ]
    if exact:
        return exact
    if query.name:
        by_name = [
            artifact
            for artifact in artifacts
            if _is_structure_path(artifact.path, artifact.kind)
            and Path(artifact.path).name == query.name
        ]
        if by_name:
            return by_name
        query_stem = query.stem.lower()
        by_field = [
            artifact
            for artifact in artifacts
            if _is_structure_path(artifact.path, artifact.kind)
            and query_stem
            and (query_stem in artifact.field.lower() or artifact.field.lower() in query_stem)
        ]
        if by_field:
            return by_field
    return []


def _stable_structure_path(path: str | None, artifacts: list[ReportArtifact]) -> str | None:
    if not path:
        return None
    matches = _matching_artifacts(path, artifacts)
    if matches:
        return matches[0].path
    if path and Path(path).expanduser().is_file():
        return path
    structure_artifacts = [
        artifact for artifact in artifacts if _is_structure_path(artifact.path, artifact.kind)
    ]
    if len(structure_artifacts) == 1:
        return structure_artifacts[0].path
    return path


def _is_structure_path(path: str, kind: str | None = None) -> bool:
    return Path(path).suffix.lower() in _STRUCTURE_SUFFIXES or (kind or "").lower() in {
        "pdb",
        "cif",
        "mmcif",
        "structure",
    }


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


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _compact(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth >= 3:
        return f"<{type(value).__name__}>"
    if isinstance(value, list):
        return [_compact(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth=depth + 1)
            for key, item in list(value.items())[:50]
        }
    return str(value)


# --------------------------------------------------------------------------- agent transcript


def _collect_agent_transcript(
    workspace: Workspace,
    *,
    workflow_id: str | None = None,
    job_id: str | None = None,
) -> AgentTranscript | None:
    """Pull the LLM-agent conversation that produced this workflow / job.

    Looks up the ``agent_runs`` row whose ``workflow_ids_json`` /
    ``job_ids_json`` contains the given id, then loads the message log,
    last validated plan, and final report. Returns ``None`` when:

    * no agent run is associated with this workflow / job (e.g. it was
      submitted as a direct-tool call), OR
    * the associated run carries no user message and no assistant
      message (purely-prevalidated submission with nothing to show).

    Failures (DB errors, malformed rows) are swallowed and return
    ``None`` — the rest of the report still renders.
    """
    if not workflow_id and not job_id:
        return None
    try:
        # Local import to avoid a circular dependency at module-load time
        # (the agents.session module pulls in heavier deps that the
        # reporting layer should not require for non-agent runs).
        from vibeproteinagent.agents.session.store import AgentSessionStore

        store = AgentSessionStore(workspace)
        run = _find_agent_run_for(store, workflow_id=workflow_id, job_id=job_id)
        if run is None:
            return None
        details = store.get_run_details(run.run_id)
    except Exception:
        return None

    messages_raw = details.get("messages") or []
    tool_calls_raw = details.get("tool_calls") or []
    plans_raw = details.get("plans") or []
    decisions_raw = details.get("decisions") or []
    summaries_raw = details.get("summaries") or []
    run_record = details.get("run") or {}

    messages: list[AgentMessage] = []
    for row in messages_raw:
        if not isinstance(row, dict):
            continue
        content = str(row.get("content") or "").strip()
        if not content:
            continue
        messages.append(
            AgentMessage(
                role=str(row.get("role") or "assistant"),
                content=content,
                timestamp_ms=_as_int(row.get("created_at")),
                model=row.get("model"),
                token_count=_as_int(row.get("token_count")),
            )
        )

    tool_calls: list[AgentToolCall] = []
    for row in tool_calls_raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("tool_name") or "").strip()
        if not name:
            continue
        args = row.get("args")
        tool_calls.append(
            AgentToolCall(
                tool_name=name,
                args=args if isinstance(args, dict) else {},
                result=row.get("result"),
                ok=bool(row.get("ok", True)),
                timestamp_ms=_as_int(row.get("created_at")),
            )
        )

    decisions: list[AgentDecision] = []
    for row in decisions_raw:
        if not isinstance(row, dict):
            continue
        decision_obj = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        next_round_raw = decision_obj.get("next_round")
        next_round = next_round_raw if isinstance(next_round_raw, dict) else {}
        notes_raw = decision_obj.get("notes")
        decisions.append(
            AgentDecision(
                round_index=int(row.get("round_index") or 0),
                decision_label=str(decision_obj.get("decision") or "decision"),
                next_round_recommended=bool(next_round.get("recommended", False)),
                reason=str(next_round.get("reason") or ""),
                notes=notes_raw if isinstance(notes_raw, list) else [],
                metrics_used=[
                    str(m) for m in (decision_obj.get("metrics_used") or [])
                    if isinstance(m, (str, int, float))
                ],
                raw=decision_obj,
                timestamp_ms=_as_int(row.get("created_at")),
            )
        )

    step_summaries: list[AgentStepSummary] = []
    for row in summaries_raw:
        if not isinstance(row, dict):
            continue
        body = row.get("summary") if isinstance(row.get("summary"), dict) else {}
        is_workflow = bool(body.get("workflow_id")) and "nodes" in body
        error = body.get("error") if isinstance(body.get("error"), dict) else None
        step_summaries.append(
            AgentStepSummary(
                kind="workflow" if is_workflow else "job",
                workflow_id=row.get("workflow_id") or body.get("workflow_id"),
                job_id=row.get("job_id") or body.get("job_id"),
                tool=body.get("tool"),
                state=body.get("state"),
                name=body.get("name"),
                error_message=(error or {}).get("message") or body.get("error_message"),
                metrics_used=[
                    str(m) for m in (body.get("metrics_used") or [])
                    if isinstance(m, (str, int, float))
                ],
                raw=body,
                timestamp_ms=_as_int(row.get("created_at")),
            )
        )

    # User query: prefer the first explicit user message; fall back to
    # ``agent_runs.user_request`` so we still have something when the
    # message log was pruned.
    user_query = ""
    for m in messages:
        if m.role == "user":
            user_query = m.content
            break
    if not user_query:
        user_query = str(run_record.get("user_request") or "").strip()
    if not user_query and not any(m.role == "assistant" for m in messages):
        # Nothing useful to surface.
        return None

    # Plan: pick the most-recently-validated plan; fall back to the last
    # plan of any status so we still show *something* even mid-run.
    plan_structured: dict[str, Any] | None = None
    validated_statuses = {"validated", "validated_next_round", "submitted"}
    for entry in reversed(plans_raw):
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in validated_statuses:
            plan_structured = entry.get("plan") if isinstance(entry.get("plan"), dict) else None
            break
    if plan_structured is None:
        for entry in reversed(plans_raw):
            if isinstance(entry, dict) and isinstance(entry.get("plan"), dict):
                plan_structured = entry["plan"]
                break

    # Plan text: take the *last assistant message before the final*
    # — that's typically where the agent explains the plan it generated.
    plan_text: str | None = None
    assistant_msgs = [m for m in messages if m.role == "assistant"]
    if len(assistant_msgs) >= 2:
        plan_text = assistant_msgs[0].content
    elif assistant_msgs and not plan_structured:
        # Only one assistant message exists; treat it as the plan
        # unless it clearly reads as a final wrap-up.
        plan_text = assistant_msgs[0].content

    # Final response: prefer agent_runs.final_report_json text; fall
    # back to the last assistant message.
    final_response: str | None = None
    final_report = run_record.get("final_report")
    if isinstance(final_report, dict):
        for key in ("summary", "final_response", "narrative", "answer", "text"):
            value = final_report.get(key)
            if isinstance(value, str) and value.strip():
                final_response = value.strip()
                break
    if not final_response and len(assistant_msgs) >= 2:
        final_response = assistant_msgs[-1].content

    return AgentTranscript(
        run_id=str(run_record.get("run_id") or run.run_id),
        user_query=user_query,
        plan_text=plan_text,
        plan_structured=plan_structured,
        final_response=final_response,
        messages=messages,
        tool_calls=tool_calls,
        decisions=decisions,
        step_summaries=step_summaries,
        model=run_record.get("model"),
        provider=run_record.get("provider"),
        started_at_ms=_as_int(run_record.get("created_at")),
        finished_at_ms=_as_int(run_record.get("updated_at")),
    )


def _find_agent_run_for(
    store: Any,
    *,
    workflow_id: str | None,
    job_id: str | None,
) -> Any | None:
    """Scan recent agent runs for one whose workflow/job lists contain our id.

    ``agent_runs`` doesn't have a reverse index from workflow → run, so we
    list the most-recent N runs and match in Python. This is O(N) but N is
    bounded (typical workspace has a small handful of agent runs); the
    sample data uses ≤3 runs per session.
    """
    try:
        runs = store.list_runs(limit=200)
    except Exception:
        return None
    for run in runs:
        wf_ids = getattr(run, "workflow_ids", None) or []
        job_ids = getattr(run, "job_ids", None) or []
        if workflow_id and workflow_id in wf_ids:
            return run
        if job_id and job_id in job_ids:
            return run
    return None


def _agent_allows_auto_selection(workspace: Workspace, workflow_id: str) -> bool:
    """Return true only for agent runs that explicitly exhausted round budget.

    Report collection is also used for direct workflows and one-off jobs. Those
    should show explored candidates, but not invent a final answer. The agent
    controller sets this marker only when max_rounds is reached without a
    submit_candidate result.
    """
    try:
        from vibeproteinagent.agents.session.store import AgentSessionStore

        store = AgentSessionStore(workspace)
        run = _find_agent_run_for(store, workflow_id=workflow_id, job_id=None)
        if run is None:
            return False
        details = store.get_run_details(run.run_id)
    except Exception:
        return False

    for row in reversed(details.get("decisions") or []):
        if not isinstance(row, dict):
            continue
        decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        fallback = decision.get("fallback_final_selection")
        if isinstance(fallback, dict):
            return fallback.get("auto_select") is True
    return False


def _agent_submission_outputs(
    workspace: Workspace,
    *,
    workflow_id: str | None,
    job_id: str | None,
) -> list[dict[str, Any]]:
    try:
        from vibeproteinagent.agents.session.store import AgentSessionStore

        store = AgentSessionStore(workspace)
        run = _find_agent_run_for(store, workflow_id=workflow_id, job_id=job_id)
        if run is None:
            return []
        details = store.get_run_details(run.run_id)
    except Exception:
        return []

    submissions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in details.get("decisions") or []:
        if not isinstance(row, dict):
            continue
        decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        submission = decision.get("submission")
        if not isinstance(submission, dict) or submission.get("terminal") is not True:
            continue
        submission_id = str(submission.get("submission_id") or "")
        key = submission_id or json.dumps(submission, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        row_submission = dict(submission)
        if decision.get("submit_job_id"):
            row_submission.setdefault("job_id", decision.get("submit_job_id"))
        row_submission.setdefault("node_id", "post_round_decision")
        submissions.append(row_submission)
    return submissions


__all__ = ["collect_report_model"]
