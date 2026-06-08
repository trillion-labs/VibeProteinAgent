from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from vibeproteinagent.candidates.merge import merge_candidate_records
from vibeproteinagent.core.errors import InvalidParamsError, NotFoundError, WorkspaceError
from vibeproteinagent.reporting.enrich import REPORT_ENRICHMENT_CALL_ID_PREFIX
from vibeproteinagent.workspace import Workspace
from vibeproteinagent.workspace.types import (
    ArtifactRecord,
    JobRecord,
    JobState,
    Priority,
    TrajectoryRecord,
    WorkflowNodeRecord,
    WorkflowRecord,
)

from .models import (
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
    "mean_iptm",
    "ptm",
    "tm_score",
    "pae",
    "mean_pae",
    "pae_interaction",
    "ipae",
    "iplddt",
    "lddt",
    "gpde",
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
    # Sequence design
    "recovery",
    "seq_recovery",
    # Biophysical / QC (protein_qc)
    "gravy",
    "pi",
    "instability_index",
    "polybasic_hits",
    "ca_bfactor_plddt",
    "atom_bfactor_plddt",
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
_REFERENCE_CANDIDATE_TOOLS = frozenset({
    "interpro_sequences",
    "go_sequences",
    "family_profile",
})
_NON_BOARD_CANDIDATE_TOOLS = _REFERENCE_CANDIDATE_TOOLS | frozenset({"submit_candidate"})
_FINALIZATION_EVALUATOR_TOOLS = frozenset({
    "boltz2_affinity",
    "ipsae",
    "prodigy",
    "protein_qc",
    "protenix_v2",
    "pxmeter",
    "pyrosetta",
    "structure_analysis",
    "tmalign",
})
_FINALIZATION_GENERATOR_TOOLS = frozenset({
    "bindcraft",
    "boltzgen",
    "esm2",
    "esm3",
    "proteina_complexa",
    "proteinmpnn",
    "pxdesign",
    "rfdiffusion",
    "rfdiffusion3",
    "rfpeptides",
})


def collect_report_model(workspace: Workspace, source_id: str, *, top_k: int = 5) -> ReportModel:
    del top_k  # top-k is a rendering concern; collection keeps the complete model.
    if source_id.startswith("traj_"):
        return _collect_trajectory(workspace, source_id)
    if source_id.startswith("wf_"):
        return _collect_workflow(workspace, source_id)
    if source_id.startswith("job_"):
        return _collect_job(workspace, source_id)
    raise InvalidParamsError(
        "report id must start with traj_, job_, or wf_",
        details={"id": source_id, "allowed_prefixes": ["traj_", "job_", "wf_"]},
    )


def _collect_job(workspace: Workspace, job_id: str) -> ReportModel:
    job = _read_job_readonly(workspace, job_id)
    if job.tool_name == "submit_candidate":
        if job.workflow_id:
            try:
                return _collect_workflow(workspace, job.workflow_id)
            except NotFoundError:
                pass
        return _collect_submit_candidate_trajectory(workspace, job)
    return _collect_single_job(workspace, job)


def _collect_single_job(workspace: Workspace, job: JobRecord) -> ReportModel:
    artifacts = _artifact_list(
        _read_artifacts_for_job_readonly(workspace, job.job_id),
        job_id=job.job_id,
    )
    node = _node_from_job(job, artifacts)
    report_nodes = [node]
    all_artifacts = list(artifacts)
    warnings = _warnings_from_job(job)
    params_by_node = {node.node_id: job.params if isinstance(job.params, dict) else {}}
    enrichment_nodes, enrichment_artifacts, enrichment_warnings, enrichment_params = (
        _report_enrichment_nodes(workspace, job.job_id, upstream=node.node_id)
    )
    report_nodes.extend(enrichment_nodes)
    all_artifacts.extend(enrichment_artifacts)
    warnings.extend(enrichment_warnings)
    params_by_node.update(enrichment_params)
    candidates = _normalize_candidates(
        report_nodes,
        all_artifacts,
        node_params=params_by_node,
    )
    submissions = _submission_outputs(report_nodes)
    final_submission = _latest_final_submission(submissions)
    final_candidates = _enrich_final_candidates_from_current_candidates(
        _candidates_from_submission(final_submission),
        candidates,
    )
    if not candidates:
        candidates = _structure_candidates(
            all_artifacts,
            source={"kind": "job", "job_id": job.job_id},
        )
    msa_blocks = _collect_msa_blocks(
        all_artifacts,
        workflow_tools={(n.tool or "").lower() for n in report_nodes},
        tool_by_job={n.job_id: (n.tool or "").lower() for n in report_nodes if n.job_id},
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
        graph=WorkflowGraph(workflow_id=job.workflow_id, nodes=report_nodes),
        candidates=candidates,
        final_candidates=final_candidates,
        final_submission=final_submission,
        submissions=submissions,
        artifacts=all_artifacts,
        msa_blocks=msa_blocks,
        warnings=warnings,
        reproducibility=_reproducibility(
            source_id=job.job_id,
            nodes=report_nodes,
            artifacts=all_artifacts,
            workspace=workspace.name,
        ),
        satisfied_semantic_facets=_satisfied_semantic_facets_from_nodes(report_nodes),
        empty_candidate_events=_empty_candidate_events_from_nodes(report_nodes),
        candidate_capability_filter_events=(
            _candidate_capability_filter_events_from_nodes(report_nodes)
        ),
        candidate_merge_events=_candidate_merge_events_from_nodes(report_nodes),
        parameter_resolution_events=_parameter_resolution_events_from_nodes(report_nodes),
        raw_summary=_raw_trajectory_summary(
            source_job_id=job.job_id,
            nodes=report_nodes,
            artifacts=all_artifacts,
        )
        if enrichment_nodes
        else _raw_job_summary(job, artifacts),
    )


def _collect_submit_candidate_trajectory(
    workspace: Workspace,
    final_job: JobRecord,
) -> ReportModel:
    """Report a direct-run query as one trajectory when its terminal submit job is reported.

    Agents often run VPA as a sequence of direct ``vpa run`` calls and finish
    with ``submit_candidate`` rather than compiling a ``wf_`` descriptor. In
    that mode the submit job is the only stable report handle the agent has, so
    ``vpa report job_<submit>`` should expand to the recorded workspace
    trajectory up to that final selection instead of rendering a one-node
    report.
    """

    jobs = _read_direct_trajectory_jobs_readonly(workspace, final_job)
    if not any(job.job_id == final_job.job_id for job in jobs):
        jobs.append(final_job)
        jobs.sort(key=lambda job: (job.submitted_at, job.job_id))
    enrichment_jobs = _read_report_enrichment_jobs_readonly(workspace, final_job.job_id)
    if enrichment_jobs:
        known_job_ids = {job.job_id for job in jobs}
        jobs.extend(job for job in enrichment_jobs if job.job_id not in known_job_ids)
        jobs.sort(key=lambda job: (job.submitted_at, job.job_id))

    report_nodes: list[WorkflowNodeSummary] = []
    all_artifacts: list[ReportArtifact] = []
    warnings: list[ReportWarning] = []
    params_by_node: dict[str, dict[str, Any]] = {}
    previous_node_id: str | None = None
    seen_node_ids: set[str] = set()

    for index, job in enumerate(jobs, start=1):
        artifacts = _artifact_list(
            _read_artifacts_for_job_readonly(workspace, job.job_id),
            job_id=job.job_id,
        )
        node = _node_from_job(job, artifacts)
        node_id = _direct_trajectory_node_id(job, index, seen_node_ids)
        seen_node_ids.add(node_id)
        upstream = [previous_node_id] if previous_node_id else []
        node = replace(
            node,
            node_id=node_id,
            upstream=upstream,
            workflow_id=None,
        )
        previous_node_id = node_id
        report_nodes.append(node)
        all_artifacts.extend(artifacts)
        warnings.extend(_warnings_from_job(job, source=node_id))
        if isinstance(job.params, dict):
            params_by_node[node_id] = job.params

    candidates = _normalize_candidates(
        report_nodes,
        all_artifacts,
        node_params=params_by_node,
    )
    submissions = _submission_outputs(report_nodes)
    final_submission = _latest_final_submission(submissions)
    final_candidates = _enrich_final_candidates_from_current_candidates(
        _candidates_from_submission(final_submission),
        candidates,
    )
    if not candidates:
        candidates = _structure_candidates(
            all_artifacts,
            source={"kind": "direct_trajectory", "job_id": final_job.job_id},
        )
    workflow_tools = {(n.tool or "").lower() for n in report_nodes}
    tool_by_job = {n.job_id: (n.tool or "").lower() for n in report_nodes if n.job_id}
    warnings.extend(
        _finalization_contract_warnings(
            report_nodes,
            candidates,
            final_candidates,
            final_submission,
        )
    )
    return ReportModel(
        summary=RunSummary(
            kind="job",
            id=final_job.job_id,
            workspace=workspace.name,
            state=final_job.state.value,
            name="direct-run trajectory",
            version=final_job.tool_version,
            submitted_at=jobs[0].submitted_at if jobs else final_job.submitted_at,
            finished_at=final_job.finished_at,
            runtime_ms=_duration_ms(
                jobs[0].submitted_at if jobs else final_job.submitted_at,
                final_job.finished_at,
            ),
        ),
        graph=WorkflowGraph(workflow_id=None, nodes=report_nodes),
        candidates=candidates,
        final_candidates=final_candidates,
        final_submission=final_submission,
        submissions=submissions,
        artifacts=all_artifacts,
        msa_blocks=_collect_msa_blocks(
            all_artifacts,
            workflow_tools=workflow_tools,
            tool_by_job=tool_by_job,
        ),
        warnings=warnings,
        reproducibility=_reproducibility(
            source_id=final_job.job_id,
            nodes=report_nodes,
            artifacts=all_artifacts,
            workspace=workspace.name,
        ),
        satisfied_semantic_facets=_satisfied_semantic_facets_from_nodes(report_nodes),
        empty_candidate_events=_empty_candidate_events_from_nodes(report_nodes),
        candidate_capability_filter_events=(
            _candidate_capability_filter_events_from_nodes(report_nodes)
        ),
        candidate_merge_events=_candidate_merge_events_from_nodes(report_nodes),
        parameter_resolution_events=_parameter_resolution_events_from_nodes(report_nodes),
        raw_summary=_raw_trajectory_summary(
            source_job_id=final_job.job_id,
            nodes=report_nodes,
            artifacts=all_artifacts,
        ),
    )


def _collect_trajectory(workspace: Workspace, trajectory_id: str) -> ReportModel:
    trajectory = _read_trajectory_readonly(workspace, trajectory_id)
    direct_jobs = _read_trajectory_direct_jobs_readonly(workspace, trajectory_id)
    workflows = _read_trajectory_workflows_readonly(workspace, trajectory_id)

    report_nodes: list[WorkflowNodeSummary] = []
    all_artifacts: list[ReportArtifact] = []
    warnings: list[ReportWarning] = []
    params_by_node: dict[str, dict[str, Any]] = {}
    seen_node_ids: set[str] = set()
    top_level_events: list[dict[str, Any]] = []
    direct_index = 0
    events: list[tuple[int, str, str, JobRecord | WorkflowRecord]] = []
    events.extend(
        (int(job.submitted_at), "job", job.job_id, job)
        for job in direct_jobs
    )
    events.extend(
        (int(wf.submitted_at), "workflow", wf.workflow_id, wf)
        for wf in workflows
    )
    events.sort(key=lambda e: (e[0], e[2]))

    for _, kind, _event_id, event in events:
        if kind == "job":
            job = event
            assert isinstance(job, JobRecord)
            direct_index += 1
            artifacts = _artifact_list(
                _read_artifacts_for_job_readonly(workspace, job.job_id),
                job_id=job.job_id,
            )
            node = _node_from_job(job, artifacts)
            node_id = _direct_trajectory_node_id(job, direct_index, seen_node_ids)
            seen_node_ids.add(node_id)
            node = replace(node, node_id=node_id, upstream=[], workflow_id=None)
            report_nodes.append(node)
            all_artifacts.extend(artifacts)
            warnings.extend(_warnings_from_job(job, source=node_id))
            if isinstance(job.params, dict):
                params_by_node[node_id] = job.params
            top_level_events.append(
                {
                    "kind": "job",
                    "id": job.job_id,
                    "submitted_at": job.submitted_at,
                    "state": job.state.value,
                    "tool": job.tool_name,
                }
            )
        else:
            wf = event
            assert isinstance(wf, WorkflowRecord)
            wf_nodes, wf_artifacts, wf_warnings, wf_params = _workflow_report_nodes(workspace, wf)
            report_nodes.extend(wf_nodes)
            all_artifacts.extend(wf_artifacts)
            warnings.extend(wf_warnings)
            params_by_node.update(wf_params)
            top_level_events.append(
                {
                    "kind": "workflow",
                    "id": wf.workflow_id,
                    "submitted_at": wf.submitted_at,
                    "state": wf.state,
                    "workflow": wf.name,
                }
            )

    candidates = _normalize_candidates(
        report_nodes,
        all_artifacts,
        node_params=params_by_node,
    )
    submissions = _submission_outputs(report_nodes)
    final_submission = _latest_final_submission(submissions)
    final_candidates = _enrich_final_candidates_from_current_candidates(
        _candidates_from_submission(final_submission),
        candidates,
    )
    if not candidates:
        candidates = _structure_candidates(
            all_artifacts,
            source={"kind": "trajectory", "trajectory_id": trajectory_id},
        )
    started = _min_not_none(
        [j.submitted_at for j in direct_jobs]
        + [w.submitted_at for w in workflows]
        + [trajectory.created_at]
    )
    finished = _max_not_none(
        [j.finished_at for j in direct_jobs if j.finished_at is not None]
        + [w.finished_at for w in workflows if w.finished_at is not None]
    )
    all_job_terminal = all(
        j.state.value in {"completed", "failed", "cancelled"} for j in direct_jobs
    )
    all_workflow_terminal = all(w.state in {"completed", "failed", "cancelled"} for w in workflows)
    state = (
        "completed"
        if final_submission is not None and all_job_terminal and all_workflow_terminal
        else "open"
    )
    workflow_tools = {(n.tool or "").lower() for n in report_nodes}
    tool_by_job = {n.job_id: (n.tool or "").lower() for n in report_nodes if n.job_id}
    warnings.extend(
        _finalization_contract_warnings(
            report_nodes,
            candidates,
            final_candidates,
            final_submission,
        )
    )
    return ReportModel(
        summary=RunSummary(
            kind="trajectory",
            id=trajectory.trajectory_id,
            workspace=workspace.name,
            state=state,
            name="adaptive trajectory",
            version=None,
            submitted_at=started,
            finished_at=finished,
            runtime_ms=_duration_ms(started, finished),
        ),
        graph=WorkflowGraph(workflow_id=None, nodes=report_nodes),
        candidates=candidates,
        final_candidates=final_candidates,
        final_submission=final_submission,
        submissions=submissions,
        artifacts=all_artifacts,
        msa_blocks=_collect_msa_blocks(
            all_artifacts,
            workflow_tools=workflow_tools,
            tool_by_job=tool_by_job,
        ),
        warnings=warnings,
        reproducibility=_reproducibility(
            source_id=trajectory_id,
            nodes=report_nodes,
            artifacts=all_artifacts,
            workspace=workspace.name,
        ),
        satisfied_semantic_facets=_satisfied_semantic_facets_from_nodes(report_nodes),
        empty_candidate_events=_empty_candidate_events_from_nodes(report_nodes),
        candidate_capability_filter_events=(
            _candidate_capability_filter_events_from_nodes(report_nodes)
        ),
        candidate_merge_events=_candidate_merge_events_from_nodes(report_nodes),
        parameter_resolution_events=_parameter_resolution_events_from_nodes(report_nodes),
        raw_summary=_raw_trajectory_report_summary(
            trajectory=trajectory,
            nodes=report_nodes,
            artifacts=all_artifacts,
            top_level_events=top_level_events,
        ),
    )


def _collect_workflow(workspace: Workspace, workflow_id: str) -> ReportModel:
    wf = _read_workflow_readonly(workspace, workflow_id)
    objective_context = _workflow_objective_context(wf)
    report_nodes, all_artifacts, warnings, params_by_node = _workflow_report_nodes(
        workspace,
        wf,
    )
    upstream = report_nodes[-1].node_id if report_nodes else None
    enrichment_nodes, enrichment_artifacts, enrichment_warnings, enrichment_params = (
        _report_enrichment_nodes(workspace, wf.workflow_id, upstream=upstream)
    )
    report_nodes.extend(enrichment_nodes)
    all_artifacts.extend(enrichment_artifacts)
    warnings.extend(enrichment_warnings)
    params_by_node.update(enrichment_params)

    candidates = _normalize_candidates(
        report_nodes, all_artifacts, node_params=params_by_node,
    )
    submissions = _submission_outputs(report_nodes)
    final_submission = _latest_final_submission(submissions)
    final_candidates = _enrich_final_candidates_from_current_candidates(
        _candidates_from_submission(final_submission),
        candidates,
    )
    if not candidates:
        candidates = _structure_candidates(
            all_artifacts, source={"kind": "workflow", "workflow_id": wf.workflow_id}
        )
    workflow_tools = {(n.tool or "").lower() for n in report_nodes}
    tool_by_job = {n.job_id: (n.tool or "").lower() for n in report_nodes if n.job_id}
    warnings.extend(
        _finalization_contract_warnings(
            report_nodes,
            candidates,
            final_candidates,
            final_submission,
        )
    )
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
            user_query=objective_context.get("user_query"),
            objective=objective_context.get("objective"),
        ),
    )


def _workflow_report_nodes(
    workspace: Workspace,
    wf: WorkflowRecord,
) -> tuple[
    list[WorkflowNodeSummary],
    list[ReportArtifact],
    list[ReportWarning],
    dict[str, dict[str, Any]],
]:
    nodes = _read_workflow_nodes_readonly(workspace, wf.workflow_id)
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
            workflow_id=wf.workflow_id,
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

    return report_nodes, all_artifacts, warnings, params_by_node


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
        for key in ("candidate_capability_filter_event", "candidate_selection_event"):
            event = output.get(key)
            if isinstance(event, dict):
                events.append(_availability_filter_event(node, event, event_type=key))
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
    *,
    event_type: str = "candidate_capability_filter_event",
) -> dict[str, Any]:
    filtered = event.get("filtered_candidates")
    if not isinstance(filtered, list):
        filtered = []
    selected_ids = [
        str(item) for item in event.get("selected_candidate_ids") or [] if item not in (None, "")
    ]
    skipped_ids = [
        str(item) for item in event.get("skipped_candidate_ids") or [] if item not in (None, "")
    ]
    return {
        "event_type": event_type,
        "node_id": node.node_id,
        "tool": node.tool,
        "status": event.get("status"),
        "reason": event.get("selection_reason") or event.get("reason"),
        "input_field": event.get("input_field"),
        "candidate_count": _as_int(event.get("input_candidate_count"))
        or _as_int(event.get("candidate_count")),
        "compatible_candidate_count": _as_int(
            event.get("compatible_candidate_count")
        )
        or (len(selected_ids) if selected_ids else None),
        "skipped_candidate_count": _as_int(event.get("skipped_candidate_count"))
        or (len(skipped_ids) if skipped_ids else None),
        "downstream_tool": event.get("downstream_tool"),
        "selection_metrics": event.get("selection_metrics")
        if isinstance(event.get("selection_metrics"), dict)
        else {},
        "selected_candidate_ids": selected_ids[:_AVAILABILITY_FILTERED_CANDIDATE_LIMIT],
        "skipped_candidate_ids": skipped_ids[:_AVAILABILITY_FILTERED_CANDIDATE_LIMIT],
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
        trajectory_id=row["trajectory_id"] if "trajectory_id" in row.keys() else None,
        state=row["state"],
        submitted_at=int(row["submitted_at"]),
        finished_at=row["finished_at"],
    )


def _read_trajectory_readonly(workspace: Workspace, trajectory_id: str) -> TrajectoryRecord:
    conn = _readonly_connection(workspace)
    try:
        row = conn.execute(
            "SELECT * FROM trajectories WHERE trajectory_id = ?",
            (trajectory_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise NotFoundError(
            f"trajectory {trajectory_id!r} not found",
            details={"trajectory_id": trajectory_id},
        )
    return TrajectoryRecord(
        trajectory_id=row["trajectory_id"],
        workspace=row["workspace"],
        user_query=row["user_query"],
        objective=row["objective"],
        state=row["state"],
        final_submit_job_id=row["final_submit_job_id"],
        report_path=row["report_path"],
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
        completed_at=row["completed_at"],
    )


def _read_trajectory_direct_jobs_readonly(
    workspace: Workspace,
    trajectory_id: str,
) -> list[JobRecord]:
    conn = _readonly_connection(workspace)
    try:
        rows = conn.execute(
            """
            SELECT * FROM jobs
             WHERE trajectory_id = ?
               AND workflow_id IS NULL
             ORDER BY submitted_at ASC, job_id ASC
            """,
            (trajectory_id,),
        ).fetchall()
    finally:
        conn.close()
    return [_job_from_row(row) for row in rows]


def _read_trajectory_workflows_readonly(
    workspace: Workspace,
    trajectory_id: str,
) -> list[WorkflowRecord]:
    conn = _readonly_connection(workspace)
    try:
        rows = conn.execute(
            """
            SELECT * FROM workflows
             WHERE trajectory_id = ?
             ORDER BY submitted_at ASC, workflow_id ASC
            """,
            (trajectory_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        WorkflowRecord(
            workflow_id=row["workflow_id"],
            name=row["name"],
            version=row["version"],
            descriptor=row["descriptor"],
            params=_json_load(row["params_json"], default={}),
            trajectory_id=row["trajectory_id"] if "trajectory_id" in row.keys() else None,
            state=row["state"],
            submitted_at=int(row["submitted_at"]),
            finished_at=row["finished_at"],
        )
        for row in rows
    ]


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
    return _job_from_row(row)


def _read_direct_trajectory_jobs_readonly(
    workspace: Workspace,
    final_job: JobRecord,
) -> list[JobRecord]:
    conn = _readonly_connection(workspace)
    try:
        rows = conn.execute(
            """
            SELECT * FROM jobs
             WHERE workflow_id IS NULL
               AND submitted_at <= ?
             ORDER BY submitted_at ASC, job_id ASC
            """,
            (final_job.submitted_at,),
        ).fetchall()
    finally:
        conn.close()
    jobs = [_job_from_row(row) for row in rows]
    cutoff = _direct_trajectory_cutoff_submission_time(jobs, final_job)
    if cutoff is not None:
        jobs = [job for job in jobs if job.submitted_at >= cutoff]
    return jobs


def _read_report_enrichment_jobs_readonly(
    workspace: Workspace,
    source_id: str,
) -> list[JobRecord]:
    prefix = f"{REPORT_ENRICHMENT_CALL_ID_PREFIX}_{_safe_enrichment_source(source_id)}_"
    conn = _readonly_connection(workspace)
    try:
        rows = conn.execute(
            """
            SELECT * FROM jobs
             WHERE call_id LIKE ?
             ORDER BY submitted_at ASC, job_id ASC
            """,
            (prefix + "%",),
        ).fetchall()
    finally:
        conn.close()
    return [_job_from_row(row) for row in rows]


def _report_enrichment_nodes(
    workspace: Workspace,
    source_id: str,
    *,
    upstream: str | None,
) -> tuple[
    list[WorkflowNodeSummary],
    list[ReportArtifact],
    list[ReportWarning],
    dict[str, dict[str, Any]],
]:
    nodes: list[WorkflowNodeSummary] = []
    artifacts: list[ReportArtifact] = []
    warnings: list[ReportWarning] = []
    params: dict[str, dict[str, Any]] = {}
    previous = upstream
    seen_node_ids: set[str] = set()
    for index, job in enumerate(
        _read_report_enrichment_jobs_readonly(workspace, source_id),
        start=1,
    ):
        job_artifacts = _artifact_list(
            _read_artifacts_for_job_readonly(workspace, job.job_id),
            job_id=job.job_id,
        )
        node = _node_from_job(job, job_artifacts)
        node_id = _direct_trajectory_node_id(job, index, seen_node_ids | {n.node_id for n in nodes})
        seen_node_ids.add(node_id)
        node = replace(
            node,
            node_id=node_id,
            upstream=[previous] if previous else [],
            workflow_id=None,
        )
        previous = node_id
        nodes.append(node)
        artifacts.extend(job_artifacts)
        warnings.extend(_warnings_from_job(job, source=node_id))
        if isinstance(job.params, dict):
            params[node_id] = job.params
    return nodes, artifacts, warnings, params


def _safe_enrichment_source(source_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in source_id)


def _direct_trajectory_cutoff_submission_time(
    jobs: list[JobRecord],
    final_job: JobRecord,
) -> int | None:
    final_output = final_job.output if isinstance(final_job.output, dict) else {}
    superseded = {
        str(final_output.get("supersedes"))
    } if final_output.get("supersedes") else set()
    current = set(superseded)
    while current:
        next_current: set[str] = set()
        for job in jobs:
            output = job.output if isinstance(job.output, dict) else {}
            submission_id = output.get("submission_id")
            if submission_id is None or str(submission_id) not in current:
                continue
            previous = output.get("supersedes")
            if previous:
                next_current.add(str(previous))
        current = next_current - superseded
        superseded.update(current)

    prior_submits: list[JobRecord] = []
    superseded_submit_times: list[int] = []
    for job in jobs:
        if job.job_id == final_job.job_id or job.tool_name != "submit_candidate":
            continue
        output = job.output if isinstance(job.output, dict) else {}
        if output.get("terminal") is not True:
            continue
        submission_id = str(output.get("submission_id") or "")
        if submission_id and submission_id in superseded:
            superseded_submit_times.append(job.submitted_at)
            continue
        prior_submits.append(job)
    if superseded_submit_times:
        return min(superseded_submit_times)
    if prior_submits:
        return max(job.submitted_at for job in prior_submits) + 1
    return None


def _job_from_row(row: sqlite3.Row) -> JobRecord:
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
        trajectory_id=get("trajectory_id"),
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


def _direct_trajectory_node_id(
    job: JobRecord,
    index: int,
    seen: set[str],
) -> str:
    base = job.workflow_node_id or job.tool_name or "tool"
    candidate = base
    if candidate not in seen:
        return candidate
    short_job = job.job_id[4:12] if job.job_id.startswith("job_") else job.job_id[:8]
    candidate = f"{base}_{short_job}"
    if candidate not in seen:
        return candidate
    return f"{base}_{index}"


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
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default
    return default if parsed is None else parsed


def _workflow_objective_context(wf: WorkflowRecord) -> dict[str, str]:
    """Natural-language context recorded on a workflow descriptor or params.

    Current workflow reports are scoped to one user query, not one node/job.
    The descriptor-level fields are preferred because they describe the whole
    process; run params are accepted as a fallback for programmatic workflows.
    """

    descriptor = _workflow_descriptor_mapping(wf.descriptor)
    user_query = _first_text(
        descriptor.get("user_query"),
        wf.params.get("user_query") if isinstance(wf.params, dict) else None,
        wf.params.get("query") if isinstance(wf.params, dict) else None,
    )
    objective = _first_text(
        descriptor.get("objective"),
        descriptor.get("design_objective"),
        wf.params.get("objective") if isinstance(wf.params, dict) else None,
        wf.params.get("design_objective") if isinstance(wf.params, dict) else None,
    )
    out: dict[str, str] = {}
    if user_query:
        out["user_query"] = user_query
    if objective:
        out["objective"] = objective
    return out


def _workflow_descriptor_mapping(descriptor_yaml: str | None) -> dict[str, Any]:
    if not descriptor_yaml:
        return {}
    try:
        raw = yaml.safe_load(descriptor_yaml) or {}
    except yaml.YAMLError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _candidate_score_map(record: dict[str, Any]) -> dict[str, Any]:
    scores = record.get("scores")
    if isinstance(scores, dict):
        out = deepcopy(scores)
        for namespace, block in list(out.items()):
            if not isinstance(block, dict):
                continue
            normalized = _score_map(block)
            if normalized:
                out[namespace] = {**block, **normalized}
        for key, value in _score_map(record).items():
            if key not in out:
                out[key] = value
        return out
    return _score_map(record)


def _candidate_artifact_metadata(record: dict[str, Any]) -> dict[str, Any]:
    artifacts = record.get("candidate_artifacts")
    if isinstance(artifacts, dict):
        return deepcopy(artifacts)
    artifacts = record.get("artifacts")
    return deepcopy(artifacts) if isinstance(artifacts, dict) else {}


def _normalize_complex_candidate_row(
    tool: str,
    row: dict[str, Any],
    node_params: dict[str, Any] | None,
) -> dict[str, Any]:
    if tool != "protenix_v2" or not isinstance(node_params, dict):
        return row
    input_json = node_params.get("input_json")
    if not isinstance(input_json, str) or not input_json:
        return row
    try:
        from vibeproteinagent.adapters.protenix_v2.adapter import (
            _input_json_complex_sequences_for_sample,
            _record_sample_name_from_native_id,
        )
        from vibeproteinagent.candidates.chain_roles import complex_chain_metadata
    except Exception:
        return row
    sample_name = _record_sample_name_from_native_id(
        str(row.get("candidate_id")) if row.get("candidate_id") else None
    )
    complex_sequences = _input_json_complex_sequences_for_sample(input_json, sample_name)
    if complex_sequences is None:
        return row
    target_sequence, binder_sequence = complex_sequences
    normalized = dict(row)
    normalized["sequence"] = binder_sequence
    artifacts = dict(normalized.get("artifacts") or {})
    protenix = dict(artifacts.get("protenix_v2") or {})
    protenix.setdefault("complex_input_mode", "input_json_target_sequence_binder_sequence")
    protenix.setdefault(
        "complex_chains",
        complex_chain_metadata(
            target_sequence=target_sequence,
            binder_sequence=binder_sequence,
            structure_path=_first_str(normalized, *_STRUCTURE_PATH_KEYS),
        ),
    )
    artifacts["protenix_v2"] = protenix
    normalized["artifacts"] = artifacts
    return normalized


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
    candidates: list[ReportCandidate] = []
    params_by_node = node_params or {}
    exclude_reference_candidates = _has_design_candidate_context(nodes)
    for node in nodes:
        output = node.output if isinstance(node.output, dict) else {}
        candidates.extend(
            _candidates_from_output(
                node, output, artifacts,
                node_params=params_by_node.get(node.node_id),
                exclude_reference_candidates=exclude_reference_candidates,
            )
        )
    _attach_qc_and_analysis(candidates, nodes, merge_global_metrics=False)
    if ranked:
        return _dedupe_candidates([*ranked, *candidates])
    return _dedupe_candidates(candidates)


def _has_design_candidate_context(nodes: list[WorkflowNodeSummary]) -> bool:
    for node in nodes:
        tool = (node.tool or "").lower()
        if not tool or tool in _NON_BOARD_CANDIDATE_TOOLS:
            continue
        return True
    return False


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
                    "candidate_artifacts": _candidate_artifact_metadata(row),
                },
                artifacts=artifacts,
            )
        )
    return out


def _enrich_final_candidates_from_current_candidates(
    final_candidates: list[ReportCandidate],
    current_candidates: list[ReportCandidate],
) -> list[ReportCandidate]:
    """Backfill final picks from candidates already collected for this report."""
    if not final_candidates:
        return final_candidates
    by_id = {candidate.candidate_id: candidate for candidate in current_candidates}
    enriched: list[ReportCandidate] = []
    for candidate in final_candidates:
        rich = by_id.get(candidate.candidate_id)
        if rich is None:
            enriched.append(candidate)
            continue
        enriched.append(
            replace(
                candidate,
                scores={**(rich.scores or {}), **(candidate.scores or {})},
                sequence=candidate.sequence or rich.sequence,
                structure_path=candidate.structure_path or rich.structure_path,
                qc_flags=candidate.qc_flags or rich.qc_flags,
                liabilities=candidate.liabilities or rich.liabilities,
                artifacts=candidate.artifacts or rich.artifacts,
            )
        )
    return enriched


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
                        "candidate_artifacts": _candidate_artifact_metadata(row),
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
    exclude_reference_candidates: bool = False,
) -> list[ReportCandidate]:
    tool = (node.tool or "").lower()
    if tool == "submit_candidate":
        return []
    if exclude_reference_candidates and tool in _REFERENCE_CANDIDATE_TOOLS:
        return []

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
        row = _normalize_complex_candidate_row(tool, row, node_params)
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
                source={
                    "node_id": node.node_id,
                    "tool": node.tool,
                    "kind": "output",
                    "candidate_artifacts": _candidate_artifact_metadata(row),
                },
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


def _finalization_contract_warnings(
    nodes: list[WorkflowNodeSummary],
    candidates: list[ReportCandidate],
    final_candidates: list[ReportCandidate],
    final_submission: dict[str, Any] | None,
) -> list[ReportWarning]:
    if not final_submission or final_submission.get("outcome") != "accepted":
        return []
    tools = {(node.tool or "").lower() for node in nodes}
    evaluator_tools = tools & _FINALIZATION_EVALUATOR_TOOLS
    if not evaluator_tools:
        return []

    source = final_submission.get("node_id") or final_submission.get("job_id")
    warnings: list[ReportWarning] = []
    if "rank_designs" not in tools and (tools & _FINALIZATION_GENERATOR_TOOLS):
        warnings.append(
            ReportWarning(
                level="warning",
                message=(
                    "The agent submitted candidates before running rank_designs to "
                    "merge validation evidence."
                ),
                source=source,
            )
        )

    final_ids = {str(c.candidate_id) for c in final_candidates if c.candidate_id}
    final_score_namespaces = {
        str(namespace)
        for candidate in final_candidates
        for namespace in candidate.scores
    }
    final_has_evaluator_scores = bool(final_score_namespaces & evaluator_tools)
    evaluator_candidate_ids = _candidate_ids_from_tools(nodes, evaluator_tools)
    evaluator_unmatched = bool(
        final_ids
        and evaluator_candidate_ids
        and evaluator_candidate_ids.isdisjoint(final_ids)
    )
    if not final_has_evaluator_scores or evaluator_unmatched:
        warnings.append(
            ReportWarning(
                level="warning",
                message=(
                    "The agent submitted candidates before merging validation "
                    "evidence. Some Protenix/ipSAE/QC results may be shown as "
                    "unmatched evidence."
                ),
                source=source,
            )
        )

    return warnings


def _candidate_ids_from_tools(
    nodes: list[WorkflowNodeSummary],
    tools: set[str] | frozenset[str],
) -> set[str]:
    out: set[str] = set()
    for node in nodes:
        if (node.tool or "").lower() not in tools:
            continue
        output = node.output if isinstance(node.output, dict) else {}
        for row in output.get("candidates") or []:
            if isinstance(row, dict) and row.get("candidate_id"):
                out.add(str(row["candidate_id"]))
    return out




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
    user_query: str | None = None,
    objective: str | None = None,
) -> dict[str, Any]:
    summary = {
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
    if user_query:
        summary["user_query"] = user_query
    if objective:
        summary["objective"] = objective
    return summary


def _raw_trajectory_summary(
    *,
    source_job_id: str,
    nodes: list[WorkflowNodeSummary],
    artifacts: list[ReportArtifact],
) -> dict[str, Any]:
    context = _trajectory_objective_context(nodes)
    summary = {
        "source_job_id": source_job_id,
        "scope": "direct_workspace_until_submit_candidate",
        "nodes": [
            {
                "node_id": node.node_id,
                "tool": node.tool,
                "state": node.state,
                "job_id": node.job_id,
                "upstream": node.upstream,
                "output": _compact(node.output),
                "artifacts": [_artifact_summary(a) for a in node.artifacts],
                "error_code": node.error_code,
                "error_message": node.error_message,
            }
            for node in nodes
        ],
        "artifacts": [_artifact_summary(a) for a in artifacts],
    }
    summary.update(context)
    return summary


def _raw_trajectory_report_summary(
    *,
    trajectory: TrajectoryRecord,
    nodes: list[WorkflowNodeSummary],
    artifacts: list[ReportArtifact],
    top_level_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    context = _trajectory_objective_context(nodes)
    summary = {
        "trajectory_id": trajectory.trajectory_id,
        "scope": "trajectory",
        "workspace": trajectory.workspace,
        "user_query": trajectory.user_query,
        "objective": trajectory.objective,
        "top_level_events": top_level_events or [],
        "nodes": [
            {
                "node_id": node.node_id,
                "tool": node.tool,
                "state": node.state,
                "job_id": node.job_id,
                "workflow_id": node.workflow_id,
                "upstream": node.upstream,
                "output": _compact(node.output),
                "artifacts": [_artifact_summary(a) for a in node.artifacts],
                "error_code": node.error_code,
                "error_message": node.error_message,
            }
            for node in nodes
        ],
        "artifacts": [_artifact_summary(a) for a in artifacts],
    }
    if "user_query" not in context:
        context["user_query"] = trajectory.user_query
    if trajectory.objective and "objective" not in context:
        context["objective"] = trajectory.objective
    summary.update(context)
    return summary


def _min_not_none(values: list[int | None]) -> int | None:
    present = [int(v) for v in values if v is not None]
    return min(present) if present else None


def _max_not_none(values: list[int | None]) -> int | None:
    present = [int(v) for v in values if v is not None]
    return max(present) if present else None


def _trajectory_objective_context(nodes: list[WorkflowNodeSummary]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in nodes:
        params = node.params if isinstance(node.params, dict) else {}
        output = node.output if isinstance(node.output, dict) else {}
        query = _first_text(
            params.get("user_query"),
            params.get("natural_language_request"),
            params.get("request"),
            params.get("query") if _looks_like_user_query(params.get("query")) else None,
            output.get("user_query"),
        )
        if query and "user_query" not in out:
            out["user_query"] = query
        objective = _first_text(
            params.get("objective"),
            params.get("design_objective"),
            output.get("objective"),
            output.get("design_objective"),
        )
        if objective and "objective" not in out:
            out["objective"] = objective
        for key in ("thresholds", "acceptance_criteria", "success_criteria"):
            value = params.get(key)
            if value not in (None, "", {}, []) and key not in out:
                out[key] = value
        if "user_query" in out and "objective" in out:
            break
    return out


def _looks_like_user_query(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith(("/", "{", "[")):
        return False
    return bool(" " in stripped or len(stripped) > 32)


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
    for key, nested in row.items():
        if key in _NESTED_SCORE_KEYS or not isinstance(nested, dict):
            continue
        nested_keys = set(nested)
        if nested_keys & (set(_SCORE_KEYS) | set(_NESTED_SCORE_KEYS)):
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
    if "iptm" not in out and "mean_iptm" in out:
        out["iptm"] = out["mean_iptm"]
    if "ipsae" not in out:
        for alias in ("ipsae_score", "ipsae_max", "interface_pae"):
            if alias in out:
                out["ipsae"] = out[alias]
                break
    if "ptm" not in out and "tm_score" in out:
        out["ptm"] = out["tm_score"]
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
    structure_path = (
        record.get("structure_path")
        if isinstance(record.get("structure_path"), str)
        else representative.structure_path
    )
    if not structure_path:
        for position in positions:
            if 0 <= position < len(candidates) and candidates[position].structure_path:
                structure_path = candidates[position].structure_path
                break
    if not artifacts:
        seen_artifacts: set[str] = set()
        for position in positions:
            if position < 0 or position >= len(candidates):
                continue
            for artifact in candidates[position].artifacts:
                if artifact.path in seen_artifacts:
                    continue
                seen_artifacts.add(artifact.path)
                artifacts.append(artifact)
    return replace(
        representative,
        structure_path=structure_path,
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


__all__ = ["collect_report_model"]
