from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

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
    candidates = _normalize_candidates(
        [node], artifacts,
        node_params={node.node_id: job.params if isinstance(job.params, dict) else {}},
    )
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
        artifacts=artifacts,
        msa_blocks=msa_blocks,
        warnings=warnings,
        reproducibility=_reproducibility(
            source_id=job.job_id,
            nodes=[node],
            artifacts=artifacts,
            workspace=workspace.name,
        ),
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

    candidates = _normalize_candidates(
        report_nodes, all_artifacts, node_params=params_by_node,
    )
    if not candidates:
        candidates = _structure_candidates(
            all_artifacts, source={"kind": "workflow", "workflow_id": wf.workflow_id}
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
        artifacts=artifacts,
        error_code=job.error_code,
        error_message=job.error_message,
        error_details=job.error_details,
    )


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
                    scores=_score_map(row),
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
                scores=_score_map(row),
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
        merged_scores: dict[str, float | int | str | None] = {}
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


def _score_map(row: dict[str, Any]) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {}
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
    seen: set[tuple[str, str | None]] = set()
    out: list[ReportCandidate] = []
    for candidate in candidates:
        key = (candidate.candidate_id, candidate.structure_path)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


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
