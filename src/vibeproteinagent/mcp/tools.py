from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vibeproteinagent.adapters import (
    ToolRegistry,
    serialize_descriptor,
)
from vibeproteinagent.adapters.descriptor import (
    resolve_input_schema_dict,
    resolve_output_schema_dict,
)
from vibeproteinagent.adapters.schema_affordance import conditional_required_summary
from vibeproteinagent.adapters.availability import attach_availability, descriptor_availability
from vibeproteinagent.adapters.visibility import is_public_tool
from vibeproteinagent.core.errors import (
    DaemonUnavailableError,
    DependencyError,
    InvalidParamsError,
    NotFoundError,
    TimeoutError_,
    VPAError,
)
from vibeproteinagent.core.types import resolve_salt
from vibeproteinagent.workers.supervisor_client import SupervisorClient
from vibeproteinagent.workspace import (
    TERMINAL_STATES,
    JobState,
    JobStore,
    LogFileManager,
    TrajectoryStore,
    Workspace,
)
from vibeproteinagent.workspace.store import ArtifactStore, now_us

from .util import (
    job_to_dict,
    resolve_input_schema,
    validate_against,
    vpa_error_to_dict,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


@dataclass
class SessionCtx:
    """Per-MCP-session state.

    One MCP session pins to a single workspace; the orchestration tools resolve
    JobStore / ArtifactStore / SupervisorClient from it lazily so any test can
    swap in a fresh workspace at construction time.
    """

    workspace: Workspace
    registry: ToolRegistry

    def store(self) -> JobStore:
        return JobStore(self.workspace)

    def artifacts(self) -> ArtifactStore:
        return ArtifactStore(self.workspace)

    @staticmethod
    def supervisor() -> SupervisorClient:
        return SupervisorClient()


# ---------------------------------------------------------------- orchestration tool fns
#
# Each of these is registered as a FastMCP tool. They return plain dicts/lists
# the SDK serialises into the MCP response. VPAError is the one exception we catch and
# re-shape into a domain-error dict — the agent can read `ok=false, code=...` instead
# of an opaque protocol error.


def _descriptor_payload(
    descriptor: Any,
    adapter_cls: type[Any] | None = None,
    *,
    schema_only: bool = False,
) -> dict[str, Any]:
    payload = serialize_descriptor(descriptor)
    if schema_only:
        input_schema = resolve_input_schema_dict(descriptor)
        payload["input_schema_resolved"] = input_schema
        payload["conditional_required"] = conditional_required_summary(input_schema)
        payload["output_schema_resolved"] = resolve_output_schema_dict(descriptor)
        return payload
    return attach_availability(payload, descriptor, adapter_cls)


def _raise_if_unavailable(
    descriptor: Any,
    adapter_cls: type[Any],
    params: dict[str, Any] | None = None,
) -> None:
    availability = descriptor_availability(descriptor, adapter_cls)
    try:
        from vibeproteinagent.install.installer import install_readiness

        install_status = install_readiness(
            (str(descriptor.name),),
            runtime_probe=False,
        )
    except Exception as exc:
        install_status = {
            "install_ready": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    if availability.get("available") is True and install_status.get("install_ready") is True:
        if str(descriptor.name) == "rfdiffusion" and params is not None:
            mode_status = _rfdiffusion_mode_readiness(params)
            if mode_status.get("ready") is not True:
                raise DependencyError(
                    "RFdiffusion mode checkpoint is not ready",
                    details={
                        "tool": descriptor.name,
                        "version": descriptor.version,
                        "mode_readiness": mode_status,
                    },
                )
        return
    raise DependencyError(
        f"tool '{descriptor.name}' is not ready to run",
        details={
            "tool": descriptor.name,
            "version": descriptor.version,
            "availability": availability,
            "install_readiness": install_status,
        },
    )


def list_tools(
    ctx: SessionCtx,
    category: str | None = None,
    *,
    include_internal: bool = False,
    schema_only: bool = False,
) -> list[dict[str, Any]]:
    """Return public tool descriptors, optionally filtered by category.

    Internal runtime/glue tools remain executable through low-level submit_job
    for authored workflows and tests, but are hidden from first-class catalogs by
    default so LLM agents do not plan with them.
    """
    try:
        ctx.registry.discover()
        return [
            _descriptor_payload(d, cls, schema_only=schema_only)
            for d, cls in ctx.registry.entries(category=category)
            if include_internal or is_public_tool(d)
        ]
    except VPAError as exc:
        return [vpa_error_to_dict(exc)]


def tool_info(
    ctx: SessionCtx,
    name: str,
    version: str = "*",
    *,
    schema_only: bool = False,
) -> dict[str, Any]:
    """Return one tool descriptor by name + PEP-440 version specifier."""
    try:
        ctx.registry.discover()
        descriptor, cls = ctx.registry.get(name, version)
        return _descriptor_payload(descriptor, cls, schema_only=schema_only)
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def submit_job(
    ctx: SessionCtx,
    tool: str,
    params: dict[str, Any],
    version: str = "*",
    call_id: str | None = None,
    priority: str = "normal",
    workflow_id: str | None = None,
    trajectory_id: str | None = None,
    force_new: bool = False,
    max_retries: int = 0,
) -> dict[str, Any]:
    """Submit a job. Validates params against the tool's input schema.

    ``force_new`` is the MCP-side parity for the CLI's ``--force-new`` flag.
    Job IDs are deterministic — re-submitting an identical request returns
    the existing job. Pass ``force_new=True`` to deliberately launch a
    fresh run (e.g. seed-reproducibility tests); implementation adds a
    random salt so the deterministic ``job_id`` no longer collides.
    """
    try:
        runtime_info = ctx.supervisor().require_workspace(ctx.workspace.name)
        if not isinstance(force_new, bool):
            raise InvalidParamsError(
                "force_new must be a boolean",
                details={"field": "force_new", "got_type": type(force_new).__name__},
            )
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise InvalidParamsError(
                "max_retries must be a non-negative integer",
                details={
                    "field": "max_retries",
                    "got_type": type(max_retries).__name__,
                },
            )
        ctx.registry.discover()
        if trajectory_id is not None:
            TrajectoryStore(ctx.workspace).get(trajectory_id)
        descriptor, cls = ctx.registry.get(tool, version)
        schema = resolve_input_schema(descriptor)
        validate_against(schema, params, tool_name=descriptor.name)
        _raise_if_unavailable(descriptor, cls, params)
        store = ctx.store()
        pre_submit_us = now_us()
        rec = store.submit(
            descriptor.name,
            descriptor.version,
            params,
            call_id=call_id,
            priority=priority,
            workspace_name=ctx.workspace.name,
            trajectory_id=trajectory_id,
            workflow_id=workflow_id,
            max_retries=max_retries,
            salt=resolve_salt(force_new),
            input_schema=schema,
        )
        idempotent_hit = (
            int(rec.submitted_at) < pre_submit_us
            and (rec.queued_at is None or int(rec.queued_at) < pre_submit_us)
        )
        if not idempotent_hit and rec.state is JobState.SUBMITTED:
            rec = store.transition(rec.job_id, JobState.QUEUED)
        return {
            **job_to_dict(rec),
            "idempotent_hit": idempotent_hit,
            **_execution_payload(ctx.workspace.name, runtime_info),
        }
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def _execution_payload(workspace_name: str, runtime_info: dict[str, Any]) -> dict[str, Any]:
    daemon_running = bool(runtime_info.get("running"))
    supervisor_workspace = runtime_info.get("workspace")
    return {
        "daemon_running": daemon_running,
        "requested_workspace": workspace_name,
        "supervisor_workspace": supervisor_workspace,
        "workspace_match": (
            daemon_running
            and isinstance(supervisor_workspace, str)
            and supervisor_workspace == workspace_name
        ),
        "execution_ready": (
            daemon_running
            and (
                supervisor_workspace is None
                or str(supervisor_workspace) == workspace_name
            )
        ),
    }


def _rfdiffusion_mode_readiness(params: dict[str, Any]) -> dict[str, Any]:
    try:
        from vibeproteinagent.adapters.rfdiffusion import mode_readiness

        return dict(mode_readiness(params))
    except VPAError:
        raise
    except Exception as exc:
        raise InvalidParamsError(
            "could not validate RFdiffusion mode readiness",
            details={"error": str(exc), "exception_type": type(exc).__name__},
        ) from exc


def get_status(ctx: SessionCtx, job_id: str) -> dict[str, Any]:
    """Return one job record."""
    try:
        rec = ctx.store().get(job_id)
        return job_to_dict(rec)
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def wait_job(ctx: SessionCtx, job_id: str, timeout_ms: int = 30_000) -> dict[str, Any]:
    """Poll until terminal or timeout. Returns the final job record dict."""
    store = ctx.store()
    deadline = time.time() + (timeout_ms / 1000.0)
    try:
        while True:
            rec = store.get(job_id)
            if rec.state in TERMINAL_STATES:
                return job_to_dict(rec)
            if time.time() >= deadline:
                return {
                    **job_to_dict(rec),
                    "timed_out": True,
                }
            time.sleep(0.25)
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def get_results(
    ctx: SessionCtx,
    job_id: str,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """Return a completed job's output dict (or a projection of it)."""
    try:
        rec = ctx.store().get(job_id)
        if rec.output is None:
            raise NotFoundError(
                f"job {rec.job_id} has no output (state={rec.state.value})",
                details={"job_id": rec.job_id, "state": rec.state.value},
            )
        out = rec.output
        if fields:
            out = {k: out.get(k) for k in fields}
        return {"job_id": rec.job_id, "output": out}
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def get_logs(ctx: SessionCtx, job_id: str, tail: int | None = None) -> str | dict[str, Any]:
    """Return stdout log text for a job."""
    try:
        rec = ctx.store().get(job_id)
        log_mgr = LogFileManager()
        job_dir = ctx.workspace.job_dir(rec.job_id)
        return log_mgr.read(job_dir, "stdout", tail=tail)
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def cancel_job(ctx: SessionCtx, job_id: str) -> dict[str, Any]:
    """Cancel a job, stopping the owning VPA worker when it is already running."""
    try:
        rec = ctx.store().get(job_id)
        client = ctx.supervisor()
        if client.is_running():
            client.require_workspace(ctx.workspace.name)
            return client.cancel(rec.job_id)
        rec = ctx.store().request_cancel(rec.job_id)
        return {"job_id": rec.job_id, "state": rec.state.value}
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def list_artifacts(ctx: SessionCtx, job_id: str) -> list[dict[str, Any]]:
    """Return artifact records for a job."""
    try:
        rec = ctx.store().get(job_id)
        records = ctx.artifacts().for_job(rec.job_id)
        return [
            {
                "field": field,
                "id": a.artifact_id,
                "kind": a.kind,
                "mime": a.mime,
                "size_bytes": a.size_bytes,
                "sha256": a.sha256,
                "path": a.pool_path,
            }
            for field, a in records.items()
        ]
    except VPAError as exc:
        return [vpa_error_to_dict(exc)]


def daemon_status(ctx: SessionCtx) -> dict[str, Any]:
    """Report whether the supervisor is running, and list its workers."""
    client = ctx.supervisor()
    if not client.is_running():
        return {"running": False}
    try:
        info = client.worker_info()
    except VPAError as exc:
        return {"running": True, "error": vpa_error_to_dict(exc)}
    return {
        "running": True,
        "pid": client.pid(),
        "socket": str(client.socket_path),
        "workers": info.get("workers", []),
    }


def _is_loaded(worker: dict[str, Any], tool: str, version: str) -> bool:
    """True iff ``worker.loaded_adapters`` contains the requested (tool, version).

    Per Phase 3.5 the worker emits each loaded adapter as ``{"name", "version",
    ...}`` (a dict). The Phase 4.3 MCP code treated entries as ``(name, version)``
    tuples; that path always returned False against today's worker info shape
    (Codex review Re13). The version comparison accepts ``"*"`` as a wildcard.
    """
    loaded = worker.get("loaded_adapters") or []
    for entry in loaded:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") != tool:
            continue
        if version in ("", "*", None):
            return True
        if entry.get("version") == version:
            return True
    return False


def warm_tool(
    ctx: SessionCtx,
    tool: str,
    version: str = "*",
    poll_timeout_s: float = 180.0,
) -> dict[str, Any]:
    """Pre-load a tool into the supervisor's warm pool.

    Phase 4.2 hand-off: the supervisor's `_op_warm` has a hard 60s timeout but heavy
    adapters (Protenix) need ~110s to cold-load. The CLI surfaces that timeout as a
    failure; here we tolerate it and poll `worker_info` for up to `poll_timeout_s`
    waiting for the (tool, version) pair to appear in `loaded_adapters` on any
    worker, so an agent doesn't have to retry by hand.

    ``success`` is aggregated across every worker the supervisor reached: ``True``
    only when at least one worker tried AND every per-worker response reports
    ``success=True`` (Codex review Re13). A mixed-success response (one OK,
    one OOM) is reported as ``ok=False``.
    """
    client = ctx.supervisor()
    if not client.is_running():
        return vpa_error_to_dict(
            DaemonUnavailableError(
                "supervisor is not running",
                details={"socket": str(client.socket_path)},
            )
        )

    warm_err: dict[str, Any] | None = None
    warmed_immediate: list[dict[str, Any]] = []
    try:
        result = client.warm(tool, version_spec=version)
        warmed_immediate = list(result.get("warmed") or [])
        if warmed_immediate:
            success = all(
                bool(w.get("success", False)) for w in warmed_immediate
            )
            return {
                "ok": success,
                "tool": tool,
                "version": version,
                "warmed": warmed_immediate,
                "via": "synchronous",
            }
    except VPAError as exc:
        warm_err = vpa_error_to_dict(exc)

    deadline = time.time() + poll_timeout_s
    while time.time() < deadline:
        try:
            info = client.worker_info()
        except VPAError:
            info = {"workers": []}
        for worker in info.get("workers", []) or []:
            if _is_loaded(worker, tool, version):
                return {
                    "ok": True,
                    "tool": tool,
                    "version": version,
                    "worker_id": worker.get("gpu_id") or worker.get("worker_id"),
                    "via": "polled",
                    "first_response_error": warm_err,
                }
        time.sleep(2.0)

    return {
        "ok": False,
        "code": "E_TIMEOUT",
        "tool": tool,
        "version": version,
        "message": (
            f"tool {tool!r} did not appear in any worker's loaded_adapters "
            f"within {poll_timeout_s:.0f}s"
        ),
        "details": {"first_response_error": warm_err},
    }


def list_workspaces(_ctx: SessionCtx) -> list[str]:
    """List all workspaces under VPA_HOME."""
    return Workspace.list()


def current_workspace(ctx: SessionCtx) -> dict[str, Any]:
    """Return the session-pinned workspace."""
    return {"name": ctx.workspace.name, "root": str(ctx.workspace.root)}


def start_trajectory(
    ctx: SessionCtx,
    user_query: str,
    objective: str | None = None,
) -> dict[str, Any]:
    """Create a trajectory for one user query."""
    try:
        rec = TrajectoryStore(ctx.workspace).create(
            user_query=user_query,
            objective=objective,
        )
        return {"trajectory": rec.model_dump(), "trajectory_id": rec.trajectory_id}
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def get_trajectory_status(ctx: SessionCtx, trajectory_id: str) -> dict[str, Any]:
    """Return completion accounting for one trajectory."""
    try:
        return {"trajectory": TrajectoryStore(ctx.workspace).status(trajectory_id)}
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def wait_trajectory(
    ctx: SessionCtx,
    trajectory_id: str,
    timeout_ms: int = 30_000,
) -> dict[str, Any]:
    """Poll all direct jobs and workflows in a trajectory until terminal."""
    store = TrajectoryStore(ctx.workspace)
    deadline = time.time() + (timeout_ms / 1000.0)
    try:
        status = store.status(trajectory_id)
        timed_out = False
        while True:
            execution_all_terminal = bool(status["execution_all_terminal"])
            if execution_all_terminal:
                break
            if timeout_ms == 0 or time.time() >= deadline:
                timed_out = not execution_all_terminal
                break
            time.sleep(0.25)
            status = store.status(trajectory_id)
        return {
            "trajectory_id": trajectory_id,
            "jobs": status["jobs"],
            "workflows": status["workflows"],
            "execution_all_terminal": status["execution_all_terminal"],
            "trajectory_complete": status["complete"],
            "unfinished_job_ids": status["unfinished_job_ids"],
            "unfinished_workflow_ids": status["unfinished_workflow_ids"],
            "final_submit_job_id": status["final_submit_job_id"],
            "report_path": status["report_path"],
            "timed_out": timed_out,
        }
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def render_report(
    ctx: SessionCtx,
    source_id: str,
    out: str | None = None,
    format: str = "html",
    replace_report: bool = False,
) -> dict[str, Any]:
    """Render a read-only report for traj_ / wf_ / job_ sources."""
    try:
        from vibeproteinagent.reporting import (
            collect_report_model,
            resolve_report_output_path,
            write_report,
        )

        model = collect_report_model(ctx.workspace, source_id)
        default_name = (
            f"{source_id}_report_bundle"
            if format == "bundle"
            else f"{source_id}_report.html"
        )
        out_path = Path(out) if out else ctx.workspace.root / default_name
        canonical_report_path = resolve_report_output_path(out_path, report_format=format)
        trajectory_store = TrajectoryStore(ctx.workspace) if source_id.startswith("traj_") else None
        if trajectory_store is not None:
            trajectory_store.preflight_report(
                source_id,
                str(canonical_report_path),
                format=format,
                replace=replace_report,
            )
        report_path = write_report(model, out_path, report_format=format)
        trajectory_status = None
        if trajectory_store is not None:
            trajectory_status = trajectory_store.record_report(
                source_id,
                str(report_path),
                source_id=source_id,
                format=format,
                replace=replace_report,
            ).model_dump()
        data: dict[str, Any] = {
            "report_html": str(report_path),
            "format": format,
            "source": {"kind": model.summary.kind, "id": model.summary.id},
            "candidate_count": len(model.candidates),
            "replace_report": replace_report,
        }
        if trajectory_status is not None:
            data["trajectory"] = trajectory_status
        return data
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def list_workflows(_ctx: SessionCtx) -> list[dict[str, Any]]:
    """Return discoverable workflow descriptors with planner-usable structure."""
    from vibeproteinagent.cli.commands.workflow import (
        _discover_descriptors,
    )
    from vibeproteinagent.workflows import load_workflow

    out: list[dict[str, Any]] = []
    for name, p in sorted(_discover_descriptors().items()):
        try:
            wf = load_workflow(p)
            out.append({
                "name": wf.name,
                "version": wf.version,
                "description": wf.description,
                "path": str(p),
                "nodes": len(wf.nodes),
                "node_count": len(wf.nodes),
                "params": {k: v.model_dump() for k, v in wf.params.items()},
                "tool_sequence": [node.tool for node in wf.nodes],
                "node_specs": [
                    {
                        "id": node.id,
                        "tool": node.tool,
                        "version": node.version,
                        "foreach": node.foreach,
                        "max_parallel": node.max_parallel,
                        "when": node.when,
                        "inputs": node.inputs,
                    }
                    for node in wf.nodes
                ],
            })
        except VPAError as exc:
            out.append({"name": name, "path": str(p),
                        "error": vpa_error_to_dict(exc)})
    return out


def _workflow_node_to_dict(n: Any) -> dict[str, Any]:
    return {
        "node_id": n.node_id,
        "tool": n.tool_name,
        "tool_version": n.tool_version,
        "state": n.state,
        "job_id": n.job_id,
        "cache_key": n.cache_key,
        "error_code": n.error_code,
        "error_message": n.error_message,
        "error_details": n.error_details,
    }


def _workflow_node_to_dict_with_output(ctx: SessionCtx, n: Any) -> dict[str, Any]:
    row = _workflow_node_to_dict(n)
    row["output"] = n.output
    if n.job_id:
        try:
            row["output"] = ctx.store().get(n.job_id).output
        except Exception:
            row["output"] = n.output
    return row


def get_workflow_status(ctx: SessionCtx, workflow_id: str) -> dict[str, Any]:
    """Return one workflow record and its node states."""
    from vibeproteinagent.workflows import WorkflowStore

    try:
        store = WorkflowStore(ctx.workspace)
        wf = store.get(workflow_id)
        nodes = store.list_nodes(workflow_id)
        return {
            "ok": wf.state == "completed",
            "workflow_id": wf.workflow_id,
            "name": wf.name,
            "version": wf.version,
            "state": wf.state,
            "submitted_at": wf.submitted_at,
            "finished_at": wf.finished_at,
            "nodes": [_workflow_node_to_dict_with_output(ctx, n) for n in nodes],
        }
    except VPAError as exc:
        return vpa_error_to_dict(exc)


def run_workflow(
    ctx: SessionCtx,
    name: str,
    params: dict[str, Any],
    *,
    trajectory_id: str | None = None,
    max_wait_s: int = 3600,
) -> dict[str, Any]:
    """Submit and synchronously run a workflow by name or path."""
    from vibeproteinagent.cli.commands.workflow import (
        _resolve_descriptor,
    )
    from vibeproteinagent.workflows import (
        WorkflowExecutor,
        load_workflow,
    )

    try:
        p = _resolve_descriptor(name)
        wf = load_workflow(p)
        if trajectory_id is not None:
            TrajectoryStore(ctx.workspace).get(trajectory_id)
        supervisor = ctx.supervisor()
        runtime_info = supervisor.require_workspace(ctx.workspace.name)
        sup = supervisor if runtime_info.get("running") else None
        executor = WorkflowExecutor(
            ctx.workspace, ctx.registry, sup, trajectory_id=trajectory_id
        )
        wf_id = executor.submit(wf, params)
        if sup is None:
            rec = executor.store.get(wf_id)
            nodes = executor.store.list_nodes(wf_id)
            return {
                "ok": False,
                "code": "E_DAEMON_UNAVAILABLE",
                "workflow_id": wf_id,
                "trajectory_id": rec.trajectory_id,
                "state": rec.state,
                "name": rec.name,
                "message": "supervisor not running; nodes were recorded but not executed",
                "requested_workspace": ctx.workspace.name,
                "supervisor_workspace": runtime_info.get("workspace"),
                "daemon_running": runtime_info.get("running"),
                "nodes": [_workflow_node_to_dict_with_output(ctx, n) for n in nodes],
            }
        try:
            final = executor.run_sync(wf_id, max_wait_s=max_wait_s)
        except TimeoutError_ as exc:
            rec = executor.store.get(wf_id)
            nodes = executor.store.list_nodes(wf_id)
            return {
                "ok": False,
                "code": exc.code,
                "workflow_id": wf_id,
                "trajectory_id": rec.trajectory_id,
                "state": rec.state,
                "name": rec.name,
                "message": exc.message,
                "details": {
                    **(exc.details or {}),
                    "requested_workspace": ctx.workspace.name,
                    "supervisor_workspace": supervisor.runtime_info().get("workspace"),
                },
                "nodes": [_workflow_node_to_dict_with_output(ctx, n) for n in nodes],
            }
        nodes = executor.store.list_nodes(wf_id)
        return {
            "ok": final.state == "completed",
            "workflow_id": wf_id,
            "trajectory_id": final.trajectory_id,
            "state": final.state,
            "name": final.name,
            "nodes": [_workflow_node_to_dict_with_output(ctx, n) for n in nodes],
        }
    except VPAError as exc:
        return vpa_error_to_dict(exc)


# ---------------------------------------------------------------- registration


def register_orchestration_tools(app: FastMCP, ctx: SessionCtx) -> list[str]:
    """Wire every orchestration function as a FastMCP tool. Returns names registered."""

    @app.tool(
        name="list_tools",
        description=(
            "List public first-class tools (optionally filter by category). "
            "Set include_internal=True for developer-only runtime/glue adapters; "
            "set schema_only=True to skip availability probes and include resolved schemas."
        ),
    )
    def _list_tools(
        category: str | None = None,
        include_internal: bool = False,
        schema_only: bool = False,
    ) -> list[dict[str, Any]]:
        return list_tools(
            ctx,
            category,
            include_internal=include_internal,
            schema_only=schema_only,
        )

    @app.tool(
        name="tool_info",
        description=(
            "Show one tool's descriptor (name + PEP-440 version spec). "
            "Set schema_only=True to skip availability probes and include resolved schemas."
        ),
    )
    def _tool_info(
        name: str,
        version: str = "*",
        schema_only: bool = False,
    ) -> dict[str, Any]:
        return tool_info(ctx, name, version, schema_only=schema_only)

    @app.tool(
        name="submit_job",
        description=(
            "Submit a tool job (validates params against the tool's input schema). "
            "Job IDs are deterministic — an identical submission returns the existing "
            "job. Pass force_new=True for an explicit fresh run; max_retries "
            "controls retryable runtime failures."
        ),
    )
    def _submit_job(
        tool: str,
        params: dict[str, Any],
        version: str = "*",
        call_id: str | None = None,
        priority: str = "normal",
        workflow_id: str | None = None,
        trajectory_id: str | None = None,
        force_new: bool = False,
        max_retries: int = 0,
    ) -> dict[str, Any]:
        return submit_job(
            ctx, tool, params, version, call_id, priority, workflow_id,
            trajectory_id=trajectory_id,
            force_new=force_new, max_retries=max_retries,
        )

    @app.tool(name="get_status", description="Return one job's record by id or call_id.")
    def _get_status(job_id: str) -> dict[str, Any]:
        return get_status(ctx, job_id)

    @app.tool(
        name="wait_job",
        description="Poll until a job reaches a terminal state or timeout_ms elapses.",
    )
    def _wait_job(job_id: str, timeout_ms: int = 30_000) -> dict[str, Any]:
        return wait_job(ctx, job_id, timeout_ms)

    @app.tool(
        name="get_results",
        description="Return a completed job's output (or a projection of it).",
    )
    def _get_results(job_id: str, fields: list[str] | None = None) -> dict[str, Any]:
        return get_results(ctx, job_id, fields)

    @app.tool(name="get_logs", description="Return a job's stdout log text.")
    def _get_logs(job_id: str, tail: int | None = None) -> str | dict[str, Any]:
        return get_logs(ctx, job_id, tail)

    @app.tool(name="cancel_job", description="Set the durable cancel_requested flag on a job.")
    def _cancel_job(job_id: str) -> dict[str, Any]:
        return cancel_job(ctx, job_id)

    @app.tool(name="list_artifacts", description="List artifacts produced by a job.")
    def _list_artifacts(job_id: str) -> list[dict[str, Any]]:
        return list_artifacts(ctx, job_id)

    @app.tool(name="daemon_status", description="Report supervisor + worker status.")
    def _daemon_status() -> dict[str, Any]:
        return daemon_status(ctx)

    @app.tool(
        name="warm_tool",
        description=(
            "Pre-load a tool into the warm pool. Polls worker_info for up to 180s "
            "to tolerate the supervisor's 60s warm RPC timeout on slow cold-loads."
        ),
    )
    def _warm_tool(tool: str, version: str = "*") -> dict[str, Any]:
        return warm_tool(ctx, tool, version)

    @app.tool(name="list_workspaces", description="List all workspaces under VPA_HOME.")
    def _list_workspaces() -> list[str]:
        return list_workspaces(ctx)

    @app.tool(
        name="current_workspace",
        description="Return the workspace pinned to this MCP session.",
    )
    def _current_workspace() -> dict[str, Any]:
        return current_workspace(ctx)

    @app.tool(
        name="start_trajectory",
        description="Create a trajectory for one user query in this MCP workspace.",
    )
    def _start_trajectory(
        user_query: str,
        objective: str | None = None,
    ) -> dict[str, Any]:
        return start_trajectory(ctx, user_query, objective)

    @app.tool(
        name="get_trajectory_status",
        description="Return query-level completion accounting for a trajectory.",
    )
    def _get_trajectory_status(trajectory_id: str) -> dict[str, Any]:
        return get_trajectory_status(ctx, trajectory_id)

    @app.tool(
        name="wait_trajectory",
        description=(
            "Poll all direct jobs and workflows in a trajectory until terminal "
            "or timeout_ms elapses."
        ),
    )
    def _wait_trajectory(
        trajectory_id: str,
        timeout_ms: int = 30_000,
    ) -> dict[str, Any]:
        return wait_trajectory(ctx, trajectory_id, timeout_ms)

    @app.tool(
        name="render_report",
        description=(
            "Render a read-only HTML report for a traj_, wf_, or job_ source. "
            "For traj_ sources, records exactly one final report unless "
            "replace_report=True."
        ),
    )
    def _render_report(
        source_id: str,
        out: str | None = None,
        format: str = "html",
        replace_report: bool = False,
    ) -> dict[str, Any]:
        return render_report(ctx, source_id, out, format, replace_report)

    @app.tool(
        name="list_workflows",
        description="List workflow descriptors visible to vpa (built-ins + user dir).",
    )
    def _list_workflows() -> list[dict[str, Any]]:
        return list_workflows(ctx)

    @app.tool(
        name="get_workflow_status",
        description="Return one workflow record and its node states.",
    )
    def _get_workflow_status(workflow_id: str) -> dict[str, Any]:
        return get_workflow_status(ctx, workflow_id)

    @app.tool(
        name="run_workflow",
        description="Submit and synchronously run a workflow by name or descriptor path.",
    )
    def _run_workflow(
        name: str,
        params: dict[str, Any],
        trajectory_id: str | None = None,
        max_wait_s: int = 3600,
    ) -> dict[str, Any]:
        return run_workflow(
            ctx, name, params, trajectory_id=trajectory_id, max_wait_s=max_wait_s
        )

    return [
        "list_tools",
        "tool_info",
        "submit_job",
        "get_status",
        "wait_job",
        "get_results",
        "get_logs",
        "cancel_job",
        "list_artifacts",
        "daemon_status",
        "warm_tool",
        "list_workspaces",
        "current_workspace",
        "start_trajectory",
        "get_trajectory_status",
        "wait_trajectory",
        "render_report",
        "list_workflows",
        "get_workflow_status",
        "run_workflow",
    ]


__all__ = [
    "SessionCtx",
    "cancel_job",
    "current_workspace",
    "daemon_status",
    "get_logs",
    "get_results",
    "get_status",
    "get_trajectory_status",
    "get_workflow_status",
    "list_artifacts",
    "list_tools",
    "list_workflows",
    "list_workspaces",
    "register_orchestration_tools",
    "render_report",
    "run_workflow",
    "start_trajectory",
    "submit_job",
    "tool_info",
    "wait_trajectory",
    "wait_job",
    "warm_tool",
]
