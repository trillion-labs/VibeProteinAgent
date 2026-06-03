from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.cli.output import (
    fail_from_exception,
    print_envelope_err,
    print_envelope_ok,
)
from vibeproteinagent.cli.render import (
    render_workflow_list,
    render_workflow_status,
)
from vibeproteinagent.core.errors import ErrorCode, NotFoundError
from vibeproteinagent.core.types import load_params_file, parse_kv_pairs
from vibeproteinagent.workers.supervisor_client import SupervisorClient
from vibeproteinagent.workflows import (
    WorkflowExecutor,
    WorkflowStore,
    load_workflow,
)
from vibeproteinagent.workspace import Workspace

workflow_app = typer.Typer(no_args_is_help=True, help="Workflow management.")


_BUILTIN_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "workflows" / "examples"
_USER_WORKFLOWS_DIR = Path.home() / ".vibeproteinagent" / "workflows"


def _ensure_workspace(name: str | None) -> Workspace:
    ws = Workspace.resolve(name)
    if not ws.db_path.exists():
        ws = Workspace.create(ws.name)
    return ws


def _registry() -> ToolRegistry:
    r = ToolRegistry()
    r.discover()
    return r


def _discover_descriptors() -> dict[str, Path]:
    """Map workflow-name → descriptor path. Searches built-ins + user dir."""
    out: dict[str, Path] = {}
    roots = (_BUILTIN_EXAMPLES_DIR, _USER_WORKFLOWS_DIR)
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob("*.yaml")):
            try:
                wf = load_workflow(p)
            except Exception:
                continue
            out[wf.name] = p
        for p in sorted(root.glob("*.yml")):
            try:
                wf = load_workflow(p)
            except Exception:
                continue
            out[wf.name] = p
    return out


def _resolve_descriptor(name_or_path: str) -> Path:
    """Treat ``name_or_path`` as a path first, then look up by descriptor name."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    found = _discover_descriptors()
    if name_or_path in found:
        return found[name_or_path]
    raise NotFoundError(
        f"workflow {name_or_path!r} not found (looked up by path then by descriptor name)",
        details={
            "name_or_path": name_or_path,
            "searched": [
                str(_BUILTIN_EXAMPLES_DIR),
                str(_USER_WORKFLOWS_DIR),
            ],
        },
    )


@workflow_app.command("list")
def workflow_list(
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """List workflow descriptors visible to vpa."""
    try:
        descs = _discover_descriptors()
        rows = []
        for name, p in sorted(descs.items()):
            try:
                wf = load_workflow(p)
                rows.append({
                    "name": wf.name,
                    "version": wf.version,
                    "description": wf.description,
                    "path": str(p),
                    "nodes": len(wf.nodes),
                })
            except Exception as exc:
                rows.append({"name": name, "path": str(p), "error": str(exc)})
        print_envelope_ok(
            {"workflows": rows}, json_mode=json_, render=render_workflow_list,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@workflow_app.command("inspect")
def workflow_inspect(
    name: str = typer.Argument(..., help="Workflow name or path."),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show a workflow descriptor."""
    try:
        p = _resolve_descriptor(name)
        wf = load_workflow(p)
        body = {
            "path": str(p),
            "name": wf.name,
            "version": wf.version,
            "description": wf.description,
            "params": {k: v.model_dump() for k, v in wf.params.items()},
            "nodes": [n.model_dump() for n in wf.nodes],
        }
        print_envelope_ok(body, json_mode=json_)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@workflow_app.command("run")
def workflow_run(
    name: str = typer.Argument(..., help="Workflow name or path."),
    param: list[str] = typer.Option(  # noqa: B008
        None, "--param", help="KEY=VALUE workflow param."
    ),
    params_file: str | None = typer.Option(None, "--params-file"),
    workspace: str | None = typer.Option(None, "--workspace"),
    follow: bool = typer.Option(False, "--follow"),
    resume: str | None = typer.Option(None, "--resume", help="Existing workflow_id."),
    max_wait_s: int = typer.Option(3600, "--max-wait-s"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Submit and run a workflow synchronously."""
    try:
        ws = _ensure_workspace(workspace)
        registry = _registry()
        supervisor = SupervisorClient()
        if not supervisor.is_running():
            supervisor = None  # type: ignore[assignment]

        if resume is not None:
            rec = WorkflowStore(ws).get(resume)
            if supervisor is None:
                print_envelope_err(
                    ErrorCode.DAEMON_UNAVAILABLE,
                    "supervisor not running; workflow resume was not executed",
                    details={"workflow_id": resume, "state": rec.state},
                    json_mode=json_,
                )
                return
            executor = WorkflowExecutor(ws, registry, supervisor)
            final = executor.run_sync(resume, max_wait_s=max_wait_s)
            _print_workflow(executor, final, json_=json_, follow=follow)
            return

        p = _resolve_descriptor(name)
        wf = load_workflow(p)

        merged: dict[str, Any] = {}
        if params_file:
            merged.update(load_params_file(params_file))
        merged.update(parse_kv_pairs(param))

        executor = WorkflowExecutor(ws, registry, supervisor)
        wf_id = executor.submit(wf, merged)
        if supervisor is None:
            # Without a supervisor we can record the workflow but cannot run it.
            store = WorkflowStore(ws)
            rec = store.get(wf_id)
            print_envelope_err(
                ErrorCode.DAEMON_UNAVAILABLE,
                "supervisor not running; nodes were recorded but not executed",
                details={"workflow_id": wf_id, "state": rec.state},
                json_mode=json_,
            )
            return

        final = executor.run_sync(wf_id, max_wait_s=max_wait_s)
        _print_workflow(executor, final, json_=json_, follow=follow)
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


@workflow_app.command("status")
def workflow_status(
    workflow_id: str = typer.Argument(..., help="Workflow id."),
    workspace: str | None = typer.Option(None, "--workspace"),
    json_: bool = typer.Option(False, "--json"),
) -> None:
    """Show a workflow record and its node states."""
    try:
        ws = _ensure_workspace(workspace)
        store = WorkflowStore(ws)
        wf = store.get(workflow_id)
        nodes = store.list_nodes(workflow_id)
        print_envelope_ok(
            {
                "workflow_id": wf.workflow_id,
                "name": wf.name,
                "version": wf.version,
                "state": wf.state,
                "submitted_at": wf.submitted_at,
                "finished_at": wf.finished_at,
                "nodes": [_node_status_dict(n) for n in nodes],
            },
            json_mode=json_,
            render=render_workflow_status,
        )
    except Exception as exc:
        fail_from_exception(exc, json_mode=json_)


def _print_workflow(
    executor: WorkflowExecutor,
    wf,  # WorkflowRecord
    *,
    json_: bool,
    follow: bool,
) -> None:
    """Final envelope for ``run`` — includes node states + outputs."""
    nodes = executor.store.list_nodes(wf.workflow_id)
    rows = []
    for n in nodes:
        out = None
        if n.job_id:
            try:
                out = executor.jobs.get(n.job_id).output
            except Exception:
                out = None
        rows.append({
            **_node_status_dict(n),
            "output": out,
        })
    print_envelope_ok(
        {
            "workflow_id": wf.workflow_id,
            "name": wf.name,
            "version": wf.version,
            "state": wf.state,
            "submitted_at": wf.submitted_at,
            "finished_at": wf.finished_at,
            "nodes": rows,
        },
        json_mode=json_,
        render=render_workflow_status,
    )


def _node_status_dict(n: Any) -> dict[str, Any]:
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


__all__ = ["workflow_app"]
