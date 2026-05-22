from __future__ import annotations

from .cache import compute_cache_key, env_fingerprint
from .descriptor import Workflow, WorkflowNode, WorkflowParam
from .executor import WorkflowExecutor
from .loader import load_workflow, serialize_workflow
from .resolver import eval_when, resolve
from .store import WorkflowStore


def run_workflow_sync(
    workflow: Workflow,
    params: dict,
    workspace,  # type: ignore[no-untyped-def]
    registry=None,  # type: ignore[no-untyped-def]
    supervisor=None,  # type: ignore[no-untyped-def]
    *,
    max_wait_s: int = 3600,
):
    """Convenience: submit + run_sync a workflow in one call."""
    from vibeproteinagent.adapters import ToolRegistry

    if registry is None:
        registry = ToolRegistry()
        registry.discover()
    executor = WorkflowExecutor(workspace, registry, supervisor)
    wf_id = executor.submit(workflow, params)
    return executor.run_sync(wf_id, max_wait_s=max_wait_s)


def build_workflow(name: str, version: str = "0.0.0", **kwargs) -> Workflow:
    """Tiny helper: ``Workflow(name=..., version=...)`` factory."""
    return Workflow(name=name, version=version, **kwargs)


__all__ = [
    "Workflow",
    "WorkflowExecutor",
    "WorkflowNode",
    "WorkflowParam",
    "WorkflowStore",
    "build_workflow",
    "compute_cache_key",
    "env_fingerprint",
    "eval_when",
    "load_workflow",
    "resolve",
    "run_workflow_sync",
    "serialize_workflow",
]
