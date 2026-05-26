from __future__ import annotations

from typing import Any

from vibeproteinagent.core.errors import InvalidParamsError

from .descriptor import VIRTUAL_OUTPUT_NODES, Workflow, WorkflowNode
from .resolver import iter_refs, resolve


class WorkflowPlanner:
    """Pure workflow planning helpers used before job dispatch."""

    @staticmethod
    def params_dict_from_workflow(wf: Workflow, params: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for k, p in wf.params.items():
            if k in params:
                merged[k] = params[k]
            elif p.default is not None:
                merged[k] = p.default
            elif p.required:
                raise InvalidParamsError(
                    f"workflow {wf.name!r} is missing required param {k!r}",
                    details={"param": k, "workflow": wf.name},
                )
            else:
                merged[k] = None
        for k, v in params.items():
            if k not in merged:
                merged[k] = v
        return merged

    @staticmethod
    def static_upstream(node: WorkflowNode) -> list[str]:
        refs = list(iter_refs(node.inputs))
        if node.when:
            refs.extend(iter_refs(node.when))
        if node.foreach:
            refs.extend(iter_refs(node.foreach))
        upstream: set[str] = set()
        for expr in refs:
            head, _, rest = expr.partition(".")
            if head not in {"outputs", "jobs"}:
                continue
            name = rest.split(".", 1)[0]
            if "[" in name:
                name = name.split("[", 1)[0]
            if name and name not in VIRTUAL_OUTPUT_NODES:
                upstream.add(name)
        return sorted(upstream)

    @staticmethod
    def resolve_iterable(expr: str, scope: dict[str, Any]) -> list[Any]:
        bare = expr.strip()
        if bare.startswith("${") and bare.endswith("}"):
            bare = bare[2:-1].strip()
        value = resolve("${" + bare + "}", scope)
        if not isinstance(value, list):
            raise InvalidParamsError(
                f"foreach iterable {expr!r} did not resolve to a list "
                f"(got {type(value).__name__})",
                details={"expr": expr, "type": type(value).__name__},
            )
        return value


params_dict_from_workflow = WorkflowPlanner.params_dict_from_workflow
static_upstream = WorkflowPlanner.static_upstream
resolve_iterable = WorkflowPlanner.resolve_iterable


__all__ = [
    "WorkflowPlanner",
    "params_dict_from_workflow",
    "resolve_iterable",
    "static_upstream",
]
