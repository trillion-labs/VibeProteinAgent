from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .descriptor import WorkflowNode

TERMINAL_NODE_STATES = frozenset({"completed", "cached", "failed", "skipped"})
SUCCESS_NODE_STATES = frozenset({"completed", "cached"})
FOREACH_CHILD_RE = re.compile(r"^(?P<parent>[^\[\]]+)\[(?P<index>\d+)\]$")


@dataclass
class NodeRuntime:
    """Per-node execution bookkeeping kept in memory while the DAG runs."""

    desc: WorkflowNode
    runtime_id: str
    upstream: list[str] = field(default_factory=list)
    state: str = "pending"
    job_id: str | None = None
    output: dict[str, Any] | None = None
    item: Any = None
    foreach_parent: str | None = None
    foreach_index: int | None = None
    cache_key: str | None = None


class WorkflowStateMachine:
    """Pure state helpers for node readiness and expression scope building."""

    @staticmethod
    def logical_id(rt: NodeRuntime) -> str:
        return rt.foreach_parent or rt.desc.id

    @staticmethod
    def upstream_ok(rt: NodeRuntime, runtime: dict[str, NodeRuntime]) -> bool:
        for up in rt.upstream:
            other = runtime.get(up)
            if other is None or not _output_available(other):
                return False
        return True

    @staticmethod
    def build_scope(
        params: dict[str, Any],
        runtime: dict[str, NodeRuntime],
        *,
        current: NodeRuntime | None = None,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        jobs: dict[str, Any] = {}
        parent_outputs: set[str] = set()
        foreach_children: list[NodeRuntime] = []
        for rt in runtime.values():
            if not _output_available(rt):
                continue
            outputs[rt.runtime_id] = rt.output
            if rt.job_id:
                jobs[rt.runtime_id] = rt.job_id
            if rt.foreach_parent is not None:
                foreach_children.append(rt)
            elif rt.desc.foreach:
                parent_outputs.add(rt.runtime_id)
        for rt in sorted(
            foreach_children,
            key=lambda child: (child.foreach_parent or "", child.foreach_index or 0),
        ):
            # The completed foreach placeholder is the canonical aggregate
            # output once it exists. Before then, expose the currently completed
            # children as a best-effort ordered list.
            if rt.foreach_parent not in parent_outputs:
                outputs.setdefault(rt.foreach_parent, []).append(rt.output)
            if rt.job_id:
                jobs.setdefault(rt.foreach_parent, []).append(rt.job_id)
        scope: dict[str, Any] = {"params": params, "outputs": outputs, "jobs": jobs}
        if current is not None and current.foreach_parent is not None:
            scope["item"] = current.item
            scope["index"] = current.foreach_index
        return scope


logical_id = WorkflowStateMachine.logical_id
upstream_ok = WorkflowStateMachine.upstream_ok
build_scope = WorkflowStateMachine.build_scope


def _output_available(rt: NodeRuntime) -> bool:
    if rt.state in SUCCESS_NODE_STATES and rt.output is not None:
        return True
    return (
        rt.state == "skipped"
        and isinstance(rt.output, dict)
        and rt.output.get("technical_failure") is False
    )


__all__ = [
    "FOREACH_CHILD_RE",
    "SUCCESS_NODE_STATES",
    "TERMINAL_NODE_STATES",
    "NodeRuntime",
    "WorkflowStateMachine",
    "build_scope",
    "logical_id",
    "upstream_ok",
]
