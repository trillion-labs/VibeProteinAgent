from __future__ import annotations

from .state import NodeRuntime, WorkflowStateMachine


class FailurePolicy:
    """Failure-policy helpers for stop/continue/skip workflow semantics."""

    @staticmethod
    def descendant_runtime_ids(
        failed_rt: NodeRuntime,
        runtime: dict[str, NodeRuntime],
    ) -> set[str]:
        failed_logical = WorkflowStateMachine.logical_id(failed_rt)
        descendants: set[str] = set()
        frontier = [rt for rt in runtime.values() if failed_logical in rt.upstream]
        while frontier:
            rt = frontier.pop()
            if rt.runtime_id in descendants:
                continue
            descendants.add(rt.runtime_id)
            rt_logical = WorkflowStateMachine.logical_id(rt)
            for other in runtime.values():
                if rt_logical in other.upstream and other.runtime_id not in descendants:
                    frontier.append(other)
        return descendants


__all__ = ["FailurePolicy"]
