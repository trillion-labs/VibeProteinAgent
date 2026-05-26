from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ParamType = Literal["str", "int", "float", "bool", "path", "list", "dict"]
OnFailure = Literal["stop", "continue", "skip"]
VIRTUAL_OUTPUT_NODES = frozenset({"prior_round"})


class WorkflowParam(BaseModel):
    """A workflow input parameter declared at the top of a descriptor."""

    model_config = ConfigDict(extra="forbid")

    type: ParamType = "str"
    required: bool = False
    default: Any | None = None
    description: str | None = None


class WorkflowNode(BaseModel):
    """One node in a workflow DAG."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    tool: str = Field(min_length=1)
    version: str = "*"
    inputs: dict[str, Any] = Field(default_factory=dict)
    cache: bool = True
    foreach: str | None = None
    max_parallel: int | None = Field(default=None, ge=1)
    when: str | None = None
    on_failure: OnFailure = "stop"
    max_retries: int = 0
    priority: str = "normal"

    @model_validator(mode="after")
    def _check_id(self) -> WorkflowNode:
        if "[" in self.id or "]" in self.id:
            raise ValueError(
                f"node id {self.id!r} must not contain brackets (reserved for foreach expansion)"
            )
        return self


class Workflow(BaseModel):
    """A workflow descriptor — the YAML-loadable / Python-builder root object."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = "0.0.0"
    description: str | None = None
    params: dict[str, WorkflowParam] = Field(default_factory=dict)
    nodes: list[WorkflowNode] = Field(default_factory=list)

    # ----------------------------------------------------------- builder API

    def node(
        self,
        id: str,
        *,
        tool: str,
        version: str = "*",
        inputs: dict[str, Any] | None = None,
        cache: bool = True,
        when: str | None = None,
        on_failure: OnFailure = "stop",
        max_retries: int = 0,
        priority: str = "normal",
        max_parallel: int | None = None,
    ) -> WorkflowNode:
        """Append a node to the workflow and return it (for chaining)."""
        n = WorkflowNode(
            id=id,
            tool=tool,
            version=version,
            inputs=inputs or {},
            cache=cache,
            max_parallel=max_parallel,
            when=when,
            on_failure=on_failure,
            max_retries=max_retries,
            priority=priority,
        )
        self.nodes.append(n)
        return n

    def fanout(
        self,
        id: str,
        *,
        tool: str,
        over: str,
        inputs: dict[str, Any] | None = None,
        version: str = "*",
        cache: bool = True,
        on_failure: OnFailure = "stop",
        max_parallel: int | None = None,
    ) -> WorkflowNode:
        """Append a foreach (fanout) node."""
        n = WorkflowNode(
            id=id,
            tool=tool,
            version=version,
            inputs=inputs or {},
            cache=cache,
            foreach=over,
            max_parallel=max_parallel,
            on_failure=on_failure,
        )
        self.nodes.append(n)
        return n

    def save(self, path: str | Path) -> None:
        """Serialise to YAML at ``path``. Runs :meth:`validate_dag` first."""
        from pathlib import Path as _Path

        from .loader import serialize_workflow

        self.validate_dag()
        _Path(str(path)).write_text(serialize_workflow(self))

    @model_validator(mode="after")
    def _check_unique_node_ids(self) -> Workflow:
        seen: set[str] = set()
        for n in self.nodes:
            if n.id in seen:
                raise ValueError(f"duplicate node id {n.id!r}")
            seen.add(n.id)
        return self

    def validate_dag(self) -> None:
        """Reject workflows with missing upstreams or dependency cycles.

        Walks every node's ``inputs`` (recursively, dict/list/scalars),
        ``foreach`` and ``when`` strings, harvests every
        ``${outputs.X.field}`` reference, and asserts that:

        1. every referenced ``X`` is a declared node id;
        2. the resulting dependency graph is acyclic (Kahn's algorithm).

        Both failure modes raise :class:`VPAError` with ``E_INVALID_PARAMS``
        — caught by the workflow CLI / executor and surfaced as an error
        envelope. Pre-Phase-9 these slipped through and only manifested
        as a runtime timeout (Codex round-3 RC2).
        """
        from vibeproteinagent.core.errors import InvalidParamsError

        from .resolver import iter_refs

        declared = {n.id for n in self.nodes}
        deps: dict[str, set[str]] = {n.id: set() for n in self.nodes}
        referenced_upstream: set[str] = set()

        for node in self.nodes:
            refs: list[str] = []
            refs.extend(iter_refs(node.inputs))
            if node.foreach is not None:
                refs.extend(iter_refs(node.foreach))
            if node.when is not None:
                refs.extend(iter_refs(node.when))
            for expr in refs:
                head, _, rest = expr.partition(".")
                if head not in {"outputs", "jobs"}:
                    continue
                upstream = rest.split(".", 1)[0]
                # Strip any bracket index on the head segment ("foo[0]" -> "foo").
                if "[" in upstream:
                    upstream = upstream.split("[", 1)[0]
                if not upstream:
                    continue
                referenced_upstream.add(upstream)
                if upstream in declared:
                    # A self-reference is a degenerate cycle and must be
                    # caught by the topological sort below.
                    deps[node.id].add(upstream)

        missing = sorted(referenced_upstream - declared - VIRTUAL_OUTPUT_NODES)
        if missing:
            raise InvalidParamsError(
                f"workflow {self.name!r} references undeclared upstream nodes",
                details={
                    "workflow": self.name,
                    "missing": missing,
                },
            )

        # Kahn's topological sort over `deps`. `deps[node]` is the set of
        # nodes ``node`` depends on; we count in-degrees and peel off
        # ready nodes one layer at a time.
        in_degree: dict[str, int] = {n.id: len(deps[n.id]) for n in self.nodes}
        ready: list[str] = [nid for nid, d in in_degree.items() if d == 0]
        ordered: list[str] = []
        # Reverse adjacency: who depends on me?
        rev: dict[str, set[str]] = {n.id: set() for n in self.nodes}
        for node_id, upstreams in deps.items():
            for up in upstreams:
                rev[up].add(node_id)

        while ready:
            cur = ready.pop()
            ordered.append(cur)
            for dependent in rev[cur]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

        unresolved = [nid for nid, d in in_degree.items() if d > 0]
        if unresolved:
            cycle = _find_cycle(deps, unresolved)
            raise InvalidParamsError(
                f"workflow {self.name!r} has a dependency cycle",
                details={
                    "workflow": self.name,
                    "cycle": cycle,
                },
            )


def _find_cycle(deps: dict[str, set[str]], unresolved: list[str]) -> list[str]:
    """Return one cycle path from ``unresolved`` over ``deps``.

    DFS from each unresolved node, recording the current path; the
    first revisit closes the cycle. Returns the cycle as a list of
    node ids in dependency order, e.g. ``["A", "B", "C", "A"]`` for
    ``A -> B -> C -> A``. Caller has guaranteed at least one cycle.
    """
    unresolved_set = set(unresolved)
    visited: set[str] = set()
    for start in unresolved:
        if start in visited:
            continue
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        on_path: set[str] = {start}
        while stack:
            node, path = stack[-1]
            # Only consider edges that stay within `unresolved_set` —
            # nodes already fully ordered can't be on a cycle.
            next_deps = [d for d in deps.get(node, ()) if d in unresolved_set]
            advanced = False
            for d in next_deps:
                if d in on_path:
                    # Found a cycle; trim to the loop and close it.
                    i = path.index(d)
                    return [*path[i:], d]
                if d not in visited:
                    stack.append((d, [*path, d]))
                    on_path.add(d)
                    advanced = True
                    break
            if not advanced:
                visited.add(node)
                on_path.discard(node)
                stack.pop()
    # Defensive fallback: caller said there's a cycle but DFS missed it.
    return sorted(unresolved)


__all__ = [
    "OnFailure",
    "ParamType",
    "Workflow",
    "WorkflowNode",
    "WorkflowParam",
]
