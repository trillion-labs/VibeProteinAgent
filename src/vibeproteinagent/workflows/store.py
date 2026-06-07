from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from vibeproteinagent.core.errors import NotFoundError
from vibeproteinagent.workspace.store import (
    Workspace,
    canonical_json,
    now_us,
)
from vibeproteinagent.workspace.types import WorkflowNodeRecord, WorkflowRecord

_VALID_WORKFLOW_STATES = frozenset(
    {"submitted", "running", "completed", "failed", "cancelled"}
)
_VALID_NODE_STATES = frozenset(
    {"pending", "ready", "running", "completed", "failed", "cached", "skipped"}
)


class WorkflowStore:
    """CRUD over the ``workflows`` and ``workflow_nodes`` tables."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self._ensure_node_columns()

    # ------------------------------------------------------------ workflows

    def create(
        self,
        *,
        name: str,
        version: str,
        descriptor_yaml: str,
        params: dict[str, Any],
        trajectory_id: str | None = None,
    ) -> str:
        """Insert a workflow row in state ``submitted``, return its workflow_id."""
        workflow_id = f"wf_{uuid.uuid4().hex}"
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO workflows (
                  workflow_id, name, version, descriptor, params_json,
                  trajectory_id, state, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'submitted', ?)
                """,
                (
                    workflow_id, name, version, descriptor_yaml,
                    canonical_json(params), trajectory_id, ts,
                ),
            )
        finally:
            conn.close()
        return workflow_id

    def get(self, workflow_id: str) -> WorkflowRecord:
        conn = self.workspace.connect()
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
        return self._row_to_wf(row)

    def list(self, *, limit: int = 100) -> list[WorkflowRecord]:
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM workflows ORDER BY submitted_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_wf(r) for r in rows]

    def update_state(
        self,
        workflow_id: str,
        state: str,
        *,
        finished: bool = False,
    ) -> None:
        if state not in _VALID_WORKFLOW_STATES:
            raise ValueError(f"unknown workflow state {state!r}")
        conn = self.workspace.connect()
        try:
            if finished:
                conn.execute(
                    "UPDATE workflows SET state = ?, finished_at = ? "
                    " WHERE workflow_id = ?",
                    (state, now_us(), workflow_id),
                )
            else:
                conn.execute(
                    "UPDATE workflows SET state = ? WHERE workflow_id = ?",
                    (state, workflow_id),
                )
        finally:
            conn.close()

    # ------------------------------------------------------------ nodes

    def add_node(
        self,
        *,
        workflow_id: str,
        node_id: str,
        tool_name: str,
        tool_version: str,
        upstream: list[str],
        params_hash: str,
        input_hash: str,
        cache_key: str,
        state: str = "pending",
    ) -> None:
        if state not in _VALID_NODE_STATES:
            raise ValueError(f"unknown node state {state!r}")
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO workflow_nodes (
                  workflow_id, node_id, tool_name, tool_version,
                  upstream, params_hash, input_hash, cache_key, state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id, node_id, tool_name, tool_version,
                    json.dumps(list(upstream)), params_hash, input_hash,
                    cache_key, state,
                ),
            )
        finally:
            conn.close()

    def update_node_state(
        self,
        workflow_id: str,
        node_id: str,
        state: str,
        *,
        job_id: str | None = None,
        cache_key: str | None = None,
        input_hash: str | None = None,
        params_hash: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error_details: dict[str, Any] | None = None,
        output: Any | None = None,
    ) -> None:
        if state not in _VALID_NODE_STATES:
            raise ValueError(f"unknown node state {state!r}")
        sets: list[str] = ["state = ?"]
        args: list[Any] = [state]
        if job_id is not None:
            sets.append("job_id = ?")
            args.append(job_id)
        if output is not None:
            sets.append("output_json = ?")
            args.append(canonical_json(output))
        if cache_key is not None:
            sets.append("cache_key = ?")
            args.append(cache_key)
        if input_hash is not None:
            sets.append("input_hash = ?")
            args.append(input_hash)
        if params_hash is not None:
            sets.append("params_hash = ?")
            args.append(params_hash)
        if state == "failed":
            sets.append("error_code = ?")
            args.append(error_code)
            sets.append("error_message = ?")
            args.append(error_message)
            sets.append("error_details_json = ?")
            args.append(canonical_json(error_details) if error_details else None)
        else:
            sets.extend([
                "error_code = NULL",
                "error_message = NULL",
                "error_details_json = NULL",
            ])
        args.append(workflow_id)
        args.append(node_id)
        conn = self.workspace.connect()
        try:
            conn.execute(
                f"UPDATE workflow_nodes SET {', '.join(sets)} "
                "WHERE workflow_id = ? AND node_id = ?",
                args,
            )
        finally:
            conn.close()

    def get_node(self, workflow_id: str, node_id: str) -> WorkflowNodeRecord:
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM workflow_nodes "
                " WHERE workflow_id = ? AND node_id = ?",
                (workflow_id, node_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise NotFoundError(
                f"workflow node {workflow_id}:{node_id} not found",
                details={"workflow_id": workflow_id, "node_id": node_id},
            )
        return self._row_to_node(row)

    def list_nodes(self, workflow_id: str) -> list[WorkflowNodeRecord]:
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM workflow_nodes WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_node(r) for r in rows]

    def find_cached_node(self, cache_key: str) -> WorkflowNodeRecord | None:
        """Return the most recent successfully-completed node with this cache_key.

        Codex round-2 C12: order by the producing job's ``finished_at`` so we
        prefer the most recent successful node — sqlite returns rows in
        unspecified order otherwise, which made the cache lookup
        non-deterministic across multiple cache hits.
        """
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                """
                SELECT wn.* FROM workflow_nodes wn
                LEFT JOIN jobs j ON j.job_id = wn.job_id
                WHERE wn.cache_key = ? AND wn.state IN ('completed','cached')
                ORDER BY COALESCE(j.finished_at, j.submitted_at, 0) DESC
                LIMIT 1
                """,
                (cache_key,),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return None
        return self._row_to_node(rows[0])

    # ------------------------------------------------------------ helpers

    def _ensure_node_columns(self) -> None:
        """Backfill workflow-node columns for workspaces created by older DDL.

        Workspace schema application uses ``CREATE TABLE IF NOT EXISTS``; adding
        columns to ``schema.sql`` alone does not mutate existing SQLite files.
        Keep this local to workflow storage so workflow status reads remain
        backward-compatible without touching unrelated workspace store code.
        """
        conn = self.workspace.connect()
        try:
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(workflow_nodes)").fetchall()
            }
            if "error_code" not in cols:
                conn.execute("ALTER TABLE workflow_nodes ADD COLUMN error_code TEXT")
            if "error_message" not in cols:
                conn.execute(
                    "ALTER TABLE workflow_nodes ADD COLUMN error_message TEXT"
                )
            if "error_details_json" not in cols:
                conn.execute(
                    "ALTER TABLE workflow_nodes ADD COLUMN error_details_json TEXT"
                )
            if "output_json" not in cols:
                conn.execute("ALTER TABLE workflow_nodes ADD COLUMN output_json TEXT")
            wf_cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(workflows)").fetchall()
            }
            if "trajectory_id" not in wf_cols:
                conn.execute("ALTER TABLE workflows ADD COLUMN trajectory_id TEXT")
        finally:
            conn.close()

    @staticmethod
    def _row_to_wf(row: sqlite3.Row) -> WorkflowRecord:
        return WorkflowRecord(
            workflow_id=row["workflow_id"],
            name=row["name"],
            version=row["version"],
            descriptor=row["descriptor"],
            params=json.loads(row["params_json"]),
            trajectory_id=row["trajectory_id"] if "trajectory_id" in row.keys() else None,
            state=row["state"],
            submitted_at=int(row["submitted_at"]),
            finished_at=row["finished_at"],
        )

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> WorkflowNodeRecord:
        edj = row["error_details_json"]
        oj = row["output_json"]
        error_details = json.loads(edj) if edj else None
        output = json.loads(oj) if oj else None
        return WorkflowNodeRecord(
            workflow_id=row["workflow_id"],
            node_id=row["node_id"],
            tool_name=row["tool_name"],
            tool_version=row["tool_version"],
            upstream=json.loads(row["upstream"]),
            params_hash=row["params_hash"],
            input_hash=row["input_hash"],
            cache_key=row["cache_key"],
            state=row["state"],
            job_id=row["job_id"],
            output=output,
            error_code=row["error_code"],
            error_message=row["error_message"],
            error_details=error_details,
        )


__all__ = ["WorkflowStore"]
