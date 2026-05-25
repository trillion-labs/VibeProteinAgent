from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from vibeproteinagent.workspace import Workspace
from vibeproteinagent.workspace.store import canonical_json, now_us

from .types import AgentRunRecord, ModeClassification

SYSTEM_PROMPT_VERSION = "vpa-agent-2026-05-24"
TOOL_SCHEMA_VERSION = "vpa-agent-tools-v1"


class AgentSessionStore:
    """Workspace-backed persistence for agent runs, messages, plans and decisions."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.ensure_schema()

    def ensure_schema(self) -> None:
        conn = self.workspace.connect()
        try:
            _create_agent_tables(conn)
        finally:
            conn.close()

    def create_run(
        self,
        user_request: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> AgentRunRecord:
        run_id = f"arun_{uuid.uuid4().hex}"
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_runs (
                  run_id, workspace, user_request, state, provider, model,
                  system_prompt_version, tool_schema_version, workflow_ids_json,
                  job_ids_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'created', ?, ?, ?, ?, '[]', '[]', ?, ?)
                """,
                (
                    run_id,
                    self.workspace.name,
                    user_request,
                    provider,
                    model,
                    SYSTEM_PROMPT_VERSION,
                    TOOL_SCHEMA_VERSION,
                    ts,
                    ts,
                ),
            )
        finally:
            conn.close()
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> AgentRunRecord:
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise KeyError(run_id)
        return _row_to_run(row)

    def update_run(
        self,
        run_id: str,
        *,
        state: str | None = None,
        classification: ModeClassification | None = None,
        workflow_ids: list[str] | None = None,
        job_ids: list[str] | None = None,
        artifacts: dict[str, Any] | None = None,
        final_report: dict[str, Any] | None = None,
        provider_snapshot: dict[str, Any] | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> AgentRunRecord:
        sets = ["updated_at = ?"]
        args: list[Any] = [now_us()]
        if state is not None:
            sets.append("state = ?")
            args.append(state)
        if classification is not None:
            sets.append("mode = ?")
            args.append(classification.mode)
            sets.append("classification_json = ?")
            args.append(canonical_json(classification.model_dump()))
        if workflow_ids is not None:
            sets.append("workflow_ids_json = ?")
            args.append(canonical_json(workflow_ids))
        if job_ids is not None:
            sets.append("job_ids_json = ?")
            args.append(canonical_json(job_ids))
        if artifacts is not None:
            sets.append("artifacts_json = ?")
            args.append(canonical_json(artifacts))
        if final_report is not None:
            sets.append("final_report_json = ?")
            args.append(canonical_json(final_report))
        if provider_snapshot is not None:
            sets.append("provider_snapshot_json = ?")
            args.append(canonical_json(provider_snapshot))
        if provider is not None:
            sets.append("provider = ?")
            args.append(provider)
        if model is not None:
            sets.append("model = ?")
            args.append(model)
        args.append(run_id)

        conn = self.workspace.connect()
        try:
            conn.execute(
                f"UPDATE agent_runs SET {', '.join(sets)} WHERE run_id = ?",
                args,
            )
        finally:
            conn.close()
        return self.get_run(run_id)

    def add_message(
        self,
        run_id: str,
        role: str,
        content: str,
        *,
        model: str | None = None,
        token_count: int | None = None,
    ) -> str:
        message_id = f"amsg_{uuid.uuid4().hex}"
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_messages (
                  message_id, run_id, role, content, model, token_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (message_id, run_id, role, content, model, token_count, now_us()),
            )
        finally:
            conn.close()
        return message_id

    def add_tool_call(
        self,
        run_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        *,
        ok: bool,
    ) -> str:
        tool_call_id = f"atool_{uuid.uuid4().hex}"
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_tool_calls (
                  tool_call_id, run_id, tool_name, args_json, result_json, ok, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    run_id,
                    tool_name,
                    canonical_json(args),
                    canonical_json(result),
                    1 if ok else 0,
                    now_us(),
                ),
            )
        finally:
            conn.close()
        return tool_call_id

    def add_plan(
        self,
        run_id: str,
        plan: dict[str, Any],
        *,
        status: str,
        validation_errors: list[dict[str, Any]] | None = None,
        workflow_id: str | None = None,
    ) -> str:
        plan_id = f"aplan_{uuid.uuid4().hex}"
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_plans (
                  plan_id, run_id, plan_json, status, validation_errors_json,
                  workflow_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    run_id,
                    canonical_json(plan),
                    status,
                    canonical_json(validation_errors or []),
                    workflow_id,
                    now_us(),
                ),
            )
        finally:
            conn.close()
        return plan_id

    def add_summary(
        self,
        run_id: str,
        summary: dict[str, Any],
        *,
        workflow_id: str | None = None,
        job_id: str | None = None,
    ) -> str:
        summary_id = f"asum_{uuid.uuid4().hex}"
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_summaries (
                  summary_id, run_id, workflow_id, job_id, summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (summary_id, run_id, workflow_id, job_id, canonical_json(summary), now_us()),
            )
        finally:
            conn.close()
        return summary_id

    def add_decision(
        self,
        run_id: str,
        decision: dict[str, Any],
        *,
        round_index: int = 0,
    ) -> str:
        decision_id = f"adec_{uuid.uuid4().hex}"
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_decisions (
                  decision_id, run_id, round_index, decision_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (decision_id, run_id, int(round_index), canonical_json(decision), now_us()),
            )
        finally:
            conn.close()
        return decision_id

    def add_faithfulness_check(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        round_index: int = 0,
        decision_id: str | None = None,
    ) -> str:
        check_id = f"adfc_{uuid.uuid4().hex}"
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO agent_faithfulness_checks (
                  check_id, run_id, round_index, decision_id, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    check_id,
                    run_id,
                    int(round_index),
                    decision_id,
                    canonical_json(result),
                    now_us(),
                ),
            )
        finally:
            conn.close()
        return check_id

    def list_runs(self, *, limit: int = 50) -> list[AgentRunRecord]:
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
        return [_row_to_run(r) for r in rows]

    def get_run_details(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        conn = self.workspace.connect()
        try:
            messages = conn.execute(
                """
                SELECT message_id, role, content, model, token_count, created_at
                FROM agent_messages
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
            tool_calls = conn.execute(
                """
                SELECT tool_call_id, tool_name, args_json, result_json, ok, created_at
                FROM agent_tool_calls
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
            plans = conn.execute(
                """
                SELECT plan_id, plan_json, status, validation_errors_json, workflow_id, created_at
                FROM agent_plans
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
            summaries = conn.execute(
                """
                SELECT summary_id, workflow_id, job_id, summary_json, created_at
                FROM agent_summaries
                WHERE run_id = ?
                ORDER BY created_at ASC
                """,
                (run_id,),
            ).fetchall()
            decisions = conn.execute(
                """
                SELECT decision_id, round_index, decision_json, created_at
                FROM agent_decisions
                WHERE run_id = ?
                ORDER BY round_index ASC, created_at ASC
                """,
                (run_id,),
            ).fetchall()
        finally:
            conn.close()
        return {
            "run": run.model_dump(),
            "messages": [
                {
                    "message_id": row["message_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "model": row["model"],
                    "token_count": row["token_count"],
                    "created_at": int(row["created_at"]),
                }
                for row in messages
            ],
            "tool_calls": [
                {
                    "tool_call_id": row["tool_call_id"],
                    "tool_name": row["tool_name"],
                    "args": json.loads(row["args_json"] or "{}"),
                    "result": json.loads(row["result_json"]) if row["result_json"] else None,
                    "ok": bool(row["ok"]),
                    "created_at": int(row["created_at"]),
                }
                for row in tool_calls
            ],
            "plans": [
                {
                    "plan_id": row["plan_id"],
                    "plan": json.loads(row["plan_json"] or "{}"),
                    "status": row["status"],
                    "validation_errors": json.loads(row["validation_errors_json"] or "[]"),
                    "workflow_id": row["workflow_id"],
                    "created_at": int(row["created_at"]),
                }
                for row in plans
            ],
            "summaries": [
                {
                    "summary_id": row["summary_id"],
                    "workflow_id": row["workflow_id"],
                    "job_id": row["job_id"],
                    "summary": json.loads(row["summary_json"] or "{}"),
                    "created_at": int(row["created_at"]),
                }
                for row in summaries
            ],
            "decisions": [
                {
                    "decision_id": row["decision_id"],
                    "round_index": int(row["round_index"]),
                    "decision": json.loads(row["decision_json"] or "{}"),
                    "created_at": int(row["created_at"]),
                }
                for row in decisions
            ],
        }


def _create_agent_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
          run_id                 TEXT PRIMARY KEY,
          workspace              TEXT NOT NULL,
          user_request           TEXT NOT NULL,
          mode                   TEXT,
          state                  TEXT NOT NULL,
          provider               TEXT,
          model                  TEXT,
          provider_snapshot_json TEXT,
          system_prompt_version  TEXT NOT NULL,
          tool_schema_version    TEXT NOT NULL,
          classification_json    TEXT,
          workflow_ids_json      TEXT NOT NULL DEFAULT '[]',
          job_ids_json           TEXT NOT NULL DEFAULT '[]',
          artifacts_json         TEXT NOT NULL DEFAULT '[]',
          final_report_json      TEXT,
          created_at             INTEGER NOT NULL,
          updated_at             INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at);

        CREATE TABLE IF NOT EXISTS agent_messages (
          message_id   TEXT PRIMARY KEY,
          run_id       TEXT NOT NULL,
          role         TEXT NOT NULL,
          content      TEXT NOT NULL,
          model        TEXT,
          token_count  INTEGER,
          created_at   INTEGER NOT NULL,
          FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_agent_messages_run ON agent_messages(run_id);

        CREATE TABLE IF NOT EXISTS agent_tool_calls (
          tool_call_id TEXT PRIMARY KEY,
          run_id       TEXT NOT NULL,
          tool_name    TEXT NOT NULL,
          args_json    TEXT NOT NULL,
          result_json  TEXT,
          ok           INTEGER NOT NULL,
          created_at   INTEGER NOT NULL,
          FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run ON agent_tool_calls(run_id);

        CREATE TABLE IF NOT EXISTS agent_plans (
          plan_id                TEXT PRIMARY KEY,
          run_id                 TEXT NOT NULL,
          plan_json              TEXT NOT NULL,
          status                 TEXT NOT NULL,
          validation_errors_json TEXT NOT NULL DEFAULT '[]',
          workflow_id            TEXT,
          created_at             INTEGER NOT NULL,
          FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS agent_summaries (
          summary_id   TEXT PRIMARY KEY,
          run_id       TEXT NOT NULL,
          workflow_id  TEXT,
          job_id        TEXT,
          summary_json  TEXT NOT NULL,
          created_at    INTEGER NOT NULL,
          FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS agent_decisions (
          decision_id   TEXT PRIMARY KEY,
          run_id        TEXT NOT NULL,
          round_index   INTEGER NOT NULL,
          decision_json TEXT NOT NULL,
          created_at    INTEGER NOT NULL,
          FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
        );
        """
    )


def _row_to_run(row: sqlite3.Row) -> AgentRunRecord:
    final = row["final_report_json"]
    provider_snapshot = row["provider_snapshot_json"]
    artifacts = json.loads(row["artifacts_json"] or "{}")
    return AgentRunRecord(
        run_id=row["run_id"],
        workspace=row["workspace"],
        user_request=row["user_request"],
        mode=row["mode"],
        state=row["state"],
        provider=row["provider"],
        model=row["model"],
        provider_snapshot=json.loads(provider_snapshot) if provider_snapshot else None,
        system_prompt_version=row["system_prompt_version"],
        tool_schema_version=row["tool_schema_version"],
        workflow_ids=json.loads(row["workflow_ids_json"] or "[]"),
        job_ids=json.loads(row["job_ids_json"] or "[]"),
        artifacts=artifacts if isinstance(artifacts, dict) else {},
        final_report=json.loads(final) if final else None,
        created_at=int(row["created_at"]),
        updated_at=int(row["updated_at"]),
    )


__all__ = [
    "SYSTEM_PROMPT_VERSION",
    "TOOL_SCHEMA_VERSION",
    "AgentSessionStore",
]
