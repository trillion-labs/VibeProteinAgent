from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from vibeproteinagent.core import config as cfg
from vibeproteinagent.core.errors import (
    ConflictError,
    InvalidParamsError,
    NotFoundError,
    WorkspaceError,
)

from .digest import normalize_params_for_digest
from .ids import derive_job_id, parse_id, validate_call_id
from .types import (
    LEGAL_TRANSITIONS,
    TERMINAL_STATES,
    ArtifactRecord,
    JobRecord,
    JobState,
    Priority,
    RunRecord,
    TrajectoryRecord,
    WorkflowRecord,
)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


# ---------------------------------------------------------------- utilities


def now_us() -> int:
    """Current wall-clock time as integer microseconds since epoch."""
    return time.time_ns() // 1000


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding used for params hashing and cache keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def params_hash(params: dict[str, Any]) -> str:
    """sha256 of canonical-JSON-encoded params."""
    return hashlib.sha256(canonical_json(params).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> tuple[str, int]:
    """Return (hex digest, size in bytes) of a file's contents."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
            size += len(buf)
    return h.hexdigest(), size


def _gpu_ids_to_str(ids: list[int] | None) -> str | None:
    if not ids:
        return None
    return ",".join(str(i) for i in ids)


def _gpu_ids_from_str(s: str | None) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.split(",") if x != ""]


def _submission_diff(
    existing: JobRecord,
    *,
    tool: str,
    version: str,
    params_hash: str,
    trajectory_id: str | None,
    workflow_id: str | None,
    workflow_node_id: str | None,
) -> dict[str, dict[str, Any]]:
    """Field-level diff between an existing job and a re-submission attempt.

    Returns an empty dict when the submissions are equivalent (which makes the
    re-submission a valid idempotent retry per Architecture v1 §5). Otherwise
    returns a ``{field: {"existing": ..., "submitted": ...}}`` map naming each
    field that differs — surfaced to the caller via ``E_CONFLICT.details.diff``.
    """
    pairs = (
        ("tool_name", existing.tool_name, tool),
        ("tool_version", existing.tool_version, version),
        ("params_hash", existing.params_hash, params_hash),
        ("trajectory_id", existing.trajectory_id, trajectory_id),
        ("workflow_id", existing.workflow_id, workflow_id),
        ("workflow_node_id", existing.workflow_node_id, workflow_node_id),
    )
    return {
        field: {"existing": old, "submitted": new}
        for field, old, new in pairs
        if old != new
    }


def _assert_call_id_match(
    existing: JobRecord,
    *,
    call_id: str,
    tool: str,
    version: str,
    params_hash: str,
    trajectory_id: str | None,
    workflow_id: str | None,
    workflow_node_id: str | None,
) -> JobRecord:
    """Single owner of "is this re-submission idempotent or a conflict?".

    Phase 7 added the diff-and-raise check to the pre-insert path; the
    SQLite-IntegrityError race handler still bypassed it (Codex round-2 RC6).
    Both paths now flow through this helper so the contract is identical
    whichever side wins the race.
    """
    diff = _submission_diff(
        existing,
        tool=tool, version=version, params_hash=params_hash,
        trajectory_id=trajectory_id,
        workflow_id=workflow_id, workflow_node_id=workflow_node_id,
    )
    if diff:
        raise ConflictError(
            f"call_id {call_id!r} already bound to job "
            f"{existing.job_id} with a different submission",
            details={
                "call_id": call_id,
                "existing_job_id": existing.job_id,
                "diff": diff,
            },
        )
    return existing


# ---------------------------------------------------------------- Workspace


class Workspace:
    """Filesystem + SQLite handle for a single named workspace."""

    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self.root = root
        self.db_path = root / "workspace.db"
        self.jobs_dir = root / "jobs"
        self.artifacts_dir = root / "artifacts"
        self.workflows_dir = root / "workflows"
        self.cache_dir = root / "cache"

    # ------------------------------------------------------------ resolution

    @classmethod
    def resolve(
        cls,
        name: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Workspace:
        """Resolve a workspace per §9: arg > $VPA_WORKSPACE > active file > 'default'."""
        if env is None:
            chosen = cfg.resolve_workspace(name)
        else:
            if name:
                chosen = name
            elif env.get(cfg.VPA_WORKSPACE_ENV):
                chosen = env[cfg.VPA_WORKSPACE_ENV]
            else:
                active = cfg.active_workspace_file()
                if active.exists():
                    txt = active.read_text().strip()
                    chosen = txt or cfg.DEFAULT_WORKSPACE
                else:
                    chosen = cfg.DEFAULT_WORKSPACE
        return cls(chosen, cfg.workspace_dir(chosen))

    # ------------------------------------------------------------ lifecycle

    @classmethod
    def create(cls, name: str) -> Workspace:
        """Create (idempotently) a workspace named ``name`` and apply the schema."""
        if not name or not isinstance(name, str):
            raise InvalidParamsError("workspace name must be a non-empty string",
                                     details={"got": name})
        root = cfg.workspace_dir(name)
        ws = cls(name, root)
        ws._ensure_dirs()
        ws.apply_schema()
        return ws

    @classmethod
    def list(cls) -> list[str]:
        """Return the names of all workspaces under ``$VPA_HOME/workspaces``."""
        root = cfg.workspaces_root()
        if not root.exists():
            return []
        return sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "workspace.db").exists()
        )

    @classmethod
    def delete(cls, name: str) -> None:
        """Permanently delete the named workspace's directory."""
        root = cfg.workspace_dir(name)
        if not root.exists():
            raise NotFoundError(f"workspace {name!r} does not exist",
                                details={"name": name})
        shutil.rmtree(root)
        active = cfg.active_workspace_file()
        if active.exists() and active.read_text().strip() == name:
            active.unlink()

    @classmethod
    def use(cls, name: str) -> None:
        """Mark ``name`` as the active workspace (writes ``$VPA_HOME/active``)."""
        root = cfg.workspace_dir(name)
        if not root.exists():
            raise NotFoundError(f"workspace {name!r} does not exist",
                                details={"name": name})
        active = cfg.active_workspace_file()
        active.parent.mkdir(parents=True, exist_ok=True)
        active.write_text(name)

    # ------------------------------------------------------------ filesystem

    def _ensure_dirs(self) -> None:
        for d in (self.root, self.jobs_dir, self.artifacts_dir,
                  self.workflows_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Path to a job's per-job directory (created on demand)."""
        d = self.jobs_dir / job_id
        (d / "outputs").mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ database

    def connect(self) -> sqlite3.Connection:
        """Open a sync SQLite connection with WAL mode and foreign keys on."""
        if not self.db_path.exists():
            raise WorkspaceError(
                f"workspace database missing at {self.db_path}",
                details={"workspace": self.name, "db_path": str(self.db_path)},
            )
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def aconnect(self) -> aiosqlite.Connection:
        """Return a coroutine that opens an async SQLite connection (await it)."""
        async def _open() -> aiosqlite.Connection:
            conn = await aiosqlite.connect(self.db_path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA synchronous = NORMAL")
            return conn

        return _open()  # type: ignore[return-value]

    def apply_schema(self) -> None:
        """Apply ``schema.sql`` and run user_version migrations idempotently."""
        self._ensure_dirs()
        ddl = _SCHEMA_PATH.read_text()
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            conn.executescript(ddl)
            uv = conn.execute("PRAGMA user_version").fetchone()[0]
            if uv < 2:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
                if "cancel_requested" not in cols:
                    conn.execute(
                        "ALTER TABLE jobs ADD COLUMN "
                        "cancel_requested INTEGER NOT NULL DEFAULT 0"
                    )
                conn.execute("PRAGMA user_version = 2")
            if uv < 3:
                # Codex round-3 Re1: structured error details (e.g. the
                # ``violations`` list from an output-schema failure) used
                # to be stringified into ``error_message`` and lost. Add
                # a dedicated TEXT column carrying the JSON-encoded dict.
                cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
                if "error_details_json" not in cols:
                    conn.execute(
                        "ALTER TABLE jobs ADD COLUMN error_details_json TEXT"
                    )
                conn.execute("PRAGMA user_version = 3")
            if uv < 4:
                # v4 retired active agent-session tables.
                conn.execute("PRAGMA user_version = 4")
            if uv < 5:
                job_cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()
                }
                if "trajectory_id" not in job_cols:
                    conn.execute("ALTER TABLE jobs ADD COLUMN trajectory_id TEXT")
                wf_cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(workflows)").fetchall()
                }
                if "trajectory_id" not in wf_cols:
                    conn.execute("ALTER TABLE workflows ADD COLUMN trajectory_id TEXT")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trajectories (
                      trajectory_id       TEXT PRIMARY KEY,
                      workspace           TEXT NOT NULL,
                      user_query          TEXT NOT NULL,
                      objective           TEXT,
                      state               TEXT NOT NULL DEFAULT 'open'
                        CHECK (state IN ('open','completed')),
                      final_submit_job_id TEXT,
                      report_path         TEXT,
                      created_at          INTEGER NOT NULL,
                      updated_at          INTEGER NOT NULL,
                      completed_at        INTEGER
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_jobs_traj ON jobs(trajectory_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_workflows_traj ON workflows(trajectory_id)"
                )
                conn.execute("PRAGMA user_version = 5")
            if uv < 6:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS trajectory_reports (
                      trajectory_id        TEXT PRIMARY KEY,
                      report_path          TEXT NOT NULL,
                      source_id            TEXT NOT NULL,
                      format               TEXT NOT NULL,
                      final_submit_job_id  TEXT,
                      created_at           INTEGER NOT NULL,
                      FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
                        ON DELETE CASCADE
                    )
                    """
                )
                conn.execute("PRAGMA user_version = 6")
        finally:
            conn.close()


# ---------------------------------------------------------------- JobStore


class JobStore:
    """CRUD + state-transition logic for the ``jobs`` table of one workspace."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    # ------------------------------------------------------------ submit

    def submit(
        self,
        tool: str,
        version: str,
        params: dict[str, Any],
        *,
        call_id: str | None = None,
        priority: str | Priority = Priority.NORMAL,
        workspace_name: str | None = None,
        trajectory_id: str | None = None,
        workflow_id: str | None = None,
        workflow_node_id: str | None = None,
        max_retries: int = 0,
        salt: str | None = None,
        input_schema: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Insert (or re-queue) a job; auto-idempotent on deterministic ``job_id``.

        Per **Architecture v1 — Errata 2** (Codex round-4 RC1): the ``job_id``
        is derived from ``sha256(canonical_submission)``, so the same
        ``(tool, version, params, workspace, trajectory_id, workflow_id,
        workflow_node_id, salt)`` always resolves to the same id. This collapses the original
        idempotency guarantee (``call_id``) and the deterministic-naming
        guarantee promised by the brief into a single mechanism:

        * Re-submitting an existing job in ``{submitted, queued, running,
          completed}`` returns the original record — idempotent.
        * Re-submitting an existing job in ``{failed, cancelled}`` re-queues
          the **same** ``job_id`` (clearing error fields and incrementing
          ``retry_count``). Same job_id continues its history; the user
          doesn't have to remember which deterministic id they're after.

        ``call_id`` remains a user-controlled alias key for systems that
        have their own naming. A re-use of the same ``call_id`` with a
        mismatched submission still raises :class:`ConflictError`
        (``E_CONFLICT``) with ``details.diff`` naming the differing field
        — silent shadowing would let an agent's bug produce stale results.

        ``salt`` is the escape hatch for the CLI's ``--force-new`` flag:
        passing a random salt produces a fresh ``job_id`` for legitimate
        seed-reproducibility re-runs.
        """
        if call_id is not None:
            validate_call_id(call_id)
        prio = Priority(priority) if not isinstance(priority, Priority) else priority
        ws_name = workspace_name or self.workspace.name

        # Codex round-6 RC4: hash file *contents* (not just path strings)
        # for any schema field annotated with ``vpa:digest``. Without a
        # schema we fall back to raw params — preserves Phase 10's
        # deterministic-id contract for tests that register adhoc tools.
        normalized = normalize_params_for_digest(params, input_schema)
        ph = params_hash(normalized)

        derived_id = derive_job_id(
            tool=tool, version=version, params=normalized, workspace=ws_name,
            trajectory_id=trajectory_id, workflow_id=workflow_id,
            workflow_node_id=workflow_node_id,
            salt=salt,
        )

        # call_id is an *alias* layer over the deterministic id. If the user
        # provided one and it's already bound, the bound job_id wins — the
        # alias is the user's anchor. _assert_call_id_match still rejects a
        # mismatched submission so a buggy caller cannot rebind silently.
        target_id = derived_id
        if call_id is not None:
            existing_alias = self._get_by_call_id(call_id)
            if existing_alias is not None:
                _assert_call_id_match(
                    existing_alias,
                    call_id=call_id,
                    tool=tool, version=version, params_hash=ph,
                    trajectory_id=trajectory_id,
                    workflow_id=workflow_id, workflow_node_id=workflow_node_id,
                )
                target_id = existing_alias.job_id

        # Deterministic-id lookup. If the row exists, decide between
        # idempotent return and in-place re-queue.
        existing_by_id = self._get_by_job_id(target_id)
        if existing_by_id is not None:
            # If the caller now supplies a call_id but the deterministic row
            # was created without one, fill it in. Backfilling a NULL alias
            # is non-destructive; if a *different* call_id is already bound
            # we leave it (first alias wins for a given deterministic id —
            # the user can read the response's ``call_id`` field).
            if call_id is not None and existing_by_id.call_id is None:
                self._bind_call_id(existing_by_id.job_id, call_id)
                existing_by_id = self._get_by_job_id(existing_by_id.job_id) or existing_by_id
            if existing_by_id.state in (JobState.FAILED, JobState.CANCELLED):
                return self._resubmit_existing(existing_by_id.job_id)
            return existing_by_id

        ts = now_us()
        log_path = str(self.workspace.job_dir(target_id) / "stdout.log")

        # Persist params.json alongside the per-job dir for forensic access.
        params_file = self.workspace.job_dir(target_id) / "params.json"
        params_file.write_text(canonical_json(params))

        conn = self.workspace.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (
                      job_id, call_id, state, tool_name, tool_version,
                      params_json, params_hash, log_path, workspace,
                      trajectory_id, workflow_id, workflow_node_id, priority,
                      max_retries, submitted_at
                    ) VALUES (?, ?, 'submitted', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_id, call_id, tool, version,
                        canonical_json(params), ph, log_path, ws_name,
                        trajectory_id, workflow_id, workflow_node_id, prio.value,
                        int(max_retries), ts,
                    ),
                )
                conn.execute("COMMIT")
            except sqlite3.IntegrityError as e:
                conn.execute("ROLLBACK")
                # Race: a concurrent submitter inserted the same target_id
                # (deterministic collision) or the same call_id between our
                # read above and our write. Resolve through the same helpers
                # as the pre-check paths so the contract is identical
                # whichever side wins.
                if call_id is not None:
                    existing_alias = self._get_by_call_id(call_id)
                    if existing_alias is not None:
                        _assert_call_id_match(
                            existing_alias,
                            call_id=call_id,
                            tool=tool, version=version, params_hash=ph,
                            trajectory_id=trajectory_id,
                            workflow_id=workflow_id,
                            workflow_node_id=workflow_node_id,
                        )
                existing_by_id = self._get_by_job_id(target_id)
                if existing_by_id is not None:
                    if existing_by_id.state in (JobState.FAILED, JobState.CANCELLED):
                        return self._resubmit_existing(existing_by_id.job_id)
                    return existing_by_id
                raise ConflictError(
                    f"job insert failed: {e}",
                    details={"call_id": call_id, "tool": tool,
                             "job_id": target_id},
                ) from e
        finally:
            conn.close()

        return self.get(target_id)

    # ------------------------------------------------------------ resubmit

    def _resubmit_existing(self, job_id: str) -> JobRecord:
        """Re-queue an existing failed/cancelled job under its current job_id.

        Used by :meth:`submit` when the deterministic-id lookup resolves to
        a terminal-non-completed row. Atomic; clears error fields and the
        cancel-requested flag; bumps ``retry_count`` so the job's history
        survives. Goes via direct SQL because ``LEGAL_TRANSITIONS`` does
        not allow ``cancelled -> queued`` and we want to keep the
        scheduler's view of legal transitions narrower than the
        idempotency-driven re-queue path.
        """
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT state FROM jobs WHERE job_id = ?", (job_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise NotFoundError(
                        f"job {job_id!r} not found",
                        details={"job_id": job_id},
                    )
                if row["state"] not in ("failed", "cancelled"):
                    conn.execute("ROLLBACK")
                    raise ConflictError(
                        f"job {job_id} cannot be resubmitted from state {row['state']!r}",
                        details={"job_id": job_id, "state": row["state"]},
                    )
                conn.execute(
                    """
                    UPDATE jobs SET state = 'queued', queued_at = ?,
                                    retry_count = retry_count + 1,
                                    error_code = NULL, error_message = NULL,
                                    error_details_json = NULL,
                                    output_json = NULL,
                                    worker_id = NULL, gpu_ids = NULL,
                                    started_at = NULL, finished_at = NULL,
                                    cancel_requested = 0
                     WHERE job_id = ?
                    """,
                    (ts, job_id),
                )
                conn.execute("COMMIT")
            except Exception:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return self.get(job_id)

    def _get_by_job_id(self, job_id: str) -> JobRecord | None:
        """Return a job row by id, or ``None`` if it does not exist."""
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_record(row) if row else None

    def _bind_call_id(self, job_id: str, call_id: str) -> None:
        """Atomically bind a ``call_id`` alias to ``job_id`` IF unbound.

        Used by :meth:`submit` when the deterministic-id lookup matches an
        existing row that has no alias yet. The ``WHERE call_id IS NULL``
        clause makes this a no-op (instead of an integrity error) when the
        row already has any alias — first alias wins.
        """
        conn = self.workspace.connect()
        try:
            try:
                conn.execute(
                    "UPDATE jobs SET call_id = ? "
                    "WHERE job_id = ? AND call_id IS NULL",
                    (call_id, job_id),
                )
            except sqlite3.IntegrityError:
                # Another row already owns this call_id; leave existing
                # binding alone. The caller's _assert_call_id_match in
                # the alias pre-check is the user-facing answer.
                pass
        finally:
            conn.close()

    # ------------------------------------------------------------ transition

    def transition(
        self,
        id_or_call: str,
        new_state: str | JobState,
        **fields: Any,
    ) -> JobRecord:
        """Move a job to ``new_state``; enforce §5 legal transitions."""
        target = JobState(new_state) if not isinstance(new_state, JobState) else new_state
        rec = self.get(id_or_call)
        cur = rec.state
        if target not in LEGAL_TRANSITIONS.get(cur, frozenset()):
            raise ConflictError(
                f"illegal transition {cur.value} -> {target.value} for job {rec.job_id}",
                details={"job_id": rec.job_id, "from": cur.value, "to": target.value},
            )

        sets: list[str] = ["state = ?"]
        args: list[Any] = [target.value]

        ts = now_us()
        if target is JobState.QUEUED:
            sets.append("queued_at = ?")
            args.append(ts)
            if cur is JobState.FAILED:
                sets.append("retry_count = retry_count + 1")
        elif target is JobState.RUNNING:
            sets.append("started_at = ?")
            args.append(ts)
        elif target in TERMINAL_STATES:
            sets.append("finished_at = ?")
            args.append(ts)

        # Optional named fields.
        for key, col in (
            ("worker_id", "worker_id"),
            ("error_code", "error_code"),
            ("error_message", "error_message"),
            ("cache_hit", "cache_hit"),
        ):
            if key in fields:
                sets.append(f"{col} = ?")
                val = fields[key]
                if key == "cache_hit":
                    val = 1 if val else 0
                args.append(val)

        if "error_details" in fields:
            # Codex round-3 Re1: structured violation details, e.g. the
            # ``violations`` list from a schema-output failure. JSON-encoded
            # for storage; ``None`` clears the column on requeue.
            sets.append("error_details_json = ?")
            details_val = fields["error_details"]
            args.append(
                canonical_json(details_val) if details_val is not None else None
            )

        if "gpu_ids" in fields:
            sets.append("gpu_ids = ?")
            args.append(_gpu_ids_to_str(fields["gpu_ids"]))

        if "output" in fields:
            sets.append("output_json = ?")
            out = fields["output"]
            args.append(canonical_json(out) if out is not None else None)

        args.append(rec.job_id)
        sql = f"UPDATE jobs SET {', '.join(sets)} WHERE job_id = ?"

        conn = self.workspace.connect()
        try:
            conn.execute(sql, args)
        finally:
            conn.close()

        return self.get(rec.job_id)

    # ------------------------------------------------------------ read

    def get(self, id_or_call: str) -> JobRecord:
        """Look up a job by job_id or call_id."""
        kind, val = parse_id(id_or_call)
        conn = self.workspace.connect()
        try:
            col = "job_id" if kind == "job_id" else "call_id"
            row = conn.execute(
                f"SELECT * FROM jobs WHERE {col} = ?", (val,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise NotFoundError(
                f"job {id_or_call!r} not found",
                details={"id": id_or_call, "kind": kind},
            )
        return self._row_to_record(row)

    def _get_by_call_id(self, call_id: str) -> JobRecord | None:
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE call_id = ?", (call_id,)
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_record(row) if row else None

    def list(
        self,
        *,
        state: str | JobState | None = None,
        tool: str | None = None,
        workflow_id: str | None = None,
        trajectory_id: str | None = None,
        since: int | None = None,
        limit: int = 100,
        order: str = "submitted_at_desc",
    ) -> list[JobRecord]:
        """List jobs with optional filters. Default order: newest first."""
        where: list[str] = []
        args: list[Any] = []
        if state is not None:
            s = state.value if isinstance(state, JobState) else state
            where.append("state = ?")
            args.append(s)
        if tool is not None:
            where.append("tool_name = ?")
            args.append(tool)
        if workflow_id is not None:
            where.append("workflow_id = ?")
            args.append(workflow_id)
        if trajectory_id is not None:
            where.append("trajectory_id = ?")
            args.append(trajectory_id)
        if since is not None:
            where.append("submitted_at >= ?")
            args.append(int(since))

        order_sql = {
            "submitted_at_desc": "submitted_at DESC",
            "submitted_at_asc": "submitted_at ASC",
        }.get(order)
        if order_sql is None:
            raise InvalidParamsError(
                f"unknown order {order!r}",
                details={"got": order, "allowed": ["submitted_at_desc", "submitted_at_asc"]},
            )
        sql = "SELECT * FROM jobs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order_sql} LIMIT ?"
        args.append(int(limit))

        conn = self.workspace.connect()
        try:
            rows = conn.execute(sql, args).fetchall()
        finally:
            conn.close()
        return [self._row_to_record(r) for r in rows]

    # ------------------------------------------------------------ cancel

    def cancel(self, id_or_call: str) -> JobRecord:
        """Legacy: alias for :meth:`request_cancel`."""
        return self.request_cancel(id_or_call)

    def request_cancel(self, id_or_call: str) -> JobRecord:
        """Set the cancel_requested flag; queued/submitted -> cancelled immediately."""
        rec = self.get(id_or_call)
        conn = self.workspace.connect()
        try:
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?",
                (rec.job_id,),
            )
        finally:
            conn.close()
        if rec.state is JobState.SUBMITTED:
            self.transition(rec.job_id, JobState.QUEUED)
            return self.transition(rec.job_id, JobState.CANCELLED)
        if rec.state is JobState.QUEUED:
            return self.transition(rec.job_id, JobState.CANCELLED)
        if rec.state is JobState.RUNNING:
            return self.get(rec.job_id)
        raise ConflictError(
            f"job {rec.job_id} cannot be cancelled from state {rec.state.value}",
            details={"job_id": rec.job_id, "state": rec.state.value},
        )

    def is_cancel_requested(self, job_id: str) -> bool:
        """Worker polls this inside RunContext.is_cancelled."""
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return False
        return bool(row["cancel_requested"])

    def cancel_running_for_shutdown(self, *, reason: str = "daemon_stopped") -> list[str]:
        """Mark running jobs cancelled when their owning daemon is stopped."""
        ts = now_us()
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                "SELECT job_id FROM jobs WHERE state = 'running'",
            ).fetchall()
            job_ids = [str(r["job_id"]) for r in rows]
            if job_ids:
                conn.execute(
                    """
                    UPDATE jobs
                       SET state = 'cancelled',
                           cancel_requested = 1,
                           finished_at = ?,
                           error_code = ?,
                           error_message = ?,
                           error_details_json = ?
                     WHERE state = 'running'
                    """,
                    (
                        ts,
                        "E_CANCELLED",
                        "daemon stopped before job completed",
                        canonical_json({"reason": reason}),
                    ),
                )
        finally:
            conn.close()
        return job_ids

    # ------------------------------------------------------------ claim/retry

    def claim_next_queued(
        self,
        worker_id: int,
        gpu_ids: list[int],
        *,
        job_id: str | None = None,
    ) -> JobRecord | None:
        """Atomically claim a queued job; mark it ``running``.

        When ``job_id`` is provided the scheduler is asking to claim a *specific*
        queued job (used by the GPU-aware placement pass — the scheduler has
        already decided which worker should get which job, so the global
        priority+FIFO winner is not necessarily the right answer).

        When ``job_id`` is ``None`` (the historical behaviour) the highest
        priority oldest queued job is claimed.
        """
        priority_order = {"high": 0, "normal": 1, "low": 2}
        ts = now_us()
        gpu_str = _gpu_ids_to_str(gpu_ids)
        conn = self.workspace.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if job_id is not None:
                    row = conn.execute(
                        "SELECT job_id FROM jobs WHERE job_id = ? AND state = 'queued'",
                        (job_id,),
                    ).fetchone()
                    if row is None:
                        conn.execute("COMMIT")
                        return None
                    target_id = row["job_id"]
                else:
                    rows = conn.execute(
                        "SELECT job_id, priority FROM jobs WHERE state = 'queued'"
                    ).fetchall()
                    if not rows:
                        conn.execute("COMMIT")
                        return None
                    rows_sorted = sorted(
                        rows,
                        key=lambda r: (priority_order.get(r["priority"], 99),
                                       r["job_id"]),
                    )
                    target_id = rows_sorted[0]["job_id"]
                conn.execute(
                    """
                    UPDATE jobs SET state = 'running', worker_id = ?, gpu_ids = ?,
                                    started_at = ?
                     WHERE job_id = ? AND state = 'queued'
                    """,
                    (worker_id, gpu_str, ts, target_id),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return self.get(target_id)

    def requeue_failed(self, job_id: str) -> JobRecord:
        """Atomically re-queue a ``failed`` job, clearing stale per-run fields.

        Raises :class:`ConflictError` if the job is not in ``failed`` state, or
        if ``retry_count`` would exceed ``max_retries``. Used by the supervisor
        for auto-retry on transient errors and by the manual ``vpa retry``
        command.
        """
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT state, retry_count, max_retries FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise NotFoundError(
                        f"job {job_id!r} not found",
                        details={"job_id": job_id},
                    )
                state = row["state"]
                retry_count = int(row["retry_count"])
                max_retries = int(row["max_retries"])
                if state != "failed":
                    conn.execute("ROLLBACK")
                    raise ConflictError(
                        f"job {job_id} cannot be requeued from state {state!r}",
                        details={"job_id": job_id, "state": state,
                                 "reason": "not_failed"},
                    )
                if retry_count >= max_retries:
                    conn.execute("ROLLBACK")
                    raise ConflictError(
                        f"job {job_id} exceeded max_retries={max_retries}",
                        details={
                            "job_id": job_id,
                            "retry_count": retry_count,
                            "max_retries": max_retries,
                            "reason": "max_retries_exceeded",
                        },
                    )
                conn.execute(
                    """
                    UPDATE jobs SET state = 'queued', queued_at = ?,
                                    retry_count = retry_count + 1,
                                    error_code = NULL, error_message = NULL,
                                    error_details_json = NULL,
                                    worker_id = NULL, gpu_ids = NULL,
                                    started_at = NULL, finished_at = NULL
                     WHERE job_id = ?
                    """,
                    (ts, job_id),
                )
                conn.execute("COMMIT")
            except Exception:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return self.get(job_id)

    def requeue_after_worker_reject(self, job_id: str) -> JobRecord:
        """Re-queue a ``failed`` job WITHOUT bumping ``retry_count``.

        Used by the supervisor when a worker rejected a dispatch with a
        transient placement error (currently only ``E_VRAM_EXHAUSTED``):
        the disagreement between scheduler and worker is *our* bug, not
        the user's, so we should not consume one of their retry budget
        slots. Codex round-2 RC5 (VRAM disagreement self-healing).
        """
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT state FROM jobs WHERE job_id = ?",
                    (job_id,),
                ).fetchone()
                if row is None:
                    conn.execute("ROLLBACK")
                    raise NotFoundError(
                        f"job {job_id!r} not found",
                        details={"job_id": job_id},
                    )
                if row["state"] != "failed":
                    conn.execute("ROLLBACK")
                    raise ConflictError(
                        f"job {job_id} cannot be requeued from state {row['state']!r}",
                        details={"job_id": job_id, "state": row["state"]},
                    )
                conn.execute(
                    """
                    UPDATE jobs SET state = 'queued', queued_at = ?,
                                    error_code = NULL, error_message = NULL,
                                    error_details_json = NULL,
                                    worker_id = NULL, gpu_ids = NULL,
                                    started_at = NULL, finished_at = NULL
                     WHERE job_id = ?
                    """,
                    (ts, job_id),
                )
                conn.execute("COMMIT")
            except Exception:
                with contextlib.suppress(sqlite3.OperationalError):
                    conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()
        return self.get(job_id)

    def bump_retry(self, job_id: str) -> int:
        """Increment retry_count; raise ConflictError if it would exceed max_retries."""
        rec = self.get(job_id)
        new_count = rec.retry_count + 1
        if new_count > rec.max_retries:
            raise ConflictError(
                f"job {job_id} exceeded max_retries={rec.max_retries}",
                details={"job_id": job_id, "retry_count": new_count,
                         "max_retries": rec.max_retries},
            )
        conn = self.workspace.connect()
        try:
            conn.execute(
                "UPDATE jobs SET retry_count = ? WHERE job_id = ?",
                (new_count, job_id),
            )
        finally:
            conn.close()
        return new_count

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        keys = set(row.keys())
        # error_details_json was added in user_version 3; older rows or
        # rows from a pre-migration read may not have the column.
        try:
            edj = row["error_details_json"]
        except (IndexError, KeyError):
            edj = None
        error_details = json.loads(edj) if edj else None
        return JobRecord(
            job_id=row["job_id"],
            call_id=row["call_id"],
            state=JobState(row["state"]),
            tool_name=row["tool_name"],
            tool_version=row["tool_version"],
            params=json.loads(row["params_json"]),
            params_hash=row["params_hash"],
            output=json.loads(row["output_json"]) if row["output_json"] else None,
            error_code=row["error_code"],
            error_message=row["error_message"],
            error_details=error_details,
            log_path=row["log_path"],
            workspace=row["workspace"],
            trajectory_id=row["trajectory_id"] if "trajectory_id" in keys else None,
            workflow_id=row["workflow_id"],
            workflow_node_id=row["workflow_node_id"],
            worker_id=row["worker_id"],
            gpu_ids=_gpu_ids_from_str(row["gpu_ids"]),
            priority=Priority(row["priority"]),
            retry_count=int(row["retry_count"]),
            max_retries=int(row["max_retries"]),
            submitted_at=int(row["submitted_at"]),
            queued_at=row["queued_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            cache_hit=bool(row["cache_hit"]),
        )


# ---------------------------------------------------------------- ArtifactStore


class TrajectoryStore:
    """CRUD and accounting for one user-query trajectory."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def create(self, *, user_query: str, objective: str | None = None) -> TrajectoryRecord:
        user_query = str(user_query or "").strip()
        if not user_query:
            raise InvalidParamsError(
                "trajectory user_query must be non-empty",
                details={"field": "user_query"},
            )
        trajectory_id = f"traj_{uuid.uuid4().hex}"
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO trajectories (
                  trajectory_id, workspace, user_query, objective, state,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'open', ?, ?)
                """,
                (trajectory_id, self.workspace.name, user_query, objective, ts, ts),
            )
        finally:
            conn.close()
        return self.get(trajectory_id)

    def get(self, trajectory_id: str) -> TrajectoryRecord:
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise NotFoundError(
                f"trajectory {trajectory_id!r} not found",
                details={"trajectory_id": trajectory_id},
            )
        return self._row_to_record(row)

    def list_jobs(self, trajectory_id: str) -> list[JobRecord]:
        self.get(trajectory_id)
        return JobStore(self.workspace).list(
            trajectory_id=trajectory_id,
            limit=10000,
            order="submitted_at_asc",
        )

    def list_workflows(self, trajectory_id: str) -> list[WorkflowRecord]:
        self.get(trajectory_id)
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                """
                SELECT * FROM workflows
                 WHERE trajectory_id = ?
                 ORDER BY submitted_at ASC, workflow_id ASC
                """,
                (trajectory_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._workflow_from_row(r) for r in rows]

    def record_report(
        self,
        trajectory_id: str,
        report_path: str,
        *,
        source_id: str | None = None,
        format: str = "html",
        replace: bool = False,
    ) -> TrajectoryRecord:
        rec = self.get(trajectory_id)
        preflight = self.preflight_report(
            trajectory_id,
            report_path,
            format=format,
            replace=replace,
        )
        final_submit_job_id = preflight["final_submit_job_id"]
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO trajectory_reports (
                  trajectory_id, report_path, source_id, format,
                  final_submit_job_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(trajectory_id) DO UPDATE SET
                  report_path = excluded.report_path,
                  source_id = excluded.source_id,
                  format = excluded.format,
                  final_submit_job_id = excluded.final_submit_job_id,
                  created_at = excluded.created_at
                """,
                (
                    trajectory_id,
                    report_path,
                    source_id or trajectory_id,
                    format,
                    final_submit_job_id,
                    ts,
                ),
            )
        finally:
            conn.close()
        status = self.status(trajectory_id)
        completed = bool(status["complete"])
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                UPDATE trajectories
                   SET report_path = ?,
                       final_submit_job_id = ?,
                       state = ?,
                       updated_at = ?,
                       completed_at = ?
                 WHERE trajectory_id = ?
                """,
                (
                    report_path,
                    status["final_submit_job_id"],
                    "completed" if completed else "open",
                    ts,
                    ts if completed else None,
                    trajectory_id,
                ),
            )
        finally:
            conn.close()
        return self.get(rec.trajectory_id)

    def preflight_report(
        self,
        trajectory_id: str,
        report_path: str,
        *,
        format: str = "html",
        replace: bool = False,
    ) -> dict[str, Any]:
        status = self.status(trajectory_id)
        final_submit_job_id = status["final_submit_job_id"]
        existing = self.report_record(trajectory_id)
        if existing is not None and not replace:
            same_path = existing["report_path"] == report_path
            same_final = existing.get("final_submit_job_id") == final_submit_job_id
            if not (same_path and same_final):
                raise ConflictError(
                    f"trajectory {trajectory_id!r} already has a final report",
                    details={
                        "trajectory_id": trajectory_id,
                        "existing_report_path": existing["report_path"],
                        "submitted_report_path": report_path,
                        "existing_final_submit_job_id": existing.get("final_submit_job_id"),
                        "submitted_final_submit_job_id": final_submit_job_id,
                        "existing_format": existing.get("format"),
                        "submitted_format": format,
                        "hint": "rerun with --replace-report to replace the recorded final report",
                    },
                )
        return {
            "trajectory_id": trajectory_id,
            "report_path": report_path,
            "format": format,
            "replace": replace,
            "existing_report": existing,
            "final_submit_job_id": final_submit_job_id,
            "idempotent": bool(
                existing is not None
                and existing["report_path"] == report_path
                and existing.get("final_submit_job_id") == final_submit_job_id
            ),
        }

    def report_record(self, trajectory_id: str) -> dict[str, Any] | None:
        self.get(trajectory_id)
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM trajectory_reports WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return {
            "trajectory_id": row["trajectory_id"],
            "report_path": row["report_path"],
            "source_id": row["source_id"],
            "format": row["format"],
            "final_submit_job_id": row["final_submit_job_id"],
            "created_at": int(row["created_at"]),
        }

    def status(
        self,
        trajectory_id: str,
        *,
        report_path: str | None = None,
    ) -> dict[str, Any]:
        rec = self.get(trajectory_id)
        jobs = self.list_jobs(trajectory_id)
        workflows = self.list_workflows(trajectory_id)
        unfinished_jobs = [j for j in jobs if j.state not in TERMINAL_STATES]
        unfinished_workflows = [
            w for w in workflows if w.state not in {"completed", "failed", "cancelled"}
        ]
        submit_jobs = [
            j for j in jobs
            if j.tool_name == "submit_candidate"
            and j.state is JobState.COMPLETED
            and isinstance(j.output, dict)
            and j.output.get("terminal") is True
        ]
        final_submit = submit_jobs[-1] if submit_jobs else None
        report_record = self.report_record(trajectory_id)
        legacy_report_count = 1 if report_record is None and rec.report_path else 0
        report_count = (1 if report_record is not None else 0) + legacy_report_count
        report_final_submit_job_id = (
            report_record.get("final_submit_job_id") if report_record else None
        )
        report_matches_final_submit = (
            final_submit is not None
            and report_record is not None
            and report_final_submit_job_id == final_submit.job_id
        )
        stale_report = (
            final_submit is not None
            and report_record is not None
            and report_final_submit_job_id != final_submit.job_id
        )
        effective_report = (
            report_path
            or (report_record["report_path"] if report_record is not None else None)
            or rec.report_path
        )
        complete = (
            not unfinished_jobs
            and not unfinished_workflows
            and final_submit is not None
            and report_count == 1
            and bool(effective_report)
            and report_matches_final_submit
        )
        execution_all_terminal = not unfinished_jobs and not unfinished_workflows
        return {
            "trajectory_id": rec.trajectory_id,
            "workspace": rec.workspace,
            "user_query": rec.user_query,
            "objective": rec.objective,
            "state": "completed" if complete else "open",
            "execution_all_terminal": execution_all_terminal,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "completed_at": rec.completed_at,
            "job_count": len(jobs),
            "workflow_count": len(workflows),
            "unfinished_job_ids": [j.job_id for j in unfinished_jobs],
            "unfinished_workflow_ids": [w.workflow_id for w in unfinished_workflows],
            "final_submit_job_id": final_submit.job_id if final_submit else None,
            "final_submit_outcome": (
                final_submit.output.get("outcome")
                if final_submit and isinstance(final_submit.output, dict)
                else None
            ),
            "report_path": effective_report,
            "report_count": report_count,
            "report_final_submit_job_id": report_final_submit_job_id,
            "report_matches_final_submit": report_matches_final_submit,
            "stale_report": stale_report,
            "report_recorded_at": (
                report_record["created_at"]
                if report_record is not None
                else (rec.updated_at if rec.report_path else None)
            ),
            "complete": complete,
            "jobs": [self._job_summary(j) for j in jobs],
            "workflows": [self._workflow_summary(w) for w in workflows],
        }

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TrajectoryRecord:
        return TrajectoryRecord(
            trajectory_id=row["trajectory_id"],
            workspace=row["workspace"],
            user_query=row["user_query"],
            objective=row["objective"],
            state=row["state"],
            final_submit_job_id=row["final_submit_job_id"],
            report_path=row["report_path"],
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
            completed_at=row["completed_at"],
        )

    @staticmethod
    def _workflow_from_row(row: sqlite3.Row) -> WorkflowRecord:
        keys = set(row.keys())
        return WorkflowRecord(
            workflow_id=row["workflow_id"],
            name=row["name"],
            version=row["version"],
            descriptor=row["descriptor"],
            params=json.loads(row["params_json"]),
            trajectory_id=row["trajectory_id"] if "trajectory_id" in keys else None,
            state=row["state"],
            submitted_at=int(row["submitted_at"]),
            finished_at=row["finished_at"],
        )

    @staticmethod
    def _job_summary(job: JobRecord) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "tool": job.tool_name,
            "state": job.state.value,
            "workflow_id": job.workflow_id,
            "workflow_node_id": job.workflow_node_id,
            "submitted_at": job.submitted_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error_code": job.error_code,
        }

    @staticmethod
    def _workflow_summary(workflow: WorkflowRecord) -> dict[str, Any]:
        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "version": workflow.version,
            "state": workflow.state,
            "submitted_at": workflow.submitted_at,
            "finished_at": workflow.finished_at,
        }


# ---------------------------------------------------------------- ArtifactStore


class ArtifactStore:
    """Content-addressed artifact pool keyed by sha256."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    # ------------------------------------------------------------ register

    def register(
        self,
        producer_job_id: str,
        field: str,
        file_path: str | Path,
        *,
        kind: str,
        mime: str | None = None,
        pinned: bool = False,
    ) -> ArtifactRecord:
        """Hash + copy a file into the dedup pool and link to ``producer_job_id``."""
        src = Path(file_path)
        if not src.is_file():
            raise NotFoundError(
                f"artifact source {src} is not a file",
                details={"path": str(src)},
            )
        digest, size = _sha256_file(src)
        ext = src.suffix.lstrip(".")
        pool_rel = Path("sha256") / digest[:2] / digest[2:4] / (
            digest[4:] + (f".{ext}" if ext else "")
        )
        pool_path = self.workspace.artifacts_dir / pool_rel
        pool_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE sha256 = ?", (digest,)
            ).fetchone()
            ts = now_us()
            if row is None:
                if not pool_path.exists():
                    shutil.copyfile(src, pool_path)
                artifact_id = f"art_{uuid.uuid4().hex}"
                conn.execute(
                    """
                    INSERT INTO artifacts (
                      artifact_id, sha256, kind, mime, size_bytes,
                      pool_path, producer_job, created_at, last_used_at, pinned
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id, digest, kind, mime, size,
                        str(pool_path), producer_job_id, ts, ts,
                        1 if pinned else 0,
                    ),
                )
            else:
                artifact_id = row["artifact_id"]
                conn.execute(
                    "UPDATE artifacts SET last_used_at = ?, pinned = MAX(pinned, ?) "
                    "WHERE artifact_id = ?",
                    (ts, 1 if pinned else 0, artifact_id),
                )

            # job_artifacts link (idempotent on (job_id, field)).
            conn.execute(
                "INSERT OR REPLACE INTO job_artifacts (job_id, field, artifact_id) "
                "VALUES (?, ?, ?)",
                (producer_job_id, field, artifact_id),
            )
        finally:
            conn.close()

        return self.get(artifact_id)

    # ------------------------------------------------------------ read

    def get(self, artifact_id: str) -> ArtifactRecord:
        """Fetch an artifact row by id."""
        conn = self.workspace.connect()
        try:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise NotFoundError(
                f"artifact {artifact_id!r} not found",
                details={"artifact_id": artifact_id},
            )
        return self._row_to_record(row)

    def for_job(self, job_id: str) -> dict[str, ArtifactRecord]:
        """Return the artifacts produced by ``job_id``, keyed by field name."""
        conn = self.workspace.connect()
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
        out: dict[str, ArtifactRecord] = {}
        for r in rows:
            out[r["field"]] = self._row_to_record(r)
        return out

    # ------------------------------------------------------------ prune

    def prune(
        self,
        *,
        older_than_s: int,
        dry_run: bool = False,
    ) -> list[ArtifactRecord]:
        """LRU-prune unpinned artifacts whose ``last_used_at`` precedes the cutoff."""
        cutoff = now_us() - int(older_than_s) * 1_000_000
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE pinned = 0 AND last_used_at < ? "
                "ORDER BY last_used_at ASC",
                (cutoff,),
            ).fetchall()
            victims = [self._row_to_record(r) for r in rows]
            if not dry_run:
                for v in victims:
                    p = Path(v.pool_path)
                    if p.exists():
                        try:
                            p.unlink()
                        except OSError:
                            pass
                    conn.execute(
                        "DELETE FROM artifacts WHERE artifact_id = ?",
                        (v.artifact_id,),
                    )
        finally:
            conn.close()
        return victims

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
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


# ---------------------------------------------------------------- RunStore


class RunStore:
    """CRUD for the ``runs`` table — one row per (job, attempt) execution.

    The supervisor calls :meth:`start_run` when it dispatches a job to a worker
    and :meth:`finish_run` when the worker has reported a terminal state.
    Architecture v1 §9 defines this table; until Phase 7 it existed in the
    schema but was never written.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def start_run(
        self,
        job_id: str,
        attempt: int,
        worker_id: int,
        gpu_ids: list[int] | None = None,
    ) -> str:
        run_id = f"run_{uuid.uuid4().hex}"
        gpu_str = _gpu_ids_to_str(gpu_ids) or ""
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                INSERT INTO runs (
                  run_id, job_id, attempt, worker_id, gpu_ids,
                  started_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, job_id, int(attempt), int(worker_id), gpu_str, ts),
            )
        finally:
            conn.close()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        exit_state: str,
        duration_ms: int | None = None,
        vram_peak_mib: int | None = None,
    ) -> None:
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                UPDATE runs SET finished_at = ?, exit_state = ?,
                                duration_ms = ?, vram_peak_mib = ?
                 WHERE run_id = ?
                """,
                (ts, exit_state, duration_ms, vram_peak_mib, run_id),
            )
        finally:
            conn.close()

    def finish_open_for_job(self, job_id: str, *, exit_state: str) -> None:
        ts = now_us()
        conn = self.workspace.connect()
        try:
            conn.execute(
                """
                UPDATE runs
                   SET finished_at = ?,
                       exit_state = ?,
                       duration_ms = CASE
                         WHEN duration_ms IS NULL THEN (? - started_at) / 1000
                         ELSE duration_ms
                       END
                 WHERE job_id = ?
                   AND finished_at IS NULL
                """,
                (ts, exit_state, ts, job_id),
            )
        finally:
            conn.close()

    def for_job(self, job_id: str) -> list[RunRecord]:
        conn = self.workspace.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM runs WHERE job_id = ? ORDER BY started_at ASC",
                (job_id,),
            ).fetchall()
        finally:
            conn.close()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            job_id=row["job_id"],
            attempt=int(row["attempt"]),
            worker_id=int(row["worker_id"]),
            gpu_ids=_gpu_ids_from_str(row["gpu_ids"]),
            started_at=int(row["started_at"]),
            finished_at=row["finished_at"],
            exit_state=row["exit_state"],
            vram_peak_mib=row["vram_peak_mib"],
            duration_ms=row["duration_ms"],
        )


# Re-exports for convenience.
__all__ = [
    "ArtifactStore",
    "JobStore",
    "RunStore",
    "TrajectoryStore",
    "Workspace",
    "canonical_json",
    "now_us",
    "params_hash",
]
