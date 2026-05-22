"""Codex round-3 Re2: warm RPC surfaces per-worker busy state + wait time.

Before this fix, ``vpa daemon warm`` blocked behind the worker's
ThreadPoolExecutor with no visibility into *why*. Operators couldn't
tell a slow cold-load from a queued backlog. The fix is additive: each
per-worker entry in the warm response carries:

  * ``worker_state_before``: "busy" or "idle" — snapshot just before the
    warm RPC is sent;
  * ``waited_for_busy_worker_ms``: wall-clock spent waiting for the
    warm RPC to return.

This unit test drives the supervisor's :meth:`_op_warm` against an
in-process fake socket so we can assert the additive fields are
populated without spawning workers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from vibeproteinagent.workers import ipc
from vibeproteinagent.workers.supervisor import Supervisor


def _bare_supervisor() -> Supervisor:
    sup = Supervisor.__new__(Supervisor)
    sup._log = SimpleNamespace(  # type: ignore[attr-defined]
        info=lambda *a, **k: None,
        warn=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    sup._worker_state = {}
    sup._submit_locks = {}
    return sup


async def _fake_warm_worker(
    socket_path: Path,
    *,
    delay_s: float,
    body: dict[str, Any] | None = None,
) -> asyncio.AbstractServer:
    """Spin up a UDS server that mimics a worker's warm response."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        req = await ipc.unpack(reader)
        await asyncio.sleep(delay_s)
        resp = ipc.make_response(
            req.get("id", ""),
            body or {
                "tool": "noop",
                "version": "1.0.0",
                "vram_used_mib": 0,
                "free_vram_mib": 1024,
                "pin_warm": False,
            },
        )
        await ipc.send(writer, resp)
        writer.close()

    return await asyncio.start_unix_server(handle, path=str(socket_path))


def _make_handle(gpu_id: int, sock: Path, *, busy: bool) -> SimpleNamespace:
    """Build a stand-in for :class:`_WorkerHandle` consumed by ``_op_warm``."""
    return SimpleNamespace(worker_id=gpu_id, gpu_id=gpu_id, socket_path=sock, busy=busy)


def test_warm_records_busy_state_and_wait(tmp_path: Path) -> None:
    """A busy worker's warm result includes worker_state_before='busy' + ms > 0."""

    async def run() -> dict[str, Any]:
        sock = tmp_path / "worker.sock"
        server = await _fake_warm_worker(sock, delay_s=0.2)
        try:
            sup = _bare_supervisor()
            sup.workers = [_make_handle(0, sock, busy=True)]  # type: ignore[attr-defined]
            resp = await sup._op_warm(
                "mid",
                {"op": "warm", "tool": "noop", "version_spec": "*"},
            )
            return resp
        finally:
            server.close()
            await server.wait_closed()

    resp = asyncio.run(run())
    warmed = resp["body"]["warmed"]
    assert len(warmed) == 1
    entry = warmed[0]
    assert entry["success"] is True
    assert entry["worker_state_before"] == "busy"
    assert entry["waited_for_busy_worker_ms"] >= 100  # ≥ 0.1 s actual wait


def test_warm_records_idle_state_when_worker_not_busy(tmp_path: Path) -> None:
    async def run() -> dict[str, Any]:
        sock = tmp_path / "worker.sock"
        server = await _fake_warm_worker(sock, delay_s=0.01)
        try:
            sup = _bare_supervisor()
            sup.workers = [_make_handle(0, sock, busy=False)]  # type: ignore[attr-defined]
            return await sup._op_warm(
                "mid",
                {"op": "warm", "tool": "noop", "version_spec": "*"},
            )
        finally:
            server.close()
            await server.wait_closed()

    resp = asyncio.run(run())
    entry = resp["body"]["warmed"][0]
    assert entry["worker_state_before"] == "idle"
    # Field still present + non-negative.
    assert entry["waited_for_busy_worker_ms"] >= 0


def test_warm_failure_keeps_additive_fields(tmp_path: Path) -> None:
    """Even when warm fails (no socket), the additive fields are present."""

    async def run() -> dict[str, Any]:
        sock = tmp_path / "missing.sock"
        sup = _bare_supervisor()
        sup.workers = [_make_handle(7, sock, busy=True)]  # type: ignore[attr-defined]
        return await sup._op_warm(
            "mid", {"op": "warm", "tool": "noop", "version_spec": "*"},
        )

    resp = asyncio.run(run())
    entry = resp["body"]["warmed"][0]
    assert entry["success"] is False
    assert entry["worker_state_before"] == "busy"
    assert entry["waited_for_busy_worker_ms"] == 0
