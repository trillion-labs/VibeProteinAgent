from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from vibeproteinagent.core.errors import (
    DaemonUnavailableError,
    ErrorCode,
    VPAError,
)
from vibeproteinagent.workers import ipc, paths


class SupervisorClient:
    """Synchronous client for talking to the supervisor over UDS."""

    def __init__(self, socket_path: Path | str | None = None,
                 pid_file: Path | str | None = None) -> None:
        self.socket_path = Path(socket_path) if socket_path else paths.supervisor_socket()
        self.pid_file = Path(pid_file) if pid_file else paths.supervisor_pid_file()

    # --------------------------------------------------------------- presence

    def pid(self) -> int | None:
        if not self.pid_file.exists():
            return None
        try:
            return int(self.pid_file.read_text().strip())
        except (ValueError, OSError):
            return None

    def is_running(self) -> bool:
        """True iff pidfile + socket exist and process is alive."""
        pid = self.pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return self.socket_path.exists()

    # --------------------------------------------------------------- request helpers

    def _call(self, body: dict[str, Any], *, timeout_s: float = 10.0) -> dict[str, Any]:
        async def _go() -> dict[str, Any]:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_unix_connection(str(self.socket_path)),
                    timeout=timeout_s,
                )
            except (FileNotFoundError, ConnectionRefusedError) as e:
                raise DaemonUnavailableError(
                    f"supervisor socket {self.socket_path} not reachable: {e}",
                    details={"socket": str(self.socket_path)},
                ) from e
            try:
                req = ipc.make_request(body)
                await ipc.send(writer, req)
                resp = await asyncio.wait_for(ipc.unpack(reader), timeout=timeout_s)
                if resp is None:
                    raise DaemonUnavailableError("supervisor closed connection")
                err = resp.get("error")
                if err:
                    exc = VPAError(
                        err.get("message", "supervisor error"),
                        details=err.get("details", {}),
                    )
                    exc.code = err.get("code") or ErrorCode.INTERNAL
                    raise exc
                return resp.get("body", {}) or {}
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

        return asyncio.run(_go())

    # --------------------------------------------------------------- API

    def submit(self, job_id: str) -> dict[str, Any]:
        return self._call({"op": "submit", "job_id": job_id})

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._call({"op": "cancel", "job_id": job_id})

    def status(self, job_id: str) -> dict[str, Any]:
        return self._call({"op": "status", "job_id": job_id})

    def worker_info(self) -> dict[str, Any]:
        return self._call({"op": "worker_info"})

    def warm(
        self,
        tool: str,
        *,
        version_spec: str = "*",
        gpu_ids: list[int] | None = None,
        timeout_s: float = 300.0,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "op": "warm",
            "tool": tool,
            "version_spec": version_spec,
        }
        if gpu_ids is not None:
            body["gpu_ids"] = [int(g) for g in gpu_ids]
        return self._call(body, timeout_s=timeout_s)

    def shutdown(self) -> dict[str, Any]:
        try:
            return self._call({"op": "shutdown"})
        except DaemonUnavailableError:
            return {"ok": False, "reason": "not running"}

    # --------------------------------------------------------------- wait helpers

    def wait_until_running(self, *, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.is_running():
                return True
            time.sleep(0.1)
        return False

    def wait_until_stopped(self, *, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.is_running():
                return True
            time.sleep(0.1)
        return False


__all__ = ["ErrorCode", "SupervisorClient"]
