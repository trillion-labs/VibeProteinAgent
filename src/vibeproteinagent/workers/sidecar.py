from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack

from vibeproteinagent.core.errors import DependencyError, TimeoutError_, VPAError
from vibeproteinagent.workers import ipc

_VPA_SRC = Path(__file__).resolve().parents[2]


def with_vpa_src_pythonpath(env: dict[str, str]) -> dict[str, str]:
    """Return ``env`` with this checkout's ``src`` path prepended.

    Compatibility sidecars often run in dedicated envs that intentionally do not
    inherit the parent ``PYTHONPATH``. They still need to import the lightweight
    VPA sidecar child and runner modules from this checkout. Prepending only the
    repo ``src`` path keeps that explicit without leaking the whole parent env.
    """
    out = dict(env)
    src = str(_VPA_SRC)
    existing = out.get("PYTHONPATH")
    if existing:
        parts = existing.split(os.pathsep)
        if src not in parts:
            out["PYTHONPATH"] = os.pathsep.join([src, *parts])
    else:
        out["PYTHONPATH"] = src
    return out


class PersistentPythonSidecar:
    """Long-lived Python subprocess for dependency-isolated adapters.

    This is the "compatibility island" primitive: the main VPA worker keeps the
    scheduler, cache, artifacts, and adapter lifecycle in the shared ``vpa`` env,
    while tools with unavoidable dependency/licence conflicts run inside a
    dedicated Python interpreter that stays warm across adapter calls.
    """

    def __init__(
        self,
        *,
        python: Path,
        env: dict[str, str],
        name: str,
        startup_timeout_s: float = 30.0,
        cwd: Path | str | None = None,
    ) -> None:
        self.python = Path(python)
        self.env = with_vpa_src_pythonpath(env)
        self.name = name
        self.startup_timeout_s = float(startup_timeout_s)
        self.cwd = Path(cwd) if cwd is not None else None
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()

    def health(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        return self.request({"op": "health"}, timeout_s=timeout_s or self.startup_timeout_s)

    def call(
        self,
        target: str,
        kwargs: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        body = {"op": "call", "target": target, "kwargs": kwargs or {}}
        return self.request(body, timeout_s=timeout_s).get("result")

    def request(self, body: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        timeout = self.startup_timeout_s if timeout_s is None else float(timeout_s)
        with self._lock:
            self._ensure_started()
            assert self._proc is not None
            assert self._proc.stdin is not None
            msg = ipc.make_request(body)
            try:
                self._proc.stdin.write(ipc.pack(msg))
                self._proc.stdin.flush()
            except BrokenPipeError as exc:
                self._mark_dead()
                raise DependencyError(f"{self.name} sidecar pipe closed before request") from exc

            resp = self._read_response(timeout)
            if resp is None:
                raise DependencyError(f"{self.name} sidecar exited without a response")
            if resp.get("error"):
                err = resp["error"]
                raise VPAError(
                    f"{self.name} sidecar error: {err.get('message')}",
                    details={
                        "sidecar": self.name,
                        "code": err.get("code"),
                        "details": err.get("details") or {},
                    },
                )
            payload = resp.get("body") or {}
            if not isinstance(payload, dict):
                raise DependencyError(f"{self.name} sidecar returned a non-object body")
            return payload

    def close(self, *, timeout_s: float = 2.0) -> None:
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is None:
                try:
                    assert proc.stdin is not None
                    proc.stdin.write(ipc.pack(ipc.make_request({"op": "shutdown"})))
                    proc.stdin.flush()
                    self._read_response(timeout_s)
                except Exception:
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=timeout_s)
            self._proc = None

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.python.is_file():
            raise DependencyError(
                f"{self.name} sidecar python does not exist: {self.python}"
            )
        self._proc = subprocess.Popen(
            [str(self.python), "-m", "vibeproteinagent.workers.sidecar_child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            env=self.env,
            cwd=str(self.cwd) if self.cwd is not None else os.getcwd(),
        )

    def _read_response(self, timeout_s: float) -> dict[str, Any] | None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        fd = proc.stdout.fileno()
        header = self._read_exact(fd, 4, timeout_s)
        if header is None:
            return None
        n = int.from_bytes(header, "big")
        payload = self._read_exact(fd, n, timeout_s)
        if payload is None:
            return None
        obj = msgpack.unpackb(payload, raw=False)
        if not isinstance(obj, dict):
            raise DependencyError(
                f"{self.name} sidecar returned non-dict frame: {type(obj).__name__}"
            )
        return obj

    def _read_exact(self, fd: int, n: int, timeout_s: float) -> bytes | None:
        chunks: list[bytes] = []
        remaining = n
        deadline = time.monotonic() + timeout_s
        while remaining:
            if self._proc is not None and self._proc.poll() is not None:
                return None
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError_(
                    f"{self.name} sidecar timed out waiting for {n} response bytes",
                    details={"sidecar": self.name, "timeout_s": timeout_s},
                )
            readable, _, _ = select.select([fd], [], [], left)
            if not readable:
                raise TimeoutError_(
                    f"{self.name} sidecar timed out waiting for response",
                    details={"sidecar": self.name, "timeout_s": timeout_s},
                )
            chunk = os.read(fd, remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _mark_dead(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.kill()

    def __enter__(self) -> PersistentPythonSidecar:
        self.health()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close(timeout_s=0.2)
        except Exception:
            pass


@dataclass(frozen=True)
class SidecarRuntimeSpec:
    """Declarative sidecar process contract for compatibility adapters."""

    name: str
    python: Path
    env: dict[str, str]
    warm_target: str | None = None
    run_target: str | None = None
    timeout_s: float = 30.0
    seed_policy: str = "per_seed"
    cwd: Path | str | None = None

    def spawn(
        self,
        sidecar_cls: type[PersistentPythonSidecar] = PersistentPythonSidecar,
    ) -> PersistentPythonSidecar:
        return sidecar_cls(
            python=self.python,
            env=self.env,
            name=self.name,
            startup_timeout_s=self.timeout_s,
            cwd=self.cwd,
        )


class SidecarPool:
    """Small cache for long-lived sidecars keyed by adapter-defined affinity."""

    def __init__(self) -> None:
        self._sidecars: dict[Any, PersistentPythonSidecar] = {}

    def get(self, key: Any, spec: SidecarRuntimeSpec) -> PersistentPythonSidecar:
        sidecar = self._sidecars.get(key)
        if sidecar is None:
            sidecar = spec.spawn()
            self._sidecars[key] = sidecar
        return sidecar

    def close_all(self) -> None:
        for sidecar in self._sidecars.values():
            sidecar.close()
        self._sidecars.clear()


__all__ = [
    "PersistentPythonSidecar",
    "SidecarPool",
    "SidecarRuntimeSpec",
    "with_vpa_src_pythonpath",
]
