"""Shared runtime contracts for adapter availability and isolated execution.

This module is intentionally lightweight. It centralizes the repeated
environment discovery and smoke-probe mechanics that adapters need, while
leaving scientific parameter mapping and output parsing inside each adapter.
"""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vibeproteinagent.core.binary import find_binary

_REPO_SRC = Path(__file__).resolve().parents[2]
_SHARED_CONDA_ROOT = Path("/opt/miniconda3")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - sidecars may run py3.10.


@dataclass(frozen=True)
class RuntimeContract:
    """Normalized runtime metadata derived from a tool descriptor."""

    tool: str
    support_status: str
    support_reason: str | None = None
    execution_isolation: str = "shared"
    environment_name: str | None = None
    environment_reason: str | None = None
    availability_probe: str | None = None

    @classmethod
    def from_descriptor(cls, descriptor: Any) -> RuntimeContract:
        support_status = getattr(
            descriptor.support_status, "value", descriptor.support_status
        )
        execution_isolation = getattr(
            descriptor.execution_isolation, "value", descriptor.execution_isolation
        )
        return cls(
            tool=str(descriptor.name),
            support_status=str(support_status),
            support_reason=descriptor.support_status_reason,
            execution_isolation=str(execution_isolation),
            environment_name=descriptor.environment_name,
            environment_reason=descriptor.environment_reason,
            availability_probe=descriptor.availability_probe,
        )


@dataclass(frozen=True)
class AvailabilityReport:
    """Structured adapter availability result used internally."""

    available: bool
    status: str
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    checked_at: str = field(default_factory=_now_iso)

    @classmethod
    def from_skip_reason(
        cls,
        reason: str | None,
        *,
        support_status: str = "verified",
        details: Mapping[str, Any] | None = None,
        checked_at: str | None = None,
    ) -> AvailabilityReport:
        available = support_status == "verified" and reason is None
        status = "available" if available else support_status
        if reason and status == "verified":
            status = "unavailable"
        return cls(
            available=available,
            status=status,
            reason=reason,
            details=dict(details or {}),
            checked_at=checked_at or _now_iso(),
        )

    def skip_reason(self) -> str | None:
        return None if self.available else self.reason or self.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "availability_status": self.status,
            "availability_reason": self.reason,
            "availability_details": dict(self.details),
            "availability_checked_at": self.checked_at,
        }


class EnvLocator:
    """Shared discovery helpers for conda envs, uv venvs, and binaries."""

    @staticmethod
    def find_conda_exe(extra_candidates: Iterable[Path] = ()) -> Path | None:
        candidates = [
            *extra_candidates,
            _SHARED_CONDA_ROOT / "bin" / "conda",
            Path.home() / "miniconda3" / "bin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
            Path("/opt/conda/bin/conda"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        for part in os.environ.get("PATH", "").split(os.pathsep):
            if not part:
                continue
            candidate = Path(part) / "conda"
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def python_in_env(env_path: Path) -> Path:
        return Path(env_path) / "bin" / "python"

    @staticmethod
    def uv_venv_python(venv_path: Path) -> Path:
        return Path(venv_path) / "bin" / "python"

    @staticmethod
    def conda_env_path(
        env_name: str,
        *,
        explicit_env_var: str | None = None,
        candidate_paths: Iterable[Path] = (),
        final_default: Path | None = None,
        conda_exe: Path | None = None,
    ) -> Path | None:
        if explicit_env_var:
            raw = os.environ.get(explicit_env_var)
            if raw:
                path = Path(raw).expanduser()
                if EnvLocator.python_in_env(path).is_file():
                    return path

        conda_exe = conda_exe or EnvLocator.find_conda_exe()
        roots: list[Path] = []
        if conda_exe is not None:
            roots.append(conda_exe.parent.parent / "envs")
        roots.extend(
            [
                _SHARED_CONDA_ROOT / "envs",
                Path.home() / ".conda" / "envs",
                Path.home() / "miniconda3" / "envs",
                Path.home() / "anaconda3" / "envs",
                Path("/opt/conda/envs"),
            ]
        )

        seen: set[Path] = set()
        for candidate in [*(Path(p) for p in candidate_paths), *(r / env_name for r in roots)]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if EnvLocator.python_in_env(candidate).is_file():
                return candidate

        if conda_exe is not None:
            try:
                proc = subprocess.run(
                    [str(conda_exe), "env", "list", "--json"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if proc.returncode == 0:
                    for raw in json.loads(proc.stdout).get("envs", []):
                        candidate = Path(raw)
                        if (
                            candidate.name == env_name
                            and EnvLocator.python_in_env(candidate).is_file()
                        ):
                            return candidate
            except Exception:
                pass

        if final_default is not None:
            return final_default
        return None

    @staticmethod
    def binary(candidates: Iterable[str], *, env_override: str | None = None) -> Path | None:
        return find_binary(candidates, env_override=env_override)


def sanitized_python_env(
    *,
    seed: int | None = None,
    env_prefix: Path | None = None,
    prepend_bin: Iterable[Path] = (),
    vendor_paths: Iterable[Path] = (),
    extra: Mapping[str, str] | None = None,
    strip_pythonpath: bool = True,
    include_vpa_src: bool = True,
) -> dict[str, str]:
    """Build a stable Python subprocess env for sidecars and probes."""

    env = os.environ.copy()
    if strip_pythonpath:
        env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)

    path_parts = [str(Path(p)) for p in prepend_bin]
    if env_prefix is not None:
        path_parts.append(str(Path(env_prefix) / "bin"))
    existing_path = env.get("PATH")
    if existing_path:
        path_parts.append(existing_path)
    env["PATH"] = os.pathsep.join(path_parts)

    pythonpath_parts = [*(str(Path(p)) for p in vendor_paths)]
    if include_vpa_src:
        pythonpath_parts.insert(0, str(_REPO_SRC))
    if pythonpath_parts:
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


@dataclass(frozen=True)
class SmokeProbe:
    """Reusable Python import/binary/checkpoint probe with stable errors."""

    label: str
    python: Path
    code: str
    env: Mapping[str, str] | None = None
    timeout_s: int = 30
    cwd: Path | str | None = None

    def run(self) -> str | None:
        try:
            proc = subprocess.run(
                [str(self.python), "-c", self.code],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=dict(self.env or os.environ),
                cwd=str(self.cwd) if self.cwd is not None else None,
            )
        except FileNotFoundError:
            return f"{self.label} python not found: {self.python}"
        except subprocess.TimeoutExpired:
            return f"{self.label} import probe timed out after {self.timeout_s}s"
        if proc.returncode == 0:
            return None
        stderr = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = stderr[-1] if stderr else f"exit code {proc.returncode}"
        return f"{self.label} import probe failed: {tail}"


def smoke_import(
    *,
    label: str,
    python: Path,
    code: str,
    env: Mapping[str, str] | None = None,
    timeout_s: int = 30,
    cwd: Path | str | None = None,
) -> str | None:
    return SmokeProbe(
        label=label,
        python=python,
        code=code,
        env=env,
        timeout_s=timeout_s,
        cwd=cwd,
    ).run()


__all__ = [
    "AvailabilityReport",
    "EnvLocator",
    "RuntimeContract",
    "SmokeProbe",
    "sanitized_python_env",
    "smoke_import",
]
