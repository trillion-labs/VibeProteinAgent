from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibeproteinagent.core.errors import ConflictError


@dataclass(frozen=True)
class InstallPaths:
    root: Path
    shared_env: Path
    sidecars_root: Path
    cache_dir: Path
    state_db: Path


class InstallState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = self._load()

    def mark(self, unit: str, status: str, details: dict[str, Any] | None = None) -> None:
        units = self.data.setdefault("units", {})
        units[unit] = {"status": status, "details": details or {}}

    def status(self, unit: str) -> str | None:
        entry = (self.data.get("units") or {}).get(unit) or {}
        value = entry.get("status")
        return str(value) if value else None

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")
        tmp.replace(self.path)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "units": {}}
        return json.loads(self.path.read_text())


@contextmanager
def install_lock(root: Path) -> Iterator[None]:
    lock_dir = root / "install"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / ".lock"
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ConflictError(
            "another VPA install is already running",
            details={"lock_path": str(lock_path)},
        ) from exc
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
        yield
    finally:
        os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
