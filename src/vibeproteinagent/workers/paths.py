from __future__ import annotations

import hashlib
import os
from pathlib import Path

_AF_UNIX_SAFE_BYTES = 100


def _chmod_private(path: Path) -> None:
    try:
        path.chmod(0o700)
    except OSError:
        pass


def _is_safe_socket_dir(path: Path) -> bool:
    # Linux sockaddr_un allows roughly 108 bytes including the trailing NUL.
    # Keep a small margin for portability and for worker_N.sock variants.
    longest = path / "supervisor.sock"
    return len(str(longest).encode("utf-8")) < _AF_UNIX_SAFE_BYTES


def _short_runtime_dir(target: Path) -> Path:
    """Return a short alias for a long runtime directory.

    ``VPA_RUNTIME_DIR`` is often placed under a deep experiment/output path so
    logs are easy to find. Unix-domain sockets, however, have a hard path-length
    limit. Use a stable per-target alias under /tmp for socket-bearing files;
    when possible the alias is a symlink back to the requested directory so logs
    still appear in the caller's runtime tree.
    """
    root = Path(os.environ.get("VPA_RUNTIME_ALIAS_ROOT", f"/tmp/vpa-runtime-{os.getuid()}"))
    root.mkdir(parents=True, exist_ok=True)
    _chmod_private(root)
    digest = hashlib.sha1(str(target.resolve()).encode("utf-8")).hexdigest()[:16]
    alias = root / digest
    if alias.exists() or alias.is_symlink():
        return alias
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        alias.mkdir(parents=True, exist_ok=True)
        _chmod_private(alias)
    return alias


def _candidate_dirs() -> list[Path]:
    cands: list[Path] = []
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        cands.append(Path(xdg) / "vpa")
    uid = os.getuid()
    run_user = Path(f"/run/user/{uid}")
    if run_user.exists():
        cands.append(run_user / "vpa")
    cands.append(Path.home() / ".vibeproteinagent" / "run")
    return cands


def runtime_dir() -> Path:
    """Resolve the per-user runtime dir, creating it with 0o700 if needed."""
    override = os.environ.get("VPA_RUNTIME_DIR")
    if override:
        d = Path(override).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        _chmod_private(d)
        return d if _is_safe_socket_dir(d) else _short_runtime_dir(d)
    for cand in _candidate_dirs():
        try:
            cand.mkdir(parents=True, exist_ok=True)
            _chmod_private(cand)
            return cand
        except OSError:
            continue
    fallback = Path.home() / ".vibeproteinagent" / "run"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def supervisor_socket() -> Path:
    return runtime_dir() / "supervisor.sock"


def worker_socket(worker_id: int) -> Path:
    return runtime_dir() / f"worker_{worker_id}.sock"


def supervisor_pid_file() -> Path:
    return runtime_dir() / "supervisor.pid"


def worker_pid_file(worker_id: int) -> Path:
    return runtime_dir() / f"worker_{worker_id}.pid"


def supervisor_log_file() -> Path:
    return runtime_dir() / "supervisor.log"


def worker_log_file(worker_id: int) -> Path:
    return runtime_dir() / f"worker_{worker_id}.log"


__all__ = [
    "runtime_dir",
    "supervisor_log_file",
    "supervisor_pid_file",
    "supervisor_socket",
    "worker_log_file",
    "worker_pid_file",
    "worker_socket",
]
