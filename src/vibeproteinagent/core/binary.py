"""Single source of truth for binary-wrapper adapters' `$PATH` probing.

Binary-backed adapters share this helper so each executable is resolved the
same way: env-var override first, then `$PATH`.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _python_env_bin_dirs() -> list[Path]:
    """Return stable binary directories for the active Python environment.

    Daemons and tests are often launched via an absolute interpreter path such
    as ``/path/to/env/bin/python`` without shell activation. In that mode
    ``$PATH`` may not include the env's ``bin`` directory even though bundled
    executables are available next to the interpreter.
    """

    dirs: list[Path] = []
    for raw in (
        Path(sys.executable).resolve().parent,
        Path(sys.prefix).resolve() / "bin",
        Path(sys.base_prefix).resolve() / "bin",
        Path(os.environ["CONDA_PREFIX"]).resolve() / "bin"
        if os.environ.get("CONDA_PREFIX")
        else None,
        Path(os.environ["VIRTUAL_ENV"]).resolve() / "bin"
        if os.environ.get("VIRTUAL_ENV")
        else None,
    ):
        if raw is None:
            continue
        if raw not in dirs:
            dirs.append(raw)
    return dirs


def find_binary(candidates: list[str] | tuple[str, ...], env_override: str | None = None) -> Path:
    """Return the absolute path to the first available binary.

    Resolution order:
      1. If ``env_override`` is set and the named env var is a non-empty path
         to an executable file, return it (lets ops point adapters at a
         non-standard build, e.g. ``VPA_MMSEQS_BINARY=/opt/mmseqs/bin/mmseqs``).
      2. Check the active Python environment's ``bin`` directory. This keeps
         binary-backed adapters deterministic when a supervisor is launched via
         an absolute conda/venv interpreter without shell activation.
      3. ``shutil.which`` each candidate in order; return the first hit.

    Raises ``FileNotFoundError`` listing the candidates and the env-var name if
    nothing is found, so the caller can surface install guidance.
    """
    if env_override:
        override = os.environ.get(env_override, "").strip()
        if override:
            p = Path(override).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return p.resolve()
            raise FileNotFoundError(
                f"{env_override}={override!r} is not an executable file"
            )
    for bin_dir in _python_env_bin_dirs():
        for name in candidates:
            p = bin_dir / name
            if p.is_file() and os.access(p, os.X_OK):
                return p.resolve()
    for name in candidates:
        which = shutil.which(name)
        if which:
            return Path(which).resolve()
    hint = f" (env override: {env_override})" if env_override else ""
    raise FileNotFoundError(
        f"none of {list(candidates)} found on $PATH{hint}"
    )


__all__ = ["find_binary"]
