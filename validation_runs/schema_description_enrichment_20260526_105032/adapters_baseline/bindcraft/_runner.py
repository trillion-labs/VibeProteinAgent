"""BindCraft sidecar runner.

This module is imported inside the dedicated BindCraft conda env by
``vibeproteinagent.workers.sidecar_child``. It deliberately avoids importing
BindCraft at module load; the upstream ``bindcraft.py`` entrypoint is executed
only when the parent adapter dispatches a run.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def run_sidecar(
    *,
    bindcraft_py: str,
    tree: str,
    settings_json: str,
    filters_json: str,
    advanced_json: str,
    seed: int | None = None,
) -> dict[str, object]:
    """Run BindCraft's CLI entrypoint inside the persistent sidecar process."""
    script = Path(bindcraft_py)
    if not script.is_file():
        raise FileNotFoundError(f"BindCraft entrypoint not found: {script}")

    argv_before = sys.argv[:]
    cwd_before = os.getcwd()
    tree_path = str(Path(tree))
    added_tree_path = False
    try:
        os.chdir(tree)
        if tree_path not in sys.path:
            sys.path.insert(0, tree_path)
            added_tree_path = True
        if seed is not None:
            import numpy as np

            np.random.seed(int(seed))
        sys.argv = [
            str(script),
            "--settings",
            str(settings_json),
            "--filters",
            str(filters_json),
            "--advanced",
            str(advanced_json),
        ]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code not in (None, 0):
                raise RuntimeError(f"BindCraft exited with code {code}") from exc
    finally:
        sys.argv = argv_before
        os.chdir(cwd_before)
        if added_tree_path:
            try:
                sys.path.remove(tree_path)
            except ValueError:
                pass

    return {"ok": True}


__all__ = ["run_sidecar"]
