"""PXDesign pipeline runner for the persistent sidecar.

This module is imported inside the dedicated ``pxdesign-pipeline-vpa`` env by
``vibeproteinagent.workers.sidecar_child``. It deliberately keeps the adapter's
main process free of the AF2/JAX/ stack while avoiding one Python
process startup per pipeline call.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any


def run_sidecar(argv: list[str], cwd: str | None = None) -> dict[str, Any]:
    """Run ``pxdesign.runner.pipeline`` with CLI-style argv.

    ``argv`` is the argument list after ``python -m pxdesign.runner.pipeline``.
    Upstream PXDesign writes its own outputs under ``--dump_dir``; the sidecar
    response only needs to report successful completion.
    """
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    try:
        if cwd:
            os.chdir(cwd)
        sys.argv = ["pxdesign.runner.pipeline", *argv]
        try:
            runpy.run_module("pxdesign.runner.pipeline", run_name="__main__")
        except SystemExit as exc:
            code = exc.code
            if code not in (None, 0):
                raise RuntimeError(
                    f"PXDesign pipeline exited with status {code}"
                ) from exc
        return {"ok": True}
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


__all__ = ["run_sidecar"]
