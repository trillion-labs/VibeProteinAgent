"""Ensure DGL's first-import banner does not pollute stdout.

DGL prints `Setting the default backend to "pytorch" ...` via ``print()``
(stdout, not stderr) on first import when ``DGLBACKEND`` is unset. That
leaks into any per-process JSON output we collect — most visibly the
matrix runner's per-case ``<case>.json`` files. ``compat.import_dgl_with_
graphbolt_guard`` now sets ``DGLBACKEND=pytorch`` before importing DGL
(via ``os.environ.setdefault``), so the banner is silenced for any code
path that uses the centralized helper.

These tests pin that contract — both the setdefault behavior (operator
override preserved) and the practical "no stdout noise" outcome.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


PYTHON = sys.executable


def _run(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a subprocess with the given env, capturing
    stdout and stderr independently."""
    return subprocess.run(
        [PYTHON, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    keep = {"PATH", "HOME", "PYTHONPATH"}
    base = {k: v for k, v in os.environ.items() if k in keep}
    base.pop("DGLBACKEND", None)
    if extra:
        base.update(extra)
    return base


def test_silence_helper_sets_dglbackend_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DGLBACKEND", raising=False)
    from vibeproteinagent.core.compat import _silence_dgl_backend_banner

    _silence_dgl_backend_banner()
    assert os.environ.get("DGLBACKEND") == "pytorch"


def test_silence_helper_preserves_operator_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``setdefault`` semantics: don't clobber an operator's choice."""
    monkeypatch.setenv("DGLBACKEND", "mxnet")
    from vibeproteinagent.core.compat import _silence_dgl_backend_banner

    _silence_dgl_backend_banner()
    assert os.environ.get("DGLBACKEND") == "mxnet"


def test_import_dgl_via_helper_produces_no_stdout_banner() -> None:
    """End-to-end: importing DGL through the helper, with no DGLBACKEND in
    the environment, must NOT leak the backend banner to stdout.

    This is the practical contract the matrix runner depends on.
    """
    code = (
        "from vibeproteinagent.core.compat import import_dgl_with_graphbolt_guard;"
        " import_dgl_with_graphbolt_guard()"
    )
    result = _run(code, env=_clean_env())
    assert result.returncode == 0, (
        f"helper raised: stderr={result.stderr[:500]}"
    )
    # The banner is the regression we're guarding against.
    banner_fragment = "Setting the default backend"
    assert banner_fragment not in result.stdout, (
        "DGL banner leaked to stdout via the helper:\n" + result.stdout[:500]
    )


def test_raw_dgl_import_without_helper_still_shows_banner() -> None:
    """Sanity check: confirm the banner DOES appear when DGL is imported
    without going through the helper and without DGLBACKEND preset. If
    this assertion ever breaks (DGL changes default behavior), the helper's
    silencing becomes redundant — at that point the helper can be retired,
    but for now we keep both signals visible.
    """
    code = "import dgl"
    result = _run(code, env=_clean_env())
    if result.returncode != 0:
        pytest.skip(f"raw dgl import not viable in this env: {result.stderr[:200]}")
    banner_fragment = "Setting the default backend"
    # Either DGL changed its behavior (banner gone — then this test should
    # be reconsidered), OR the banner is present — we accept either to keep
    # the helper test (above) as the binding contract.
    if banner_fragment not in result.stdout:
        pytest.skip(
            "DGL no longer prints the backend banner by default — helper "
            "silencing is now redundant but harmless."
        )
    # If we reach here, the banner IS present in raw import, confirming the
    # helper's silencing is doing real work.
    assert banner_fragment in result.stdout
