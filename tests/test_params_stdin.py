"""Codex round-2 RC1: ``--params-file -`` reads JSON from stdin.

Before Phase 8, ``-`` was opened as a filesystem path and the open(2) failed
with ``E_INVALID_PARAMS`` even though the README documented stdin support.
The fix is in :func:`vibeproteinagent.core.types.load_params_file`: special-case
``"-"`` to call :func:`sys.stdin.read` and parse as JSON. Both ``vpa run`` and
``vpa workflow run`` go through the same helper.
"""

from __future__ import annotations

import io
import sys

import pytest

from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.core.types import load_params_file


def test_dash_reads_json_from_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"payload": "hi"}'))
    # In a non-TTY (StringIO) we should accept the JSON.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    obj = load_params_file("-")
    assert obj == {"payload": "hi"}


def test_dash_with_invalid_json_raises_invalid_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("not-json"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(InvalidParamsError) as exc:
        load_params_file("-")
    assert exc.value.details.get("path") == "-"
    assert "parse_error" in exc.value.details


def test_dash_requires_object_at_top_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("[1, 2, 3]"))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(InvalidParamsError) as exc:
        load_params_file("-")
    assert exc.value.details.get("got_type") == "list"


def test_dash_refuses_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """If stdin is a TTY, refuse instead of hanging the user's terminal."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    with pytest.raises(InvalidParamsError) as exc:
        load_params_file("-")
    assert "TTY" in exc.value.message or "tty" in exc.value.message.lower()
