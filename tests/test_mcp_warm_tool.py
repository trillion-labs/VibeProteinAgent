"""Codex review Re13: ``warm_tool`` MCP facade must aggregate success properly
and recognise the worker's ``loaded_adapters`` dict shape.

The previous implementation:

1. Returned ``ok=True`` as soon as the synchronous ``warm`` call returned a
   non-empty list — even when every per-worker entry reported ``success=False``.
2. Treated each ``loaded_adapters`` entry as a ``(name, version)`` tuple,
   while the worker actually emits ``{"name", "version", ...}`` dicts (per
   Phase 3.5 worker_info contract), so the polling fallback never matched.

Fixed shape: ``ok`` is ``True`` iff there was at least one worker tried AND
every per-worker response has ``success=True``. The polling fallback uses
:func:`mcp.tools._is_loaded` which reads the dict form correctly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.mcp.tools import SessionCtx, _is_loaded, warm_tool
from vibeproteinagent.workspace import Workspace


@pytest.fixture
def ctx(tmp_workspace: Workspace) -> SessionCtx:
    return SessionCtx(workspace=tmp_workspace, registry=ToolRegistry())


def test_is_loaded_recognises_dict_entries() -> None:
    worker = {
        "gpu_id": 0,
        "loaded_adapters": [
            {"name": "noop", "version": "1.0.0", "vram_used_mib": 0},
            {"name": "proteinmpnn", "version": "1.0.1", "vram_used_mib": 14},
        ],
    }
    assert _is_loaded(worker, "noop", "1.0.0") is True
    assert _is_loaded(worker, "noop", "*") is True
    assert _is_loaded(worker, "proteinmpnn", "1.0.1") is True
    assert _is_loaded(worker, "proteinmpnn", "9.9.9") is False
    assert _is_loaded(worker, "nope", "*") is False


def test_warm_tool_all_success_returns_ok_true(
    ctx: SessionCtx, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.is_running.return_value = True
    fake_client.socket_path = "/dev/null"
    fake_client.warm.return_value = {
        "warmed": [
            {"gpu_id": 0, "success": True, "vram_used_mib": 64},
            {"gpu_id": 1, "success": True, "vram_used_mib": 64},
        ]
    }
    monkeypatch.setattr(
        SessionCtx, "supervisor", staticmethod(lambda: fake_client)
    )
    out = warm_tool(ctx, "noop", "1.0.0")
    assert out["ok"] is True
    assert out["via"] == "synchronous"
    assert len(out["warmed"]) == 2


def test_warm_tool_mixed_returns_ok_false(
    ctx: SessionCtx, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.is_running.return_value = True
    fake_client.socket_path = "/dev/null"
    fake_client.warm.return_value = {
        "warmed": [
            {"gpu_id": 0, "success": True, "vram_used_mib": 64},
            {"gpu_id": 1, "success": False, "error_code": "E_VRAM_EXHAUSTED"},
        ]
    }
    monkeypatch.setattr(
        SessionCtx, "supervisor", staticmethod(lambda: fake_client)
    )
    out = warm_tool(ctx, "noop", "1.0.0")
    assert out["ok"] is False
    assert len(out["warmed"]) == 2


def test_warm_tool_daemon_offline(
    ctx: SessionCtx, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.is_running.return_value = False
    fake_client.socket_path = "/run/missing.sock"
    monkeypatch.setattr(
        SessionCtx, "supervisor", staticmethod(lambda: fake_client)
    )
    out = warm_tool(ctx, "noop", "1.0.0")
    assert out["ok"] is False
    assert out["code"] == "E_DAEMON_UNAVAILABLE"
