from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from vibeproteinagent.core.errors import VPAError
from vibeproteinagent.workers import supervisor_client
from vibeproteinagent.workers.supervisor_client import SupervisorClient


class _Writer:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


def test_supervisor_client_preserves_rpc_error_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fake_open_unix_connection(_path: str):
        return object(), _Writer()

    async def fake_unpack(_reader):
        return {
            "error": {
                "code": "E_INVALID_PARAMS",
                "message": "bad request",
                "details": {"field": "tool"},
            }
        }

    async def fake_send(_writer, _request) -> None:
        pass

    monkeypatch.setattr(asyncio, "open_unix_connection", fake_open_unix_connection)
    monkeypatch.setattr(supervisor_client.ipc, "unpack", fake_unpack)
    monkeypatch.setattr(supervisor_client.ipc, "send", fake_send)

    client = SupervisorClient(socket_path=tmp_path / "supervisor.sock")
    with pytest.raises(VPAError) as ei:
        client._call({"op": "warm", "tool": ""})

    assert ei.value.code == "E_INVALID_PARAMS"
    assert ei.value.details == {"field": "tool"}

