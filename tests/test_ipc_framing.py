from __future__ import annotations

import asyncio
import io

import msgpack
import pytest

from vibeproteinagent.core.errors import VPAError
from vibeproteinagent.workers import ipc


def _reader_from_bytes(data: bytes) -> asyncio.StreamReader:
    """Build a StreamReader pre-filled with ``data`` and EOF."""
    loop = asyncio.new_event_loop()
    try:
        reader = asyncio.StreamReader(loop=loop)
        reader.feed_data(data)
        reader.feed_eof()
        return reader
    finally:
        loop.close()


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.new_event_loop().run_until_complete(coro)


def test_pack_format_is_length_prefixed() -> None:
    msg = {"kind": "request", "id": "abc", "body": {"op": "ping"}}
    blob = ipc.pack(msg)
    assert len(blob) >= 4
    n = int.from_bytes(blob[:4], "big")
    assert n == len(blob) - 4
    body = msgpack.unpackb(blob[4:], raw=False)
    assert body == msg


def test_pack_unpack_roundtrip() -> None:
    msg = ipc.make_request({"op": "submit", "job_id": "job_" + "0" * 32})
    blob = ipc.pack(msg)

    async def go() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(blob)
        reader.feed_eof()
        out = await ipc.unpack(reader)
        assert out == msg

    _run(go())


def test_unpack_eof_returns_none() -> None:
    async def go() -> None:
        reader = asyncio.StreamReader()
        reader.feed_eof()
        out = await ipc.unpack(reader)
        assert out is None

    _run(go())


def test_unpack_incomplete_body_raises() -> None:
    msg = {"x": 1, "y": "two"}
    payload = msgpack.packb(msg, use_bin_type=True)
    header = len(payload).to_bytes(4, "big")
    # Feed header + half the body.
    truncated = header + payload[: len(payload) // 2]

    async def go() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(truncated)
        reader.feed_eof()
        with pytest.raises(VPAError):
            await ipc.unpack(reader)

    _run(go())


def test_pack_too_large_raises() -> None:
    # Forge an oversized header without paying msgpack cost.
    fake_header = (ipc.MAX_PAYLOAD_BYTES + 1).to_bytes(4, "big")

    async def go() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(fake_header)
        reader.feed_eof()
        with pytest.raises(VPAError):
            await ipc.unpack(reader)

    _run(go())


def test_make_request_and_response_shapes() -> None:
    req = ipc.make_request({"op": "ping"})
    assert req["kind"] == "request"
    assert isinstance(req["id"], str) and len(req["id"]) == 32
    assert req["body"] == {"op": "ping"}

    resp = ipc.make_response(req["id"], {"ok": True})
    assert resp["kind"] == "response"
    assert resp["id"] == req["id"]
    assert resp["body"] == {"ok": True}
    assert "error" not in resp

    err = ipc.make_response(req["id"], {}, error=ipc.make_error("E_TEST", "nope"))
    assert err["error"]["code"] == "E_TEST"


def test_error_from_exception_vpa() -> None:
    e = VPAError("boom", details={"reason": "test"})
    body = ipc.error_from_exception(e)
    assert body["code"] == e.code
    assert body["message"] == "boom"
    assert body["details"] == {"reason": "test"}


def test_unpack_non_dict_raises() -> None:
    payload = msgpack.packb(["not", "a", "dict"], use_bin_type=True)
    blob = len(payload).to_bytes(4, "big") + payload

    async def go() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(blob)
        reader.feed_eof()
        with pytest.raises(VPAError):
            await ipc.unpack(reader)

    _run(go())


_ = io  # keep imports stable
