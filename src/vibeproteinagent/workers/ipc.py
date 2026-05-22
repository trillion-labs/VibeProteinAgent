from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import msgpack

from vibeproteinagent.core.errors import ErrorCode, VPAError

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024  # 64 MiB

# Envelope kinds (Architecture v1 §4).
KIND_REQUEST = "request"
KIND_RESPONSE = "response"
KIND_STREAM = "stream"
KIND_CANCEL = "cancel"


def new_msg_id() -> str:
    """Fresh message correlation id (32-char hex)."""
    return uuid.uuid4().hex


def pack(msg: dict[str, Any]) -> bytes:
    """Pack a message: 4-byte big-endian length prefix + msgpack body."""
    payload = msgpack.packb(msg, use_bin_type=True)
    n = len(payload)
    if n > MAX_PAYLOAD_BYTES:
        raise VPAError(
            f"frame too large: {n} > {MAX_PAYLOAD_BYTES}",
            details={"size": n, "limit": MAX_PAYLOAD_BYTES},
        )
    return n.to_bytes(4, "big") + payload


async def unpack(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Read one length-prefixed msgpack frame; return ``None`` on EOF."""
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as e:
        if not e.partial:
            return None
        raise VPAError(
            f"incomplete header (got {len(e.partial)} bytes)",
            details={"got": len(e.partial)},
        ) from e
    n = int.from_bytes(header, "big")
    if n > MAX_PAYLOAD_BYTES:
        raise VPAError(
            f"frame too large: {n} > {MAX_PAYLOAD_BYTES}",
            details={"size": n, "limit": MAX_PAYLOAD_BYTES},
        )
    try:
        body = await reader.readexactly(n)
    except asyncio.IncompleteReadError as e:
        raise VPAError(
            f"incomplete body (got {len(e.partial)}/{n})",
            details={"got": len(e.partial), "expected": n},
        ) from e
    obj = msgpack.unpackb(body, raw=False)
    if not isinstance(obj, dict):
        raise VPAError(
            "frame body is not a dict envelope",
            details={"type": type(obj).__name__},
        )
    return obj


async def send(writer: asyncio.StreamWriter, msg: dict[str, Any]) -> None:
    """Pack + write + drain."""
    writer.write(pack(msg))
    await writer.drain()


async def recv(reader: asyncio.StreamReader) -> dict[str, Any] | None:
    """Alias of :func:`unpack`."""
    return await unpack(reader)


def make_request(body: dict[str, Any], *, msg_id: str | None = None) -> dict[str, Any]:
    """Build a request envelope."""
    return {
        "kind": KIND_REQUEST,
        "id": msg_id or new_msg_id(),
        "ts": time.time(),
        "body": body,
    }


def make_response(
    msg_id: str,
    body: dict[str, Any],
    *,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a response envelope (success or error)."""
    env: dict[str, Any] = {
        "kind": KIND_RESPONSE,
        "id": msg_id,
        "ts": time.time(),
        "body": body,
    }
    if error is not None:
        env["error"] = error
    return env


def make_stream(msg_id: str, event: dict[str, Any]) -> dict[str, Any]:
    """Build a stream-event envelope."""
    return {
        "kind": KIND_STREAM,
        "id": msg_id,
        "ts": time.time(),
        "body": event,
    }


def make_error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compact error body for inclusion in a response envelope."""
    return {"code": code, "message": message, "details": details or {}}


def error_from_exception(exc: BaseException) -> dict[str, Any]:
    """Convert a VPAError (or any exception) into an error body."""
    if isinstance(exc, VPAError):
        return make_error(exc.code, exc.message, exc.details)
    return make_error(ErrorCode.INTERNAL, str(exc) or exc.__class__.__name__, {})


__all__ = [
    "KIND_CANCEL",
    "KIND_REQUEST",
    "KIND_RESPONSE",
    "KIND_STREAM",
    "MAX_PAYLOAD_BYTES",
    "error_from_exception",
    "make_error",
    "make_request",
    "make_response",
    "make_stream",
    "new_msg_id",
    "pack",
    "recv",
    "send",
    "unpack",
]
