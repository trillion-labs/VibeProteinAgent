from __future__ import annotations

import importlib
import sys
import traceback
from typing import Any

import msgpack

from vibeproteinagent.core.errors import ErrorCode
from vibeproteinagent.workers import ipc


def _read_exact(n: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = n
    stream = sys.stdin.buffer
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv() -> dict[str, Any] | None:
    header = _read_exact(4)
    if header is None:
        return None
    n = int.from_bytes(header, "big")
    if n > ipc.MAX_PAYLOAD_BYTES:
        raise RuntimeError(f"frame too large: {n} > {ipc.MAX_PAYLOAD_BYTES}")
    body = _read_exact(n)
    if body is None:
        return None
    obj = msgpack.unpackb(body, raw=False)
    if not isinstance(obj, dict):
        raise RuntimeError(f"frame body is not a dict: {type(obj).__name__}")
    return obj


def _send(msg: dict[str, Any]) -> None:
    sys.__stdout__.buffer.write(ipc.pack(msg))
    sys.__stdout__.buffer.flush()


def _resolve_callable(target: str) -> Any:
    if ":" not in target:
        raise ValueError("target must be 'module:function'")
    module_name, func_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _handle(body: dict[str, Any]) -> dict[str, Any]:
    op = body.get("op")
    if op == "health":
        return {"ok": True}
    if op == "shutdown":
        return {"ok": True, "shutdown": True}
    if op != "call":
        raise ValueError(f"unknown op {op!r}")

    target = body.get("target")
    kwargs = body.get("kwargs") or {}
    if not isinstance(target, str):
        raise ValueError("call requires string target")
    if not isinstance(kwargs, dict):
        raise ValueError("call kwargs must be an object")
    func = _resolve_callable(target)
    result = func(**kwargs)
    return {"ok": True, "result": result}


def main() -> int:
    # Keep stdout reserved for framed responses. Third-party prints from imported
    # scientific stacks go to stderr, so they cannot corrupt the msgpack stream.
    sys.stdout = sys.stderr

    while True:
        req = _recv()
        if req is None:
            return 0
        mid = str(req.get("id") or "")
        try:
            body = req.get("body") or {}
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            payload = _handle(body)
            _send(ipc.make_response(mid, payload))
            if payload.get("shutdown"):
                return 0
        except Exception as exc:  # pragma: no cover - exercised through parent
            _send(
                ipc.make_response(
                    mid,
                    {},
                    error=ipc.make_error(
                        ErrorCode.INTERNAL,
                        str(exc) or exc.__class__.__name__,
                        {
                            "type": type(exc).__name__,
                            "traceback": traceback.format_exc(limit=8),
                        },
                    ),
                )
            )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
