from __future__ import annotations

import asyncio
from pathlib import Path

from vibeproteinagent.workspace import LogFileManager


def test_open_streams_writes_appendable(tmp_path: Path) -> None:
    mgr = LogFileManager()
    out, err = mgr.open_streams(tmp_path)
    try:
        out.write(b"hello stdout\n")
        err.write(b"hello stderr\n")
    finally:
        out.close()
        err.close()
    # Re-open and append again to verify O_APPEND.
    out2, _ = mgr.open_streams(tmp_path)
    try:
        out2.write(b"line two\n")
    finally:
        out2.close()
    assert mgr.read(tmp_path, "stdout") == "hello stdout\nline two\n"
    assert mgr.read(tmp_path, "stderr") == "hello stderr\n"


def test_read_tail(tmp_path: Path) -> None:
    mgr = LogFileManager()
    out, _ = mgr.open_streams(tmp_path)
    try:
        for i in range(10):
            out.write(f"line {i}\n".encode())
    finally:
        out.close()
    tail = mgr.read(tmp_path, "stdout", tail=3)
    assert tail == "line 7\nline 8\nline 9\n"


def test_tail_follow_yields_lines_and_terminates(tmp_path: Path) -> None:
    mgr = LogFileManager()
    job_dir = tmp_path / "job_dir"
    job_dir.mkdir()

    async def producer() -> None:
        out, _ = mgr.open_streams(job_dir)
        try:
            await asyncio.sleep(0.05)
            out.write(b"first\n")
            await asyncio.sleep(0.05)
            out.write(b"second\n")
            await asyncio.sleep(0.05)
        finally:
            out.close()
        mgr.mark_terminated(job_dir)

    async def consume() -> list[str | None]:
        events: list[str | None] = []
        async for line in mgr.tail_follow(job_dir, "stdout"):
            events.append(line)
            if line is None:
                break
        return events

    async def driver() -> list[str | None]:
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consume())
        await producer_task
        return await asyncio.wait_for(consumer_task, timeout=3.0)

    events = asyncio.run(driver())
    # We expect at least the two written lines and the terminal sentinel.
    text_lines = [e for e in events if isinstance(e, str)]
    assert any(line.startswith("first") for line in text_lines)
    assert any(line.startswith("second") for line in text_lines)
    assert events[-1] is None


def test_mark_terminated_creates_marker(tmp_path: Path) -> None:
    mgr = LogFileManager()
    mgr.mark_terminated(tmp_path)
    assert (tmp_path / "terminated").exists()
