from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import IO

try:
    import inotify_simple  # type: ignore[import-not-found]
    _HAS_INOTIFY = True
except ImportError:  # pragma: no cover - env-dependent
    inotify_simple = None
    _HAS_INOTIFY = False


_TERMINATED_MARKER = "terminated"


class LogFileManager:
    """Open, append, and tail-follow ``stdout.log`` / ``stderr.log`` per job."""

    def open_streams(self, job_dir: str | Path) -> tuple[IO[bytes], IO[bytes]]:
        """Return line-buffered, append-only writers for stdout.log and stderr.log."""
        d = Path(job_dir)
        d.mkdir(parents=True, exist_ok=True)
        return self._open(d / "stdout.log"), self._open(d / "stderr.log")

    @staticmethod
    def _open(path: Path) -> IO[bytes]:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        return os.fdopen(fd, "ab", buffering=0)

    # ------------------------------------------------------------ read

    def read(
        self,
        job_dir: str | Path,
        stream: str = "stdout",
        *,
        tail: int | None = None,
    ) -> str:
        """Read the full ``stdout`` or ``stderr`` log; optionally only the last ``tail`` lines."""
        path = Path(job_dir) / f"{stream}.log"
        if not path.exists():
            return ""
        text = path.read_text(errors="replace")
        if tail is None:
            return text
        lines = text.splitlines(keepends=True)
        return "".join(lines[-int(tail):])

    # ------------------------------------------------------------ tail-follow

    async def tail_follow(
        self,
        job_dir: str | Path,
        stream: str = "stdout",
    ) -> AsyncIterator[str | None]:
        """Yield appended lines; yield ``None`` once when termination marker is observed."""
        d = Path(job_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{stream}.log"
        marker = d / _TERMINATED_MARKER

        # Touch the file so we have something to open.
        path.touch(exist_ok=True)
        loop = asyncio.get_running_loop()

        with path.open("rb") as f:
            # Start at the beginning so callers see any pre-existing content.
            pending = b""
            terminated_emitted = False
            while True:
                chunk = await loop.run_in_executor(None, f.read)
                if chunk:
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        yield line.decode("utf-8", errors="replace") + "\n"
                    continue

                if marker.exists() and not terminated_emitted:
                    # Drain any partial line, then signal end.
                    if pending:
                        yield pending.decode("utf-8", errors="replace")
                        pending = b""
                    terminated_emitted = True
                    yield None
                    return
                await asyncio.sleep(0.05)

    # ------------------------------------------------------------ terminate

    def mark_terminated(self, job_dir: str | Path) -> None:
        """Write the termination marker and fsync to make it visible to tailers."""
        d = Path(job_dir)
        d.mkdir(parents=True, exist_ok=True)
        marker = d / _TERMINATED_MARKER
        fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, b"1\n")
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["LogFileManager"]
