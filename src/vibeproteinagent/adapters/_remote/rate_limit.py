from __future__ import annotations

from dataclasses import dataclass
from threading import BoundedSemaphore


@dataclass
class ConcurrencyLimiter:
    max_concurrency: int = 3

    def __post_init__(self) -> None:
        self._sem = BoundedSemaphore(max(1, int(self.max_concurrency)))

    def __enter__(self) -> ConcurrencyLimiter:
        self._sem.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._sem.release()
