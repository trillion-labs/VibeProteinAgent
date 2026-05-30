from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PriorRoundContext:
    """Prior round prompt/executor routing state."""

    round_number: int
    prior_decision_reason: str
    survivor_snapshot_id: str | None = None
    summary: dict[str, Any] | None = None
    prompt_context: dict[str, Any] | None = None
    prompt_digest: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        digest = self.prompt_context or self.prompt_digest
        object.__setattr__(self, "prompt_context", digest)
        object.__setattr__(self, "prompt_digest", digest)
