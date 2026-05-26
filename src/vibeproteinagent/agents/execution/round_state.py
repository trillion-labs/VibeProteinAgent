from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PriorRoundContext:
    round_number: int
    surviving_candidates: list[dict[str, Any]]
    prior_decision_reason: str
    summary: dict[str, Any] | None = None

