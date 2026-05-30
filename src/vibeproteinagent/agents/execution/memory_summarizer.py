from __future__ import annotations

from typing import Any

from .memory_bank import MemoryBank
from .prompt_context import PromptContextBuilder


class MemorySummarizer:
    """Compatibility facade for bounded, lossy prompt context generation."""

    def summarize(
        self,
        *,
        previous_digest: dict[str, Any] | None,
        latest_round_record: dict[str, Any],
        memory_bank: MemoryBank | None = None,
    ) -> dict[str, Any]:
        return PromptContextBuilder().build(
            previous_digest=previous_digest,
            latest_round_record=latest_round_record,
            memory_bank=memory_bank,
        )


__all__ = ["MemorySummarizer", "PromptContextBuilder"]
