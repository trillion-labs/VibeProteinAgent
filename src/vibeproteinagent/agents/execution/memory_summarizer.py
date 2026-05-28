from __future__ import annotations

from typing import Any

from .memory_bank import MemoryBank


class MemorySummarizer:
    """Bounded prompt digest generator; never mutates MemoryBank state."""

    def summarize(
        self,
        *,
        previous_digest: dict[str, Any] | None,
        latest_round_record: dict[str, Any],
        memory_bank: MemoryBank | None = None,
    ) -> dict[str, Any]:
        try:
            previous = previous_digest if isinstance(previous_digest, dict) else {}
            return {
                "previous": {
                    "round_count": previous.get("round_count", 0),
                    "survivor_pool_summary": previous.get("survivor_pool_summary", {}),
                },
                "latest_round": {
                    "round_index": latest_round_record.get("round_index"),
                    "workflow_id": latest_round_record.get("workflow_id"),
                    "state": latest_round_record.get("state"),
                    "empty_candidate_events": latest_round_record.get(
                        "empty_candidate_events", []
                    )[:5],
                    "skipped_nodes": latest_round_record.get("skipped_nodes", [])[:10],
                    "current_round_candidate_summary": latest_round_record.get(
                        "current_round_candidate_summary", {}
                    ),
                    "survivor_pool_summary": latest_round_record.get(
                        "survivor_pool_summary", {}
                    ),
                    "branch_decision": latest_round_record.get("branch_decision"),
                },
                "round_count": max(
                    int(previous.get("round_count") or 0),
                    int(latest_round_record.get("round_index") or 0) + 1,
                ),
            }
        except Exception:
            if memory_bank is not None:
                return memory_bank.bounded_digest()
            return {
                "round_count": 0,
                "recent_rounds": [],
                "survivor_pool_summary": {"count": 0, "items": [], "truncated": False},
            }


__all__ = ["MemorySummarizer"]
