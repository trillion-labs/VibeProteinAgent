from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _candidate_summary(candidates: list[dict[str, Any]], *, limit: int = 10) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for candidate in candidates[:limit]:
        capabilities = candidate.get("capabilities")
        items.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "source": candidate.get("source"),
                "capabilities": dict(capabilities) if isinstance(capabilities, dict) else {},
                "has_sequence": bool(candidate.get("sequence")),
                "structure_path": candidate.get("structure_path"),
                "backbone_path": candidate.get("backbone_path"),
                "full_structure_path": candidate.get("full_structure_path"),
                "score_keys": sorted(candidate.get("scores") or {})
                if isinstance(candidate.get("scores"), dict)
                else [],
            }
        )
    return {
        "count": len(candidates),
        "items": items,
        "truncated": len(candidates) > limit,
    }


def _empty_events(summary: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node in summary.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        output = node.get("output")
        if isinstance(output, dict) and isinstance(output.get("empty_candidate_event"), dict):
            events.append(dict(output["empty_candidate_event"]))
    return events


def _skipped_nodes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    for node in summary.get("nodes") or []:
        if not isinstance(node, dict) or node.get("state") != "skipped":
            continue
        output = node.get("output")
        skipped.append(
            {
                "node_id": node.get("node_id"),
                "tool": node.get("tool"),
                "skip_kind": output.get("skip_kind") if isinstance(output, dict) else None,
                "skip_reason": output.get("skip_reason") if isinstance(output, dict) else None,
            }
        )
    return skipped


@dataclass
class MemoryBank:
    """Deterministic source-of-truth memory for multi-round agent execution."""

    round_records: list[dict[str, Any]] = field(default_factory=list)
    controller_branches: list[dict[str, Any]] = field(default_factory=list)
    raw_decision_outputs: list[dict[str, Any]] = field(default_factory=list)
    survivor_pool: list[dict[str, Any]] = field(default_factory=list)

    def record_round(
        self,
        *,
        round_index: int,
        workflow_summary: dict[str, Any],
        current_round_candidates: list[dict[str, Any]],
        survivor_pool: list[dict[str, Any]],
        max_round_reached: bool,
        branch_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.survivor_pool = [dict(candidate) for candidate in survivor_pool]
        record = {
            "round_index": round_index,
            "workflow_id": workflow_summary.get("workflow_id"),
            "state": workflow_summary.get("state"),
            "empty_candidate_events": _empty_events(workflow_summary),
            "skipped_nodes": _skipped_nodes(workflow_summary),
            "current_round_candidates": [dict(candidate) for candidate in current_round_candidates],
            "current_round_candidate_summary": _candidate_summary(current_round_candidates),
            "survivor_pool": [dict(candidate) for candidate in survivor_pool],
            "survivor_pool_summary": _candidate_summary(survivor_pool),
            "max_round_reached": max_round_reached,
            "branch_decision": dict(branch_decision) if isinstance(branch_decision, dict) else None,
            "workflow_summary": workflow_summary,
        }
        self.round_records.append(record)
        return record

    def record_branch(self, round_index: int, branch_decision: dict[str, Any]) -> None:
        entry = {"round_index": round_index, "branch_decision": dict(branch_decision)}
        self.controller_branches.append(entry)
        if self.round_records and self.round_records[-1].get("round_index") == round_index:
            self.round_records[-1]["branch_decision"] = dict(branch_decision)

    def record_decision(self, round_index: int, decision: dict[str, Any]) -> None:
        entry = {"round_index": round_index, "decision": dict(decision)}
        self.raw_decision_outputs.append(entry)
        if self.round_records and self.round_records[-1].get("round_index") == round_index:
            decisions = self.round_records[-1].setdefault("raw_decision_outputs", [])
            if isinstance(decisions, list):
                decisions.append(dict(decision))

    def latest_round_record(self) -> dict[str, Any] | None:
        return self.round_records[-1] if self.round_records else None

    def bounded_digest(self, *, max_rounds: int = 3) -> dict[str, Any]:
        rounds = self.round_records[-max_rounds:]
        return {
            "round_count": len(self.round_records),
            "recent_rounds": [
                {
                    "round_index": record.get("round_index"),
                    "workflow_id": record.get("workflow_id"),
                    "state": record.get("state"),
                    "empty_candidate_events": record.get("empty_candidate_events", [])[:5],
                    "skipped_nodes": record.get("skipped_nodes", [])[:10],
                    "current_round_candidate_summary": record.get(
                        "current_round_candidate_summary", {}
                    ),
                    "survivor_pool_summary": record.get("survivor_pool_summary", {}),
                    "max_round_reached": record.get("max_round_reached"),
                    "branch_decision": record.get("branch_decision"),
                }
                for record in rounds
            ],
            "survivor_pool_summary": _candidate_summary(self.survivor_pool),
        }


__all__ = ["MemoryBank"]
