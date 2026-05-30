from __future__ import annotations

import copy
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from vibeproteinagent.workspace.store import now_us


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
        if not isinstance(output, dict):
            continue
        if isinstance(output.get("empty_candidate_event"), dict):
            events.append(dict(output["empty_candidate_event"]))
        empty_facets = output.get("empty_facets")
        if isinstance(empty_facets, list):
            for facet in empty_facets:
                if isinstance(facet, dict):
                    events.append(dict(facet))
    return events


def _capability_filter_events(summary: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for node in summary.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        output = node.get("output")
        if not isinstance(output, dict):
            continue
        event = output.get("candidate_capability_filter_event")
        if isinstance(event, dict):
            events.append(
                {
                    "node_id": node.get("node_id"),
                    "tool": node.get("tool"),
                    **dict(event),
                }
            )
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
class RawMemoryEvent:
    event_id: str
    run_id: str
    round_index: int
    workflow_id: str | None
    event_type: str
    payload: dict[str, Any]
    created_at: int


@dataclass
class RoundLedgerRecord:
    round_index: int
    workflow_id: str | None
    workflow_summary_raw: dict[str, Any]
    candidate_snapshot_id: str | None
    survivor_snapshot_id: str | None
    decision_event_ids: list[str] = field(default_factory=list)
    branch_event_ids: list[str] = field(default_factory=list)


@dataclass
class MemoryBank:
    """Lossless in-process ledger for multi-round agent execution."""

    raw_events: list[RawMemoryEvent] = field(default_factory=list)
    round_ledgers: list[RoundLedgerRecord] = field(default_factory=list)
    round_records: list[dict[str, Any]] = field(default_factory=list)
    controller_branches: list[dict[str, Any]] = field(default_factory=list)
    raw_decision_outputs: list[dict[str, Any]] = field(default_factory=list)
    session_store: Any | None = None

    def record_event(
        self,
        *,
        run_id: str,
        round_index: int,
        workflow_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> RawMemoryEvent:
        event = RawMemoryEvent(
            event_id=f"amev_{uuid.uuid4().hex}",
            run_id=run_id,
            round_index=int(round_index),
            workflow_id=workflow_id,
            event_type=event_type,
            payload=copy.deepcopy(payload),
            created_at=now_us(),
        )
        self.raw_events.append(event)
        if self.session_store is not None and hasattr(self.session_store, "add_memory_event"):
            self.session_store.add_memory_event(asdict(event))
        return event

    def record_round_raw(
        self,
        *,
        run_id: str,
        round_index: int,
        workflow_id: str | None,
        workflow_summary_raw: dict[str, Any],
        candidate_snapshot_id: str | None,
        survivor_snapshot_id: str | None,
    ) -> RoundLedgerRecord:
        event = self.record_event(
            run_id=run_id,
            round_index=round_index,
            workflow_id=workflow_id,
            event_type="workflow_round_raw",
            payload=workflow_summary_raw,
        )
        record = RoundLedgerRecord(
            round_index=int(round_index),
            workflow_id=workflow_id,
            workflow_summary_raw=copy.deepcopy(workflow_summary_raw),
            candidate_snapshot_id=candidate_snapshot_id,
            survivor_snapshot_id=survivor_snapshot_id,
        )
        self.round_ledgers.append(record)
        legacy = {
            "round_index": round_index,
            "workflow_id": workflow_id,
            "state": workflow_summary_raw.get("state"),
            "raw_event_id": event.event_id,
            "candidate_snapshot_id": candidate_snapshot_id,
            "survivor_snapshot_id": survivor_snapshot_id,
            "workflow_summary_raw": copy.deepcopy(workflow_summary_raw),
            "empty_candidate_events": _empty_events(workflow_summary_raw),
            "candidate_capability_filter_events": _capability_filter_events(
                workflow_summary_raw
            ),
            "skipped_nodes": _skipped_nodes(workflow_summary_raw),
        }
        self.round_records.append(legacy)
        return record

    def record_branch(
        self,
        round_index: int,
        branch_decision: dict[str, Any],
        *,
        run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> None:
        entry = {"round_index": round_index, "branch_decision": dict(branch_decision)}
        self.controller_branches.append(entry)
        if run_id is not None:
            event = self.record_event(
                run_id=run_id,
                round_index=round_index,
                workflow_id=workflow_id,
                event_type="branch_decision",
                payload=dict(branch_decision),
            )
            if self.round_ledgers and self.round_ledgers[-1].round_index == round_index:
                self.round_ledgers[-1].branch_event_ids.append(event.event_id)
        if self.round_records and self.round_records[-1].get("round_index") == round_index:
            self.round_records[-1]["branch_decision"] = dict(branch_decision)

    def record_decision(
        self,
        round_index: int,
        decision: dict[str, Any],
        *,
        run_id: str | None = None,
        workflow_id: str | None = None,
    ) -> None:
        entry = {"round_index": round_index, "decision": dict(decision)}
        self.raw_decision_outputs.append(entry)
        if run_id is not None:
            event = self.record_event(
                run_id=run_id,
                round_index=round_index,
                workflow_id=workflow_id,
                event_type="decision",
                payload=dict(decision),
            )
            if self.round_ledgers and self.round_ledgers[-1].round_index == round_index:
                self.round_ledgers[-1].decision_event_ids.append(event.event_id)
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
                    "candidate_capability_filter_events": record.get(
                        "candidate_capability_filter_events", []
                    )[:10],
                    "skipped_nodes": record.get("skipped_nodes", [])[:10],
                    "current_round_candidate_summary": record.get(
                        "current_round_candidate_summary", {}
                    ),
                    "current_round_merge_summary": record.get(
                        "current_round_merge_summary", {}
                    ),
                    "survivor_pool_summary": record.get("survivor_pool_summary", {}),
                    "survivor_merge_summary": record.get("survivor_merge_summary", {}),
                    "max_round_reached": record.get("max_round_reached"),
                    "branch_decision": record.get("branch_decision"),
                }
                for record in rounds
            ],
            "survivor_pool_summary": (
                dict(self.round_records[-1].get("survivor_pool_summary") or {})
                if self.round_records
                else _candidate_summary([])
            ),
        }


__all__ = ["MemoryBank", "RawMemoryEvent", "RoundLedgerRecord"]
