from __future__ import annotations

from typing import Any

from .memory_bank import MemoryBank

_SUMMARY_LIMIT = 10
_CARD_LIMIT = 20
_FAILURE_LIMIT = 12
_BLOCKED_KEYS = {
    "candidate",
    "candidates",
    "current_round_candidates",
    "survivor_pool",
    "surviving_candidates",
    "sequence",
    "sequences",
    "scores",
    "structure_path",
    "structure_paths",
    "backbone_path",
    "backbone_paths",
    "full_structure_path",
    "full_structure_paths",
}


class PromptContextBuilder:
    """Build lossy prompt context from round memory.

    The output is safe to pass back into planning prompts: it preserves counts,
    capabilities, compact trajectory, and failure signals while omitting raw
    candidates, full sequences, raw scores, and structure path payloads.
    """

    def build(
        self,
        *,
        previous_digest: dict[str, Any] | None,
        latest_round_record: dict[str, Any],
        memory_bank: MemoryBank | None = None,
    ) -> dict[str, Any]:
        try:
            previous = previous_digest if isinstance(previous_digest, dict) else {}
            latest = _latest_round_context(latest_round_record)
            round_count = max(
                _safe_int(previous.get("round_count"), default=0),
                _safe_int(latest_round_record.get("round_index"), default=0) + 1,
            )
            previous_survivor = _previous_survivor_summary(previous)
            return {
                "round_count": round_count,
                "previous": {
                    "round_count": _safe_int(previous.get("round_count"), default=0),
                    "survivor_summary": previous_survivor,
                    "survivor_pool_summary": previous_survivor,
                },
                "latest_round": latest,
                "survivor_summary": latest["survivor_summary"],
                "survivor_pool_summary": latest["survivor_summary"],
            }
        except Exception:
            record = _fallback_record(memory_bank)
            if record is not None and record is not latest_round_record:
                try:
                    return self.build(
                        previous_digest=previous_digest,
                        latest_round_record=record,
                        memory_bank=None,
                    )
                except Exception:
                    pass
            return _empty_context()


def _latest_round_context(record: dict[str, Any]) -> dict[str, Any]:
    candidate_summary = _candidate_summary(
        record.get("current_round_candidate_summary"),
    )
    survivor_summary = _candidate_summary(
        record.get("survivor_pool_summary"),
    )
    current_merge_summary = _merge_summary(record.get("current_round_merge_summary"))
    survivor_merge_summary = _merge_summary(record.get("survivor_merge_summary"))
    merge_summary = {
        "current_round": current_merge_summary,
        "survivor": survivor_merge_summary,
    }
    capability_events = _safe_cards(
        record.get("candidate_capability_filter_events"), limit=_SUMMARY_LIMIT
    )
    empty_events = _safe_cards(record.get("empty_candidate_events"), limit=5)
    skipped_nodes = _safe_cards(record.get("skipped_nodes"), limit=_SUMMARY_LIMIT)
    failure_cards = _failure_cards(record)
    latest = {
        "round_index": record.get("round_index"),
        "workflow_id": record.get("workflow_id"),
        "state": record.get("state"),
        "candidate_summary": candidate_summary,
        "survivor_summary": survivor_summary,
        "capability_summary": _capability_summary(
            record.get("capability_summary"),
            candidate_summary,
            survivor_summary,
            capability_events,
        ),
        "merge_summary": merge_summary,
        "failure_cards": failure_cards,
        "trajectory_cards": _trajectory_cards(record),
        "next_move_hints": _next_move_hints(
            record,
            candidate_summary=candidate_summary,
            survivor_summary=survivor_summary,
            failure_cards=failure_cards,
        ),
        "branch_decision": _safe_card(record.get("branch_decision")),
        # Backward-compatible aliases for callers that still read the old digest.
        "empty_candidate_events": empty_events,
        "candidate_capability_filter_events": capability_events,
        "skipped_nodes": skipped_nodes,
        "current_round_candidate_summary": candidate_summary,
        "survivor_pool_summary": survivor_summary,
        "current_round_merge_summary": current_merge_summary,
        "survivor_merge_summary": survivor_merge_summary,
    }
    return latest


def _candidate_summary(
    summary: Any,
    *,
    limit: int = _SUMMARY_LIMIT,
) -> dict[str, Any]:
    count = 0
    truncated = False
    source_items: list[Any] = []
    if isinstance(summary, dict):
        count = _safe_int(summary.get("count"), default=0)
        truncated = bool(summary.get("truncated"))
        items = summary.get("items")
        if isinstance(items, list):
            source_items = items
    items = [_candidate_card(item) for item in source_items[:limit] if isinstance(item, dict)]
    if count == 0 and source_items:
        count = len(source_items)
    return {
        "count": count,
        "items": items,
        "truncated": truncated or len(source_items) > limit,
    }


def _candidate_card(candidate: dict[str, Any]) -> dict[str, Any]:
    capabilities = _capabilities(candidate.get("capabilities"))
    source = _source_card(candidate.get("source"))
    score_keys = _score_keys(candidate)
    card: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id"),
        "source": source,
        "capabilities": capabilities,
        "available_modalities": _available_modalities(candidate, capabilities),
        "has_sequence": bool(candidate.get("has_sequence") or candidate.get("sequence")),
        "score_keys": score_keys,
    }
    sequence = candidate.get("sequence")
    if isinstance(sequence, str):
        card["sequence_length"] = len(sequence)
    elif isinstance(candidate.get("sequence_length"), int):
        card["sequence_length"] = candidate["sequence_length"]
    return {key: value for key, value in card.items() if value not in (None, {}, [])}


def _source_card(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    allowed = {
        "tool",
        "node_id",
        "job_id",
        "adapter",
        "workflow_id",
        "source_kind",
        "local_id",
    }
    return {
        key: _safe_scalar(value)
        for key, value in source.items()
        if key in allowed and _safe_scalar(value) is not None
    }


def _capabilities(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): bool(item)
        for key, item in value.items()
        if isinstance(item, bool) or str(key).startswith("has_")
    }


def _available_modalities(
    candidate: dict[str, Any], capabilities: dict[str, bool]
) -> list[str]:
    modalities = [key.removeprefix("has_") for key, value in capabilities.items() if value]
    if candidate.get("has_sequence") or candidate.get("sequence"):
        modalities.append("sequence")
    return sorted(set(modalities))


def _score_keys(candidate: dict[str, Any]) -> list[str]:
    value = candidate.get("score_keys")
    if isinstance(value, list):
        return sorted(str(item) for item in value if isinstance(item, str))
    scores = candidate.get("scores")
    if isinstance(scores, dict):
        return sorted(str(key) for key in scores)
    return []


def _capability_summary(
    canonical_summary: Any,
    candidate_summary: dict[str, Any],
    survivor_summary: dict[str, Any],
    capability_events: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(canonical_summary, dict) and canonical_summary:
        summary = _safe_capability_summary(canonical_summary)
        if summary:
            return summary
    return {
        "current_round": _capability_counts(candidate_summary),
        "survivor_pool": _capability_counts(survivor_summary),
        "filter_event_count": len(capability_events),
    }


def _capability_counts(summary: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for item in summary.get("items") or []:
        if not isinstance(item, dict):
            continue
        capabilities = item.get("capabilities")
        if not isinstance(capabilities, dict):
            continue
        for key, value in capabilities.items():
            if value is True:
                counts[key] = counts.get(key, 0) + 1
    return {
        "count": _safe_int(summary.get("count"), default=0),
        "capability_counts": counts,
        "truncated": bool(summary.get("truncated")),
    }


def _safe_capability_summary(value: dict[str, Any]) -> dict[str, Any]:
    summary = _safe_capability_value(value, depth=0)
    return summary if isinstance(summary, dict) else {}


def _safe_capability_value(value: Any, *, depth: int) -> Any:
    scalar = _safe_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, dict) and depth < 4:
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _is_blocked_key(str(key)):
                continue
            safe_value = _safe_capability_value(item, depth=depth + 1)
            if safe_value is not None:
                out[str(key)] = safe_value
        return out
    if isinstance(value, list) and depth < 3:
        safe_items = []
        for item in value[:5]:
            safe_value = _safe_capability_value(item, depth=depth + 1)
            if safe_value is not None:
                safe_items.append(safe_value)
        return safe_items
    return None


def _merge_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if _is_blocked_key(key):
            continue
        if key == "duplicate_groups" and isinstance(item, list):
            out["duplicate_group_count"] = len(item)
            continue
        if key == "conflicts" and isinstance(item, list):
            out["conflict_count"] = len(item)
            continue
        scalar = _safe_scalar(item)
        if scalar is not None:
            out[str(key)] = scalar
        elif isinstance(item, list):
            out[f"{key}_count"] = len(item)
        elif isinstance(item, dict):
            out[f"{key}_key_count"] = len(item)
    return out


def _failure_cards(record: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for event in _as_list(record.get("failure_cards")):
        card = _safe_card(event)
        if card:
            cards.append(card)
    for event in _as_list(record.get("empty_candidate_events")):
        card = _safe_card(event)
        card["kind"] = "empty_candidate_event"
        cards.append(card)
    for event in _as_list(record.get("candidate_capability_filter_events")):
        card = _safe_card(event)
        card["kind"] = "capability_filter"
        cards.append(card)
    for node in _as_list(record.get("skipped_nodes")):
        card = _safe_card(node)
        card["kind"] = "skipped_node"
        cards.append(card)
    workflow_summary = _workflow_summary(record)
    if isinstance(workflow_summary, dict):
        for node in _as_list(workflow_summary.get("nodes")):
            if not isinstance(node, dict) or node.get("state") not in {"failed", "error"}:
                continue
            output = node.get("output") if isinstance(node.get("output"), dict) else {}
            cards.append(
                {
                    "kind": "failed_node",
                    "node_id": node.get("node_id"),
                    "tool": node.get("tool"),
                    "state": node.get("state"),
                    "error": _safe_card(output.get("error")),
                }
            )
    return [_drop_empty(card) for card in cards[:_FAILURE_LIMIT]]


def _trajectory_cards(record: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    workflow_summary = _workflow_summary(record)
    if isinstance(workflow_summary, dict):
        for node in _as_list(workflow_summary.get("nodes"))[:_CARD_LIMIT]:
            if not isinstance(node, dict):
                continue
            output = node.get("output") if isinstance(node.get("output"), dict) else {}
            card: dict[str, Any] = {
                "node_id": node.get("node_id"),
                "tool": node.get("tool"),
                "state": node.get("state"),
            }
            candidates = output.get("candidates")
            if isinstance(candidates, list):
                card["candidate_count"] = len(candidates)
            elif _safe_scalar(output.get("candidate_count")) is not None:
                card["candidate_count"] = output.get("candidate_count")
            if isinstance(output.get("empty_candidate_event"), dict):
                card["empty_candidate_event"] = True
            event = output.get("candidate_capability_filter_event")
            if isinstance(event, dict):
                card["capability_filter_status"] = event.get("status")
                card["compatible_candidate_count"] = event.get(
                    "compatible_candidate_count"
                )
            cards.append(_drop_empty(card))
    branch = _safe_card(record.get("branch_decision"))
    if branch:
        cards.append({"kind": "branch_decision", **branch})
    return cards[:_CARD_LIMIT]


def _workflow_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    raw = record.get("workflow_summary_raw")
    if isinstance(raw, dict):
        return raw
    legacy = record.get("workflow_summary")
    if isinstance(legacy, dict):
        return legacy
    return None


def _next_move_hints(
    record: dict[str, Any],
    *,
    candidate_summary: dict[str, Any],
    survivor_summary: dict[str, Any],
    failure_cards: list[dict[str, Any]],
) -> list[str]:
    hints: list[str] = []
    branch = record.get("branch_decision")
    if isinstance(branch, dict):
        action = branch.get("action")
        reason = branch.get("reason")
        if action:
            hint = f"Controller chose {action}"
            if reason:
                hint = f"{hint}: {reason}"
            hints.append(hint)
    candidate_count = _safe_int(candidate_summary.get("count"), default=0)
    survivor_count = _safe_int(survivor_summary.get("count"), default=0)
    if candidate_count == 0 and survivor_count > 0:
        hints.append("Use survivor pool; latest round produced no candidates.")
    elif candidate_count == 0 and survivor_count == 0:
        hints.append("No candidate pool is available; select a producer or fix upstream failures.")
    if failure_cards and candidate_count == 0:
        hints.append("Address failure cards before adding candidate-consuming tools.")
    return hints[:5]


def _previous_survivor_summary(previous: dict[str, Any]) -> dict[str, Any]:
    latest = previous.get("latest_round")
    if isinstance(latest, dict):
        for key in ("survivor_summary", "survivor_pool_summary"):
            if isinstance(latest.get(key), dict):
                return _candidate_summary(latest[key])
    for key in ("survivor_summary", "survivor_pool_summary"):
        if isinstance(previous.get(key), dict):
            return _candidate_summary(previous[key])
    return {"count": 0, "items": [], "truncated": False}


def _safe_cards(value: Any, *, limit: int) -> list[dict[str, Any]]:
    return [_safe_card(item) for item in _as_list(value)[:limit] if isinstance(item, dict)]


def _safe_card(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if not isinstance(value, dict) or depth > 2:
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if _is_blocked_key(str(key)):
            continue
        safe_value = _safe_value(item, depth=depth + 1)
        if safe_value is not None:
            out[str(key)] = safe_value
    return out


def _safe_value(value: Any, *, depth: int) -> Any:
    scalar = _safe_scalar(value)
    if scalar is not None:
        return scalar
    if isinstance(value, list):
        safe_items = []
        for item in value[:5]:
            item_scalar = _safe_scalar(item)
            if item_scalar is not None:
                safe_items.append(item_scalar)
            elif isinstance(item, dict) and depth < 2:
                safe_items.append(_safe_card(item, depth=depth + 1))
        return safe_items
    if isinstance(value, dict) and depth < 2:
        return _safe_card(value, depth=depth + 1)
    return None


def _safe_scalar(value: Any) -> Any:
    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:240]
    return None


def _safe_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _is_blocked_key(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized in _BLOCKED_KEYS
        or normalized.endswith("_path")
        or normalized.endswith("_paths")
    )


def _drop_empty(card: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in card.items() if value not in (None, {}, [])}


def _fallback_record(memory_bank: MemoryBank | None) -> dict[str, Any] | None:
    if memory_bank is None:
        return None
    try:
        return memory_bank.latest_round_record()
    except Exception:
        return None


def _empty_context() -> dict[str, Any]:
    empty_summary = {"count": 0, "items": [], "truncated": False}
    latest = {
        "candidate_summary": empty_summary,
        "survivor_summary": empty_summary,
        "capability_summary": {
            "current_round": {"count": 0, "capability_counts": {}, "truncated": False},
            "survivor_pool": {"count": 0, "capability_counts": {}, "truncated": False},
            "filter_event_count": 0,
        },
        "merge_summary": {"current_round": {}, "survivor": {}},
        "failure_cards": [],
        "trajectory_cards": [],
        "next_move_hints": [],
        "current_round_candidate_summary": empty_summary,
        "survivor_pool_summary": empty_summary,
        "current_round_merge_summary": {},
        "survivor_merge_summary": {},
    }
    return {
        "round_count": 0,
        "previous": {
            "round_count": 0,
            "survivor_summary": empty_summary,
            "survivor_pool_summary": empty_summary,
        },
        "latest_round": latest,
        "survivor_summary": empty_summary,
        "survivor_pool_summary": empty_summary,
    }


__all__ = ["PromptContextBuilder"]
