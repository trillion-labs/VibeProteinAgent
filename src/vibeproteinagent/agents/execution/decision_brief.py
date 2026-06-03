from __future__ import annotations

from typing import Any

from .candidate_pool import (
    candidate_pool_items_from_summary,
    safe_candidate_pool_stats_from_summary,
    safe_candidate_ref_cards_from_summary,
)

_DECISION_OPTIONS = [
    "submit_candidate",
    "next_round_with_surviving_candidate",
    "next_round_with_no_acceptable_candidate",
]


class DecisionBriefBuilder:
    """Build a compact decision-task view without transferring candidate payloads."""

    def build(
        self,
        *,
        round_record: dict[str, Any] | None,
        prompt_context: dict[str, Any] | None,
        outcome_summary: dict[str, Any] | None,
        success_criteria: dict[str, Any] | None,
        deliverable_contract: dict[str, Any] | None,
        local_decision: dict[str, Any] | None,
        round_index: int,
        max_rounds: int,
    ) -> dict[str, Any]:
        try:
            record = round_record if isinstance(round_record, dict) else {}
            context = prompt_context if isinstance(prompt_context, dict) else {}
            summary = outcome_summary if isinstance(outcome_summary, dict) else {}
            contract = deliverable_contract if isinstance(deliverable_contract, dict) else {}
            local = local_decision if isinstance(local_decision, dict) else {}
            latest = context.get("latest_round")
            if not isinstance(latest, dict):
                latest = {}
            current_summary = _summary_dict(latest.get("candidate_summary"))
            survivor_summary = _summary_dict(
                latest.get("survivor_summary") or record.get("survivor_pool_summary")
            )
            workflow_summary = _raw_workflow_summary(record)
            current_cards = safe_candidate_ref_cards_from_summary(
                workflow_summary,
                deliverable_contract=contract,
            )
            survivor_cards = _safe_survivor_ref_cards_from_summary(
                survivor_summary,
                deliverable_contract=contract,
            )
            recommended = _merge_recommended_cards(current_cards, survivor_cards)
            current_pool_stats = safe_candidate_pool_stats_from_summary(
                workflow_summary,
                deliverable_contract=contract,
            )
            pool_stats = _combined_pool_stats(
                current_pool_stats=current_pool_stats,
                current_candidate_ids=_candidate_ids_from_workflow_summary(
                    workflow_summary
                ),
                survivor_cards=survivor_cards,
                previous_survivor_pool_stats=_summary_dict(
                    record.get("previous_survivor_pool_stats")
                ),
            )
            required = _string_list(contract.get("required_candidate_capabilities"))
            optional = _string_list(contract.get("optional_candidate_capabilities"))
            return {
                "version": "decision_brief_v1",
                "round": {
                    "index": _safe_int(round_index, default=0),
                    "number": _safe_int(round_index, default=0) + 1,
                    "max_rounds": _safe_int(max_rounds, default=0),
                    "remaining_rounds": max(
                        0,
                        _safe_int(max_rounds, default=0)
                        - _safe_int(round_index, default=0)
                        - 1,
                    ),
                    "workflow_id": _safe_string(
                        record.get("workflow_id") or summary.get("workflow_id")
                    ),
                    "workflow_state": _safe_string(
                        record.get("state") or summary.get("state")
                    ),
                },
                "user_goal": {
                    "deliverable": _safe_string(
                        contract.get("answer_type") or contract.get("deliverable_kind")
                    ),
                    "requested_candidate_count": _requested_candidate_count(contract),
                    "required_capabilities": required,
                    "optional_capabilities": optional,
                },
                "decision_options": list(_DECISION_OPTIONS),
                "candidate_pool": _candidate_pool_summary(
                    current_summary=current_summary,
                    survivor_summary=survivor_summary,
                    pool_stats=pool_stats,
                ),
                "recommended_candidates": recommended,
                "blocking_issues": _blocking_issues(latest),
                "round_diagnosis": _round_diagnosis(record, latest),
                "controller_hint": _controller_hint(local),
                "success_criteria_keys": sorted(
                    str(key) for key in (success_criteria or {}).keys()
                ),
            }
        except Exception:
            return self.empty(
                round_index=round_index,
                max_rounds=max_rounds,
                local_decision=local_decision,
            )

    def empty(
        self,
        *,
        round_index: int = 0,
        max_rounds: int = 0,
        local_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        local = local_decision if isinstance(local_decision, dict) else {}
        return {
            "version": "decision_brief_v1",
            "round": {
                "index": _safe_int(round_index, default=0),
                "number": _safe_int(round_index, default=0) + 1,
                "max_rounds": _safe_int(max_rounds, default=0),
                "remaining_rounds": max(
                    0,
                    _safe_int(max_rounds, default=0) - _safe_int(round_index, default=0) - 1,
                ),
                "workflow_id": None,
                "workflow_state": None,
            },
            "user_goal": {
                "deliverable": None,
                "requested_candidate_count": None,
                "required_capabilities": [],
                "optional_capabilities": [],
            },
            "decision_options": list(_DECISION_OPTIONS),
            "candidate_pool": {
                "current_round_count": 0,
                "survivor_count": 0,
                "submit_ready_count": 0,
                "missing_capability_count": 0,
                "truncated": False,
            },
            "recommended_candidates": [],
            "blocking_issues": [],
            "round_diagnosis": {
                "empty_candidate_events": 0,
                "failed_nodes": 0,
                "skipped_nodes": 0,
                "capability_filter_events": [],
            },
            "controller_hint": _controller_hint(local),
            "success_criteria_keys": [],
        }


def _raw_workflow_summary(record: dict[str, Any]) -> dict[str, Any]:
    raw = record.get("workflow_summary_raw")
    if isinstance(raw, dict) and isinstance(raw.get("nodes"), list):
        return raw
    return {}


def _summary_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_survivor_ref_cards_from_summary(
    summary: dict[str, Any],
    *,
    deliverable_contract: dict[str, Any],
    limit: int = 10,
) -> list[dict[str, Any]]:
    items = summary.get("items")
    if not isinstance(items, list):
        return []
    required = _string_list(deliverable_contract.get("required_candidate_capabilities"))
    cards: list[dict[str, Any]] = []
    for item in items:
        if len(cards) >= limit:
            break
        if not isinstance(item, dict):
            continue
        candidate_id = _safe_string(item.get("candidate_id"))
        if candidate_id is None:
            continue
        source = item.get("source")
        if not isinstance(source, dict):
            source = {}
        capabilities = _safe_capabilities(item)
        missing = sorted(cap for cap in required if cap not in capabilities)
        card = {
            "candidate_id": candidate_id,
            "job_id": _safe_string(source.get("job_id")),
            "rank": len(cards) + 1,
            "source_tool": _safe_string(source.get("tool")),
            "capabilities_satisfied": not missing,
            "matched_capabilities": sorted(capabilities),
            "missing_capabilities": missing,
            "evidence": _survivor_evidence(item, capabilities),
            "warnings": _survivor_warnings(missing),
            "source_surface": "survivor_summary",
        }
        cards.append({key: value for key, value in card.items() if value is not None})
    return cards


def _merge_recommended_cards(
    current_cards: list[dict[str, Any]],
    survivor_cards: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in [*current_cards, *survivor_cards]:
        candidate_id = _safe_string(card.get("candidate_id")) if isinstance(card, dict) else None
        if candidate_id is None or candidate_id in seen:
            continue
        seen.add(candidate_id)
        merged.append(dict(card))
        if len(merged) >= limit:
            break
    for rank, card in enumerate(merged, start=1):
        card["rank"] = rank
    return merged


def _combined_pool_stats(
    *,
    current_pool_stats: dict[str, int],
    current_candidate_ids: set[str],
    survivor_cards: list[dict[str, Any]],
    previous_survivor_pool_stats: dict[str, Any],
) -> dict[str, int]:
    submit_ready_count = _safe_int(
        current_pool_stats.get("submit_ready_count"),
        default=0,
    )
    missing_capability_count = _safe_int(
        current_pool_stats.get("missing_capability_count"),
        default=0,
    )
    if previous_survivor_pool_stats:
        return {
            "submit_ready_count": submit_ready_count
            + _safe_int(
                previous_survivor_pool_stats.get("submit_ready_count"),
                default=0,
            ),
            "missing_capability_count": missing_capability_count
            + _safe_int(
                previous_survivor_pool_stats.get("missing_capability_count"),
                default=0,
            ),
        }
    for card in survivor_cards:
        candidate_id = _safe_string(card.get("candidate_id"))
        if candidate_id is None or candidate_id in current_candidate_ids:
            continue
        if card.get("capabilities_satisfied") is False:
            missing_capability_count += 1
        else:
            submit_ready_count += 1
    return {
        "submit_ready_count": submit_ready_count,
        "missing_capability_count": missing_capability_count,
    }


def _candidate_ids_from_workflow_summary(summary: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in candidate_pool_items_from_summary(summary):
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            continue
        candidate_id = _safe_string(candidate.get("candidate_id"))
        if candidate_id is not None:
            ids.add(candidate_id)
    return ids


def _safe_capabilities(item: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    raw = item.get("capabilities")
    explicitly_false: set[str] = set()
    if isinstance(raw, dict):
        out.update(
            str(name)
            for name, value in raw.items()
            if str(name).startswith("has_") and value is True
        )
        explicitly_false.update(
            str(name)
            for name, value in raw.items()
            if str(name).startswith("has_") and value is False
        )
    elif isinstance(raw, list):
        out.update(str(value) for value in raw if isinstance(value, str))
    if item.get("has_sequence") is True and "has_sequence" not in explicitly_false:
        out.add("has_sequence")
    if item.get("has_structure") is True and "has_structure" not in explicitly_false:
        out.add("has_structure")
    score_keys = item.get("score_keys")
    if isinstance(score_keys, list) and score_keys and "has_candidate_scores" not in explicitly_false:
        out.add("has_candidate_scores")
    return out


def _survivor_evidence(item: dict[str, Any], capabilities: set[str]) -> list[str]:
    evidence = ["survivor"]
    if "has_sequence" in capabilities:
        evidence.append("sequence_available")
    if "has_backbone_structure" in capabilities:
        evidence.append("backbone_structure_available")
    if "has_full_structure" in capabilities:
        evidence.append("full_structure_available")
    if "has_structure" in capabilities:
        evidence.append("structure_available")
    if "has_candidate_scores" in capabilities:
        evidence.append("candidate_scores_available")
    score_keys = item.get("score_keys")
    if isinstance(score_keys, list) and score_keys:
        evidence.append("scored")
    return evidence


def _survivor_warnings(missing: list[str]) -> list[str]:
    if not missing:
        return []
    return ["missing_required_capabilities: " + ", ".join(missing)]


def _candidate_pool_summary(
    *,
    current_summary: dict[str, Any],
    survivor_summary: dict[str, Any],
    pool_stats: dict[str, int],
) -> dict[str, Any]:
    return {
        "current_round_count": _safe_int(current_summary.get("count"), default=0),
        "survivor_count": _safe_int(survivor_summary.get("count"), default=0),
        "submit_ready_count": _safe_int(pool_stats.get("submit_ready_count"), default=0),
        "missing_capability_count": _safe_int(
            pool_stats.get("missing_capability_count"),
            default=0,
        ),
        "truncated": bool(
            current_summary.get("truncated")
            or survivor_summary.get("truncated")
        ),
    }


def _blocking_issues(latest: dict[str, Any]) -> list[dict[str, Any]]:
    cards = latest.get("failure_cards")
    if not isinstance(cards, list):
        return []
    out: list[dict[str, Any]] = []
    for card in cards[:5]:
        if not isinstance(card, dict):
            continue
        out.append(
            {
                key: value
                for key, value in {
                    "kind": _safe_string(card.get("kind")),
                    "node_id": _safe_string(card.get("node_id")),
                    "tool": _safe_string(card.get("tool")),
                    "phase": _safe_string(card.get("phase")),
                    "error_kind": _safe_string(card.get("error_kind")),
                    "message": _safe_string(card.get("message")),
                    "retryable": card.get("retryable") if isinstance(card.get("retryable"), bool) else None,
                }.items()
                if value is not None
            }
        )
    return out


def _round_diagnosis(record: dict[str, Any], latest: dict[str, Any]) -> dict[str, Any]:
    return {
        "empty_candidate_events": len(_list(record.get("empty_candidate_events"))),
        "failed_nodes": _failed_node_count(_raw_workflow_summary(record)),
        "skipped_nodes": len(_list(record.get("skipped_nodes"))),
        "capability_filter_events": _capability_filter_event_notes(latest, record),
    }


def _failed_node_count(workflow_summary: dict[str, Any]) -> int:
    count = 0
    for node in _list(workflow_summary.get("nodes")):
        if isinstance(node, dict) and node.get("state") in {"failed", "error"}:
            count += 1
    return count


def _capability_filter_event_notes(
    latest: dict[str, Any],
    record: dict[str, Any],
) -> list[str]:
    events = latest.get("candidate_capability_filter_events")
    if not isinstance(events, list):
        events = record.get("candidate_capability_filter_events")
    notes: list[str] = []
    for event in _list(events)[:5]:
        if not isinstance(event, dict):
            continue
        tool = _safe_string(event.get("tool"))
        skipped = event.get("skipped_candidate_count")
        compatible = event.get("compatible_candidate_count")
        if isinstance(skipped, int) and skipped > 0:
            notes.append(f"{tool or 'candidate consumer'} skipped {skipped} candidate(s)")
        elif isinstance(compatible, int):
            notes.append(f"{tool or 'candidate consumer'} compatible candidates: {compatible}")
        else:
            status = _safe_string(event.get("status"))
            if status:
                notes.append(f"{tool or 'candidate consumer'} capability filter: {status}")
    return notes


def _controller_hint(local: dict[str, Any]) -> dict[str, Any]:
    next_round = local.get("next_round")
    reason = None
    if isinstance(next_round, dict):
        reason = _safe_string(next_round.get("reason"))
    return {
        "local_decision": _safe_string(local.get("decision")),
        "reason": reason,
    }


def _requested_candidate_count(contract: dict[str, Any]) -> int | None:
    for key in ("requested_candidate_count", "num_candidates", "candidate_count", "top_k"):
        value = contract.get(key)
        count = _safe_int(value, default=-1)
        if count >= 1:
            return count
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(str(item) for item in value if isinstance(item, str) and item)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
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


__all__ = ["DecisionBriefBuilder"]
