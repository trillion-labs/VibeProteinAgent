from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vibeproteinagent.agents.planning.candidate_merge import (
    merge_candidate_records,
    summarize_candidate_merge_audit,
)


def _is_candidate_v1(candidate: Any) -> bool:
    return (
        isinstance(candidate, dict)
        and candidate.get("schema_version") == "candidate_v1"
        and isinstance(candidate.get("candidate_id"), str)
        and isinstance(candidate.get("source"), dict)
        and isinstance(candidate["source"].get("tool"), str)
        and isinstance(candidate["source"].get("local_id"), str)
    )


def candidate_capability_names(candidate: dict[str, Any], *, key: str) -> set[str]:
    del key
    capabilities: set[str] = set()
    raw = candidate.get("capabilities")
    explicitly_false: set[str] = set()
    if isinstance(raw, dict):
        capabilities.update(
            str(name)
            for name, value in raw.items()
            if name.startswith("has_") and value is True
        )
        explicitly_false.update(
            str(name)
            for name, value in raw.items()
            if name.startswith("has_") and value is False
        )
    elif isinstance(raw, list):
        capabilities.update(str(item) for item in raw if isinstance(item, str))
    if (
        "has_sequence" not in explicitly_false
        and isinstance(candidate.get("sequence"), str)
        and candidate.get("sequence").strip()
    ):
        capabilities.add("has_sequence")
    if (
        "has_backbone_structure" not in explicitly_false
        and (candidate.get("structure_path") or candidate.get("backbone_path"))
    ):
        capabilities.add("has_backbone_structure")
    if (
        "has_full_structure" not in explicitly_false
        and (candidate.get("full_structure_path") or candidate.get("structure_path"))
    ):
        capabilities.add("has_full_structure")
    scores = candidate.get("scores")
    if "has_candidate_scores" not in explicitly_false and isinstance(scores, dict) and scores:
        capabilities.add("has_candidate_scores")
    return capabilities


def _canonicalize_candidate_pool_item(
    item: Any,
    *,
    container_key: str,
    tool: str,
    job_id: str,
    index: int,
) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if _is_candidate_v1(item):
        if not _material_candidate_capabilities(
            candidate_capability_names(item, key=container_key)
        ):
            return None
        return dict(item)
    capabilities = candidate_capability_names(item, key=container_key)
    if not _material_candidate_capabilities(capabilities):
        return None
    explicit_candidate_id = item.get("candidate_id")
    local_id = str(explicit_candidate_id or item.get("id") or f"{container_key}:{index}")
    source_tool = tool or str(item.get("tool") or container_key)
    candidate_id = (
        str(explicit_candidate_id).strip()
        if explicit_candidate_id is not None and str(explicit_candidate_id).strip()
        else f"{source_tool}:{job_id or 'unknown'}:{local_id}"
    )
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": source_tool,
            "local_id": local_id,
        },
        "capabilities": {cap: True for cap in sorted(capabilities)},
    }
    for field in (
        "sequence",
        "structure_path",
        "backbone_path",
        "full_structure_path",
        "scores",
        "artifacts",
    ):
        if item.get(field) is not None:
            candidate[field] = item[field]
    return candidate


def candidate_pool_items_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return _merge_candidate_pool_items(_raw_candidate_pool_items_from_summary(summary))


def candidate_pool_merge_summary_from_summary(summary: dict[str, Any]) -> dict[str, int]:
    result = _merge_candidate_pool_items_result(_raw_candidate_pool_items_from_summary(summary))
    return summarize_candidate_merge_audit(result.audit)


def _raw_candidate_pool_items_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node in summary.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        output = node.get("output")
        if not isinstance(output, dict):
            continue
        job_id = str(node.get("job_id") or "")
        tool = str(node.get("tool") or "")
        for key in _candidate_container_keys(output):
            values = output.get(key)
            if not isinstance(values, list):
                continue
            for index, item in enumerate(values):
                candidate = _canonicalize_candidate_pool_item(
                    item,
                    container_key=key,
                    tool=tool,
                    job_id=job_id,
                    index=index,
                )
                if candidate is None:
                    continue
                candidates.append(
                    {
                        "candidate": candidate,
                        "job_id": job_id,
                        "tool": tool,
                        "container_key": key,
                        "index": index,
                    }
                )
    return candidates


def _candidate_container_keys(output: dict[str, Any]) -> tuple[str, ...]:
    canonical = tuple(
        key
        for key in ("candidates", "ranked_candidates")
        if isinstance(output.get(key), list) and output.get(key)
    )
    if canonical:
        return canonical
    return tuple(
        key
        for key in ("representative_sequences",)
        if isinstance(output.get(key), list) and output.get(key)
    )


def _merge_candidate_pool_items(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _merge_candidate_pool_items_result(raw_items).items


@dataclass(frozen=True)
class _CandidatePoolItemsMergeResult:
    items: list[dict[str, Any]]
    audit: dict[str, Any]


def _merge_candidate_pool_items_result(
    raw_items: list[dict[str, Any]],
) -> _CandidatePoolItemsMergeResult:
    candidates = [
        item["candidate"]
        for item in raw_items
        if isinstance(item.get("candidate"), dict)
    ]
    result = merge_candidate_records(
        candidates,
        strip_private_keys=False,
        include_private_positions=True,
    )
    merged_items: list[dict[str, Any]] = []
    for merged_candidate in result.candidates:
        representative_position = _candidate_merge_representative_position(merged_candidate)
        representative_item = (
            raw_items[representative_position]
            if representative_position is not None and 0 <= representative_position < len(raw_items)
            else None
        )
        if not isinstance(representative_item, dict):
            positions = _candidate_merge_positions(merged_candidate)
            representative_item = (
                raw_items[positions[0]]
                if positions and 0 <= positions[0] < len(raw_items)
                else {}
            )
        merged_items.append(
            {
                **representative_item,
                "candidate": _strip_candidate_merge_private(merged_candidate),
            }
        )
    return _CandidatePoolItemsMergeResult(items=merged_items, audit=result.audit)


def _candidate_merge_positions(candidate: dict[str, Any]) -> list[int]:
    positions = candidate.get("__candidate_merge_positions")
    if not isinstance(positions, list):
        return []
    out: list[int] = []
    for item in positions:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _candidate_merge_representative_position(candidate: dict[str, Any]) -> int | None:
    try:
        return int(candidate.get("__candidate_merge_representative_position"))
    except (TypeError, ValueError):
        return None


def _strip_candidate_merge_private(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if not str(key).startswith("__")}


def _material_candidate_capabilities(capabilities: set[str]) -> set[str]:
    return {
        capability
        for capability in capabilities
        if capability
        in {
            "has_sequence",
            "has_backbone_structure",
            "has_full_structure",
            "has_structure",
        }
    }


def merge_candidate_pool(
    survivor_pool: list[dict[str, Any]],
    current_round_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return merge_candidate_pool_result(survivor_pool, current_round_candidates).candidates


def merge_candidate_pool_result(
    survivor_pool: list[dict[str, Any]],
    current_round_candidates: list[dict[str, Any]],
):
    return merge_candidate_records(
        [
            [candidate for candidate in survivor_pool if _is_candidate_v1(candidate)],
            [candidate for candidate in current_round_candidates if _is_candidate_v1(candidate)],
        ]
    )


def safe_candidate_ref_cards_from_summary(
    summary: dict[str, Any],
    *,
    deliverable_contract: dict[str, Any] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    contract = deliverable_contract if isinstance(deliverable_contract, dict) else {}
    required = _required_capabilities(contract)
    cards: list[dict[str, Any]] = []
    for rank, item in enumerate(candidate_pool_items_from_summary(summary), start=1):
        if len(cards) >= limit:
            break
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            continue
        container_key = str(item.get("container_key") or "")
        capabilities = candidate_capability_names(candidate, key=container_key)
        missing = sorted(cap for cap in required if cap not in capabilities)
        card = {
            "candidate_id": _safe_string(candidate.get("candidate_id")),
            "job_id": _safe_string(item.get("job_id")),
            "rank": rank,
            "index": _safe_int(item.get("index")),
            "source_tool": _safe_string(item.get("tool")),
            "container_key": container_key or None,
            "capabilities_satisfied": not missing,
            "matched_capabilities": sorted(capabilities),
            "missing_capabilities": missing,
            "evidence": _candidate_evidence(capabilities, container_key, str(item.get("tool") or "")),
            "warnings": _candidate_warnings(item, missing),
        }
        cards.append({key: value for key, value in card.items() if value is not None})
    return cards


def safe_candidate_pool_stats_from_summary(
    summary: dict[str, Any],
    *,
    deliverable_contract: dict[str, Any] | None = None,
) -> dict[str, int]:
    contract = deliverable_contract if isinstance(deliverable_contract, dict) else {}
    required = _required_capabilities(contract)
    submit_ready_count = 0
    missing_capability_count = 0
    for item in candidate_pool_items_from_summary(summary):
        candidate = item.get("candidate")
        if not isinstance(candidate, dict):
            continue
        capabilities = candidate_capability_names(
            candidate,
            key=str(item.get("container_key") or ""),
        )
        missing = [cap for cap in required if cap not in capabilities]
        if missing:
            missing_capability_count += 1
        else:
            submit_ready_count += 1
    return {
        "submit_ready_count": submit_ready_count,
        "missing_capability_count": missing_capability_count,
    }


def safe_candidate_pool_stats_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    deliverable_contract: dict[str, Any] | None = None,
) -> dict[str, int]:
    contract = deliverable_contract if isinstance(deliverable_contract, dict) else {}
    required = _required_capabilities(contract)
    submit_ready_count = 0
    missing_capability_count = 0
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        capabilities = candidate_capability_names(candidate, key="")
        missing = [cap for cap in required if cap not in capabilities]
        if missing:
            missing_capability_count += 1
        else:
            submit_ready_count += 1
    return {
        "submit_ready_count": submit_ready_count,
        "missing_capability_count": missing_capability_count,
    }


def _required_capabilities(contract: dict[str, Any]) -> list[str]:
    raw = contract.get("required_candidate_capabilities")
    if not isinstance(raw, list):
        return []
    return sorted(str(item) for item in raw if isinstance(item, str) and item)


def _candidate_evidence(capabilities: set[str], container_key: str, tool: str) -> list[str]:
    evidence: list[str] = []
    if container_key == "ranked_candidates" or tool == "rank_designs":
        evidence.append("ranked")
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
    return evidence


def _candidate_warnings(item: dict[str, Any], missing: list[str]) -> list[str]:
    warnings: list[str] = []
    if not item.get("job_id"):
        warnings.append("missing_node_job_id")
    if missing:
        warnings.append("missing_required_capabilities")
    return warnings


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value[:240]
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
