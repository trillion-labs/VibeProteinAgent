from __future__ import annotations

import json
import re
from typing import Any


def sequence_relevance_score(
    *,
    metadata: Any,
    reviewed: bool | None,
    length: int | None,
    intent_query: str,
    term_match_score: float = 0.0,
    evidence_quality_score: float = 0.0,
    diversity_bonus: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    text = _metadata_text(metadata)
    query_tokens = _tokens(intent_query)
    if query_tokens:
        overlap = len(query_tokens & _tokens(text)) / max(1, len(query_tokens))
    else:
        overlap = 0.0
    reviewed_swissprot_bonus = 1.0 if reviewed is True else 0.0
    fragment_penalty = 1.0 if "fragment" in text.lower() else 0.0
    length_penalty = 1.0 if length is not None and (length < 40 or length > 2500) else 0.0
    length_or_fragment_penalty = min(1.0, max(fragment_penalty, length_penalty))
    weighted_components = {
        "term_match_score": 0.35 * max(0.0, min(1.0, term_match_score)),
        "annotation_keyword_score": 0.25 * overlap,
        "reviewed_swissprot_bonus": 0.15 * reviewed_swissprot_bonus,
        "evidence_quality_score": 0.10 * max(0.0, min(1.0, evidence_quality_score)),
        "diversity_bonus": 0.10 * max(0.0, min(1.0, diversity_bonus)),
        "length_or_fragment_penalty": -0.05 * length_or_fragment_penalty,
    }
    score = max(
        0.0,
        min(
            1.0,
            sum(weighted_components.values()),
        ),
    )
    return round(score, 6), {
        "term_match_score": round(max(0.0, min(1.0, term_match_score)), 6),
        "annotation_keyword_score": round(overlap, 6),
        "reviewed_swissprot_bonus": reviewed_swissprot_bonus,
        "evidence_quality_score": round(max(0.0, min(1.0, evidence_quality_score)), 6),
        "diversity_bonus": round(max(0.0, min(1.0, diversity_bonus)), 6),
        "length_or_fragment_penalty": round(length_or_fragment_penalty, 6),
        "weighted_components": {
            key: round(value, 6) for key, value in weighted_components.items()
        },
    }


def top_k_by_sequence_relevance(
    rows: list[dict[str, Any]],
    *,
    intent_query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        score = row.get("sequence_relevance_score")
        if not isinstance(score, (int, float)):
            score = row.get("semantic_relevance")
        if not isinstance(score, (int, float)):
            score = 0.0
        scored.append({"index": index, "score": float(score), "row": row})
    scored.sort(key=lambda item: (-item["score"], item["index"]))
    return [item["row"] for item in scored[: max(1, top_k)]]


def intent_quota_top_k(
    rows: list[dict[str, Any]],
    *,
    max_candidates: int,
    intent_key: str = "intent_id",
) -> list[dict[str, Any]]:
    if max_candidates <= 0 or not rows:
        return []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        intent_id = str(row.get(intent_key) or "default")
        grouped.setdefault(intent_id, []).append(row)
    for intent_id, items in grouped.items():
        grouped[intent_id] = top_k_by_sequence_relevance(
            items,
            intent_query=intent_id,
            top_k=len(items),
        )
    intents = sorted(grouped)
    quota = max(1, max_candidates // max(1, len(intents)))
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for intent_id in intents:
        items = grouped[intent_id]
        selected.extend(items[:quota])
        leftovers.extend(items[quota:])
    remaining = max_candidates - len(selected)
    if remaining > 0:
        selected.extend(
            top_k_by_sequence_relevance(
                leftovers,
                intent_query="remaining",
                top_k=remaining,
            )
        )
    return selected[:max_candidates]


def _metadata_text(metadata: Any) -> str:
    if metadata is None:
        return ""
    try:
        return json.dumps(metadata, sort_keys=True)
    except TypeError:
        return str(metadata)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }
