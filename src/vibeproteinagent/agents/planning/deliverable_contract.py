from __future__ import annotations

import re
from typing import Any

_COUNTED_CANDIDATE_RE = re.compile(
    r"\b(\d[\d,]*)\s+"
    r"(?:candidate|candidates|design|designs|sequence|sequences)\b",
    re.IGNORECASE,
)
_REQUESTED_COUNT_PARAM_KEYS = (
    "requested_candidate_count",
    "num_candidates",
    "candidate_count",
    "top_k",
)


def build_deliverable_contract(
    request: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the planner-facing deliverable contract for an agent request."""

    request_params = params or {}
    lower = request.lower()
    caps: list[str] = []
    answer_type = "plain_language_answer"
    terminal_submission_allowed = False
    plain_language_answer_allowed = True
    sequence_requested = _has_any_word(
        lower,
        (
            "sequence",
            "sequences",
            "fasta",
            "representative",
            "homolog",
            "ortholog",
            "enzyme",
            "activity",
            "rank",
            "candidate",
            "candidates",
        ),
    )
    structure_requested = (
        "full structure" in lower
        or "full-structure" in lower
        or "3d structure" in lower
        or "three-dimensional structure" in lower
        or _has_any_word(
            lower,
            (
                "structure",
                "structures",
                "fold",
                "model",
                "complex",
                "interface",
                "backbone",
                "scaffold",
                "pdb",
                "cif",
            ),
        )
    )
    structure_param_supplied = any(
        request_params.get(field)
        for field in (
            "target_pdb",
            "pdb_path",
            "structure_path",
            "cif_path",
            "target_cif",
        )
    )
    candidate_action_requested = _has_any_word(
        lower,
        (
            "design",
            "generate",
            "generation",
            "binder",
            "backbone",
            "inverse-fold",
            "inverse",
            "refold",
            "rank",
            "candidate",
            "candidates",
        ),
    )
    if _has_any_word(
        lower,
        (
            "report",
            "analysis",
            "analyze",
            "compare",
            "summarize",
        ),
    ):
        answer_type = "workflow_report"
        plain_language_answer_allowed = True
    if sequence_requested:
        answer_type = "candidate_submission"
        caps = ["has_sequence"]
        terminal_submission_allowed = True
    if structure_requested:
        answer_type = "candidate_submission"
        caps = ["has_sequence", "has_full_structure"]
        terminal_submission_allowed = True
    if structure_param_supplied and candidate_action_requested:
        answer_type = "candidate_submission"
        caps = ["has_sequence", "has_full_structure"]
        terminal_submission_allowed = True
    rationale = "request asks for explanation without a concrete candidate deliverable"
    if answer_type == "workflow_report":
        rationale = "request asks for analysis or reporting rather than direct candidate submission"
    elif structure_requested:
        rationale = "request mentions structure/model/interface evidence, so terminal candidates require sequence and full structure"
    elif structure_param_supplied and candidate_action_requested:
        rationale = "request supplies a structure target for a candidate-generating action, so terminal candidates require sequence and full structure"
    elif sequence_requested:
        rationale = "request asks for sequence, representative, ranking, enzyme, activity, or candidate evidence"
    contract = {
        "version": "deliverable_contract_v1",
        "source": "agent_request_local",
        "deliverable_contract_source": "fallback",
        "answer_type": answer_type,
        "deliverable_kind": answer_type,
        "required_candidate_capabilities": caps,
        "optional_candidate_capabilities": ["has_candidate_scores"],
        "terminal_submission_allowed": terminal_submission_allowed,
        "plain_language_answer_allowed": plain_language_answer_allowed,
        "rationale": rationale,
    }
    requested_count = _requested_candidate_count(request, request_params)
    if requested_count is not None:
        contract["requested_candidate_count"] = requested_count
    return contract


def _requested_candidate_count(request: str, params: dict[str, Any]) -> int | None:
    for key in _REQUESTED_COUNT_PARAM_KEYS:
        count = _positive_int_or_none(params.get(key))
        if count is not None:
            return count
    for match in _COUNTED_CANDIDATE_RE.finditer(request):
        count = _positive_int_or_none(match.group(1))
        if count is not None:
            return count
    return None


def _positive_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value.is_integer() and value > 0 else None
    if isinstance(value, str):
        stripped = value.strip().replace(",", "")
        if not re.fullmatch(r"\d+", stripped):
            return None
        parsed = int(stripped)
        return parsed if parsed > 0 else None
    return None


def _has_any_word(lower: str, words: tuple[str, ...]) -> bool:
    tokens = set(re.findall(r"[a-z0-9_+-]+", lower))
    return any(word in tokens for word in words)


__all__ = ["build_deliverable_contract"]
