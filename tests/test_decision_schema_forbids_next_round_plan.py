from __future__ import annotations

import json
import re
from typing import Any

import pytest

from vibeproteinagent.agents.evaluation import DecisionSchemaViolation, validate_decision_v1
from vibeproteinagent.agents.planning.prompts import DECISION_SYSTEM_PROMPT


def _next_round_decision() -> dict[str, Any]:
    return {
        "decision": "next_round",
        "reasoning": "More local exploration is justified.",
        "next_round": {
            "reason": "Carry forward survivors and broaden the search.",
            "recommended": True,
            "changes": ["rank prior survivors against fresh candidates"],
        },
        "metrics_used": ["rank_designs.score"],
    }


def _json_examples() -> list[dict[str, Any]]:
    return [
        json.loads(match.group(1))
        for match in re.finditer(
            r"```json\s*(\{.*?\})\s*```",
            DECISION_SYSTEM_PROMPT,
            flags=re.DOTALL,
        )
    ]


def test_next_round_plan_is_rejected_by_decision_schema() -> None:
    payload = _next_round_decision()
    payload["next_round"]["plan"] = {"workflow_plan": {"nodes": []}}

    with pytest.raises(DecisionSchemaViolation) as exc_info:
        validate_decision_v1(payload)

    assert any(
        violation.path == "/next_round/plan"
        and violation.issue == "additional_property_not_allowed"
        for violation in exc_info.value.violations
    )


def test_next_round_reason_without_plan_passes_decision_schema() -> None:
    validate_decision_v1(_next_round_decision())


def test_decision_prompt_examples_still_validate_after_plan_removal() -> None:
    examples = _json_examples()
    # Item 3 adds the additive answer decision example.
    assert len(examples) == 5
    for example in examples:
        validate_decision_v1(example)
