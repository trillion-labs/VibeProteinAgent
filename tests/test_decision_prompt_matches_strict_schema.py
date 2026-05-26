from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from typing import Any

from vibeproteinagent.agents.evaluation import DecisionEngine, validate_decision_v1
from vibeproteinagent.agents.planning.prompts import DECISION_SYSTEM_PROMPT
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse


def _json_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for match in re.finditer(
        r"```json\s*(\{.*?\})\s*```",
        DECISION_SYSTEM_PROMPT,
        flags=re.DOTALL,
    ):
        examples.append(json.loads(match.group(1)))
    return examples


class _ExampleDecisionProvider:
    provider_name = "fake"

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del messages, kwargs
        return ProviderResponse(
            provider="fake",
            model="fake",
            content=json.dumps(self.payload),
        )


def test_decision_prompt_static_content_matches_strict_schema() -> None:
    prompt = DECISION_SYSTEM_PROMPT
    prompt_lower = prompt.lower()

    # These phrases describe the legacy compat-era shape that strict
    # decision_v1 rejects: branch payloads emitted together with null/empty
    # placeholders for unchosen decisions.
    banned_branch_phrases = [
        "always include",
        "include all branches",
        "leave unused as null",
        "set unused to null",
        "all three branches",
        "unused branches",
    ]
    for phrase in banned_branch_phrases:
        assert phrase not in prompt_lower

    submit_candidate_section = prompt.split("For `submit_candidate`", 1)[1].split(
        "For `next_round`", 1
    )[0]
    assert "reasoning" not in submit_candidate_section
    assert "decision_v1" in prompt
    assert "Put `reasoning` at the top level" in prompt


def test_decision_prompt_json_examples_validate_against_decision_v1() -> None:
    examples = _json_examples()

    assert {example["decision"] for example in examples} == {
        "submit_candidate",
        "next_round",
        "no_acceptable_candidate",
        "stop",
        "answer",
    }
    for example in examples:
        validate_decision_v1(example)


def test_decision_engine_accepts_every_prompt_example_in_strict_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")
    examples = _json_examples()

    for example in examples:
        engine = DecisionEngine(_ExampleDecisionProvider(example))
        decision = asyncio.run(
            engine.decide(
                {"state": "completed", "metrics_used": ["score"]},
                success_criteria={"return_final": True},
                model="fake",
            )
        )
        assert decision["decision"] == example["decision"]
