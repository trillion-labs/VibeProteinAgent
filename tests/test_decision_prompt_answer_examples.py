from __future__ import annotations

import json
import re
from typing import Any

from vibeproteinagent.agents.evaluation import validate_decision_v1
from vibeproteinagent.agents.planning.prompts import DECISION_SYSTEM_PROMPT


_BASE_DECISION_PROMPT_WORD_COUNT = 317

_ORIGINAL_EXAMPLES = [
    """{
  "decision": "submit_candidate",
  "reasoning": "The top ranked design satisfies the requested stability and binding criteria.",
  "submit_candidate": {
    "candidates": [
      {
        "job_id": "job_rank",
        "candidate_id": "design_1",
        "justification": "Best combined rank with acceptable quality checks.",
        "scores_to_highlight": ["rank_designs.score"]
      }
    ]
  },
  "metrics_used": ["rank_designs.score"]
}""",
    """{
  "decision": "next_round_with_surviving_candidate",
  "reasoning": "The current candidates are promising but lack enough sequence diversity.",
  "next_round": {
    "reason": "Explore a broader design set before final submission.",
    "recommended": true,
    "changes": ["increase diversity filtering", "rerank by interface quality"]
  },
  "metrics_used": ["sequence_analysis.diversity", "rank_designs.score"]
}""",
    """{
  "decision": "next_round_with_no_acceptable_candidate",
  "reasoning": "All reviewed candidates fail the stated acceptance threshold, but another local search round may still be useful.",
  "next_round": {
    "reason": "No candidate meets the minimum confidence or quality criteria.",
    "recommended": true,
    "changes": ["try alternate producers or broaden the search"]
  },
  "metrics_used": ["rank_designs.score", "quality_flags"]
}""",
]


def _json_example_strings() -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(
            r"```json\s*(\{.*?\})\s*```",
            DECISION_SYSTEM_PROMPT,
            flags=re.DOTALL,
        )
    ]


def _json_examples() -> list[dict[str, Any]]:
    return [json.loads(example) for example in _json_example_strings()]


def test_decision_prompt_describes_intent_only_branching() -> None:
    prompt = DECISION_SYSTEM_PROMPT
    assert "Emit decision\nintent only" in prompt
    assert "deterministic controller branch resolver" in prompt


def test_decision_prompt_contains_three_intent_names() -> None:
    prompt = DECISION_SYSTEM_PROMPT
    assert "submit_candidate" in prompt
    assert "next_round_with_surviving_candidate" in prompt
    assert "next_round_with_no_acceptable_candidate" in prompt


def test_decision_prompt_has_exactly_three_json_examples() -> None:
    assert len(_json_examples()) == 3


def test_all_decision_prompt_examples_validate_against_schema() -> None:
    for example in _json_examples():
        validate_decision_v1(example)


def test_decision_examples_match_intent_contract_byte_for_byte() -> None:
    assert _json_example_strings() == _ORIGINAL_EXAMPLES


def test_decision_prompt_word_count_delta_is_bounded() -> None:
    word_count = len(re.findall(r"\b\S+\b", DECISION_SYSTEM_PROMPT))
    assert -80 <= word_count - _BASE_DECISION_PROMPT_WORD_COUNT <= 80
