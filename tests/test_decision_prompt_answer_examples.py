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
  "decision": "next_round",
  "reasoning": "The current candidates are promising but lack enough sequence diversity.",
  "next_round": {
    "reason": "Explore a broader design set before final submission.",
    "recommended": true,
    "changes": ["increase diversity filtering", "rerank by interface quality"]
  },
  "metrics_used": ["sequence_analysis.diversity", "rank_designs.score"]
}""",
    """{
  "decision": "no_acceptable_candidate",
  "reasoning": "All reviewed candidates fail the stated acceptance threshold.",
  "no_acceptable_candidate": {
    "reason": "No candidate meets the minimum confidence or quality criteria."
  },
  "metrics_used": ["rank_designs.score", "quality_flags"]
}""",
    """{
  "decision": "stop",
  "reasoning": "The workflow is complete and no additional local round is justified.",
  "metrics_used": ["workflow_state"]
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


def test_decision_prompt_contains_answer_guidance() -> None:
    prompt = DECISION_SYSTEM_PROMPT
    assert 'decision="answer"' in prompt
    assert "`answer.text`" in prompt


def test_decision_prompt_contains_answer_heuristic_language() -> None:
    prompt = DECISION_SYSTEM_PROMPT
    assert any(phrase in prompt for phrase in ["What is", "compare", "explain"])


def test_decision_prompt_has_exactly_five_json_examples() -> None:
    assert len(_json_examples()) == 5


def test_all_decision_prompt_examples_validate_against_schema() -> None:
    for example in _json_examples():
        validate_decision_v1(example)


def test_original_four_decision_examples_are_unchanged_byte_for_byte() -> None:
    assert _json_example_strings()[:4] == _ORIGINAL_EXAMPLES


def test_answer_example_contains_text_and_evidence_fields() -> None:
    answer = _json_examples()[-1]["answer"]
    assert "text" in answer
    assert "evidence" in answer


def test_decision_prompt_word_count_delta_is_bounded() -> None:
    word_count = len(re.findall(r"\b\S+\b", DECISION_SYSTEM_PROMPT))
    assert 80 <= word_count - _BASE_DECISION_PROMPT_WORD_COUNT <= 150
