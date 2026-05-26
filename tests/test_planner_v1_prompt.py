from __future__ import annotations

import re

from vibeproteinagent.agents.planning.prompts import PLANNER_IR_SYSTEM_PROMPT


def test_planner_ir_prompt_omits_executor_wiring_terms() -> None:
    forbidden = [
        "${outputs",
        "${jobs",
        "${item",
        "foreach",
        "workflow_ref",
        "submit_final",
        "kept_sequences",
        "output_fields",
    ]

    for term in forbidden:
        assert term not in PLANNER_IR_SYSTEM_PROMPT


def test_planner_ir_prompt_mentions_schema_and_contains_json_example() -> None:
    assert PLANNER_IR_SYSTEM_PROMPT.count("planner_ir_v1") >= 1
    assert '"version": "planner_ir_v1"' in PLANNER_IR_SYSTEM_PROMPT
    assert '"nodes": [' in PLANNER_IR_SYSTEM_PROMPT
    assert '"relationship": "candidate_source"' in PLANNER_IR_SYSTEM_PROMPT


def test_planner_ir_prompt_word_count_budget() -> None:
    words = re.findall(r"\b\w+\b", PLANNER_IR_SYSTEM_PROMPT)
    assert 700 <= len(words) <= 1100
