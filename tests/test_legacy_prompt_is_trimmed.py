from __future__ import annotations

import re
from pathlib import Path

from vibeproteinagent.agents.planning.prompts import PLANNER_SYSTEM_PROMPT


def test_legacy_prompt_excludes_obsolete_executor_ir_rules() -> None:
    forbidden = [
        "output_fields",
        "kept_sequences",
        "affinity_pred_value",
        "confidence_score",
        "${outputs",
        "${jobs",
        "${item",
        "foreach",
        "workflow_ref",
        "submit_final",
    ]
    for term in forbidden:
        assert term not in PLANNER_SYSTEM_PROMPT


def test_legacy_prompt_has_source_marker() -> None:
    source = Path("src/vibeproteinagent/agents/planning/prompts.py").read_text(
        encoding="utf-8",
    )
    assert "# LEGACY: used only when VPA_PLANNER_IR=off." in source


def test_legacy_prompt_word_count_is_in_budget() -> None:
    words = re.findall(r"\b\w+\b", PLANNER_SYSTEM_PROMPT)
    assert 700 <= len(words) <= 1000
