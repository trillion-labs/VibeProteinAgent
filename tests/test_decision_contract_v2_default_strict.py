from __future__ import annotations

import pytest

from vibeproteinagent.agents.config import (
    DECISION_CONTRACT_VERSION,
    NORMALIZATION_DEFAULTS,
    resolve_decision_normalization_mode,
)
from vibeproteinagent.agents.evaluation import DecisionSchemaViolation
from vibeproteinagent.agents.execution.controller import (
    _submit_candidate_params_from_decision,
)


def _misnested_reasoning_decision() -> dict:
    return {
        "decision": "submit_candidate",
        "submit_candidate": {
            "candidates": [
                {
                    "job_id": "rank_job",
                    "candidate_id": "cand_1",
                    "justification": "best score",
                    "reasoning": "This reasoning belongs at the top level.",
                }
            ]
        },
        "metrics_used": ["rank_designs.score"],
    }


def test_decision_contract_v2_defaults_to_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPA_NORMALIZE_DECISION_REASONING", raising=False)

    assert DECISION_CONTRACT_VERSION == "vpa-agent-decision-v2"
    assert NORMALIZATION_DEFAULTS["vpa-agent-decision-v1"]["decision_reasoning"] == "compat"
    assert NORMALIZATION_DEFAULTS["vpa-agent-decision-v2"]["decision_reasoning"] == "strict"
    assert resolve_decision_normalization_mode() == "strict"


def test_decision_reasoning_compat_override_still_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "compat")
    assert resolve_decision_normalization_mode() == "compat"


def test_default_strict_rejects_misnested_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VPA_NORMALIZE_DECISION_REASONING", raising=False)

    with pytest.raises(DecisionSchemaViolation):
        _submit_candidate_params_from_decision(_misnested_reasoning_decision())
