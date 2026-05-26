from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

import pytest

from vibeproteinagent.agents.config import AgentConfig
from vibeproteinagent.agents.evaluation import DecisionEngine, DecisionSchemaViolation
from vibeproteinagent.agents.execution.controller import (
    AgentController,
    _submit_candidate_params_from_decision,
)
from vibeproteinagent.agents.provider import ChatMessage, ProviderResponse
from vibeproteinagent.workspace import Workspace


class _DecisionProvider:
    provider_name = "fake"

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = payloads
        self.messages: list[list[ChatMessage]] = []

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ProviderResponse:
        del kwargs
        self.messages.append(list(messages))
        payload = self.payloads[min(len(self.messages) - 1, len(self.payloads) - 1)]
        return ProviderResponse(provider="fake", model="fake", content=json.dumps(payload))


def _invalid_decision() -> dict[str, Any]:
    return {
        "decision": "submit_candidate",
        "submit_candidate": {
            "candidates": [
                {
                    "job_id": "job_rank",
                    "candidate_id": "design_1",
                    "justification": "top ranked",
                    "reasoning": "misnested",
                }
            ]
        },
        "metrics_used": ["score"],
    }


def _valid_decision() -> dict[str, Any]:
    return {
        "decision": "submit_candidate",
        "reasoning": "top-level reasoning",
        "submit_candidate": {
            "candidates": [
                {
                    "job_id": "job_rank",
                    "candidate_id": "design_1",
                    "justification": "top ranked",
                }
            ]
        },
        "metrics_used": ["score"],
    }


async def _decision_entry(
    controller: AgentController,
    provider: _DecisionProvider,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    async def _factory(
        attempt: int,
        retry_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await DecisionEngine(provider).decide(
            {"state": "completed", "metrics_used": ["score"]},
            success_criteria={"return_final": True},
            model="fake",
            idempotency_key=f"decision:{attempt}",
            retry_error=retry_context,
        )

    return await controller._run_llm_entry(
        phase="decision_round_0",
        failures=failures,
        factory=_factory,
    )


def test_strict_decision_retry_injects_structured_error(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")
    provider = _DecisionProvider([_invalid_decision(), _valid_decision()])
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))

    decision = asyncio.run(_decision_entry(controller, provider))

    assert decision["decision"] == "submit_candidate"
    assert len(provider.messages) == 2
    retry_user_payload = json.loads(provider.messages[1][1].content)
    assert "previous_decision_error" in retry_user_payload
    assert (
        retry_user_payload["previous_decision_error"]["structured_error"]["error_kind"]
        == "decision_schema_violation"
    )
    assert "reasoning_inside_candidate_forbidden" in json.dumps(retry_user_payload)


def test_compat_decision_does_not_retry_and_lift_still_handles_payload(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "compat")
    caplog.set_level(logging.WARNING, logger="vibeproteinagent.audit")
    provider = _DecisionProvider([_invalid_decision()])
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=3))

    decision = asyncio.run(_decision_entry(controller, provider))
    params = _submit_candidate_params_from_decision(decision)

    assert len(provider.messages) == 1
    assert params["reasoning"] == "misnested"
    assert any(
        getattr(record, "pattern_key", None)
        == "decision.reasoning.lifted_from_candidate"
        for record in caplog.records
    )


def test_strict_decision_retry_cap_raises_final_schema_violation(
    tmp_workspace: Workspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VPA_NORMALIZE_DECISION_REASONING", "strict")
    provider = _DecisionProvider([_invalid_decision()])
    controller = AgentController(tmp_workspace, config=AgentConfig(max_entries=2))

    with pytest.raises(DecisionSchemaViolation):
        asyncio.run(_decision_entry(controller, provider))

    assert len(provider.messages) == 2
