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
            prompt_context={
                "latest_round": {
                    "state": "completed",
                    "candidate_summary": {"count": 1, "items": []},
                }
            },
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


def test_decision_llm_receives_prompt_context_not_workflow_summary() -> None:
    provider = _DecisionProvider(
        [
            {
                "decision": "next_round_with_surviving_candidate",
                "reasoning": "review prompt context",
                "next_round": {"recommended": False, "reason": "done"},
                "metrics_used": [],
            }
        ]
    )
    summary = {
        "state": "completed",
        "nodes": [
            {
                "node_id": "merge",
                "tool": "merge_candidates",
                "output": {
                    "candidates": [
                        {
                            "schema_version": "candidate_v1",
                            "candidate_id": "cand_x",
                            "sequence": "ACDEFGHIKLMNPQRSTVWY",
                            "scores": {"rank_designs": {"ranking_score": 0.91}},
                            "structure_path": "/tmp/raw.pdb",
                            "source": {"tool": "go_sequences", "local_id": "x"},
                            "__candidate_merge_positions": [0, 1],
                        }
                    ],
                    "merge_audit": {
                        "dedup_policy": "keep_all",
                        "output_count": 1,
                        "duplicates_removed": 1,
                        "duplicate_groups": [
                            {
                                "identity_key": "candidate_id:cand_x",
                                "versions": [
                                    {
                                        "source": {
                                            "tool": "go_sequences",
                                            "native_id": "ACC1",
                                        }
                                    }
                                ],
                            }
                        ],
                        "conflicts": [{"field": "scores.semantic_relevance"}],
                    },
                },
            }
        ],
    }

    asyncio.run(
        DecisionEngine(provider).decide(
            summary,
            success_criteria={"review": True},
            prompt_context={
                "latest_round": {
                    "state": "completed",
                    "candidate_summary": {
                        "count": 1,
                        "items": [
                            {
                                "candidate_id": "cand_x",
                                "source": {"tool": "go_sequences"},
                                "score_keys": ["rank_designs"],
                            }
                        ],
                    },
                    "merge_summary": {
                        "current_round": {
                            "output_count": 1,
                            "duplicate_group_count": 1,
                        }
                    },
                }
            },
            model="fake",
        )
    )

    payload = json.loads(provider.messages[0][1].content)
    serialized = json.dumps(payload, sort_keys=True)
    assert "summary" not in payload
    assert payload["prompt_context"]["latest_round"]["candidate_summary"]["count"] == 1
    assert "ACDEFGHIKLMNPQRSTVWY" not in serialized
    assert "ranking_score" not in serialized
    assert "/tmp/raw.pdb" not in serialized
    assert "duplicate_groups" not in serialized
    assert "versions" not in serialized
    assert "__candidate_merge_positions" not in serialized


def test_decision_safe_summary_is_not_decision_llm_candidate_source() -> None:
    provider = _DecisionProvider(
        [
            {
                "decision": "next_round_with_no_acceptable_candidate",
                "reasoning": "no prompt context candidates",
                "next_round": {"recommended": False, "reason": "done"},
                "metrics_used": [],
            }
        ]
    )

    asyncio.run(
        DecisionEngine(provider).decide(
            {
                "state": "completed",
                "nodes": [
                    {
                        "output": {
                            "candidates": [
                                {
                                    "schema_version": "candidate_v1",
                                    "candidate_id": "raw_candidate",
                                    "sequence": "MMMMMMMMMMMMMMMM",
                                    "scores": {"raw": {"score": 1.0}},
                                }
                            ]
                        }
                    }
                ],
            },
            success_criteria={"review": True},
            model="fake",
        )
    )

    payload = json.loads(provider.messages[0][1].content)
    assert payload["prompt_context"] == {}
    assert "raw_candidate" not in json.dumps(payload)
    assert "MMMMMMMMMMMMMMMM" not in json.dumps(payload)
