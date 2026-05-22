from __future__ import annotations

import json
from typing import Any, cast

from vibeproteinagent.agents.planning.prompts import DECISION_SYSTEM_PROMPT
from vibeproteinagent.agents.provider import ChatMessage
from vibeproteinagent.agents.provider.types import ChatProvider
from vibeproteinagent.core.errors import InvalidParamsError


class DecisionEngine:
    """Boundary-only result evaluation and next-round decision helper."""

    def __init__(self, provider: ChatProvider | None = None) -> None:
        self.provider = provider

    async def decide(
        self,
        summary: dict[str, Any],
        *,
        success_criteria: dict[str, Any] | None = None,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        local = self.local_decision(summary, success_criteria=success_criteria)
        if self.provider is None:
            return local
        payload = {
            "summary": summary,
            "success_criteria": success_criteria or {},
            "local_decision": local,
        }
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=DECISION_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=idempotency_key,
        )
        try:
            decision = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise InvalidParamsError(
                "decision model returned non-JSON content",
                details={"error": str(exc), "content": response.content[:1000]},
            ) from exc
        decision.setdefault("decision", local["decision"])
        decision.setdefault("next_round", local["next_round"])
        decision.setdefault("metrics_used", local.get("metrics_used", []))
        decision.setdefault("notes", [])
        return cast(dict[str, Any], decision)

    def local_decision(
        self,
        summary: dict[str, Any],
        *,
        success_criteria: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = summary.get("state")
        criteria = success_criteria or {}
        metrics_used = summary.get("metrics_used") or []
        if state != "completed":
            decision = "workflow_not_completed"
            recommended = False
            reason = f"state is {state!r}"
        elif criteria:
            decision = "completed_review_metrics"
            recommended = False
            reason = "workflow completed; criteria should be reviewed against compact metrics"
        else:
            decision = "completed_no_additional_round_recommended"
            recommended = False
            reason = "workflow completed and no explicit success criteria require another round"
        return {
            "decision": decision,
            "next_round": {"recommended": recommended, "changes": [], "reason": reason},
            "metrics_used": metrics_used,
            "notes": [],
        }
