from __future__ import annotations

import json
import logging
from typing import Any, cast

from vibeproteinagent.agents.config import resolve_decision_normalization_mode
from vibeproteinagent.agents.planning.prompts import DECISION_SYSTEM_PROMPT
from vibeproteinagent.agents.provider import ChatMessage
from vibeproteinagent.agents.provider.types import ChatProvider
from vibeproteinagent.core.errors import InvalidParamsError

from .decision_validator import DecisionSchemaViolation, validate_decision_v1

audit_log = logging.getLogger("vibeproteinagent.audit")


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
        retry_error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        local = self.local_decision(summary, success_criteria=success_criteria)
        if self.provider is None:
            return local
        payload = {
            "summary": summary,
            "success_criteria": success_criteria or {},
            "local_decision": local,
        }
        if retry_error is not None:
            payload["previous_decision_error"] = retry_error
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
        if not isinstance(decision, dict):
            raise InvalidParamsError(
                "decision model must return a JSON object",
                details={"got_type": type(decision).__name__},
            )
        try:
            validate_decision_v1(decision)
        except DecisionSchemaViolation as exc:
            mode = resolve_decision_normalization_mode()
            if mode == "compat":
                first = exc.violations[0] if exc.violations else None
                audit_log.warning(
                    "audit.decision_schema_violation_in_compat_mode",
                    extra={
                        "pattern_key": first.issue if first else "unknown",
                        "structured_error": exc.structured_error,
                    },
                )
            else:
                raise
        return self._normalize_decision(decision, local)

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

    def _normalize_decision(
        self,
        decision: dict[str, Any],
        local: dict[str, Any],
    ) -> dict[str, Any]:
        label = decision.get("decision")
        allowed = {"submit_candidate", "next_round", "no_acceptable_candidate", "stop", "answer"}
        if label not in allowed:
            label = "next_round" if _next_round_recommended(decision) else "stop"
        next_round = decision.get("next_round")
        if not isinstance(next_round, dict):
            next_round = {}
        if label == "next_round":
            next_round.setdefault("recommended", True)
        else:
            next_round.setdefault("recommended", False)
        next_round.setdefault("changes", [])
        next_round.setdefault(
            "reason",
            local.get("next_round", {}).get("reason", ""),
        )
        submit = decision.get("submit_candidate")
        if not isinstance(submit, dict):
            submit = None
        reasoning = decision.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            reasoning = str(local.get("next_round", {}).get("reason") or "Decision normalized.")
        notes = decision.get("notes")
        notes_text = notes if isinstance(notes, str) else None
        normalized: dict[str, Any] = {
            "decision": label,
            "reasoning": reasoning.strip(),
            "metrics_used": decision.get("metrics_used") or local.get("metrics_used", []),
        }
        if notes_text:
            normalized["notes"] = notes_text
        if label == "submit_candidate":
            if submit is None:
                raise InvalidParamsError(
                    "submit_candidate decision missing submit_candidate payload",
                    details={"decision": decision},
                )
            normalized["submit_candidate"] = submit
            return cast(dict[str, Any], normalized)
        if label == "next_round":
            normalized["next_round"] = next_round
            return cast(dict[str, Any], normalized)
        if label == "no_acceptable_candidate":
            no_candidate = decision.get("no_acceptable_candidate")
            if not isinstance(no_candidate, dict):
                no_candidate = {"reason": next_round.get("reason") or reasoning.strip()}
            normalized["no_acceptable_candidate"] = no_candidate
            return cast(dict[str, Any], normalized)
        if label == "answer":
            answer = decision.get("answer")
            if not isinstance(answer, dict):
                raise InvalidParamsError(
                    "answer decision missing answer payload",
                    details={"decision": decision},
                )
            normalized["answer"] = answer
            return cast(dict[str, Any], normalized)
        return cast(dict[str, Any], normalized)


def _next_round_recommended(decision: dict[str, Any]) -> bool:
    next_round = decision.get("next_round")
    return isinstance(next_round, dict) and next_round.get("recommended") is True
