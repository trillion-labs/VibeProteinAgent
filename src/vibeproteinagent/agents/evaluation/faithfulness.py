from __future__ import annotations

import json
from typing import Any

from vibeproteinagent.agents.provider import ChatMessage
from vibeproteinagent.agents.provider.types import ChatProvider
from vibeproteinagent.core.errors import InvalidParamsError


FAITHFULNESS_SYSTEM_PROMPT = """You are a faithfulness checker for a scientific
workflow agent. You are given the original user request, the workflow output
records that the agent's tools actually produced, and the agent's own
success_criteria + decision JSON. Your job is to find any numeric, structural,
or factual claim in success_criteria or decision that is NOT supported by
either the user request text or the workflow outputs.

For every distinct claim that mentions a number, length, position, threshold,
sequence, identifier, or named tool, emit one entry in the `claims` array:

  - claim: a short string quoting or summarizing the claim
  - source: one of "user_query", "workflow_outputs", "unsupported"
  - evidence: an exact substring (for user_query) or JSON path (for
              workflow_outputs) that justifies the claim, or null
  - issue:  one short sentence if source = "unsupported", else null

A claim is "unsupported" if the value cannot be found in the user_query text
and cannot be derived from the workflow_outputs.

Then set:
  - unsupported_count: integer
  - verdict: "grounded" if unsupported_count == 0 else "ungrounded"

Return ONLY a JSON object matching the contract. No prose, no markdown, no
thinking text. First character `{`, last character `}`."""


class FaithfulnessChecker:
    """Post-decision LLM verifier that grades planner/decision claims against
    the actual user request and workflow outputs."""

    def __init__(self, provider: ChatProvider | None = None) -> None:
        self.provider = provider

    async def check(
        self,
        *,
        user_request: str,
        success_criteria: dict[str, Any] | None,
        workflow_outputs: dict[str, Any],
        decision: dict[str, Any],
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if self.provider is None:
            return {
                "verdict": "skipped",
                "unsupported_count": 0,
                "claims": [],
                "reason": "no provider configured",
            }
        payload = {
            "user_query": user_request,
            "workflow_outputs": workflow_outputs,
            "claims_subject": {
                "success_criteria": success_criteria or {},
                "decision": decision,
            },
        }
        response = await self.provider.chat(
            [
                ChatMessage(role="system", content=FAITHFULNESS_SYSTEM_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, sort_keys=True)),
            ],
            model=model,
            response_format={"type": "json_object"},
            idempotency_key=idempotency_key,
        )
        try:
            result = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise InvalidParamsError(
                "faithfulness checker returned non-JSON content",
                details={"error": str(exc), "content": response.content[:1000]},
            ) from exc
        if not isinstance(result, dict):
            raise InvalidParamsError(
                "faithfulness checker must return a JSON object",
                details={"got_type": type(result).__name__},
            )
        return _normalize(result)


def _normalize(result: dict[str, Any]) -> dict[str, Any]:
    claims = result.get("claims")
    if not isinstance(claims, list):
        claims = []
    norm_claims: list[dict[str, Any]] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        source = c.get("source")
        if source not in {"user_query", "workflow_outputs", "unsupported"}:
            source = "unsupported"
        norm_claims.append(
            {
                "claim": str(c.get("claim", "")),
                "source": source,
                "evidence": c.get("evidence"),
                "issue": c.get("issue") if source == "unsupported" else None,
            }
        )
    unsupported_count = sum(1 for c in norm_claims if c["source"] == "unsupported")
    verdict = "grounded" if unsupported_count == 0 else "ungrounded"
    return {
        "verdict": verdict,
        "unsupported_count": unsupported_count,
        "claims": norm_claims,
    }
