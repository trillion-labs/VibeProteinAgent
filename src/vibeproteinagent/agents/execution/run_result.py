from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentRunResult:
    kind: Literal[
        "candidate",
        "answer",
        "no_acceptable",
        "no_available_candidate",
        "stopped",
        "failed",
    ]
    submitted_candidate: dict[str, Any] | None = None
    text: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.kind == "answer" and self.text is None:
            raise ValueError("AgentRunResult(kind='answer') requires text")

    @classmethod
    def from_answer_decision(cls, decision: dict[str, Any]) -> AgentRunResult:
        answer = decision.get("answer") or {}
        if not isinstance(answer, dict):
            answer = {}
        evidence = answer.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = []
        return cls(
            kind="answer",
            text=answer.get("text"),
            evidence=evidence,
            reasoning=str(decision.get("reasoning") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "reasoning": self.reasoning,
        }
        if self.kind == "no_available_candidate":
            payload["status"] = "no_available_candidate"
            payload["technical_failure"] = False
        if self.kind == "no_acceptable":
            payload["status"] = "no_acceptable_candidate"
            payload["technical_failure"] = False
        if self.submitted_candidate is not None:
            payload["submitted_candidate"] = self.submitted_candidate
        if self.text is not None:
            payload["text"] = self.text
        if self.evidence:
            payload["evidence"] = self.evidence
        elif self.kind == "answer":
            payload["evidence"] = []
        return payload
