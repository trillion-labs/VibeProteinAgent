from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from vibeproteinagent.core.errors import ErrorCode, VPAError


@dataclass(frozen=True)
class DecisionViolation:
    path: str
    issue: str
    expected: str


class DecisionSchemaViolation(VPAError):
    code = ErrorCode.INVALID_PARAMS

    def __init__(
        self,
        violations: list[DecisionViolation],
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.violations = violations
        self.payload = payload or {}
        self.structured_error = {
            "error_kind": "decision_schema_violation",
            "violations": [
                {
                    "path": item.path,
                    "issue": item.issue,
                    "expected": item.expected,
                }
                for item in violations
            ],
            "fix_instruction": (
                "Re-emit the decision JSON with all reasoning consolidated into "
                "a single top-level `decision.reasoning` string. Remove "
                "`reasoning` from every candidate ref."
            ),
        }
        super().__init__(
            "decision payload failed decision_v1 schema validation",
            details={
                "structured_error": self.structured_error,
                "payload": self.payload,
            },
        )


@lru_cache(maxsize=1)
def decision_v1_schema() -> dict[str, Any]:
    schema_path = (
        Path(__file__).resolve().parents[2] / "schemas" / "decision_v1.json"
    )
    return json.loads(schema_path.read_text())


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(decision_v1_schema())


def validate_decision_v1(payload: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(payload), key=lambda err: list(err.path))
    if not errors:
        return
    raise DecisionSchemaViolation(
        [_violation_from_error(error) for error in errors],
        payload=payload,
    )


def _violation_from_error(error: ValidationError) -> DecisionViolation:
    path = _json_pointer(list(error.path))
    if error.validator == "required":
        missing = _missing_required_property(error)
        if missing == "reasoning" and path == "":
            return DecisionViolation(
                path="/reasoning",
                issue="reasoning_missing_at_top_level",
                expected="Place a non-empty `reasoning` string at the top level.",
            )
        return DecisionViolation(
            path=f"{path}/{missing}" if missing else path or "/",
            issue="required_property_missing",
            expected=f"Required property {missing!r} must be present.",
        )
    if error.validator == "additionalProperties":
        extra = _additional_property(error)
        extra_path = f"{path}/{_escape_json_pointer(extra)}" if extra else path
        if extra == "reasoning" and _is_candidate_ref_path(path):
            return DecisionViolation(
                path=extra_path,
                issue="reasoning_inside_candidate_forbidden",
                expected=(
                    "Place `reasoning` at the top level of the decision object; "
                    "do not include it inside candidate refs."
                ),
            )
        return DecisionViolation(
            path=extra_path,
            issue="additional_property_not_allowed",
            expected="Remove fields not declared by decision_v1.json.",
        )
    if error.validator == "enum" and path == "/decision":
        return DecisionViolation(
            path=path,
            issue="invalid_decision_enum",
            expected=(
                "decision must be one of submit_candidate, "
                "next_round_with_surviving_candidate, "
                "next_round_with_no_acceptable_candidate."
            ),
        )
    if error.validator == "minItems":
        return DecisionViolation(
            path=path or "/",
            issue="array_too_short",
            expected="submit_candidate.candidates must contain at least one candidate ref.",
        )
    return DecisionViolation(
        path=path or "/",
        issue=str(error.validator),
        expected=error.message,
    )


def _json_pointer(parts: list[Any]) -> str:
    if not parts:
        return ""
    return "/" + "/".join(_escape_json_pointer(str(part)) for part in parts)


def _escape_json_pointer(value: str | None) -> str:
    if value is None:
        return ""
    return value.replace("~", "~0").replace("/", "~1")


def _missing_required_property(error: ValidationError) -> str | None:
    match = re.search(r"'([^']+)' is a required property", error.message)
    return match.group(1) if match else None


def _additional_property(error: ValidationError) -> str | None:
    match = re.search(r"'([^']+)' was unexpected", error.message)
    return match.group(1) if match else None


def _is_candidate_ref_path(path: str) -> bool:
    return bool(re.fullmatch(r"/submit_candidate/candidates/\d+", path))
