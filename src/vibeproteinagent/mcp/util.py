from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from vibeproteinagent.adapters import ToolDescriptor
from vibeproteinagent.adapters.descriptor import resolve_input_schema_dict
from vibeproteinagent.core.errors import InvalidParamsError, VPAError
from vibeproteinagent.workspace import JobRecord


def resolve_input_schema(descriptor: ToolDescriptor) -> dict[str, Any]:
    """Resolve a descriptor's `input_schema` to a real JSON-Schema dict.

    Thin alias kept for backwards compatibility; delegates to the
    canonical helper in :mod:`vibeproteinagent.adapters.descriptor` so
    CLI, MCP, and the workspace digest helper all see the same schema.
    """
    return resolve_input_schema_dict(descriptor)


def validate_against(schema: dict[str, Any], params: dict[str, Any]) -> None:
    """Validate `params` against a JSON Schema, raising InvalidParamsError on failure."""
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(params), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    first = errors[0]
    field = ".".join(str(p) for p in first.absolute_path) or "<root>"
    raise InvalidParamsError(
        f"params failed validation: {first.message} at '{field}'",
        details={
            "field": field,
            "errors": [
                {
                    "field": ".".join(str(p) for p in e.absolute_path) or "<root>",
                    "message": e.message,
                }
                for e in errors
            ],
        },
    )


def job_to_dict(rec: JobRecord) -> dict[str, Any]:
    """Serialize a JobRecord to a plain dict suitable for MCP responses."""
    return {
        "job_id": rec.job_id,
        "call_id": rec.call_id,
        "state": rec.state.value,
        "tool": rec.tool_name,
        "version": rec.tool_version,
        "params": rec.params,
        "params_hash": rec.params_hash,
        "output": rec.output,
        "error_code": rec.error_code,
        "error_message": rec.error_message,
        "error_details": rec.error_details,
        "log_path": rec.log_path,
        "workspace": rec.workspace,
        "workflow_id": rec.workflow_id,
        "workflow_node_id": rec.workflow_node_id,
        "worker_id": rec.worker_id,
        "gpu_ids": rec.gpu_ids,
        "priority": rec.priority.value,
        "retry_count": rec.retry_count,
        "max_retries": rec.max_retries,
        "submitted_at": rec.submitted_at,
        "queued_at": rec.queued_at,
        "started_at": rec.started_at,
        "finished_at": rec.finished_at,
        "cache_hit": rec.cache_hit,
    }


def vpa_error_to_dict(exc: VPAError) -> dict[str, Any]:
    """Shape a VPAError into the dict MCP tools return on domain failure."""
    return {
        "ok": False,
        "code": exc.code,
        "message": exc.message,
        "details": exc.details,
    }


__all__ = [
    "job_to_dict",
    "resolve_input_schema",
    "validate_against",
    "vpa_error_to_dict",
]
