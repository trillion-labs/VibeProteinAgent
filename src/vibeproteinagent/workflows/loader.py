from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from vibeproteinagent.core.errors import InvalidParamsError

from .descriptor import Workflow


def load_workflow(path: Path | str) -> Workflow:
    """Parse a YAML workflow descriptor at ``path`` into a Workflow."""
    p = Path(path)
    if not p.is_file():
        raise InvalidParamsError(
            f"workflow descriptor not found: {p}",
            details={"path": str(p)},
        )
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise InvalidParamsError(
            f"workflow descriptor at {p} is not valid YAML: {exc}",
            details={"path": str(p), "error": str(exc)},
        ) from exc
    if not isinstance(raw, dict):
        raise InvalidParamsError(
            f"workflow descriptor at {p} must be a mapping",
            details={"path": str(p), "type": type(raw).__name__},
        )
    try:
        wf = Workflow.model_validate(raw)
    except ValidationError as exc:
        raise InvalidParamsError(
            f"workflow descriptor at {p} failed validation: {exc.errors()[0]['msg']}",
            details={"path": str(p), "errors": exc.errors()},
        ) from exc
    # Reject DAG-level errors (missing upstream, cycle) at load time so
    # they surface as ``E_INVALID_PARAMS`` instead of as a runtime timeout
    # (Codex round-3 RC2).
    try:
        wf.validate_dag()
    except InvalidParamsError as exc:
        details = dict(exc.details or {})
        details["path"] = str(p)
        raise InvalidParamsError(exc.message, details=details) from exc
    return wf


def serialize_workflow(wf: Workflow) -> str:
    """Render ``wf`` back to canonical YAML."""
    body: dict[str, Any] = {
        "name": wf.name,
        "version": wf.version,
    }
    if wf.description is not None:
        body["description"] = wf.description
    if wf.params:
        body["params"] = {k: _param_to_dict(v) for k, v in wf.params.items()}
    body["nodes"] = [_node_to_dict(n) for n in wf.nodes]
    return yaml.safe_dump(body, sort_keys=False)


def _param_to_dict(p: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"type": p.type}
    if p.required:
        out["required"] = True
    if p.default is not None:
        out["default"] = p.default
    if p.description is not None:
        out["description"] = p.description
    return out


def _node_to_dict(n: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": n.id, "tool": n.tool}
    if n.version and n.version != "*":
        out["version"] = n.version
    if n.inputs:
        out["inputs"] = n.inputs
    if not n.cache:
        out["cache"] = False
    if n.foreach:
        out["foreach"] = n.foreach
    if n.when:
        out["when"] = n.when
    if n.on_failure != "stop":
        out["on_failure"] = n.on_failure
    if n.max_retries:
        out["max_retries"] = n.max_retries
    if n.priority != "normal":
        out["priority"] = n.priority
    return out


__all__ = ["load_workflow", "serialize_workflow"]
