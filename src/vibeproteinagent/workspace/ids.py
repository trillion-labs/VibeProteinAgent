from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from vibeproteinagent.core.errors import InvalidParamsError

_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{32}$")
_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,128}$")


def _canonical(obj: Any) -> str:
    """Deterministic JSON encoding used for ID derivation.

    Mirrors :func:`vibeproteinagent.workspace.store.canonical_json` — kept
    local so the ID module has no upward import on the store module.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def derive_job_id(
    *,
    tool: str,
    version: str,
    params: dict[str, Any],
    workspace: str,
    trajectory_id: str | None = None,
    workflow_id: str | None = None,
    workflow_node_id: str | None = None,
    salt: str | None = None,
) -> str:
    """Return the deterministic ``job_<32hex>`` ID for a submission.

    Per **Architecture v1 — Errata 2**, the job_id is the first 32 hex
    characters of ``sha256`` over a canonical-JSON encoding of the
    submission. The same ``(tool, version, params, workspace, trajectory,
    workflow)`` tuple always resolves to the same id; ``salt`` is the ``--force-new``
    escape hatch for explicit seed-reproducibility re-runs.

    ``version`` must be the *resolved* concrete version (e.g. ``"1.0.0"``),
    not a PEP-440 spec like ``"*"`` — two clients asking for ``"*"`` should
    not collide if they happen to resolve to different versions.
    """
    canon = _canonical({
        "tool": tool,
        "version": version,
        "params": params,
        "workspace": workspace,
        "trajectory_id": trajectory_id,
        "workflow_id": workflow_id,
        "workflow_node_id": workflow_node_id,
        "salt": salt,
    })
    return "job_" + hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]


def is_job_id(s: str) -> bool:
    """True iff ``s`` matches the job_id format.

    The regex accepts both legacy UUIDv7-derived rows (from before Errata 2)
    and the sha256-derived ids; both render as 32 lowercase hex characters
    after the ``job_`` prefix.
    """
    return bool(_JOB_ID_RE.match(s))


def validate_call_id(s: str) -> str:
    """Validate a call_id against the §5 regex; return it on success, raise on failure."""
    if not isinstance(s, str) or not _CALL_ID_RE.match(s):
        raise InvalidParamsError(
            f"invalid call_id: {s!r} (must match {_CALL_ID_RE.pattern})",
            details={"field": "call_id", "got": s},
        )
    return s


def parse_id(s: str) -> tuple[str, str]:
    """Classify an identifier as ``("job_id", s)`` or ``("call_id", s)``.

    Raises :class:`InvalidParamsError` if it matches neither.
    """
    if not isinstance(s, str) or not s:
        raise InvalidParamsError("id must be a non-empty string", details={"got": s})
    if is_job_id(s):
        return ("job_id", s)
    if _CALL_ID_RE.match(s):
        return ("call_id", s)
    raise InvalidParamsError(
        f"id {s!r} is neither a valid job_id nor call_id",
        details={"got": s},
    )
