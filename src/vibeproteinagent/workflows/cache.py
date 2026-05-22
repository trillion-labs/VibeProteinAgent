from __future__ import annotations

import hashlib
from typing import Any

from vibeproteinagent.workspace.digest import normalize_params_for_digest
from vibeproteinagent.workspace.store import canonical_json

from .descriptor import WorkflowNode

_DESCRIPTOR_VERSION = "1.0.0"


def env_fingerprint() -> dict[str, Any]:
    """Return the workflow-cache environment fingerprint.

    Includes ``torch_version``, ``cuda_version``, and ``descriptor_version``.
    All three are stringified so different runtimes don't accidentally collide
    on ``None``.
    """
    try:
        import torch
        torch_version = str(getattr(torch, "__version__", "unknown"))
        cuda_version = str(getattr(torch.version, "cuda", None))
    except Exception:
        torch_version = "unavailable"
        cuda_version = "unavailable"
    return {
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "descriptor_version": _DESCRIPTOR_VERSION,
    }


def compute_cache_key(
    node: WorkflowNode,
    resolved_inputs: dict[str, Any],
    env: dict[str, Any],
    *,
    tool_version_resolved: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> str:
    """Stable sha256 over (tool, resolved version, resolved params, env).

    The cache key is the architecture-spec content-address: identical tool +
    version + inputs + env always produce the same key, regardless of dict
    ordering or JSON whitespace.

    Codex round-6 RC4: path-typed inputs marked with the ``vpa:digest``
    schema annotation are replaced with their file-content sha256 before
    hashing, so overwriting a fixture in place invalidates the cache.
    """
    normalized = normalize_params_for_digest(resolved_inputs, input_schema)
    payload = {
        "tool": node.tool,
        "version": tool_version_resolved or node.version,
        "params": normalized,
        "env": env,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = ["compute_cache_key", "env_fingerprint"]
