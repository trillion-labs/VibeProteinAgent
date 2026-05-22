from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from vibeproteinagent.core.errors import InvalidParamsError

# Codex round-6 RC4: cache keys (job_id + workflow cache_key) must be
# sensitive to file *contents* — not just to path strings. Overwriting a
# fixture in place used to silently re-use the old cached result. The
# digest convention is a custom JSON-Schema keyword ``vpa:digest`` on
# fields whose canonical value is a path on disk:
#
# * ``"vpa:digest": "sha256"`` — required path. Missing file raises
#   ``E_INVALID_PARAMS``.
# * ``"vpa:digest": "sha256-if-path"`` — value may be a path *or* an
#   inline literal; digest only when the value is a string and an
#   existing file. Otherwise pass through.
#
# Digested values are encoded as ``vpa:sha256:<64hex>`` strings so a
# reader can tell at a glance that this slot was replaced — and so a
# later raw-path collision with that literal is impossible.

_DIGEST_PREFIX = "vpa:sha256:"
_MAX_DIGEST_BYTES = 256 * 1024 * 1024  # 256 MiB hard cap; see digest.md note


def _sha256_file(path: Path) -> tuple[str, int]:
    """Streaming sha256 of ``path``. Returns ``(hexdigest, size_bytes)``.

    Raises :class:`InvalidParamsError` (``E_INVALID_PARAMS``) when the
    file exceeds :data:`_MAX_DIGEST_BYTES` — pathological inputs (model
    checkpoints accidentally passed as params, etc.) should fail loud
    rather than silently stall a submission on multi-GiB I/O.
    """
    size = 0
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            size += len(buf)
            if size > _MAX_DIGEST_BYTES:
                raise InvalidParamsError(
                    f"file too large to digest: {path} "
                    f"({size} bytes > {_MAX_DIGEST_BYTES} cap)",
                    details={"path": str(path), "size_bytes": size,
                             "cap_bytes": _MAX_DIGEST_BYTES},
                )
            h.update(buf)
    return h.hexdigest(), size


def _digest_path(value: str) -> str:
    p = Path(value).expanduser()
    if not p.is_file():
        raise InvalidParamsError(
            f"path-digested input does not exist: {value}",
            details={"missing_path": value},
        )
    hexd, _size = _sha256_file(p)
    return f"{_DIGEST_PREFIX}{hexd}"


def normalize_params_for_digest(
    params: dict[str, Any],
    schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return a copy of ``params`` with path-typed fields replaced by digests.

    Walks the top-level ``properties`` of a JSON Schema; any field annotated
    with the custom ``vpa:digest`` keyword is rewritten to its content
    digest. Fields without the annotation pass through unchanged.

    With ``schema=None`` the function is a no-op — JobStore.submit and
    workflow cache fall back to raw-string hashing for synthetic / schema-less
    callers (preserving Phase 10's deterministic-id contract for tests that
    register adhoc tools).

    Errors:

    * Missing required path → ``E_INVALID_PARAMS`` with ``details.missing_path``.
    * Oversized file → ``E_INVALID_PARAMS`` with ``details.size_bytes``.
    """
    if not isinstance(params, dict) or not isinstance(schema, dict):
        return params
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return params
    out = dict(params)
    for field, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        mode = prop.get("vpa:digest")
        if mode not in ("sha256", "sha256-if-path"):
            continue
        value = out.get(field)
        if value is None:
            continue
        if mode == "sha256":
            if not isinstance(value, str):
                continue
            out[field] = _digest_path(value)
            continue
        # sha256-if-path: only digest when the value is a string AND
        # resolves to a real file. Otherwise pass through (inline yaml
        # body, inline FASTA, etc.).
        if isinstance(value, str):
            try:
                p = Path(value).expanduser()
            except (OSError, ValueError):
                continue
            if p.is_file():
                hexd, size = _sha256_file(p)
                _ = size
                out[field] = f"{_DIGEST_PREFIX}{hexd}"
    return out


__all__ = [
    "normalize_params_for_digest",
]
