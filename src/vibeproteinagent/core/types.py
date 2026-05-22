from __future__ import annotations

import json
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vibeproteinagent.core.errors import InvalidParamsError


def resolve_salt(force_new: bool) -> str | None:
    """Translate the user-facing ``force_new`` flag into a JobStore salt.

    Used by every submission surface (CLI ``--force-new``, MCP ``submit_job``,
    dynamic ``run_<tool>``) so the deterministic-id escape hatch stays one
    implementation. The salt is an opaque random hex string fed to
    :func:`derive_job_id`; agents and CLI users see only the boolean.
    """
    return uuid.uuid4().hex if force_new else None


T = TypeVar("T")


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class OkEnvelope(BaseModel, Generic[T]):  # noqa: UP046
    model_config = ConfigDict(extra="forbid")

    ok: Literal[True] = True
    data: T


class ErrEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: Literal[False] = False
    error: ErrorBody


Envelope = OkEnvelope[Any] | ErrEnvelope


def ok(data: Any) -> dict[str, Any]:
    return OkEnvelope[Any](data=data).model_dump()


def err(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return ErrEnvelope(
        error=ErrorBody(code=code, message=message, details=details or {})
    ).model_dump()


class RunContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    job_id: str
    workspace_root: Path
    job_dir: Path
    fusion_inputs: dict[str, Any] | None = None


# ---------------------------------------------------------------- CLI input helpers


def coerce_scalar(text: str) -> Any:
    """Coerce a KEY=VALUE string into bool / int / float / list / str.

    JSON-like literal lists (e.g. ``[1,2,3]`` or ``["a","b"]``) are parsed via ``json.loads``.
    Otherwise: ``true``/``false`` -> bool; integer literal -> int; float literal -> float;
    everything else stays a string.
    """
    s = text
    low = s.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    stripped = s.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return s


def parse_kv_pair(pair: str) -> tuple[str, Any]:
    """Parse one ``KEY=VALUE`` token; coerce the value with :func:`coerce_scalar`."""
    if "=" not in pair:
        raise InvalidParamsError(
            f"expected KEY=VALUE, got {pair!r}",
            details={"got": pair, "hint": "use key=value with no spaces around '='"},
        )
    key, _, value = pair.partition("=")
    key = key.strip()
    if not key:
        raise InvalidParamsError(
            f"empty key in pair {pair!r}",
            details={"got": pair},
        )
    return key, coerce_scalar(value)


def parse_kv_pairs(pairs: list[str] | None) -> dict[str, Any]:
    """Parse a list of ``KEY=VALUE`` tokens into a dict (later wins)."""
    out: dict[str, Any] = {}
    for p in pairs or []:
        k, v = parse_kv_pair(p)
        out[k] = v
    return out


def load_params_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML params object from disk, or ``-`` for stdin.

    ``path == "-"`` reads JSON from stdin. The README documents this; before
    Phase 8 it was treated as a literal filesystem path and the open(2) failed.
    Refusing TTY input avoids hanging the CLI when a user pastes ``-`` without
    actually piping anything.
    """
    if str(path) == "-":
        if sys.stdin.isatty():
            raise InvalidParamsError(
                "--params-file - requires JSON on stdin, but stdin is a TTY",
                details={"path": "-", "hint": "pipe JSON in: echo '{\"k\":\"v\"}' | vpa run …"},
            )
        text = sys.stdin.read()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InvalidParamsError(
                f"--params-file - did not parse as JSON: {exc}",
                details={"path": "-", "parse_error": str(exc)},
            ) from exc
        if not isinstance(obj, dict):
            raise InvalidParamsError(
                "--params-file - must contain a JSON object",
                details={"path": "-", "got_type": type(obj).__name__},
            )
        return obj

    p = Path(path)
    if not p.is_file():
        raise InvalidParamsError(
            f"params file {p} does not exist",
            details={"path": str(p)},
        )
    text = p.read_text()
    suffix = p.suffix.lower()
    try:
        if suffix == ".json":
            obj = json.loads(text)
        elif suffix in (".yaml", ".yml"):
            obj = yaml.safe_load(text)
        else:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                obj = yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise InvalidParamsError(
            f"params file {p} did not parse: {exc}",
            details={"path": str(p), "parse_error": str(exc)},
        ) from exc
    if not isinstance(obj, dict):
        raise InvalidParamsError(
            f"params file {p} must be a JSON/YAML object",
            details={"path": str(p), "got_type": type(obj).__name__},
        )
    return obj


def iter_batch_records(spec: str) -> Iterator[dict[str, Any]]:
    """Yield one params dict per JSONL record. ``spec='-'`` reads stdin."""
    if spec == "-":
        source: Iterator[str] = iter(sys.stdin)
    else:
        p = Path(spec)
        if not p.is_file():
            raise InvalidParamsError(
                f"batch file {p} does not exist",
                details={"path": str(p)},
            )
        source = iter(p.read_text().splitlines())

    for lineno, raw in enumerate(source, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InvalidParamsError(
                f"batch line {lineno} is not valid JSON: {exc}",
                details={"line": lineno, "parse_error": str(exc)},
            ) from exc
        if not isinstance(obj, dict):
            raise InvalidParamsError(
                f"batch line {lineno} must be a JSON object",
                details={"line": lineno, "got_type": type(obj).__name__},
            )
        yield obj
