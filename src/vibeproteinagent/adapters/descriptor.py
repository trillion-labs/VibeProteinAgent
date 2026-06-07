from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised by Python 3.10 sidecars
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042
        pass

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vibeproteinagent.core.errors import InvalidParamsError

from ._error_guides import enrich_errors

_SCHEMA_PATH = Path(__file__).parent / "_schema" / "tool-descriptor.schema.json"


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text())


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


class Category(StrEnum):
    STRUCTURE_PREDICTION = "structure_prediction"
    PROTEIN_DESIGN = "protein_design"
    SEQUENCE_ENGINEERING = "sequence_engineering"
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    INVERSE_FOLDING = "inverse_folding"
    EMBEDDING = "embedding"
    SCORING = "scoring"
    DOCKING = "docking"
    MD = "md"
    MISC = "misc"


class Runtime(StrEnum):
    PYTHON = "python"
    DOCKER = "docker"


class Backend(StrEnum):
    PYTORCH = "pytorch"
    JAX = "jax"
    BIOPYTHON = "biopython"
    NUMPY = "numpy"
    BINARY = "binary"
    MIXED = "mixed"


class Precision(StrEnum):
    FP32 = "fp32"
    TF32 = "tf32"
    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"


class StartupCost(StrEnum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class ExecutionIsolation(StrEnum):
    SHARED_ENV = "shared_env"
    PERSISTENT_SIDECAR = "persistent_sidecar"
    SUBPROCESS = "subprocess"
    DOCKER = "docker"


class SupportStatus(StrEnum):
    VERIFIED = "verified"
    PARTIAL = "partial"
    BLOCKED_DEP = "blocked_dep"
    BLOCKED_LICENSE = "blocked_license"
    BLOCKED_UNAVAILABLE = "blocked_unavailable"


class FusionOutputKind(StrEnum):
    CUDA = "torch.Tensor (cuda)"
    CPU = "torch.Tensor (cpu)"
    DISK = "disk"


class WeightURI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=1)


class Fusion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_kind: FusionOutputKind | None = None
    input_kinds: list[str] | None = None


class Sharding(BaseModel):
    """Planner-facing metadata for candidate/job-level fan-out.

    This describes safe workflow shapes. It is intentionally not an optimizer:
    VPA does not rewrite user-authored params from this metadata alone.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Named sharding strategy understood by planners/docs.",
    )
    shardable_fields: list[str] = Field(min_length=1)
    recommended_shard_size: int = Field(ge=1)
    default_max_parallel: int | None = Field(default=None, ge=1)

    @field_validator("shardable_fields")
    @classmethod
    def _validate_shardable_fields(cls, v: list[str]) -> list[str]:
        pat = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        seen: set[str] = set()
        for name in v:
            if not pat.match(name):
                raise ValueError(
                    f"shardable_fields entry '{name}' is not a valid parameter name"
                )
            if name in seen:
                raise ValueError(f"duplicate shardable_fields entry '{name}'")
            seen.add(name)
        return v


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^[A-Za-z0-9._+-]+$")
    category: Category
    description: str = Field(min_length=1, max_length=280)
    runtime: Runtime
    backend: Backend
    gpu_required: bool
    gpus_required: int = Field(default=1, ge=1, le=8)
    vram_min_gib: int | None = Field(default=None, ge=1)
    startup_cost: StartupCost
    estimated_runtime_s: int = Field(ge=0)
    warm_load_capable: bool = Field(
        description=(
            "True only when warm/preload keeps meaningful runtime state resident "
            "across calls, not merely when the adapter object can be constructed."
        )
    )
    pin_warm: bool = False
    execution_isolation: ExecutionIsolation = ExecutionIsolation.SHARED_ENV
    environment_name: str | None = None
    environment_reason: str | None = Field(default=None, max_length=400)
    support_status: SupportStatus = SupportStatus.VERIFIED
    support_status_reason: str | None = Field(default=None, max_length=500)
    availability_probe: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_.]*:[a-zA-Z_][A-Za-z0-9_]*$",
    )
    distributed_entrypoint: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Optional adapter-level distributed runner. Required before "
            "gpus_required > 1 jobs may be admitted to multi-worker dispatch."
        ),
    )
    supported_tasks: list[str] = Field(min_length=1)
    input_schema: str | dict[str, Any]
    output_schema: str | dict[str, Any]
    supported_precisions: list[Precision] = Field(min_length=1)
    weight_uris: list[WeightURI] | None = None
    supports_batching: bool = False
    planner_visible: bool = True
    audit_visible: bool = True
    incompatible_with: list[str] = Field(
        default_factory=list,
        description="Tool names that must not be co-resident in the same worker process.",
    )
    fusion: Fusion | None = None
    sharding: Sharding | None = None

    @field_validator("incompatible_with")
    @classmethod
    def _validate_incompatible_names(cls, v: list[str]) -> list[str]:
        pat = re.compile(r"^[a-z][a-z0-9_]*$")
        for n in v:
            if not pat.match(n):
                raise ValueError(
                    f"incompatible_with entry '{n}' is not a valid tool name "
                    f"(must match ^[a-z][a-z0-9_]*$)"
                )
        return v
    entry_point: str | None = Field(
        default=None,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$",
    )


def _read_descriptor_file(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        loaded = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError:
            loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise InvalidParamsError(
            f"descriptor at {path} did not parse to an object",
            details={"path": str(path), "got_type": type(loaded).__name__},
        )
    return loaded


def load_descriptor(path: Path) -> ToolDescriptor:
    """Load a tool descriptor from YAML/JSON, validating against the JSON Schema first.

    Raises InvalidParamsError (E_INVALID_PARAMS) on schema or pydantic failures, with
    `details` pointing at the offending field where possible.
    """
    path = Path(path)
    try:
        raw = _read_descriptor_file(path)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise InvalidParamsError(
            f"descriptor at {path} is not valid YAML/JSON: {exc}",
            details={"path": str(path), "parse_error": str(exc)},
        ) from exc

    errors = sorted(_validator().iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        field_path = ".".join(str(p) for p in first.absolute_path) or "<root>"
        raise InvalidParamsError(
            f"descriptor {path} failed JSON Schema: {first.message} at '{field_path}'",
            details={
                "path": str(path),
                "field": field_path,
                "schema_message": first.message,
                "all_errors": [
                    {
                        "field": ".".join(str(p) for p in e.absolute_path) or "<root>",
                        "message": e.message,
                    }
                    for e in errors
                ],
            },
        )

    try:
        return ToolDescriptor.model_validate(raw)
    except Exception as exc:  # pragma: no cover - schema should catch first
        raise InvalidParamsError(
            f"descriptor {path} failed pydantic validation: {exc}",
            details={"path": str(path), "pydantic_error": str(exc)},
        ) from exc


def serialize_descriptor(d: ToolDescriptor) -> dict[str, Any]:
    """Round-trip a ToolDescriptor back to a JSON-Schema-compatible dict."""
    out = d.model_dump(mode="json", exclude_none=True)
    if d.vram_min_gib is None:
        out["vram_min_gib"] = None
    if d.planner_visible is True:
        out.pop("planner_visible", None)
    if d.audit_visible is True:
        out.pop("audit_visible", None)
    return out


# ---------------------------------------------------------------- params validation


def resolve_input_schema_dict(descriptor: ToolDescriptor) -> dict[str, Any]:
    """Return the descriptor's ``input_schema`` as a fully-loaded dict.

    Single source of truth for "what does this tool's input look like?" —
    used by the CLI, MCP server, and the workspace digest helper. Inline
    dict schemas pass through; ``"path/to/file.json"`` references are
    resolved relative to the adapter module.
    """
    schema = descriptor.input_schema
    if isinstance(schema, dict):
        return schema
    return _load_schema_file(descriptor, schema, kind="input_schema")


def resolve_output_schema_dict(descriptor: ToolDescriptor) -> dict[str, Any]:
    """Return the descriptor's ``output_schema`` as a fully-loaded dict.

    Wildcard/free-form output schemas return an empty dict, which callers
    should treat as "not statically checkable".
    """
    schema = getattr(descriptor, "output_schema", None)
    if not schema or schema == "*":
        return {}
    if isinstance(schema, dict):
        return schema
    return _load_schema_file(descriptor, schema, kind="output_schema")


def _load_schema_file(
    descriptor: ToolDescriptor, schema_ref: str, *, kind: str
) -> dict[str, Any]:
    """Locate ``schema_ref`` relative to the descriptor's adapter module."""
    import importlib

    candidates: list[Path] = []
    if descriptor.entry_point and ":" in descriptor.entry_point:
        module_name = descriptor.entry_point.rsplit(":", 1)[0]
        try:
            mod = importlib.import_module(module_name)
            mod_file = getattr(mod, "__file__", None)
            if mod_file:
                candidates.append(Path(mod_file).parent / schema_ref)
        except ImportError:
            pass
    # Fallback: in-tree adapter dir.
    candidates.append(Path(__file__).parent / descriptor.name / schema_ref)
    for c in candidates:
        if c.is_file():
            return json.loads(c.read_text())
    raise InvalidParamsError(
        f"{kind} {schema_ref!r} not found for tool {descriptor.name!r}",
        details={"schema_ref": schema_ref, "tried": [str(c) for c in candidates]},
    )


_VALIDATOR_CACHE: dict[tuple[str, str, str], Draft202012Validator | None] = {}


def _cache_key(descriptor: Any, kind: str) -> tuple[str, str, str] | None:
    """Compute a cache key, or ``None`` if the descriptor lacks name/version.

    Synthetic descriptors used in tests (SimpleNamespace stand-ins) may not
    populate ``version``; we just skip caching for those instead of caching
    a stale entry under a partially-empty key.
    """
    name = getattr(descriptor, "name", None)
    version = getattr(descriptor, "version", None)
    if not name or not version:
        return None
    return (str(name), str(version), kind)


def input_validator(descriptor: ToolDescriptor) -> Draft202012Validator:
    """Return a :class:`Draft202012Validator` for ``descriptor.input_schema``.

    Single owner of "build a JSON Schema validator from a ToolDescriptor":
    used by ``vpa run`` and by the workflow executor before submitting any
    node's job. Centralising it keeps the two paths in lockstep. Per-tool
    validators are cached because both validation and file-loading cost
    add up under burst-load (Phase 8 RC3 wires this into every job's
    terminal path).
    """
    cache_key = _cache_key(descriptor, "input")
    if cache_key is not None and cache_key in _VALIDATOR_CACHE:
        cached = _VALIDATOR_CACHE[cache_key]
        if cached is not None:
            return cached
    schema = descriptor.input_schema
    if isinstance(schema, str):
        schema = _load_schema_file(descriptor, schema, kind="input_schema")
    Draft202012Validator.check_schema(schema)
    v = Draft202012Validator(schema)
    if cache_key is not None:
        _VALIDATOR_CACHE[cache_key] = v
    return v


def output_validator(descriptor: ToolDescriptor) -> Draft202012Validator | None:
    """Return a :class:`Draft202012Validator` for ``descriptor.output_schema``.

    Returns ``None`` when the descriptor explicitly opts out of output
    validation by setting ``output_schema`` to the wildcard ``"*"`` or
    leaving it empty — defensive for adapters whose outputs are
    intentionally free-form. Validators are cached per ``(name, version)``
    so the supervisor's terminal path doesn't re-read the schema file on
    every job completion.
    """
    cache_key = _cache_key(descriptor, "output")
    if cache_key is not None and cache_key in _VALIDATOR_CACHE:
        return _VALIDATOR_CACHE[cache_key]
    schema = getattr(descriptor, "output_schema", None)
    if not schema:
        if cache_key is not None:
            _VALIDATOR_CACHE[cache_key] = None
        return None
    if isinstance(schema, str):
        if schema == "*":
            if cache_key is not None:
                _VALIDATOR_CACHE[cache_key] = None
            return None
        schema = _load_schema_file(descriptor, schema, kind="output_schema")
    if not isinstance(schema, dict) or not schema:
        if cache_key is not None:
            _VALIDATOR_CACHE[cache_key] = None
        return None
    Draft202012Validator.check_schema(schema)
    v = Draft202012Validator(schema)
    if cache_key is not None:
        _VALIDATOR_CACHE[cache_key] = v
    return v


class InvalidOutputError(InvalidParamsError):
    """Raised when an adapter's output dict fails its declared output schema.

    Distinct error code so callers (and the supervisor) can tell adapter-output
    bugs apart from caller-supplied invalid params. See :data:`ErrorCode.INVALID_OUTPUT`.
    """

    code = "E_INVALID_OUTPUT"


def validate_params(
    descriptor: ToolDescriptor, params: dict[str, Any]
) -> None:
    """Validate ``params`` against the tool's input schema.

    Raises :class:`InvalidParamsError` (``E_INVALID_PARAMS``) listing every
    schema error, with ``details.field`` pointing at the first offending
    field. Single call site for tool-input validation; the CLI's ``run``
    command and the workflow executor's pre-submit step both go through
    here so the two surfaces can't drift.

    Error messages are enriched with retry-recovery hints (constraint info,
    schema description / examples, and per-tool alternative-tool suggestions)
    via :mod:`._error_guides` so the planner LLM has actionable context on
    retry. Enrichment fires only on validation failure (no impact on the
    happy path).
    """
    validator = input_validator(descriptor)
    errors = sorted(validator.iter_errors(params), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    raw_errors = [
        {
            "field": ".".join(str(p) for p in e.absolute_path) or "<root>",
            "message": e.message,
        }
        for e in errors
    ]
    schema = resolve_input_schema_dict(descriptor)
    enriched_errors = enrich_errors(
        tool_name=descriptor.name,
        schema=schema,
        raw_errors=raw_errors,
        params=params,
    )
    first_field = raw_errors[0]["field"]
    first_message = enriched_errors[0]["message"]
    raise InvalidParamsError(
        f"params for {descriptor.name} failed validation: {first_message} at '{first_field}'",
        details={
            "tool": descriptor.name,
            "field": first_field,
            "errors": enriched_errors,
        },
    )


def drop_optional_none_params(
    descriptor: ToolDescriptor,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Return params with optional ``None`` values omitted.

    LLM planners often use JSON ``null`` to mean "use the tool default". VPA
    tool schemas express defaults by making a field optional, not by accepting
    null unless the schema explicitly says so. This helper keeps that boundary
    behavior consistent across planner validation and workflow execution.
    """

    schema = resolve_input_schema_dict(descriptor)
    required = set(schema.get("required") or [])
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    if not properties:
        return params
    out = dict(params)
    for name, value in list(params.items()):
        if value is not None or name in required:
            continue
        prop = properties.get(name)
        if not isinstance(prop, dict) or _schema_allows_null(prop):
            continue
        out.pop(name, None)
    return out


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    typ = schema.get("type")
    if typ == "null":
        return True
    if isinstance(typ, list) and "null" in typ:
        return True
    return False


def validate_output(
    descriptor: ToolDescriptor, output: dict[str, Any]
) -> None:
    """Validate an adapter's output dict against its declared ``output_schema``.

    Mirrors :func:`validate_params`. Raises :class:`InvalidOutputError`
    (``E_INVALID_OUTPUT``) on schema failure — the supervisor catches this
    in its post-adapter finalisation path and marks the job ``failed`` with
    ``details.violations`` listing every field error. Skips validation when
    the descriptor declares no schema, or its schema is the wildcard ``"*"``.
    """
    validator = output_validator(descriptor)
    if validator is None:
        return
    errors = sorted(validator.iter_errors(output), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    first = errors[0]
    field = ".".join(str(p) for p in first.absolute_path) or "<root>"
    violations = [
        {
            "field": ".".join(str(p) for p in e.absolute_path) or "<root>",
            "message": e.message,
        }
        for e in errors
    ]
    raise InvalidOutputError(
        f"output for {descriptor.name} failed validation: {first.message} at '{field}'",
        details={
            "tool": descriptor.name,
            "field": field,
            "violations": violations,
        },
    )
