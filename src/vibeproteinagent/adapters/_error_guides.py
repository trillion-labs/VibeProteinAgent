"""Validation-error enrichment for plan-stage retries.

When tool param validation fails, plain JSON Schema messages like
"'X' does not match '^GO:[0-9]{7}$' at 'go_id'" don't give the LLM enough
context to recover. This module enriches each error with:

- A format hint derived from the schema constraint (pattern, enum, type, range).
- Description and examples from the schema metadata, if present.
- Alternative tools / fallback strategies for fields with known LLM failure
  patterns (per-tool registry).

Token-cost note: enrichments only fire on validation FAILURE in the retry
boundary. The tool catalog sent on the happy path is unchanged. The
enrichment averages ~50-200 extra characters per error message.

This module lives in the adapters layer (alongside descriptor.py) because
the field-level guide registry is tool-specific. Adapters do not import
from agents/ in this codebase, so the agent-layer retry path can pick up
the enriched messages without inverting the layer.
"""

from __future__ import annotations

import re
from typing import Any

from vibeproteinagent.adapters.schema_affordance import matching_conditional_required_hints

# Per-(tool_name, field_path) registry of recovery hints. Add an entry only
# when an LLM frequently misuses a field AND there is a clearly better tool
# or fallback that the LLM can pick instead. Keep entries short — they are
# concatenated into already-busy retry error messages.
#
# field_path uses dot-notation (matching what validate_params already emits
# via .absolute_path). For top-level fields this is just the field name.
FIELD_ERROR_GUIDES: dict[tuple[str, str], dict[str, Any]] = {
    ("go_sequences", "go_id"): {
        "alternative_tools": [
            "interpro_sequences (keyword-based protein retrieval)",
            "literature_search (Europe PMC search for the GO ID by topic)",
        ],
        "fallback_hint": (
            "If you do not know an exact GO ID for the user's keyword, use "
            "interpro_sequences or literature_search to discover the right "
            "identifier instead of guessing or emitting a placeholder."
        ),
    },
    ("protenix_v2", "msa_backend"): {
        "fallback_hint": (
            "`mmseqs2_gpu` is a VPA tool name, not a valid msa_backend value. "
            "Use `auto`, `remote`, or `none`; run mmseqs2_gpu separately only "
            "when you need an explicit MSA artifact."
        ),
    },
}


def _navigate_subschema(schema: dict[str, Any], path: list[str]) -> dict[str, Any] | None:
    """Walk a JSON Schema following property names.

    Tool input schemas are typically flat (one level of properties), but we
    also handle nested object fields via repeated `properties` lookup. If the
    path cannot be resolved, return None — the caller falls back to using
    only the raw error message.
    """
    if not isinstance(schema, dict):
        return None
    current: Any = schema
    for segment in path:
        if not isinstance(current, dict):
            return None
        properties = current.get("properties")
        if isinstance(properties, dict) and segment in properties:
            current = properties[segment]
            continue
        # Unrecognized path component (e.g., array index) — bail.
        return None
    return current if isinstance(current, dict) else None


def _format_constraint_hint(subschema: dict[str, Any]) -> str | None:
    """Pull constraint info from the subschema in a compact human form."""
    hints: list[str] = []
    pattern = subschema.get("pattern")
    if isinstance(pattern, str):
        hints.append(f"must match regex `{pattern}`")
    enum = subschema.get("enum")
    if isinstance(enum, list) and enum:
        if len(enum) <= 6:
            hints.append(f"must be one of {enum}")
        else:
            hints.append(f"must be one of (first 6 shown) {enum[:6]} ...")
    typ = subschema.get("type")
    if isinstance(typ, list):
        hints.append(f"type one of {typ}")
    elif isinstance(typ, str):
        hints.append(f"type `{typ}`")
    if "minimum" in subschema or "maximum" in subschema:
        lo = subschema.get("minimum")
        hi = subschema.get("maximum")
        hints.append(f"numeric range [{lo}, {hi}]")
    if "minLength" in subschema or "maxLength" in subschema:
        lo = subschema.get("minLength")
        hi = subschema.get("maxLength")
        hints.append(f"length [{lo}, {hi}]")
    return "; ".join(hints) if hints else None


def _format_examples(subschema: dict[str, Any]) -> str | None:
    examples = subschema.get("examples")
    if not isinstance(examples, list) or not examples:
        return None
    return ", ".join(repr(e) for e in examples[:3])


_ADDITIONAL_PROPERTIES_RE = re.compile(
    r"\(([^)]+?)\s+(?:was|were)\s+unexpected\)"
)


def _candidate_for(rejected: str, valid: list[str]) -> str | None:
    """Cheap nearest-name suggestion for an unexpected field.

    No external dependency: prefers substring containment, then shared 3-char
    prefix. Returns the first candidate (or None). Intentionally simple — the
    full `Valid fields` list always accompanies this hint so the LLM has full
    context even when the suggestion misses.
    """
    rl = rejected.lower()
    for v in valid:
        if rl in v.lower() or v.lower() in rl:
            return v
    if len(rejected) >= 3:
        for v in valid:
            if v[:3].lower() == rejected[:3].lower():
                return v
    return None


def _format_property_suggestion(
    raw_message: str,
    subschema: dict[str, Any] | None,
) -> str | None:
    """For 'Additional properties not allowed (X was unexpected)' errors,
    list the valid property names and offer a did-you-mean for each rejected
    name. Without this hint the LLM only learns its guess was wrong, not
    what the actual field names are — which we've observed produces stuck
    retries (same guess 3x in a row) on tools like bindcraft/go_sequences.
    """
    if "Additional properties are not allowed" not in raw_message:
        return None
    if not isinstance(subschema, dict):
        return None
    props = subschema.get("properties")
    if not isinstance(props, dict) or not props:
        return None
    valid: list[str] = list(props.keys())

    rejected: list[str] = []
    m = _ADDITIONAL_PROPERTIES_RE.search(raw_message)
    if m:
        rejected = [
            r.strip().strip("'").strip('"')
            for r in m.group(1).split(",")
            if r.strip()
        ]

    pieces: list[str] = []
    suggestions: list[str] = []
    for r in rejected:
        cand = _candidate_for(r, valid)
        if cand:
            suggestions.append(f"`{r}` → maybe `{cand}`")
    if suggestions:
        pieces.append("Did you mean: " + "; ".join(suggestions))

    # Cap the displayed list so a 20-field schema doesn't bloat every retry.
    shown = valid if len(valid) <= 12 else valid[:12]
    suffix = "" if len(valid) <= 12 else f", ... ({len(valid) - 12} more)"
    pieces.append("Valid fields: " + ", ".join(shown) + suffix)
    return ". ".join(pieces) + "."


def _format_root_schema_hint(
    *,
    field_path: str,
    raw_message: str,
    subschema: dict[str, Any] | None,
    params: dict[str, Any] | None = None,
) -> str | None:
    if field_path != "<root>" or not isinstance(subschema, dict):
        return None
    if "not valid under any of the given schemas" not in raw_message and (
        "is a required property" not in raw_message
    ):
        return None
    props = subschema.get("properties")
    valid_fields = list(props) if isinstance(props, dict) else []
    required = _root_required_options(subschema)
    pieces: list[str] = []
    if params:
        conditional_hints = matching_conditional_required_hints(subschema, params)
        pieces.extend(conditional_hints)
    if required:
        pieces.append(
            "Root schema guidance: satisfy one required input set: "
            + "; ".join(required)
        )
    if valid_fields:
        shown = valid_fields if len(valid_fields) <= 12 else valid_fields[:12]
        suffix = "" if len(valid_fields) <= 12 else f", ... ({len(valid_fields) - 12} more)"
        pieces.append("Valid root fields: " + ", ".join(shown) + suffix)
    return ". ".join(pieces) + "." if pieces else None


def _root_required_options(schema: dict[str, Any]) -> list[str]:
    options: list[str] = []
    required = schema.get("required")
    if isinstance(required, list) and required:
        options.append(", ".join(str(item) for item in required if isinstance(item, str)))
    for keyword in ("oneOf", "anyOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            branch_required = variant.get("required")
            if isinstance(branch_required, list) and branch_required:
                options.append(
                    ", ".join(
                        str(item) for item in branch_required if isinstance(item, str)
                    )
                )
    return [item for item in options if item]


def _format_description(subschema: dict[str, Any]) -> str | None:
    desc = subschema.get("description")
    if isinstance(desc, str) and desc.strip():
        stripped = desc.strip()
        if len(stripped) > 200:
            stripped = stripped[:197] + "..."
        return stripped
    return None


def enrich_error_message(
    *,
    tool_name: str,
    field_path: str,
    raw_message: str,
    subschema: dict[str, Any] | None,
    params: dict[str, Any] | None = None,
) -> str:
    """Build a retry-friendly enriched error message.

    The returned string starts with the raw schema error so existing log
    parsers and the JSON `details` payload continue to read the original
    message as a prefix. Hints are appended in order: constraint, examples,
    description, alternative tools / fallback.
    """
    parts: list[str] = [raw_message]
    sub = subschema or {}

    prop_hint = _format_property_suggestion(raw_message, sub)
    if prop_hint:
        parts.append(prop_hint)

    root_hint = _format_root_schema_hint(
        field_path=field_path,
        raw_message=raw_message,
        subschema=sub,
        params=params,
    )
    if root_hint:
        parts.append(root_hint)

    constraint = _format_constraint_hint(sub)
    if constraint:
        parts.append(f"Constraint: {constraint}.")

    examples = _format_examples(sub)
    if examples:
        parts.append(f"Examples: {examples}.")

    desc = _format_description(sub)
    if desc:
        parts.append(f"Field description: {desc}")

    guide = FIELD_ERROR_GUIDES.get((tool_name, field_path))
    if guide:
        alternatives = guide.get("alternative_tools")
        if isinstance(alternatives, list) and alternatives:
            parts.append("Alternative tools: " + "; ".join(alternatives) + ".")
        fallback = guide.get("fallback_hint")
        if isinstance(fallback, str) and fallback:
            parts.append(fallback)

    return " ".join(parts)


def enrich_errors(
    *,
    tool_name: str,
    schema: dict[str, Any],
    raw_errors: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a new errors list with each `message` enriched and the original
    raw text retained under `raw_message`.

    Existing callers that read `error["message"]` get the richer text
    automatically; callers that need the original (for log-parsing or for
    a strict diff against pre-enrichment behavior) can read `raw_message`.
    """
    enriched: list[dict[str, Any]] = []
    for err in raw_errors:
        field_path = err.get("field", "")
        raw_message = err.get("message", "")
        path_segments = [seg for seg in field_path.split(".") if seg and seg != "<root>"]
        subschema = _navigate_subschema(schema, path_segments)
        new_message = enrich_error_message(
            tool_name=tool_name,
            field_path=field_path,
            raw_message=raw_message,
            subschema=subschema,
            params=params,
        )
        enriched.append(
            {
                **err,
                "message": new_message,
                "raw_message": raw_message,
            }
        )
    return enriched
