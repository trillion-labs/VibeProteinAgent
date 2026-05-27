"""Tests for plan-stage validation-error enrichment.

When a tool's input schema rejects the planner LLM's emitted params,
the resulting error message must carry enough context (constraint,
examples, description, alternative tools) for the LLM to recover on
retry. This is the Plan-stage fail-resistance fix discussed during
the first canonical-coverage matrix cycle.

Enrichment fires only on validation FAILURE; the happy path is
unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeproteinagent.adapters._error_guides import (
    FIELD_ERROR_GUIDES,
    _candidate_for,
    _format_constraint_hint,
    _format_examples,
    _format_description,
    _format_property_suggestion,
    _format_root_schema_hint,
    _navigate_subschema,
    enrich_error_message,
    enrich_errors,
)
from vibeproteinagent.adapters.descriptor import ToolDescriptor, validate_params
from vibeproteinagent.core.errors import InvalidParamsError


# --- _navigate_subschema ----------------------------------------------


def test_navigate_subschema_top_level_property() -> None:
    schema = {
        "properties": {
            "go_id": {"type": "string", "pattern": "^GO:[0-9]{7}$"},
        }
    }
    assert _navigate_subschema(schema, ["go_id"]) == {
        "type": "string",
        "pattern": "^GO:[0-9]{7}$",
    }


def test_navigate_subschema_unknown_path_returns_none() -> None:
    schema = {"properties": {"go_id": {}}}
    assert _navigate_subschema(schema, ["unknown_field"]) is None


def test_navigate_subschema_empty_path_returns_root() -> None:
    schema = {"properties": {"a": {}}}
    assert _navigate_subschema(schema, []) == schema


# --- constraint formatting ---------------------------------------------


def test_format_constraint_pattern() -> None:
    sub = {"pattern": "^GO:[0-9]{7}$"}
    out = _format_constraint_hint(sub)
    assert out is not None
    assert "regex" in out
    assert "^GO:[0-9]{7}$" in out


def test_format_constraint_short_enum_listed_in_full() -> None:
    sub = {"enum": ["mf", "bp", "cc"]}
    out = _format_constraint_hint(sub)
    assert out is not None
    assert "['mf', 'bp', 'cc']" in out


def test_format_constraint_long_enum_truncated() -> None:
    sub = {"enum": [str(i) for i in range(20)]}
    out = _format_constraint_hint(sub)
    assert out is not None
    assert "first 6 shown" in out


def test_format_constraint_numeric_range() -> None:
    sub = {"type": "integer", "minimum": 0, "maximum": 1000}
    out = _format_constraint_hint(sub)
    assert out is not None
    assert "[0, 1000]" in out


def test_format_constraint_no_signal_returns_none() -> None:
    assert _format_constraint_hint({}) is None


# --- examples + description -------------------------------------------


def test_format_examples_returns_first_three() -> None:
    sub = {"examples": ["a", "b", "c", "d", "e"]}
    out = _format_examples(sub)
    assert out == "'a', 'b', 'c'"


def test_format_examples_missing_returns_none() -> None:
    assert _format_examples({}) is None


def test_format_description_trims_long_text() -> None:
    sub = {"description": "x" * 300}
    out = _format_description(sub)
    assert out is not None
    assert len(out) <= 200
    assert out.endswith("...")


def test_format_description_strips_whitespace() -> None:
    sub = {"description": "  hello  "}
    assert _format_description(sub) == "hello"


# --- enrich_error_message ---------------------------------------------


def test_enrich_message_includes_constraint_and_alternative_tools() -> None:
    out = enrich_error_message(
        tool_name="go_sequences",
        field_path="go_id",
        raw_message="'dehydratase_go_id' does not match '^GO:[0-9]{7}$'",
        subschema={
            "type": "string",
            "pattern": "^GO:[0-9]{7}$",
            "description": "Gene Ontology identifier in GO:NNNNNNN format.",
        },
    )
    assert "does not match '^GO:[0-9]{7}$'" in out
    assert "regex" in out
    assert "Gene Ontology identifier" in out
    # Per-tool guide for (go_sequences, go_id) ships in this PR.
    assert "interpro_sequences" in out


def test_enrich_message_falls_back_to_raw_when_no_subschema() -> None:
    out = enrich_error_message(
        tool_name="unknown_tool",
        field_path="weird",
        raw_message="some raw error",
        subschema=None,
    )
    assert out == "some raw error"


def test_enrich_message_unknown_tool_field_has_no_guide() -> None:
    out = enrich_error_message(
        tool_name="some_tool",
        field_path="some_field",
        raw_message="bad value",
        subschema={"type": "integer"},
    )
    assert "interpro_sequences" not in out
    assert "type `integer`" in out


# --- enrich_errors batch -----------------------------------------------


def test_enrich_errors_preserves_raw_message() -> None:
    raw = [{"field": "go_id", "message": "raw msg"}]
    schema = {"properties": {"go_id": {"pattern": "^GO:.*$"}}}
    out = enrich_errors(tool_name="go_sequences", schema=schema, raw_errors=raw)
    assert out[0]["raw_message"] == "raw msg"
    assert "raw msg" in out[0]["message"]
    # Enrichment for go_sequences.go_id should add alternative tools.
    assert "interpro_sequences" in out[0]["message"]


# --- end-to-end via validate_params on a real tool descriptor ---------


def _load_go_sequences_descriptor() -> ToolDescriptor:
    """Load the real go_sequences descriptor from its YAML on disk."""
    from vibeproteinagent.adapters.descriptor import load_descriptor

    adapter_root = Path(__file__).resolve().parents[1]
    descriptor_path = (
        adapter_root
        / "src"
        / "vibeproteinagent"
        / "adapters"
        / "go_sequences"
        / "descriptor.yaml"
    )
    return load_descriptor(descriptor_path)


def test_validate_params_raises_enriched_error_on_real_tool() -> None:
    """The matrix-cycle failure case: go_id placeholder fails regex."""
    descriptor = _load_go_sequences_descriptor()
    with pytest.raises(InvalidParamsError) as excinfo:
        validate_params(descriptor, {"go_id": "dehydratase_go_id"})

    err = excinfo.value
    msg = str(err)
    # Constraint info is present.
    assert "regex" in msg or "pattern" in msg
    assert "^GO:[0-9]{7}$" in msg
    # Per-tool alternative-tool guidance is present (the matrix fix).
    assert "interpro_sequences" in msg or "literature_search" in msg

    # raw_message is preserved in details for parsers that need the
    # original schema error text.
    details_errors = err.details["errors"]
    assert details_errors
    assert "raw_message" in details_errors[0]
    assert details_errors[0]["raw_message"] != details_errors[0]["message"]


def test_validate_params_happy_path_unchanged_on_real_tool() -> None:
    """When validation passes, no enrichment runs (no token cost)."""
    descriptor = _load_go_sequences_descriptor()
    # Valid GO ID format. Should not raise.
    validate_params(descriptor, {"go_id": "GO:0070204"})


# --- registry sanity --------------------------------------------------


def test_field_error_guides_registry_contains_go_sequences_go_id() -> None:
    """The first known LLM failure pattern (observed in matrix cycle)."""
    assert ("go_sequences", "go_id") in FIELD_ERROR_GUIDES
    guide = FIELD_ERROR_GUIDES[("go_sequences", "go_id")]
    assert "alternative_tools" in guide
    assert any("interpro_sequences" in t for t in guide["alternative_tools"])


def test_field_error_guides_keys_are_tuples_of_strings() -> None:
    """Defensive: catch accidental key-format drift."""
    for key, value in FIELD_ERROR_GUIDES.items():
        assert isinstance(key, tuple) and len(key) == 2
        tool, field = key
        assert isinstance(tool, str) and tool
        assert isinstance(field, str) and field
        assert isinstance(value, dict)


# --- _format_property_suggestion --------------------------------------
#
# Observed matrix-v3 failure: codex/claude/qwen all hallucinated field names
# (`reviewed_only`, `go_aspect`, `max_candidates`, `length`) and retried with
# either the same wrong name or a different wrong name. The plain JSON
# Schema rejection message just says "Additional properties not allowed (X
# was unexpected)" — it never tells the LLM what the real field names are.
# These tests pin the "valid fields + did-you-mean" enrichment that closes
# that gap.


def test_property_suggestion_lists_valid_fields_on_additional_property_error() -> None:
    schema = {
        "properties": {
            "go_id": {"type": "string"},
            "aspect": {"type": "string"},
            "reviewed": {"type": "boolean"},
            "max_sequences": {"type": "integer"},
        }
    }
    msg = "Additional properties are not allowed ('reviewed_only' was unexpected)"
    hint = _format_property_suggestion(msg, schema)
    assert hint is not None
    assert "Did you mean" in hint
    assert "reviewed_only" in hint
    assert "reviewed" in hint  # the correct field
    assert "Valid fields" in hint
    assert "go_id" in hint and "aspect" in hint


def test_property_suggestion_handles_multiple_rejected_names() -> None:
    schema = {
        "properties": {
            "go_id": {"type": "string"},
            "aspect": {"type": "string"},
            "reviewed": {"type": "boolean"},
        }
    }
    # JSON Schema phrasing varies between "were unexpected" and "was unexpected".
    msg = "Additional properties are not allowed ('go_aspect', 'reviewed_only' were unexpected)"
    hint = _format_property_suggestion(msg, schema)
    assert hint is not None
    # Did-you-mean for both rejected names
    assert "go_aspect" in hint and "aspect" in hint
    assert "reviewed_only" in hint and "reviewed" in hint


def test_property_suggestion_returns_none_for_unrelated_errors() -> None:
    schema = {"properties": {"go_id": {"type": "string"}}}
    assert _format_property_suggestion("'X' does not match pattern", schema) is None
    assert _format_property_suggestion("'go_id' is a required property", schema) is None


def test_property_suggestion_returns_none_when_schema_has_no_properties() -> None:
    msg = "Additional properties are not allowed ('foo' was unexpected)"
    assert _format_property_suggestion(msg, {}) is None
    assert _format_property_suggestion(msg, None) is None
    assert _format_property_suggestion(msg, {"properties": {}}) is None


def test_property_suggestion_truncates_long_field_lists() -> None:
    # Bindcraft has ~14 fields — full dump would balloon every retry msg.
    schema = {"properties": {f"field_{i:02d}": {"type": "string"} for i in range(20)}}
    msg = "Additional properties are not allowed ('xyz' was unexpected)"
    hint = _format_property_suggestion(msg, schema)
    assert hint is not None
    assert "8 more" in hint  # 20 - 12 capped = 8 hidden


def test_property_suggestion_no_close_match_still_lists_valid_fields() -> None:
    schema = {
        "properties": {
            "target_pdb": {"type": "string"},
            "candidates": {"type": "array"},
        }
    }
    msg = "Additional properties are not allowed ('max_candidates' was unexpected)"
    hint = _format_property_suggestion(msg, schema)
    assert hint is not None
    assert "Valid fields" in hint
    assert "target_pdb" in hint and "candidates" in hint
    # `max_candidates` shares substring "candidates" so candidate match should fire
    assert "maybe `candidates`" in hint


def test_root_schema_hint_surfaces_oneof_required_fields() -> None:
    schema = {
        "properties": {
            "target_pdb": {"type": "string"},
            "target_yaml": {"type": "string"},
            "candidates": {"type": "array"},
        },
        "oneOf": [
            {"required": ["target_pdb"]},
            {"required": ["target_yaml"]},
            {"required": ["candidates"]},
        ],
    }
    hint = _format_root_schema_hint(
        field_path="<root>",
        raw_message="{} is not valid under any of the given schemas",
        subschema=schema,
    )
    assert hint is not None
    assert "target_pdb" in hint
    assert "target_yaml" in hint
    assert "candidates" in hint
    assert "Valid root fields" in hint


def test_candidate_for_substring_match_first() -> None:
    assert _candidate_for("max_candidates", ["target_pdb", "candidates"]) == "candidates"


def test_candidate_for_prefix_match_fallback() -> None:
    # `seqlen` is not a substring of `sequence_length`, but they share the
    # `seq` prefix → prefix fallback fires.
    assert _candidate_for("seqlen", ["sequence_length", "max_sequences"]) == "sequence_length"
    # Pure prefix case with no substring overlap of the full rejected name.
    assert _candidate_for("seq", ["sequence_length", "other"]) == "sequence_length"


def test_candidate_for_no_match_returns_none() -> None:
    assert _candidate_for("xyz", ["target_pdb", "candidates"]) is None


def test_enrich_error_message_includes_property_suggestion_for_additional_property() -> None:
    """End-to-end: enrich_error_message wires up _format_property_suggestion."""
    schema = {
        "properties": {
            "go_id": {"type": "string"},
            "aspect": {"type": "string"},
            "reviewed": {"type": "boolean"},
        }
    }
    enriched = enrich_error_message(
        tool_name="go_sequences",
        field_path="<root>",
        raw_message="Additional properties are not allowed ('reviewed_only' was unexpected)",
        subschema=schema,
    )
    assert "Additional properties are not allowed" in enriched  # raw preserved
    assert "Did you mean" in enriched
    assert "reviewed_only" in enriched
    assert "reviewed" in enriched
    assert "Valid fields" in enriched
