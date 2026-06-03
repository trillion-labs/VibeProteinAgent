"""Coverage tests for the per-field description / examples enrichment.

The plan-stage validation enrichment (`_error_guides.py`) reads each input
field's `description`, `examples`, `pattern`, `enum`, etc. directly from
the tool's input.json schema and appends them to retry error messages.
For that to actually help the planner LLM recover, the schemas have to
*have* description and examples.

These tests enforce the contract:

1. Every top-level property in every tool's `<tool>.input.json` carries a
   non-empty `description`. (Fields nested under `$defs` are exempt.)
2. Fields with a `pattern` (regex) or with a 2+ value `enum` carry at
   least one `examples` entry.
3. Descriptions stay within reasonable length bounds (10-500 chars).
4. Each example value validates against the enclosing field's own
   subschema (catches "wrong type" or "doesn't match regex" example
   bugs).
5. Every input schema is a valid JSON Schema Draft 2020-12 document.
6. The `required` count per schema matches the documented pre-PR
   baseline (this PR is a metadata-only sweep — it must not change
   required-field semantics).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ADAPTERS = Path(__file__).resolve().parents[1] / "src" / "vibeproteinagent" / "adapters"


def _iter_input_schemas() -> list[tuple[str, Path, dict]]:
    out: list[tuple[str, Path, dict]] = []
    for tool_dir in sorted(ADAPTERS.iterdir()):
        if not tool_dir.is_dir() or tool_dir.name.startswith("_"):
            continue
        schemas_dir = tool_dir / "schemas"
        if not schemas_dir.exists():
            continue
        for schema_path in sorted(schemas_dir.glob("*.input.json")):
            schema = json.loads(schema_path.read_text())
            out.append((tool_dir.name, schema_path, schema))
    return out


# Pre-PR required-field counts. The mechanical schema enrichment PR
# MUST NOT change these numbers — every required-list change goes
# through a separate, semantics-aware review.
EXPECTED_REQUIRED_COUNTS: dict[str, int] = {
    "bindcraft": 0,
    "boltz2_affinity": 0,
    "boltzgen": 0,
    "esm2": 0,
    "esm3": 0,
    "family_profile": 0,
    "foldseek": 1,
    "function_term_resolver": 1,
    "go_sequences": 1,
    "go_term_search": 1,
    "interpro_entry_lookup": 1,
    "interpro_entry_search": 1,
    "interpro_sequences": 1,
    "ipsae": 3,
    "ligandmpnn": 1,
    "literature_search": 1,
    "merge_candidates": 1,
    "mmseqs2_gpu": 1,
    "mpnn_to_protenix_input": 1,
    "noop": 1,
    "ontology_lookup": 1,
    "prodigy": 2,
    "protein_qc": 0,
    "proteina_complexa": 1,
    "proteinmpnn": 0,
    "protenix_v2": 0,
    "pxdesign": 0,
    "pxmeter": 0,
    "pyrosetta": 1,
    "rank_designs": 0,
    "rfdiffusion": 0,
    "rfdiffusion3": 1,
    "rfpeptides": 1,
    "search_sequence": 0,
    "sequence_analysis": 0,
    "sequence_edit": 1,
    "sequence_to_protenix_input": 0,
    "structure_analysis": 0,
    "submit_candidate": 3,
    "template_select": 1,
    "tmalign": 0,
    "vina": 2,
}


_ALL_SCHEMAS = _iter_input_schemas()


@pytest.mark.parametrize("tool_name,_path,schema", _ALL_SCHEMAS, ids=[t[0] for t in _ALL_SCHEMAS])
def test_every_top_level_property_has_description(
    tool_name: str, _path: Path, schema: dict
) -> None:
    """Every property in `properties` carries a non-empty description.

    Subschemas under `$defs` (helper definitions, e.g. the canonical
    Candidate v1 record) are exempt — their fields are documented in
    the canonical schema, not duplicated here.
    """
    props = schema.get("properties", {})
    missing: list[str] = []
    for name, sub in props.items():
        if not isinstance(sub, dict):
            continue
        desc = sub.get("description")
        if not isinstance(desc, str) or not desc.strip():
            missing.append(name)
    assert not missing, f"{tool_name}: fields missing description: {missing}"


@pytest.mark.parametrize("tool_name,_path,schema", _ALL_SCHEMAS, ids=[t[0] for t in _ALL_SCHEMAS])
def test_pattern_or_multi_enum_fields_have_examples(
    tool_name: str, _path: Path, schema: dict
) -> None:
    """Fields with `pattern` or a 2+ value `enum` must carry examples.

    These are the high-value fields for the planner LLM: when validation
    fails on a regex or an enum, the examples are the single most
    useful piece of recovery information.
    """
    props = schema.get("properties", {})
    missing: list[str] = []
    for name, sub in props.items():
        if not isinstance(sub, dict):
            continue
        has_pattern = isinstance(sub.get("pattern"), str)
        enum = sub.get("enum")
        has_multi_enum = isinstance(enum, list) and len(enum) >= 2
        if not (has_pattern or has_multi_enum):
            continue
        examples = sub.get("examples")
        if not (isinstance(examples, list) and examples):
            missing.append(name)
    assert not missing, (
        f"{tool_name}: fields with pattern/enum but no examples: {missing}"
    )


@pytest.mark.parametrize("tool_name,_path,schema", _ALL_SCHEMAS, ids=[t[0] for t in _ALL_SCHEMAS])
def test_descriptions_within_reasonable_length(
    tool_name: str, _path: Path, schema: dict
) -> None:
    """10-500 characters per description (avoid empty or essay-length)."""
    props = schema.get("properties", {})
    bad: list[tuple[str, int]] = []
    for name, sub in props.items():
        if not isinstance(sub, dict):
            continue
        desc = sub.get("description")
        if not isinstance(desc, str):
            continue
        n = len(desc)
        if n < 10 or n > 500:
            bad.append((name, n))
    assert not bad, f"{tool_name}: descriptions out of [10,500] bounds: {bad}"


def _validate_example(parent_schema: dict, field_schema: dict, example) -> list[str]:
    """Validate `example` against `field_schema` and return error messages.

    The field subschema may carry `$ref` pointers to the parent's
    `$defs` block (e.g. rank_designs rows). We re-attach the parent's
    `$defs` so those refs resolve, and drop vendor-specific extension
    keys (e.g. `vpa:digest`) which are descriptive metadata, not schema
    vocabulary.
    """
    cleaned = {
        k: v
        for k, v in field_schema.items()
        if not k.startswith("vpa:") and k != "$comment"
    }
    defs = parent_schema.get("$defs")
    if isinstance(defs, dict):
        cleaned.setdefault("$defs", defs)
    validator = Draft202012Validator(cleaned)
    return [e.message for e in validator.iter_errors(example)]


@pytest.mark.parametrize("tool_name,_path,schema", _ALL_SCHEMAS, ids=[t[0] for t in _ALL_SCHEMAS])
def test_examples_validate_against_field_schema(
    tool_name: str, _path: Path, schema: dict
) -> None:
    """Every `examples` entry must validate against its field subschema."""
    props = schema.get("properties", {})
    failures: list[str] = []
    for name, sub in props.items():
        if not isinstance(sub, dict):
            continue
        examples = sub.get("examples")
        if not isinstance(examples, list):
            continue
        for i, example in enumerate(examples):
            errors = _validate_example(schema, sub, example)
            if errors:
                failures.append(
                    f"{name}[{i}]={example!r}: {errors[0]}"
                )
    assert not failures, f"{tool_name}: invalid examples: {failures}"


@pytest.mark.parametrize("tool_name,_path,schema", _ALL_SCHEMAS, ids=[t[0] for t in _ALL_SCHEMAS])
def test_full_schema_is_valid_draft_2020_12(
    tool_name: str, _path: Path, schema: dict
) -> None:
    """The enriched schema is still a structurally valid JSON Schema."""
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("tool_name,_path,schema", _ALL_SCHEMAS, ids=[t[0] for t in _ALL_SCHEMAS])
def test_required_count_matches_pre_pr_baseline(
    tool_name: str, _path: Path, schema: dict
) -> None:
    """Required-field count per tool must match the pre-PR baseline.

    This sweep is metadata-only: it must not silently expand or shrink
    the set of required fields. Required-list changes belong in a
    separate semantics-aware PR with explicit migration guidance.
    """
    actual = len(schema.get("required", []))
    expected = EXPECTED_REQUIRED_COUNTS.get(tool_name)
    assert expected is not None, (
        f"{tool_name}: not in EXPECTED_REQUIRED_COUNTS baseline; "
        "if you added a new adapter, update the baseline in this test."
    )
    assert actual == expected, (
        f"{tool_name}: required count drifted from {expected} → {actual}. "
        "Schema enrichment must not change required-field semantics."
    )


# --- sanity: the set of tools tested matches what's on disk --------


def test_all_input_schemas_covered() -> None:
    """All 38 expected tools have an input.json that the sweep saw."""
    on_disk = {name for name, _p, _s in _ALL_SCHEMAS}
    expected = set(EXPECTED_REQUIRED_COUNTS.keys())
    assert on_disk == expected, (
        f"Schema set drift: on-disk={on_disk - expected}, "
        f"missing-from-baseline={expected - on_disk}"
    )


def test_total_coverage_meets_targets() -> None:
    """Aggregate coverage meets the PR targets (≥95% desc, ≥80% examples-on-constrained)."""
    total_fields = 0
    with_description = 0
    constrained_fields = 0
    constrained_with_examples = 0
    for _tool, _path, schema in _ALL_SCHEMAS:
        props = schema.get("properties", {})
        for _name, sub in props.items():
            if not isinstance(sub, dict):
                continue
            total_fields += 1
            if isinstance(sub.get("description"), str) and sub["description"].strip():
                with_description += 1
            has_pattern = isinstance(sub.get("pattern"), str)
            enum = sub.get("enum")
            has_multi_enum = isinstance(enum, list) and len(enum) >= 2
            if has_pattern or has_multi_enum:
                constrained_fields += 1
                if isinstance(sub.get("examples"), list) and sub["examples"]:
                    constrained_with_examples += 1
    desc_pct = (100 * with_description) // total_fields if total_fields else 0
    ex_pct = (
        (100 * constrained_with_examples) // constrained_fields
        if constrained_fields
        else 100
    )
    assert desc_pct >= 95, f"description coverage {desc_pct}% < 95% target"
    assert ex_pct >= 80, f"examples-on-constrained coverage {ex_pct}% < 80% target"
