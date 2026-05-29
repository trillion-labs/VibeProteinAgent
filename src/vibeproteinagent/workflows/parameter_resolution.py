from __future__ import annotations

from copy import deepcopy
from typing import Any

from vibeproteinagent.adapters.descriptor import resolve_input_schema_dict
from vibeproteinagent.agents.planning.parameter_ownership import (
    RUNTIME_SOURCES,
    STRONG_USER_SOURCES,
    TOOL_PARAMETER_POLICIES,
    policy_for,
)
from vibeproteinagent.core.errors import InvalidParamsError

from .parameter_derivers import (
    candidate_input_json,
    candidate_input_yaml,
    candidate_sequence,
    candidate_structure_path,
    is_canonical_candidate,
    ordered_length_range,
    protein_chain_ids_from_structure,
)

CANDIDATE_STREAM_FIELDS = frozenset(
    {"candidates", "candidate_sets", "predictions", "analyses", "qc_results"}
)
CHAIN_SELECTOR_FIELDS = frozenset(
    {"chains", "target_chain", "binder_chain", "chain_id", "motif_chain"}
)


def normalize_tool_parameters(
    *,
    tool: str,
    inputs: dict[str, Any],
    provenance: dict[str, str] | None = None,
    original_inputs: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    resolved = deepcopy(inputs)
    sources = infer_input_provenance(
        resolved,
        provenance=provenance,
        original_inputs=original_inputs,
    )
    events: list[dict[str, Any]] = []
    candidates = _canonical_candidates(resolved.get("candidates"))
    downstream = _is_downstream_node(resolved, sources)

    if candidates:
        _derive_source_artifacts(tool, resolved, sources, events, candidates[0])
        _derive_chain_selectors(tool, resolved, sources, events)

    _derive_unambiguous_chain_selectors(tool, resolved, sources, events, downstream=downstream)
    _validate_structural_selectors(tool, resolved, sources, events, downstream=downstream)
    _normalize_length_constraints(tool, resolved, sources, events)
    _enforce_policy_permissions(tool, resolved, sources, events, downstream=downstream)
    return resolved, sources, events


def audit_adapter_defaults(
    *,
    descriptor: Any,
    resolved_inputs: dict[str, Any],
    sources: dict[str, str],
) -> list[dict[str, Any]]:
    """Best-effort audit for schema defaults without mutating adapter inputs."""

    try:
        schema = resolve_input_schema_dict(descriptor)
    except Exception:
        return []
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return []
    events: list[dict[str, Any]] = []
    for field, field_schema in sorted(properties.items()):
        if field in resolved_inputs:
            continue
        if not isinstance(field_schema, dict) or "default" not in field_schema:
            continue
        sources.setdefault(str(field), "adapter_default")
        events.append(
            _event(
                field=str(field),
                source_before="adapter_default",
                source_after="adapter_default",
                reason="adapter schema default will be used",
                value_summary=_summarize_value(field_schema.get("default")),
            )
        )
    return events


def infer_input_provenance(
    inputs: dict[str, Any],
    *,
    provenance: dict[str, str] | None = None,
    original_inputs: dict[str, Any] | None = None,
) -> dict[str, str]:
    out = {
        str(field): str(source)
        for field, source in (provenance or {}).items()
        if isinstance(field, str) and isinstance(source, str)
    }
    original_inputs = original_inputs if isinstance(original_inputs, dict) else {}
    for field, value in inputs.items():
        if field in out:
            continue
        original_value = original_inputs.get(field)
        if _contains_reference(original_value):
            out[field] = "compiled_wiring"
        elif _looks_runtime_wired(field, value):
            out[field] = "compiled_wiring"
        else:
            out[field] = "planner_literal"
    return out


def _is_downstream_node(inputs: dict[str, Any], sources: dict[str, str]) -> bool:
    if any(source in RUNTIME_SOURCES for source in sources.values()):
        return True
    return bool(_canonical_candidates(inputs.get("candidates")))


def _enforce_policy_permissions(
    tool: str,
    inputs: dict[str, Any],
    sources: dict[str, str],
    events: list[dict[str, Any]],
    *,
    downstream: bool,
) -> None:
    del events
    for field, value in list(inputs.items()):
        policy = policy_for(tool, field)
        source = sources.get(field, "planner_literal")
        if policy.ownership == "candidate_stream":
            _validate_candidate_stream_field(tool, field, value, source)
            continue
        if source != "planner_literal":
            continue
        planner_allowed = (
            policy.downstream_planner_allowed if downstream else policy.root_planner_allowed
        )
        runtime_owned_downstream = (
            downstream
            and policy.derive_when_upstream
            and policy.ownership != "structural_selector"
            and not policy.soft_preference
        )
        if not planner_allowed or runtime_owned_downstream:
            status = (
                "failed_planner_literal_runtime_owned"
                if runtime_owned_downstream
                else "failed_planner_literal_not_allowed"
            )
            raise _parameter_conflict(
                tool=tool,
                field=field,
                source=source,
                ownership=policy.ownership,
                status=status,
                value=value,
            )


def _validate_candidate_stream_field(
    tool: str,
    field: str,
    value: Any,
    source: str,
) -> None:
    if source in RUNTIME_SOURCES:
        return
    if source in STRONG_USER_SOURCES:
        return
    if source == "planner_literal":
        raise _parameter_conflict(
            tool=tool,
            field=field,
            source=source,
            ownership="candidate_stream",
            status="failed_planner_literal_not_allowed",
            value=value,
        )


def _parameter_conflict(
    *,
    tool: str,
    field: str,
    source: str,
    ownership: str,
    status: str,
    value: Any | None = None,
    details: dict[str, Any] | None = None,
) -> InvalidParamsError:
    event = {
        "field": field,
        "source": source,
        "ownership": ownership,
        "status": status,
        "technical_failure": False,
        "tool": tool,
    }
    if value is not None:
        event["value_summary"] = _summarize_value(value)
    if details:
        event.update(details)
    return InvalidParamsError(
        f"{tool}.{field} violates parameter ownership policy",
        details={"parameter_conflict_event": event},
    )


def _derive_source_artifacts(
    tool: str,
    inputs: dict[str, Any],
    sources: dict[str, str],
    events: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> None:
    for field in _source_artifact_fields(tool):
        policy = policy_for(tool, field)
        if not policy.derive_when_upstream:
            continue
        value = _derive_field_from_candidate(field, candidate)
        if value is None:
            continue
        _set_runtime_value(
            inputs,
            sources,
            events,
            field=field,
            value=value,
            source_after="upstream_candidate",
            reason=f"{tool}.{field} derived from upstream canonical candidate",
        )


def _derive_chain_selectors(
    tool: str,
    inputs: dict[str, Any],
    sources: dict[str, str],
    events: list[dict[str, Any]],
) -> None:
    if tool not in {"proteinmpnn", "ligandmpnn"}:
        return
    pdb_path = inputs.get("pdb_path")
    if not isinstance(pdb_path, str) or not pdb_path:
        return
    available = protein_chain_ids_from_structure(pdb_path)
    if not available:
        return
    source = sources.get("chains", "adapter_default")
    current = inputs.get("chains")
    if source in STRONG_USER_SOURCES and current:
        requested = _string_list(current)
        missing = sorted(set(requested) - set(available))
        if missing:
            raise InvalidParamsError(
                f"{tool}.chains contains chains not present in selected structure",
                details={
                    "parameter_conflict_event": {
                        "field": "chains",
                        "user_value": requested,
                        "derived_available_values": available,
                        "missing": missing,
                        "status": "failed_user_value_incompatible",
                        "technical_failure": False,
                    }
                },
            )
        events.append(
            _event(
                field="chains",
                source_before=source,
                source_after=source,
                reason=f"{tool}.chains user/known value validated against structure",
                value_summary=requested,
            )
        )
        return
    _set_runtime_value(
        inputs,
        sources,
        events,
        field="chains",
        value=available,
        source_after="runtime_derived",
        reason=f"{tool}.chains inferred from protein ATOM chains in selected structure",
    )


def _derive_unambiguous_chain_selectors(
    tool: str,
    inputs: dict[str, Any],
    sources: dict[str, str],
    events: list[dict[str, Any]],
    *,
    downstream: bool,
) -> None:
    if not downstream:
        return
    structure_path = _selected_structure_path(tool, inputs)
    if not structure_path:
        return
    available = protein_chain_ids_from_structure(structure_path)
    if len(available) != 1:
        return
    chain = available[0]
    policy_fields = {
        field
        for field, policy in TOOL_PARAMETER_POLICIES.get(tool, {}).items()
        if field in CHAIN_SELECTOR_FIELDS and policy.derive_when_upstream
    }
    for field in CHAIN_SELECTOR_FIELDS | policy_fields:
        if field == "chains" and tool in {"proteinmpnn", "ligandmpnn"}:
            continue
        if field not in inputs and field not in policy_fields:
            continue
        policy = policy_for(tool, field)
        if not policy.derive_when_upstream:
            continue
        source = sources.get(field, "adapter_default")
        if source in STRONG_USER_SOURCES:
            continue
        value = [chain] if field == "chains" else chain
        _set_runtime_value(
            inputs,
            sources,
            events,
            field=field,
            value=value,
            source_after="runtime_derived",
            reason=f"{tool}.{field} inferred from the only protein chain in selected structure",
        )


def _validate_structural_selectors(
    tool: str,
    inputs: dict[str, Any],
    sources: dict[str, str],
    events: list[dict[str, Any]],
    *,
    downstream: bool,
) -> None:
    if not downstream:
        return
    structure_path = _selected_structure_path(tool, inputs)
    available = protein_chain_ids_from_structure(structure_path) if structure_path else []
    for field, value in list(inputs.items()):
        policy = policy_for(tool, field)
        if policy.ownership != "structural_selector":
            continue
        if field in CHAIN_SELECTOR_FIELDS:
            if field == "chains" and tool in {"proteinmpnn", "ligandmpnn"}:
                continue
            if not available:
                continue
            requested = _string_list(value)
            if not requested:
                continue
            source = sources.get(field, "planner_literal")
            missing = sorted(set(requested) - set(available))
            if missing and source in STRONG_USER_SOURCES:
                raise _parameter_conflict(
                    tool=tool,
                    field=field,
                    source=source,
                    ownership=policy.ownership,
                    status="failed_user_value_incompatible",
                    value=requested,
                    details={
                        "user_value": requested,
                        "derived_available_values": available,
                        "missing": missing,
                    },
                )
            if not missing and source in STRONG_USER_SOURCES:
                events.append(
                    _event(
                        field=field,
                        source_before=source,
                        source_after=source,
                        reason=f"{tool}.{field} user/known value validated against structure",
                        value_summary=requested,
                    )
                )
            continue
        if sources.get(field) == "planner_literal" and _is_residue_like_selector(field):
            events.append(
                _event(
                    field=field,
                    source_before="planner_literal",
                    source_after="planner_literal",
                    reason="structural residue selector deferred to adapter validation",
                    value_summary=_summarize_value(value),
                )
            )


def _normalize_length_constraints(
    tool: str,
    inputs: dict[str, Any],
    sources: dict[str, str],
    events: list[dict[str, Any]],
) -> None:
    if tool != "rfdiffusion":
        return
    if "total_length" in inputs and inputs.get("total_length") is not None:
        total = _int_or_none(inputs.get("total_length"))
        if total is not None:
            _set_runtime_value(
                inputs,
                sources,
                events,
                field="length_min",
                value=total,
                source_after=sources.get("total_length", "runtime_derived"),
                reason="rfdiffusion total_length fixes length_min",
                allow_strong_user_override=True,
            )
            _set_runtime_value(
                inputs,
                sources,
                events,
                field="length_max",
                value=total,
                source_after=sources.get("total_length", "runtime_derived"),
                reason="rfdiffusion total_length fixes length_max",
                allow_strong_user_override=True,
            )
        return
    length_range = ordered_length_range(
        _int_or_none(inputs.get("length_min")),
        _int_or_none(inputs.get("length_max")),
    )
    if length_range is not None:
        low, high = length_range
        if inputs.get("length_min") != low:
            _set_runtime_value(
                inputs,
                sources,
                events,
                field="length_min",
                value=low,
                source_after=sources.get("length_min", "runtime_derived"),
                reason="rfdiffusion length range normalized in ascending order",
                allow_strong_user_override=True,
            )
        if inputs.get("length_max") != high:
            _set_runtime_value(
                inputs,
                sources,
                events,
                field="length_max",
                value=high,
                source_after=sources.get("length_max", "runtime_derived"),
                reason="rfdiffusion length range normalized in ascending order",
                allow_strong_user_override=True,
            )

    binder_range = ordered_length_range(
        _int_or_none(inputs.get("binder_length_min")),
        _int_or_none(inputs.get("binder_length_max")),
    )
    if binder_range is None:
        return
    low, high = binder_range
    if inputs.get("binder_length_min") != low:
        _set_runtime_value(
            inputs,
            sources,
            events,
            field="binder_length_min",
            value=low,
            source_after=sources.get("binder_length_min", "runtime_derived"),
            reason="rfdiffusion binder length range normalized in ascending order",
            allow_strong_user_override=True,
        )
    if inputs.get("binder_length_max") != high:
        _set_runtime_value(
            inputs,
            sources,
            events,
            field="binder_length_max",
            value=high,
            source_after=sources.get("binder_length_max", "runtime_derived"),
            reason="rfdiffusion binder length range normalized in ascending order",
            allow_strong_user_override=True,
        )


def _selected_structure_path(tool: str, inputs: dict[str, Any]) -> str | None:
    del tool
    for field in ("pdb_path", "target_pdb", "structure_path", "motif_pdb"):
        value = inputs.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _is_residue_like_selector(field: str) -> bool:
    lowered = field.lower()
    return any(
        token in lowered
        for token in ("residue", "hotspot", "pocket", "position")
    )


def _set_runtime_value(
    inputs: dict[str, Any],
    sources: dict[str, str],
    events: list[dict[str, Any]],
    *,
    field: str,
    value: Any,
    source_after: str,
    reason: str,
    allow_strong_user_override: bool = False,
) -> None:
    before = inputs.get(field)
    source_before = sources.get(field, "adapter_default")
    if (
        field in inputs
        and source_before in STRONG_USER_SOURCES
        and not allow_strong_user_override
    ):
        return
    if field in inputs and before == value and sources.get(field) == source_after:
        return
    inputs[field] = value
    sources[field] = source_after
    events.append(
        _event(
            field=field,
            source_before=source_before,
            source_after=source_after,
            reason=reason,
            value_summary=_summarize_value(value),
            replaced_value_summary=_summarize_value(before) if before is not None else None,
        )
    )


def _source_artifact_fields(tool: str) -> tuple[str, ...]:
    return {
        "proteinmpnn": ("pdb_path",),
        "ligandmpnn": ("pdb_path", "sequence"),
        "rfdiffusion": ("target_pdb", "target_sequence"),
        "rfdiffusion3": ("target_pdb", "target_sequence"),
        "rfpeptides": ("target_pdb", "target_sequence"),
        "bindcraft": ("target_pdb", "target_sequence"),
        "pxdesign": ("target_pdb",),
        "proteina_complexa": ("target_pdb", "target_sequence", "motif_pdb", "motif_sequence"),
        "boltzgen": ("target_pdb",),
        "protenix_v2": ("input_json",),
        "boltz2_affinity": ("input_yaml", "sequences"),
        "structure_analysis": ("structure_path", "sequence"),
        "sequence_analysis": ("sequence",),
        "protein_qc": ("sequence", "pdb_path"),
    }.get(tool, ())


def _derive_field_from_candidate(field: str, candidate: dict[str, Any]) -> Any:
    if field in {"pdb_path", "target_pdb", "structure_path", "motif_pdb"}:
        return candidate_structure_path(candidate, prefer_pdb=True)
    if field in {"sequence", "target_sequence", "motif_sequence"}:
        return candidate_sequence(candidate)
    if field == "input_json":
        return candidate_input_json(candidate)
    if field == "input_yaml":
        return candidate_input_yaml(candidate)
    if field == "sequences":
        sequence = candidate_sequence(candidate)
        if not sequence:
            return None
        return [{"id": candidate.get("candidate_id") or "candidate", "sequence": sequence}]
    return None


def _canonical_candidates(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if is_canonical_candidate(item)]


def _contains_reference(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith("${") and value.endswith("}")
    if isinstance(value, dict):
        return any(_contains_reference(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_reference(child) for child in value)
    return False


def _looks_runtime_wired(field: str, value: Any) -> bool:
    if field in {"candidates", "candidate_sets", "predictions", "analyses", "qc_results"}:
        return isinstance(value, list)
    return False


def _event(
    *,
    field: str,
    source_before: str,
    source_after: str,
    reason: str,
    value_summary: Any,
    replaced_value_summary: Any | None = None,
) -> dict[str, Any]:
    event = {
        "field": field,
        "source_before": source_before,
        "source_after": source_after,
        "reason": reason,
        "value_summary": value_summary,
    }
    if replaced_value_summary is not None:
        event["replaced_value_summary"] = replaced_value_summary
    return event


def _summarize_value(value: Any) -> Any:
    if isinstance(value, list):
        if len(value) <= 8:
            return value
        return [*value[:8], f"...({len(value)} total)"]
    if isinstance(value, dict):
        return {"keys": sorted(str(key) for key in value)[:12]}
    return value


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


__all__ = [
    "audit_adapter_defaults",
    "infer_input_provenance",
    "normalize_tool_parameters",
]
