from __future__ import annotations

from typing import ClassVar

import pytest

from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workflows.parameter_resolution import (
    audit_adapter_defaults,
    normalize_tool_parameters,
)


def _candidate(structure_path: str, sequence: str = "ACDE") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand",
        "source": {"tool": "pxdesign"},
        "structure_path": structure_path,
        "sequence": sequence,
        "scores": {"pxdesign": {"score": 1.0}},
        "artifacts": {"pxdesign": {"pdb_path": structure_path}},
    }


def _protenix_candidate(cif_path: str, pdb_path: str | None, sequence: str = "ACDE") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_protenix",
        "source": {"tool": "protenix_v2"},
        "structure_path": cif_path,
        "sequence": sequence,
        "scores": {"protenix_v2": {"plddt": 91.0}},
        "artifacts": {
            "protenix_v2": {
                "predicted_cif": cif_path,
                "predicted_pdb": pdb_path,
            }
        },
    }


def _pdb(tmp_path) -> str:
    path = tmp_path / "complex.pdb"
    path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00 10.00           C",
                "ATOM      2  CA  GLY B   1      1.000   0.000   0.000  1.00 10.00           C",
                "HETATM    3  C1  LIG C   1      2.000   0.000   0.000  1.00 10.00           C",
            ]
        )
        + "\n"
    )
    return str(path)


def _single_chain_pdb(tmp_path) -> str:
    path = tmp_path / "single_chain_b.pdb"
    path.write_text(
        "ATOM      1  CA  ALA B   1      0.000   0.000   0.000  1.00 10.00           C\n"
    )
    return str(path)


def test_downstream_proteinmpnn_derives_path_and_chains(tmp_path) -> None:
    pdb_path = _pdb(tmp_path)
    inputs, provenance, events = normalize_tool_parameters(
        tool="proteinmpnn",
        inputs={
            "candidates": [_candidate(pdb_path)],
            "chains": ["Z"],
            "num_designs": 2,
        },
        provenance={
            "candidates": "compiled_wiring",
            "chains": "planner_literal",
            "num_designs": "planner_literal",
        },
        original_inputs={"candidates": "${outputs.design.candidates}", "chains": ["Z"]},
    )

    assert inputs["pdb_path"] == pdb_path
    assert inputs["chains"] == ["A", "B"]
    assert provenance["pdb_path"] == "upstream_candidate"
    assert provenance["chains"] == "runtime_derived"
    assert {event["field"] for event in events} >= {"pdb_path", "chains"}


def test_pdb_only_downstream_prefers_protenix_predicted_pdb() -> None:
    candidate = _protenix_candidate("/tmp/predicted.cif", "/tmp/predicted.pdb")

    inputs, provenance, _events = normalize_tool_parameters(
        tool="proteinmpnn",
        inputs={"candidates": [candidate], "num_designs": 2},
        provenance={"candidates": "compiled_wiring", "num_designs": "planner_literal"},
        original_inputs={"candidates": "${outputs.protenix.candidates}"},
    )

    assert inputs["pdb_path"] == "/tmp/predicted.pdb"
    assert provenance["pdb_path"] == "upstream_candidate"


def test_true_pdb_only_downstream_rejects_cif_without_predicted_pdb() -> None:
    candidate = _protenix_candidate("/tmp/predicted.cif", None)

    with pytest.raises(InvalidParamsError) as exc_info:
        normalize_tool_parameters(
            tool="ligandmpnn",
            inputs={"candidates": [candidate], "num_designs": 2},
            provenance={"candidates": "compiled_wiring", "num_designs": "planner_literal"},
            original_inputs={"candidates": "${outputs.protenix.candidates}"},
        )

    event = exc_info.value.details["parameter_conflict_event"]
    assert event["field"] == "pdb_path"
    assert event["status"] == "failed_missing_pdb_artifact"
    assert event["structure_path"] == "/tmp/predicted.cif"


def test_proteinmpnn_keeps_cif_fallback_when_no_predicted_pdb() -> None:
    candidate = _protenix_candidate("/tmp/predicted.cif", None)

    inputs, provenance, _events = normalize_tool_parameters(
        tool="proteinmpnn",
        inputs={"candidates": [candidate], "num_designs": 2},
        provenance={"candidates": "compiled_wiring", "num_designs": "planner_literal"},
        original_inputs={"candidates": "${outputs.protenix.candidates}"},
    )

    assert inputs["pdb_path"] == "/tmp/predicted.cif"
    assert provenance["pdb_path"] == "upstream_candidate"


def test_pyrosetta_derives_predicted_pdb_from_candidate_stream() -> None:
    candidate = _protenix_candidate("/tmp/predicted.cif", "/tmp/predicted.pdb")

    inputs, provenance, _events = normalize_tool_parameters(
        tool="pyrosetta",
        inputs={"candidates": [candidate], "mode": "score"},
        provenance={"candidates": "compiled_wiring", "mode": "planner_literal"},
        original_inputs={"candidates": "${outputs.protenix.candidates}"},
    )

    assert inputs["pdb_path"] == "/tmp/predicted.pdb"
    assert provenance["pdb_path"] == "upstream_candidate"


def test_cif_capable_downstream_keeps_protenix_cif() -> None:
    candidate = _protenix_candidate("/tmp/predicted.cif", "/tmp/predicted.pdb")

    inputs, provenance, _events = normalize_tool_parameters(
        tool="protein_qc",
        inputs={"candidates": [candidate], "metrics": ["plddt"]},
        provenance={"candidates": "compiled_wiring", "metrics": "planner_literal"},
        original_inputs={"candidates": "${outputs.protenix.candidates}"},
    )

    assert inputs["pdb_path"] == "/tmp/predicted.cif"
    assert provenance["pdb_path"] == "upstream_candidate"


def test_user_explicit_chains_are_preserved_when_compatible(tmp_path) -> None:
    pdb_path = _pdb(tmp_path)
    inputs, provenance, events = normalize_tool_parameters(
        tool="proteinmpnn",
        inputs={"pdb_path": pdb_path, "candidates": [_candidate(pdb_path)], "chains": ["B"]},
        provenance={
            "pdb_path": "known_input",
            "candidates": "compiled_wiring",
            "chains": "known_input",
        },
        original_inputs={"candidates": "${outputs.design.candidates}"},
    )

    assert inputs["pdb_path"] == pdb_path
    assert inputs["chains"] == ["B"]
    assert provenance["chains"] == "known_input"
    assert any(event["reason"].endswith("validated against structure") for event in events)


def test_user_explicit_chains_fail_when_incompatible(tmp_path) -> None:
    pdb_path = _pdb(tmp_path)

    with pytest.raises(InvalidParamsError) as excinfo:
        normalize_tool_parameters(
            tool="proteinmpnn",
            inputs={"pdb_path": pdb_path, "candidates": [_candidate(pdb_path)], "chains": ["Z"]},
            provenance={
                "pdb_path": "known_input",
                "candidates": "compiled_wiring",
                "chains": "known_input",
            },
            original_inputs={"candidates": "${outputs.design.candidates}"},
        )

    event = excinfo.value.details["parameter_conflict_event"]
    assert event["field"] == "chains"
    assert event["missing"] == ["Z"]
    assert event["derived_available_values"] == ["A", "B"]


def test_root_planner_literal_is_preserved_without_upstream() -> None:
    inputs, provenance, events = normalize_tool_parameters(
        tool="proteinmpnn",
        inputs={"pdb_path": "/tmp/root.pdb", "chains": ["A"], "num_designs": 4},
        provenance={
            "pdb_path": "planner_literal",
            "chains": "planner_literal",
            "num_designs": "planner_literal",
        },
        original_inputs={},
    )

    assert inputs["pdb_path"] == "/tmp/root.pdb"
    assert inputs["chains"] == ["A"]
    assert provenance["chains"] == "planner_literal"
    assert events == []


def test_rfdiffusion_length_ranges_are_normalized() -> None:
    inputs, provenance, events = normalize_tool_parameters(
        tool="rfdiffusion",
        inputs={"length_min": 90, "length_max": 60},
        provenance={"length_min": "planner_literal", "length_max": "planner_literal"},
        original_inputs={},
    )

    assert inputs["length_min"] == 60
    assert inputs["length_max"] == 90
    assert provenance["length_min"] == "planner_literal"
    assert {event["field"] for event in events} == {"length_min", "length_max"}


def test_planner_literal_candidate_stream_is_rejected() -> None:
    with pytest.raises(InvalidParamsError) as excinfo:
        normalize_tool_parameters(
            tool="rank_designs",
            inputs={"candidates": [{"candidate_id": "fake"}], "top_k": 1},
            provenance={"candidates": "planner_literal", "top_k": "planner_literal"},
            original_inputs={"candidates": [{"candidate_id": "fake"}], "top_k": 1},
        )

    event = excinfo.value.details["parameter_conflict_event"]
    assert event["field"] == "candidates"
    assert event["ownership"] == "candidate_stream"
    assert event["status"] == "failed_planner_literal_not_allowed"


def test_runtime_candidate_stream_is_allowed_when_wired() -> None:
    inputs, provenance, events = normalize_tool_parameters(
        tool="rank_designs",
        inputs={"candidates": [], "top_k": 1},
        provenance={"candidates": "compiled_wiring", "top_k": "planner_literal"},
        original_inputs={"candidates": "${outputs.external_candidates.candidates}", "top_k": 1},
    )

    assert inputs["candidates"] == []
    assert provenance["candidates"] == "compiled_wiring"
    assert events == []


def test_downstream_planner_literal_source_artifact_without_derivation_fails() -> None:
    with pytest.raises(InvalidParamsError) as excinfo:
        normalize_tool_parameters(
            tool="protein_qc",
            inputs={
                "candidates": [],
                "pdb_path": "/tmp/fake_from_planner.pdb",
            },
            provenance={
                "candidates": "compiled_wiring",
                "pdb_path": "planner_literal",
            },
            original_inputs={
                "candidates": "${outputs.design.candidates}",
                "pdb_path": "/tmp/fake_from_planner.pdb",
            },
        )

    event = excinfo.value.details["parameter_conflict_event"]
    assert event["field"] == "pdb_path"
    assert event["status"] == "failed_planner_literal_runtime_owned"


def test_known_target_chain_fails_when_incompatible(tmp_path) -> None:
    pdb_path = _pdb(tmp_path)

    with pytest.raises(InvalidParamsError) as excinfo:
        normalize_tool_parameters(
            tool="bindcraft",
            inputs={"target_pdb": pdb_path, "target_chain": "Z", "candidates": []},
            provenance={
                "target_pdb": "known_input",
                "target_chain": "known_input",
                "candidates": "compiled_wiring",
            },
            original_inputs={"candidates": "${outputs.design.candidates}"},
        )

    event = excinfo.value.details["parameter_conflict_event"]
    assert event["field"] == "target_chain"
    assert event["missing"] == ["Z"]
    assert event["derived_available_values"] == ["A", "B"]


def test_downstream_single_chain_target_selector_is_runtime_derived(tmp_path) -> None:
    pdb_path = _single_chain_pdb(tmp_path)
    inputs, provenance, events = normalize_tool_parameters(
        tool="bindcraft",
        inputs={"target_pdb": pdb_path, "candidates": []},
        provenance={"target_pdb": "known_input", "candidates": "compiled_wiring"},
        original_inputs={"candidates": "${outputs.design.candidates}"},
    )

    assert inputs["target_chain"] == "B"
    assert provenance["target_chain"] == "runtime_derived"
    assert any(event["field"] == "target_chain" for event in events)


def test_residue_selector_is_deferred_to_adapter_validation(tmp_path) -> None:
    pdb_path = _pdb(tmp_path)
    _, _, events = normalize_tool_parameters(
        tool="bindcraft",
        inputs={
            "target_pdb": pdb_path,
            "target_hotspots": ["A10"],
            "candidates": [],
        },
        provenance={
            "target_pdb": "known_input",
            "target_hotspots": "planner_literal",
            "candidates": "compiled_wiring",
        },
        original_inputs={"candidates": "${outputs.design.candidates}"},
    )

    assert any(
        event["field"] == "target_hotspots"
        and event["reason"] == "structural residue selector deferred to adapter validation"
        for event in events
    )


def test_adapter_defaults_are_audited_without_mutating_inputs() -> None:
    class Descriptor:
        input_schema: ClassVar[dict] = {
            "type": "object",
            "properties": {
                "temperature": {"type": "number", "default": 0.1},
                "num_designs": {"type": "integer", "default": 4},
            },
        }

    resolved_inputs = {"num_designs": 2}
    events = audit_adapter_defaults(
        descriptor=Descriptor(),
        resolved_inputs=resolved_inputs,
        sources={"num_designs": "planner_literal"},
    )

    assert resolved_inputs == {"num_designs": 2}
    assert events == [
        {
            "field": "temperature",
            "source_before": "adapter_default",
            "source_after": "adapter_default",
            "reason": "adapter schema default will be used",
            "value_summary": 0.1,
        }
    ]
