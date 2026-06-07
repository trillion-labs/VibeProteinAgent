from __future__ import annotations

from vibeproteinagent.workflows.parameter_policies import (
    fallback_policy,
    planner_ownership_hints,
    policy_for,
)


def test_high_risk_fields_have_explicit_ownership() -> None:
    assert policy_for("proteinmpnn", "pdb_path").ownership == "source_artifact"
    assert policy_for("proteinmpnn", "chains").ownership == "structural_selector"
    assert policy_for("rank_designs", "candidates").ownership == "candidate_stream"
    assert policy_for("rfdiffusion", "length_min").ownership == "length_constraint"


def test_fallback_policy_classifies_common_field_names() -> None:
    assert fallback_policy("candidates").ownership == "candidate_stream"
    assert fallback_policy("target_pdb").ownership == "source_artifact"
    assert fallback_policy("num_designs").ownership == "budget"
    assert fallback_policy("target_chain").ownership == "structural_selector"


def test_planner_hints_distinguish_runtime_and_planner_fields() -> None:
    hints = planner_ownership_hints(
        "proteinmpnn",
        ["pdb_path", "chains", "num_designs", "candidates"],
    )

    assert hints["pdb_path"] == "runtime_derived_when_downstream_or_known_input"
    assert hints["chains"] == "runtime_derived_when_downstream_or_known_input"
    assert hints["num_designs"] == "planner_allowed"
    assert hints["candidates"] == "compiler_controller_wired"
