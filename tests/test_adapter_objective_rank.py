from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator, ValidationError as JSONSchemaValidationError

from vibeproteinagent.adapters import ToolRegistry, null_context
from vibeproteinagent.adapters.objective_rank.adapter import (
    ObjectiveRankAdapter,
    ObjectiveRankInput,
)
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _adapter(tmp_path: Path) -> ObjectiveRankAdapter:
    return ObjectiveRankAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))


def _completed_job(workspace: Workspace, tool: str, output: dict) -> str:
    store = JobStore(workspace)
    rec = store.submit(tool, "1.0.0", {"test": tool})
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED, output=output)
    return rec.job_id


def test_objective_rank_is_registered_as_public_tool() -> None:
    registry = ToolRegistry()
    registry.discover(refresh=True)

    descriptor, adapter_cls = registry.get("objective_rank")

    assert descriptor.name == "objective_rank"
    assert adapter_cls is ObjectiveRankAdapter
    assert any(descriptor.name == item.name for item in registry.list())


def test_objective_rank_output_schema_requires_ranked_row_identity() -> None:
    schema = json.loads(
        Path(
            "src/vibeproteinagent/adapters/objective_rank/schemas/"
            "objective_rank.output.json"
        ).read_text()
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    payload = {
        "ranking_intent": "objective",
        "objective_name": "fold_stability",
        "formula_text": "-1 * ${pyrosetta.mean_ddg}",
        "score_direction": "higher",
        "gate_summary": ["pyrosetta.mean_ddg < 0"],
        "candidate_count": 1,
        "ranked_json": "/tmp/ranked.json",
        "ranked_tsv": "/tmp/ranked.tsv",
        "runtime_ms": 1,
        "ranked_candidates": [
            {
                "rank": 1,
                "candidate_id": "dup",
                "objective_score": 2.0,
                "objective_name": "fold_stability",
                "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                "formula_text": "-1 * ${pyrosetta.mean_ddg}",
                "score_direction": "higher",
                "score_components": [],
                "passed_gates": [],
                "failed_gates": [],
                "tie_breaker_values": {},
                "candidate_ref": {"candidate_id": "dup"},
            }
        ],
    }

    with pytest.raises(JSONSchemaValidationError):
        validator.validate(payload)

    row = payload["ranked_candidates"][0]
    row["ranked_candidate_id"] = "dup__v1_a1b2c3d4"
    row["candidate_ref"]["ranked_candidate_id"] = "dup__v1_a1b2c3d4"
    validator.validate(payload)


def test_objective_rank_ranks_by_explicit_stability_formula(tmp_path: Path) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="fold_stability",
            candidates=[
                {
                    "candidate_id": "weak",
                    "scores": {
                        "pyrosetta": {"mean_ddg": -2.0},
                        "protenix_v2": {"plddt": 91.0},
                        "tmalign": {"tm_score": 0.98},
                    },
                },
                {
                    "candidate_id": "strong",
                    "scores": {
                        "pyrosetta": {"mean_ddg": -8.5},
                        "protenix_v2": {"plddt": 89.0},
                        "tmalign": {"tm_score": 0.97},
                    },
                },
            ],
            gates=[
                {"id": "stabilizing", "metric": "pyrosetta.mean_ddg", "op": "<", "value": 0},
                {"id": "confident", "metric": "protenix_v2.plddt", "op": ">=", "value": 80},
                {"id": "same_fold", "metric": "tmalign.tm_score", "op": ">=", "value": 0.95},
            ],
            score_formula="-1 * ${pyrosetta.mean_ddg}",
        ),
        null_context(tmp_path),
    )

    assert out.ranking_intent == "objective"
    assert out.ranked_candidates[0]["candidate_id"] == "strong"
    assert out.ranked_candidates[0]["objective_score"] == pytest.approx(8.5)
    assert out.ranked_candidates[0]["failed_gates"] == []
    assert Path(out.ranked_json).exists()
    assert Path(out.ranked_tsv).exists()


def test_objective_rank_applies_gates_before_score_order(tmp_path: Path) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="fold_stability",
            candidates=[
                {"candidate_id": "high_score_fail", "scores": {"pyrosetta": {"mean_ddg": -20}, "protenix_v2": {"plddt": 70}}},
                {"candidate_id": "lower_score_pass", "scores": {"pyrosetta": {"mean_ddg": -3}, "protenix_v2": {"plddt": 85}}},
            ],
            gates=[
                {"id": "confident", "metric": "protenix_v2.plddt", "op": ">=", "value": 80}
            ],
            score_formula="-1 * ${pyrosetta.mean_ddg}",
        ),
        null_context(tmp_path),
    )

    assert [row["candidate_id"] for row in out.ranked_candidates] == [
        "lower_score_pass",
        "high_score_fail",
    ]
    assert out.ranked_candidates[1]["failed_gates"][0]["id"] == "confident"


def test_objective_rank_assigns_unique_ranked_ids_for_duplicate_logical_candidates(
    tmp_path: Path,
) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="duplicate_identity",
            candidates=[
                {
                    "candidate_id": "dup",
                    "sequence": "AAAA",
                    "structure_path": "/tmp/a.pdb",
                    "scores": {"pyrosetta": {"mean_ddg": -1.0}},
                },
                {
                    "candidate_id": "dup",
                    "sequence": "CCCC",
                    "structure_path": "/tmp/b.pdb",
                    "scores": {"pyrosetta": {"mean_ddg": -2.0}},
                },
            ],
            score_formula="-1 * ${pyrosetta.mean_ddg}",
        ),
        null_context(tmp_path),
    )

    assert [row["candidate_id"] for row in out.ranked_candidates] == ["dup", "dup"]
    ranked_ids = [row["ranked_candidate_id"] for row in out.ranked_candidates]
    assert len(set(ranked_ids)) == 2
    assert all(ranked_id.startswith("dup__v") for ranked_id in ranked_ids)
    for row in out.ranked_candidates:
        assert row["candidate_ref"]["candidate_id"] == "dup"
        assert row["candidate_ref"]["ranked_candidate_id"] == row["ranked_candidate_id"]
        assert row["identity_key"]
        assert row["material_key"]


@pytest.mark.parametrize(
    ("policy", "expected_score", "warning"),
    [
        ("fail", None, "score metric missing"),
        ("warn", 0.0, "using 0"),
        ("zero", 0.0, ""),
        ("ignore", 0.0, ""),
    ],
)
def test_objective_rank_missing_formula_metric_policies(
    tmp_path: Path,
    policy: str,
    expected_score: float | None,
    warning: str,
) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="missing_policy",
            candidates=[{"candidate_id": "cand", "scores": {"protenix_v2": {"plddt": 90}}}],
            score_formula="${pyrosetta.mean_ddg}",
            missing=policy,
        ),
        null_context(tmp_path),
    )

    assert out.ranked_candidates[0]["objective_score"] == expected_score
    joined = " ".join(out.ranked_candidates[0]["warnings"])
    assert warning in joined


def test_objective_rank_missing_gate_warn_and_ignore_do_not_fail(tmp_path: Path) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="gate_policy",
            candidates=[{"candidate_id": "cand", "score": 1.0}],
            gates=[
                {"id": "warn_missing", "metric": "missing.warn", "op": ">=", "value": 1, "missing": "warn"},
                {"id": "ignore_missing", "metric": "missing.ignore", "op": ">=", "value": 1, "missing": "ignore"},
            ],
            score_formula="${score}",
        ),
        null_context(tmp_path),
    )

    assert out.ranked_candidates[0]["failed_gates"] == []
    assert out.ranked_candidates[0]["passed_gates"][0]["id"] == "warn_missing"


def test_objective_rank_missing_gate_zero_evaluates_with_zero(tmp_path: Path) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="gate_policy",
            candidates=[{"candidate_id": "cand", "score": 1.0}],
            gates=[
                {
                    "id": "missing_as_zero",
                    "metric": "missing.value",
                    "op": "==",
                    "value": 0,
                    "missing": "zero",
                }
            ],
            score_formula="${score}",
        ),
        null_context(tmp_path),
    )

    gate = out.ranked_candidates[0]["passed_gates"][0]
    assert gate["id"] == "missing_as_zero"
    assert gate["missing"] is True
    assert gate["missing_policy"] == "zero"
    assert gate["used_value"] == pytest.approx(0.0)


def test_objective_rank_formula_missing_override_beats_global_policy(
    tmp_path: Path,
) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="formula_policy",
            candidates=[{"candidate_id": "cand", "score": 2.0}],
            score_formula="${score} + ${missing.metric}",
            missing="fail",
            formula_missing={"missing.metric": "zero"},
        ),
        null_context(tmp_path),
    )

    row = out.ranked_candidates[0]
    assert row["objective_score"] == pytest.approx(2.0)
    missing_component = next(
        component
        for component in row["score_components"]
        if component["metric"] == "missing.metric"
    )
    assert missing_component["missing"] is True
    assert missing_component["missing_policy"] == "zero"
    assert missing_component["used_value"] == pytest.approx(0.0)


def test_objective_rank_missing_gate_fail_marks_candidate_failed(tmp_path: Path) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="gate_policy",
            candidates=[{"candidate_id": "cand", "score": 1.0}],
            gates=[
                {"id": "required", "metric": "missing.required", "op": ">=", "value": 1}
            ],
            score_formula="${score}",
        ),
        null_context(tmp_path),
    )

    assert out.ranked_candidates[0]["failed_gates"][0]["id"] == "required"


def test_objective_rank_uses_deterministic_tie_breakers(tmp_path: Path) -> None:
    out = _adapter(tmp_path).run(
        ObjectiveRankInput(
            objective_name="tie_break",
            candidates=[
                {"candidate_id": "b", "scores": {"protenix_v2": {"plddt": 80}}, "score": 1.0},
                {"candidate_id": "a", "scores": {"protenix_v2": {"plddt": 90}}, "score": 1.0},
            ],
            score_formula="${score}",
            tie_breakers=[{"metric": "protenix_v2.plddt", "direction": "higher"}],
        ),
        null_context(tmp_path),
    )

    assert [row["candidate_id"] for row in out.ranked_candidates] == ["a", "b"]


def test_objective_rank_rejects_unsafe_formula_syntax(tmp_path: Path) -> None:
    with pytest.raises(InvalidParamsError):
        _adapter(tmp_path).run(
            ObjectiveRankInput(
                objective_name="unsafe",
                candidates=[{"candidate_id": "cand", "score": 1.0}],
                score_formula="__import__('os').system('true')",
            ),
            null_context(tmp_path),
        )

    with pytest.raises(InvalidParamsError):
        _adapter(tmp_path).run(
            ObjectiveRankInput(
                objective_name="unsafe",
                candidates=[{"candidate_id": "cand", "score": 1.0}],
                score_formula="${score} ** 2",
            ),
            null_context(tmp_path),
        )


def test_objective_rank_reads_workspace_job_ids_and_preserves_evidence(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_a",
                    "source": {"tool": "protenix_v2", "local_id": "a", "index": 0},
                    "sequence": "AAAA",
                    "structure_path": "/tmp/a.cif",
                    "scores": {"protenix_v2": {"plddt": 88.0}},
                    "artifacts": {"protenix_v2": {"predicted_cif": "/tmp/a.cif"}},
                }
            ]
        },
    )

    out = _adapter(tmp_workspace.root).run(
        ObjectiveRankInput(
            job_ids=[job_id],
            objective_name="confidence",
            score_formula="${protenix_v2.plddt}",
        ),
        null_context(tmp_workspace.job_dir("job_objective_rank")),
    )

    row = out.ranked_candidates[0]
    assert row["candidate_id"] == "cand_a"
    assert row["sequence"] == "AAAA"
    assert row["structure_path"] == "/tmp/a.cif"
    assert row["scores"]["protenix_v2"]["plddt"] == 88.0
    assert row["scores"]["objective_rank"]["objective_score"] == pytest.approx(88.0)
