from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator, ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

from vibeproteinagent.adapters import AdapterRunContext, null_context
from vibeproteinagent.adapters.submit_candidate.adapter import (
    CandidateRef,
    SubmitCandidateAdapter,
    SubmitCandidateInput,
)
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import JobState, JobStore, TrajectoryStore, Workspace


def _completed_job(workspace: Workspace, output: dict, *, tool: str = "objective_rank") -> str:
    store = JobStore(workspace)
    rec = store.submit(tool, "1.0.0", {"top_k": 1})
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED, output=output)
    return rec.job_id


def _adapter(tmp_workspace: Workspace) -> SubmitCandidateAdapter:
    return SubmitCandidateAdapter(
        config={},
        weights_dir=tmp_workspace.root / "weights",
        device=torch.device("cpu"),
    )


def _selection_summary() -> dict[str, object]:
    return {
        "headline": "Best recorded evidence among submitted candidates.",
        "why_selected": "The selected candidate best matched the stated objective using recorded tool outputs.",
        "key_points": ["Passed required checks", "Had the strongest available evidence"],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"job_id": "job_x", "candidate_id": "", "justification": "x"},
        {"job_id": "job_x", "ranked_candidate_id": "", "justification": "x"},
    ],
)
def test_candidate_ref_rejects_empty_string_selectors(payload: dict) -> None:
    with pytest.raises(ValidationError):
        CandidateRef(**payload)


def test_submit_candidate_output_schema_documents_selection_evidence() -> None:
    schema = json.loads(
        Path(
            "src/vibeproteinagent/adapters/submit_candidate/schemas/"
            "submit_candidate.output.json"
        ).read_text()
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    payload = {
        "submission_id": "sub_1",
        "outcome": "accepted",
        "terminal": True,
        "reasoning": "final",
        "selection_summary": _selection_summary(),
        "auto_selected": False,
        "next_report_command": "vpa report job_1",
        "runtime_ms": 1,
        "candidates_resolved": [{"candidate_id": "cand_a"}],
    }

    with pytest.raises(JSONSchemaValidationError):
        validator.validate(payload)

    payload["candidates_resolved"][0]["selection_evidence"] = {
        "status": "unranked_submission",
        "source_tool": "manual",
    }
    validator.validate(payload)


def test_submit_candidate_resolves_candidate_id(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {
                    "candidate_id": "design_a",
                    "rank": 1,
                    "score": 1.2,
                    "objective_score": 1.2,
                    "objective_name": "test_objective",
                    "score_formula": "${score}",
                    "passed_gates": [{"id": "score_present", "metric": "score"}],
                    "failed_gates": [],
                    "sequence": "ACDEFGHIK",
                    "plddt": 88.0,
                    "scores": {
                        "objective_rank": {
                            "rank": 1,
                            "objective_score": 1.2,
                            "objective_name": "test_objective",
                            "score_formula": "${score}",
                            "passed_gates": [{"id": "score_present", "metric": "score"}],
                            "failed_gates": [],
                        }
                    },
                }
            ]
        },
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "candidate_id": "design_a",
                    "justification": "best score",
                    "scores_to_highlight": ["score", "plddt"],
                }
            ],
            reasoning="final pick",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    resolved = out.candidates_resolved[0]
    assert out.terminal is True
    assert out.auto_selected is False
    assert out.selection_summary == _selection_summary()
    assert out.next_report_command.startswith("vpa report job_000000")
    assert resolved["candidate_id"] == "design_a"
    assert resolved["score"] == 1.2
    assert resolved["selection_evidence"]["status"] == "objective_ranked"
    assert resolved["objective_rank"] == 1
    assert resolved["objective_score"] == pytest.approx(1.2)
    assert resolved["objective_name"] == "test_objective"
    assert resolved["passed_gates"] == [{"id": "score_present", "metric": "score"}]
    assert resolved["scores_highlighted"] == {"score": 1.2, "plddt": 88.0}


def test_submit_candidate_rejects_ambiguous_duplicate_candidate_id(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {
                    "candidate_id": "dup",
                    "ranked_candidate_id": "dup__v1_a1b2c3d4",
                    "rank": 1,
                    "sequence": "AAAA",
                    "objective_score": 2.0,
                },
                {
                    "candidate_id": "dup",
                    "ranked_candidate_id": "dup__v2_e5f6a7b8",
                    "rank": 2,
                    "sequence": "CCCC",
                    "objective_score": 1.0,
                },
            ]
        },
    )

    with pytest.raises(InvalidParamsError) as excinfo:
        _adapter(tmp_workspace).run(
            SubmitCandidateInput(
                outcome="accepted",
                candidates=[
                    {
                        "job_id": job_id,
                        "candidate_id": "dup",
                        "justification": "ambiguous duplicate",
                    }
                ],
                reasoning="x",
                selection_summary=_selection_summary(),
            ),
            null_context(tmp_workspace.job_dir("job_submit")),
        )

    assert "ambiguous" in str(excinfo.value)
    assert "dup__v1_a1b2c3d4" in str(excinfo.value.details)
    assert "dup__v2_e5f6a7b8" in str(excinfo.value.details)


def test_submit_candidate_resolves_ranked_candidate_id_and_preserves_objective_evidence(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {
                    "candidate_id": "dup",
                    "ranked_candidate_id": "dup__v1_a1b2c3d4",
                    "rank": 1,
                    "sequence": "AAAA",
                    "objective_score": 2.0,
                    "objective_name": "fold_stability",
                    "score_formula": "-1 * ${pyrosetta.mean_ddg}",
                    "formula_text": "-1 * ${pyrosetta.mean_ddg}",
                    "score_direction": "higher",
                    "gate_summary": ["pyrosetta.mean_ddg < 0"],
                    "passed_gates": [{"id": "stabilizing"}],
                    "failed_gates": [],
                    "score_components": [{"metric": "pyrosetta.mean_ddg", "value": -2.0}],
                    "tie_breaker_values": {"protenix_v2.plddt": 90.0},
                },
                {
                    "candidate_id": "dup",
                    "ranked_candidate_id": "dup__v2_e5f6a7b8",
                    "rank": 2,
                    "sequence": "CCCC",
                    "objective_score": 1.0,
                },
            ]
        },
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "ranked_candidate_id": "dup__v1_a1b2c3d4",
                    "justification": "exact ranked row",
                }
            ],
            reasoning="final",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    resolved = out.candidates_resolved[0]
    assert resolved["candidate_id"] == "dup"
    assert resolved["ranked_candidate_id"] == "dup__v1_a1b2c3d4"
    assert resolved["sequence"] == "AAAA"
    evidence = resolved["selection_evidence"]
    assert evidence["status"] == "objective_ranked"
    assert evidence["ranked_candidate_id"] == "dup__v1_a1b2c3d4"
    assert evidence["objective_rank"] == 1
    assert evidence["objective_score"] == pytest.approx(2.0)
    assert evidence["gate_summary"] == ["pyrosetta.mean_ddg < 0"]
    assert evidence["tie_breaker_values"] == {"protenix_v2.plddt": 90.0}


def test_submit_candidate_next_report_command_prefers_trajectory(
    tmp_workspace: Workspace,
) -> None:
    traj = TrajectoryStore(tmp_workspace).create(user_query="design trajectory")
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {"candidate_id": "design_a", "rank": 1, "sequence": "ACDE"}
            ]
        },
    )
    store = JobStore(tmp_workspace)
    submit_job = store.submit(
        "submit_candidate",
        "1.0.0",
        {
            "outcome": "accepted",
            "candidates": [
                {
                    "job_id": job_id,
                    "candidate_id": "design_a",
                    "justification": "best",
                }
            ],
            "reasoning": "final",
        },
        trajectory_id=traj.trajectory_id,
    )
    ctx = AdapterRunContext(
        job_id=submit_job.job_id,
        workspace_root=tmp_workspace.root,
        job_dir=tmp_workspace.job_dir(submit_job.job_id),
        is_cancelled=lambda: False,
        log=lambda _line: None,
        emit_progress=lambda _pct: None,
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "candidate_id": "design_a",
                    "justification": "best",
                }
            ],
            reasoning="final",
            selection_summary=_selection_summary(),
        ),
        ctx,
    )

    assert out.next_report_command.startswith(f"vpa report {traj.trajectory_id} ")


def test_submit_candidate_resolves_zero_based_index(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "designs": [
                {"candidate_id": "first", "sequence": "AAAA"},
                {"candidate_id": "second", "sequence": "CCCC"},
            ]
        },
        tool="sequence_edit",
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {"job_id": job_id, "index": 1, "justification": "second design"}
            ],
            reasoning="index fallback",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    assert out.candidates_resolved[0]["candidate_id"] == "second"
    assert out.candidates_resolved[0]["index"] == 1
    assert out.candidates_resolved[0]["selection_evidence"]["status"] == "unranked_submission"


def test_submit_candidate_resolves_rank_and_dotted_score_highlights(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {
                    "candidate_id": "second",
                    "rank": 2,
                    "sequence": "CCCC",
                    "scores": {
                        "boltz2_affinity": {
                            "affinity_pred_value": -0.8,
                            "affinity_probability_binary": 0.7,
                        }
                    },
                },
                {
                    "candidate_id": "first",
                    "rank": 1,
                    "sequence": "AAAA",
                    "score": 1.3,
                    "scores": {
                        "boltz2_affinity": {
                            "affinity_pred_value": -1.1,
                            "affinity_probability_binary": 0.8,
                        }
                    },
                },
            ]
        },
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "rank": 1,
                    "justification": "top ranked",
                    "scores_to_highlight": [
                        "score",
                        "boltz2_affinity.affinity_pred_value",
                    ],
                }
            ],
            reasoning="rank selector",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    resolved = out.candidates_resolved[0]
    assert resolved["candidate_id"] == "first"
    assert resolved["selector"]["rank"] == 1
    assert resolved["scores_highlighted"] == {
        "score": 1.3,
        "boltz2_affinity.affinity_pred_value": -1.1,
    }


def test_submit_candidate_ranked_functional_design_keeps_structure_and_highlights(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {
                    "candidate_id": "v40",
                    "rank": 1,
                    "score": 2.4,
                    "sequence": "ACDEFA",
                    "structure_path": "/tmp/v40.cif",
                    "scores": {
                        "protenix_v2": {"plddt": 79.512},
                        "protein_qc": {
                            "plddt": 79.512,
                            "ca_bfactor_plddt": 83.363,
                        },
                        "tmalign": {"tm_score_normalized": 0.88058},
                    },
                }
            ]
        },
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "rank": 1,
                    "justification": "best functional design",
                    "scores_to_highlight": [
                        "protenix_v2.plddt",
                        "protein_qc.ca_bfactor_plddt",
                        "tmalign.tm_score_normalized",
                    ],
                }
            ],
            reasoning="accepted functional design",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit_functional")),
    )

    resolved = out.candidates_resolved[0]
    assert resolved["candidate_id"] == "v40"
    assert resolved["structure_path"] == "/tmp/v40.cif"
    assert resolved["scores_highlighted"] == {
        "protenix_v2.plddt": 79.512,
        "protein_qc.ca_bfactor_plddt": 83.363,
        "tmalign.tm_score_normalized": 0.88058,
    }


def test_submit_candidate_validates_empty_outcomes() -> None:
    with pytest.raises(ValidationError, match="accepted submission requires"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[],
            reasoning="x",
            selection_summary=_selection_summary(),
        )

    out = SubmitCandidateInput(
        outcome="no_acceptable_candidate",
        candidates=[],
        reasoning="all candidates failed criteria",
        selection_summary=None,
    )
    assert out.outcome == "no_acceptable_candidate"


def test_submit_candidate_requires_selection_summary_for_accepted(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(tmp_workspace, {"designs": [{"candidate_id": "a"}]})

    with pytest.raises(ValidationError, match="selection_summary"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "candidate_id": "a", "justification": "best"}],
            reasoning="accepted",
        )

    with pytest.raises(ValidationError, match="must not include selection_summary"):
        SubmitCandidateInput(
            outcome="no_acceptable_candidate",
            candidates=[],
            reasoning="none passed",
            selection_summary=_selection_summary(),
        )

    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "candidate_id": "a", "justification": "best"}],
            reasoning="accepted",
            selection_summary={
                "headline": "Best recorded evidence.",
                "why_selected": "The final candidate had the strongest available evidence.",
                "key_points": ["Passed checks", ""],
            },
        )


def test_submit_candidate_rejects_missing_selector(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(tmp_workspace, {"designs": [{"candidate_id": "a"}]})
    with pytest.raises(ValidationError, match="candidate ref requires"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "justification": "missing selector"}],
            reasoning="x",
            selection_summary=_selection_summary(),
        )

    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "candidate_id": "",
                    "justification": "empty logical selector",
                }
            ],
            reasoning="x",
            selection_summary=_selection_summary(),
        )

    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "ranked_candidate_id": "",
                    "justification": "empty ranked selector",
                }
            ],
            reasoning="x",
            selection_summary=_selection_summary(),
        )

    with pytest.raises(ValidationError, match="exactly one selector"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "candidate_id": "a",
                    "index": 0,
                    "justification": "ambiguous selector",
                }
            ],
            reasoning="x",
            selection_summary=_selection_summary(),
        )


def test_submit_candidate_rejects_unresolved_candidate(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(tmp_workspace, {"designs": [{"candidate_id": "a"}]})

    with pytest.raises(InvalidParamsError, match="candidate_id not found"):
        _adapter(tmp_workspace).run(
            SubmitCandidateInput(
                outcome="accepted",
                candidates=[
                    {"job_id": job_id, "candidate_id": "missing", "justification": "x"}
                ],
                reasoning="x",
                selection_summary=_selection_summary(),
            ),
            null_context(tmp_workspace.job_dir("job_submit")),
        )


def test_submit_candidate_does_not_treat_empty_bindcraft_target_as_candidate(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "status": "no_acceptable_candidate",
            "target_pdb": "/tmp/target.pdb",
            "candidates": [],
            "designs": [],
        },
    )

    with pytest.raises(InvalidParamsError, match="no resolvable candidates"):
        _adapter(tmp_workspace).run(
            SubmitCandidateInput(
                outcome="accepted",
                candidates=[
                    {"job_id": job_id, "index": 0, "justification": "should not fallback"}
                ],
                reasoning="x",
                selection_summary=_selection_summary(),
            ),
            null_context(tmp_workspace.job_dir("job_submit")),
        )
