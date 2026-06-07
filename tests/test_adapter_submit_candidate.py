from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import AdapterRunContext, null_context
from vibeproteinagent.adapters.submit_candidate.adapter import (
    SubmitCandidateAdapter,
    SubmitCandidateInput,
)
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import JobState, JobStore, TrajectoryStore, Workspace


def _completed_job(workspace: Workspace, output: dict) -> str:
    store = JobStore(workspace)
    rec = store.submit("rank_designs", "1.0.0", {"top_k": 1})
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


def test_submit_candidate_resolves_candidate_id(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {
                    "candidate_id": "design_a",
                    "rank": 1,
                    "score": 1.2,
                    "sequence": "ACDEFGHIK",
                    "plddt": 88.0,
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
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    resolved = out.candidates_resolved[0]
    assert out.terminal is True
    assert out.auto_selected is False
    assert out.next_report_command.startswith("vpa report job_000000")
    assert resolved["candidate_id"] == "design_a"
    assert resolved["score"] == 1.2
    assert resolved["scores_highlighted"] == {"score": 1.2, "plddt": 88.0}


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
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {"job_id": job_id, "index": 1, "justification": "second design"}
            ],
            reasoning="index fallback",
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    assert out.candidates_resolved[0]["candidate_id"] == "second"
    assert out.candidates_resolved[0]["index"] == 1


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


def test_submit_candidate_validates_empty_outcomes() -> None:
    with pytest.raises(ValidationError, match="accepted submission requires"):
        SubmitCandidateInput(outcome="accepted", candidates=[], reasoning="x")

    out = SubmitCandidateInput(
        outcome="no_acceptable_candidate",
        candidates=[],
        reasoning="all candidates failed criteria",
    )
    assert out.outcome == "no_acceptable_candidate"


def test_submit_candidate_rejects_missing_selector(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(tmp_workspace, {"designs": [{"candidate_id": "a"}]})
    with pytest.raises(ValidationError, match="candidate ref requires"):
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "justification": "missing selector"}],
            reasoning="x",
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
            ),
            null_context(tmp_workspace.job_dir("job_submit")),
        )
