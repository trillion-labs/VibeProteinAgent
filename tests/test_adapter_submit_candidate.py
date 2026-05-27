from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.submit_candidate.adapter import (
    SubmitCandidateAdapter,
    SubmitCandidateInput,
)
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import JobState, JobStore, Workspace


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
    assert resolved["candidate_id"] == "design_a"
    assert resolved["score"] == 1.2
    assert resolved["scores_highlighted"] == {"score": 1.2, "plddt": 88.0}


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
