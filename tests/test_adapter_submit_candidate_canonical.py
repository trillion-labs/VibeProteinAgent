from __future__ import annotations

import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.submit_candidate.adapter import (
    SubmitCandidateAdapter,
    SubmitCandidateInput,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _completed_job(workspace: Workspace, output: dict, *, tool: str = "go_sequences") -> str:
    store = JobStore(workspace)
    rec = store.submit(tool, "1.0.0", {"top_k": 1})
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED, output=output)
    return rec.job_id


def _adapter(workspace: Workspace) -> SubmitCandidateAdapter:
    return SubmitCandidateAdapter(
        config={},
        weights_dir=workspace.root / "weights",
        device=torch.device("cpu"),
    )


def _selection_summary() -> dict[str, object]:
    return {
        "headline": "Best recorded evidence among submitted candidates.",
        "why_selected": "The selected candidate best matched the stated objective using recorded tool outputs.",
        "key_points": ["Passed required checks", "Had the strongest available evidence"],
    }


def _canonical_candidate(candidate_id: str, index: int, sequence: str, score: float) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "go_sequences",
            "node_id": None,
            "job_id": None,
            "local_id": f"seq_{index:04d}",
            "native_id": None,
            "index": index,
        },
        "sequence": sequence,
        "scores": {"go_sequences": {"score": score, "length": len(sequence)}},
        "artifacts": {},
    }


def test_submit_candidate_resolves_canonical_by_candidate_id(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {"candidates": [_canonical_candidate("cand_a", 0, "ACDE", 0.7)]},
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[
                {
                    "job_id": job_id,
                    "candidate_id": "cand_a",
                    "justification": "canonical candidate",
                    "scores_to_highlight": ["score"],
                }
            ],
            reasoning="submit canonical",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    resolved = out.candidates_resolved[0]
    assert resolved["candidate_id"] == "cand_a"
    assert resolved["sequence"] == "ACDE"
    assert resolved["score"] == 0.7
    assert resolved["scores_highlighted"] == {"score": 0.7}


def test_submit_candidate_resolves_canonical_by_index(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "candidates": [
                _canonical_candidate("cand_a", 0, "AAAA", 0.1),
                _canonical_candidate("cand_b", 1, "CCCC", 0.9),
            ]
        },
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "index": 1, "justification": "second canonical"}],
            reasoning="submit canonical index",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    assert out.candidates_resolved[0]["candidate_id"] == "cand_b"
    assert out.candidates_resolved[0]["sequence"] == "CCCC"


def test_submit_candidate_legacy_resolution_still_works(tmp_workspace: Workspace) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {"ranked_candidates": [{"candidate_id": "legacy", "sequence": "ACDE", "score": 1.2}]},
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "candidate_id": "legacy", "justification": "legacy"}],
            reasoning="submit legacy",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    assert out.candidates_resolved[0]["candidate_id"] == "legacy"
    assert out.candidates_resolved[0]["sequence"] == "ACDE"


def test_submit_candidate_preserves_ranked_candidate_artifact_metadata(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "ranked_candidates": [
                {
                    "rank": 1,
                    "candidate_id": "cand_complex",
                    "sequence": "ACDEF",
                    "structure_path": "/tmp/complex.cif",
                    "score": 1.2,
                    "scores": {"protenix_v2": {"iptm": 0.65}},
                    "artifacts": {
                        "protenix_v2": {
                            "complex_chains": [
                                {"role": "target", "input_order": 0, "sequence_length": 809},
                                {"role": "binder", "input_order": 1, "sequence_length": 5},
                            ]
                        }
                    },
                }
            ]
        },
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "rank": 1, "justification": "best complex"}],
            reasoning="submit ranked complex",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    resolved = out.candidates_resolved[0]
    assert resolved["candidate_id"] == "cand_complex"
    assert resolved["sequence"] == "ACDEF"
    assert resolved["candidate_artifacts"]["protenix_v2"]["complex_chains"][1]["role"] == "binder"


def test_submit_candidate_canonical_candidates_win_over_legacy_lists(
    tmp_workspace: Workspace,
) -> None:
    job_id = _completed_job(
        tmp_workspace,
        {
            "candidates": [_canonical_candidate("shared", 0, "CANONICAL", 0.8)],
            "ranked_candidates": [{"candidate_id": "shared", "sequence": "LEGACY", "score": 9.9}],
        },
    )

    out = _adapter(tmp_workspace).run(
        SubmitCandidateInput(
            outcome="accepted",
            candidates=[{"job_id": job_id, "candidate_id": "shared", "justification": "wins"}],
            reasoning="canonical first",
            selection_summary=_selection_summary(),
        ),
        null_context(tmp_workspace.job_dir("job_submit")),
    )

    assert out.candidates_resolved[0]["sequence"] == "CANONICAL"
