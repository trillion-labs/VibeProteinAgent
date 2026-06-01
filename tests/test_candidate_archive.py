from __future__ import annotations

from vibeproteinagent.agents.execution.candidate_archive import CandidateArchive
from vibeproteinagent.agents.session import AgentSessionStore
from vibeproteinagent.workspace import Workspace


def _summary(candidate: dict, *, workflow_id: str = "wf_1") -> dict:
    return _summary_from_output({"representative_sequences": [candidate]}, workflow_id=workflow_id)


def _summary_from_output(output: dict, *, workflow_id: str = "wf_1") -> dict:
    return {
        "workflow_id": workflow_id,
        "state": "completed",
        "nodes": [
            {
                "node_id": "analysis",
                "tool": "sequence_analysis",
                "job_id": "job_analysis",
                "state": "completed",
                "output": output,
            }
        ],
    }


def test_ingest_raw_workflow_candidates_preserves_nested_scores(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    snapshot = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary(
            {
                "candidate_id": "cand_x",
                "sequence": "ACDE",
                "scores": {"sequence_analysis": {"gravy": -0.1}},
            }
        ),
    )

    materialized = archive.materialize(snapshot.snapshot_id)

    assert materialized[0]["scores"]["sequence_analysis"]["gravy"] == -0.1
    assert materialized[0]["source"]["node_id"] == "analysis"


def test_ingest_rejects_compact_scores_string(tmp_workspace: Workspace) -> None:
    archive = CandidateArchive(tmp_workspace)
    snapshot = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary(
            {
                "candidate_id": "cand_bad",
                "sequence": "ACDE",
                "scores": "<dict>",
            }
        ),
    )

    assert archive.materialize(snapshot.snapshot_id) == []
    cards = archive.failure_cards(snapshot.snapshot_id)
    assert cards[0]["field"] == "scores"
    assert cards[0]["error_kind"] == "compact_candidate_field"


def test_archive_failure_cards_are_scoped_to_snapshot_round(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    clean = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_ok", "sequence": "ACDE"}),
    )
    rejected = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=1,
        workflow_id="wf_2",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_bad", "sequence": "FGHI", "scores": "<dict>"}
        ),
    )

    assert archive.failure_cards(clean.snapshot_id) == []
    assert archive.failure_cards(rejected.snapshot_id)[0]["round_index"] == 1
    archive_wide = archive.failure_cards(None)
    assert {card["round_index"] for card in archive_wide} == {1}
    assert any(card["field"] == "scores" for card in archive_wide)


def test_representative_sequences_promoted_only_with_valid_dict_fields(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    snapshot = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary(
            {
                "id": "rep_1",
                "sequence": "ACDE",
                "scores": {"sequence_analysis": {"charge": 0}},
                "artifacts": {"sequence_analysis": {"path": "/tmp/a.json"}},
            }
        ),
    )

    candidate = archive.materialize(snapshot.snapshot_id)[0]

    assert candidate["schema_version"] == "candidate_v1"
    assert candidate["scores"]["sequence_analysis"]["charge"] == 0
    assert candidate["artifacts"]["sequence_analysis"]["path"] == "/tmp/a.json"


def test_archive_prefers_canonical_candidates_over_representative_sequences(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    snapshot = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary_from_output(
            {
                "candidates": [
                    {
                        "schema_version": "candidate_v1",
                        "candidate_id": "cand_scored",
                        "source": {
                            "tool": "sequence_analysis",
                            "local_id": "representative_0000",
                        },
                        "sequence": "ACDE",
                        "scores": {"sequence_analysis": {"charge": 0}},
                    }
                ],
                "representative_sequences": [
                    {
                        "candidate_id": "representative:1:abc",
                        "sequence": "ACDE",
                    }
                ],
            }
        ),
    )

    materialized = archive.materialize(snapshot.snapshot_id)

    assert [candidate["candidate_id"] for candidate in materialized] == ["cand_scored"]
    assert materialized[0]["scores"]["sequence_analysis"]["charge"] == 0


def test_archive_failure_cards_scope_by_run_and_workflow_not_round_only(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    clean = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_target",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_ok", "sequence": "ACDE"},
            workflow_id="wf_target",
        ),
    )
    same_round_other_workflow = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_other",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_bad_wf", "sequence": "FGHI", "scores": "<dict>"},
            workflow_id="wf_other",
        ),
    )
    archive.ingest_workflow_round(
        run_id="arun_2",
        round_index=0,
        workflow_id="wf_target",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_bad_run", "sequence": "KLMN", "artifacts": "<dict>"},
            workflow_id="wf_target",
        ),
    )

    assert archive.failure_cards(clean.snapshot_id) == []
    scoped = archive.failure_cards(same_round_other_workflow.snapshot_id)
    assert "scores" in {card["field"] for card in scoped}
    assert "artifacts" not in {card["field"] for card in scoped}
    assert {card["field"] for card in archive.failure_cards(None)} >= {
        "scores",
        "artifacts",
    }


def test_archive_failure_cards_none_remains_archive_wide(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_bad_1", "sequence": "ACDE", "scores": "<dict>"}
        ),
    )
    archive.ingest_workflow_round(
        run_id="arun_2",
        round_index=0,
        workflow_id="wf_2",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_bad_2", "sequence": "FGHI", "artifacts": "<dict>"},
            workflow_id="wf_2",
        ),
    )

    assert {card["field"] for card in archive.failure_cards(None)} >= {
        "scores",
        "artifacts",
    }


def test_survivor_snapshot_merges_prior_and_current_candidates(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    first = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_a", "sequence": "ACDE"}),
    )
    survivors = archive.merge_survivors(
        run_id="arun_1",
        round_index=0,
        previous_snapshot_id=None,
        current_snapshot_id=first.snapshot_id,
    )
    second = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=1,
        workflow_id="wf_2",
        raw_workflow_summary=_summary({"candidate_id": "cand_b", "sequence": "FGHI"}),
    )
    merged = archive.merge_survivors(
        run_id="arun_1",
        round_index=1,
        previous_snapshot_id=survivors.snapshot_id,
        current_snapshot_id=second.snapshot_id,
    )

    assert [c["candidate_id"] for c in archive.materialize(merged.snapshot_id)] == [
        "cand_a",
        "cand_b",
    ]
    assert merged.workflow_ids == ["wf_1", "wf_2"]


def test_survivor_snapshot_workflow_ids_union_previous_and_current(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    first = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_a",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_a", "sequence": "ACDE"},
            workflow_id="wf_a",
        ),
    )
    survivors = archive.merge_survivors(
        run_id="arun_1",
        round_index=0,
        previous_snapshot_id=None,
        current_snapshot_id=first.snapshot_id,
    )
    second = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=1,
        workflow_id="wf_b",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_b", "sequence": "FGHI"},
            workflow_id="wf_b",
        ),
    )
    merged = archive.merge_survivors(
        run_id="arun_1",
        round_index=1,
        previous_snapshot_id=survivors.snapshot_id,
        current_snapshot_id=second.snapshot_id,
    )

    assert merged.workflow_ids == ["wf_a", "wf_b"]


def test_survivor_record_workflow_uses_representative_source_version(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    first = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_old",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_x", "sequence": "ACDE"},
            workflow_id="wf_old",
        ),
    )
    survivors = archive.merge_survivors(
        run_id="arun_1",
        round_index=0,
        previous_snapshot_id=None,
        current_snapshot_id=first.snapshot_id,
    )
    second = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=1,
        workflow_id="wf_new",
        raw_workflow_summary=_summary(
            {
                "candidate_id": "cand_x",
                "sequence": "ACDE",
                "scores": {"sequence_analysis": {"gravy": -0.1}},
                "capabilities": {"has_sequence": True, "has_candidate_scores": True},
            },
            workflow_id="wf_new",
        ),
    )
    merged = archive.merge_survivors(
        run_id="arun_1",
        round_index=1,
        previous_snapshot_id=survivors.snapshot_id,
        current_snapshot_id=second.snapshot_id,
    )

    materialized = archive.materialize(merged.snapshot_id)
    record = archive.records_by_version[merged.candidate_refs[0].version_id]

    assert len(materialized) == 1
    assert materialized[0]["scores"]["sequence_analysis"]["gravy"] == -0.1
    assert record.workflow_id == "wf_new"
    assert record.raw_source["source_workflow_ids"] == ["wf_old", "wf_new"]
    assert record.raw_source["representative_workflow_id"] == "wf_new"
    assert merged.workflow_ids == ["wf_old", "wf_new"]


def test_persisted_snapshot_round_trips_workflow_ids(
    tmp_workspace: Workspace,
) -> None:
    store = AgentSessionStore(tmp_workspace)
    run = store.create_run("archive persistence")
    archive = CandidateArchive(tmp_workspace, session_store=store)
    snapshot = archive.ingest_workflow_round(
        run_id=run.run_id,
        round_index=0,
        workflow_id="wf_persisted",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_persisted", "sequence": "ACDE"},
            workflow_id="wf_persisted",
        ),
    )

    persisted = store.get_candidate_snapshot(snapshot.snapshot_id)
    assert persisted is not None
    assert persisted["run_id"] == run.run_id
    assert persisted["workflow_ids"] == ["wf_persisted"]

    reloaded = CandidateArchive(tmp_workspace, session_store=store)
    assert reloaded.materialize(snapshot.snapshot_id)[0]["candidate_id"] == "cand_persisted"
    assert reloaded.failure_cards(snapshot.snapshot_id) == []


def test_materialize_executor_outputs_uses_archive_snapshot(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    snapshot = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "ACDE"}),
    )

    scope = archive.materialize_executor_outputs(
        snapshot_id=snapshot.snapshot_id,
        round_number=2,
        reason="continue",
        summary={"workflow_id": "wf_1"},
    )

    assert scope["outputs"]["prior_round"]["candidates"][0]["candidate_id"] == "cand_x"
    assert scope["outputs"]["prior_round"]["reason"] == "continue"


def test_snapshot_materialization_is_immutable_after_same_candidate_id_update(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    first = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "ACDE"}),
    )
    archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=1,
        workflow_id="wf_2",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "FGHI"}),
    )

    assert archive.materialize(first.snapshot_id)[0]["sequence"] == "ACDE"


def test_snapshot_materialization_not_poisoned_by_later_rejected_same_candidate_id(
    tmp_workspace: Workspace,
) -> None:
    archive = CandidateArchive(tmp_workspace)
    first = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "ACDE"}),
    )
    archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=1,
        workflow_id="wf_2",
        raw_workflow_summary=_summary(
            {"candidate_id": "cand_x", "sequence": "FGHI", "scores": "<dict>"}
        ),
    )

    assert archive.materialize(first.snapshot_id)[0]["sequence"] == "ACDE"


def test_persisted_snapshot_hydrates_exact_version_refs_after_archive_reload(
    tmp_workspace: Workspace,
) -> None:
    store = AgentSessionStore(tmp_workspace)
    run = store.create_run("persisted archive")
    archive = CandidateArchive(tmp_workspace, session_store=store)
    first = archive.ingest_workflow_round(
        run_id=run.run_id,
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "ACDE"}),
    )
    archive.ingest_workflow_round(
        run_id=run.run_id,
        round_index=1,
        workflow_id="wf_2",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "FGHI"}),
    )

    reloaded = CandidateArchive(tmp_workspace, session_store=store)

    assert reloaded.materialize(first.snapshot_id)[0]["sequence"] == "ACDE"


def test_materialize_does_not_call_latest_record_lookup(tmp_workspace: Workspace) -> None:
    archive = CandidateArchive(tmp_workspace)
    snapshot = archive.ingest_workflow_round(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "ACDE"}),
    )

    assert not hasattr(archive, "_latest_record")
    assert archive.materialize(snapshot.snapshot_id)[0]["candidate_id"] == "cand_x"


def test_cross_run_same_candidate_id_does_not_pollute_snapshot(
    tmp_workspace: Workspace,
) -> None:
    store = AgentSessionStore(tmp_workspace)
    first_run = store.create_run("first archive")
    second_run = store.create_run("second archive")
    first_archive = CandidateArchive(tmp_workspace, session_store=store)
    first = first_archive.ingest_workflow_round(
        run_id=first_run.run_id,
        round_index=0,
        workflow_id="wf_1",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "AAAA"}),
    )
    second_archive = CandidateArchive(tmp_workspace, session_store=store)
    second_archive.ingest_workflow_round(
        run_id=second_run.run_id,
        round_index=0,
        workflow_id="wf_2",
        raw_workflow_summary=_summary({"candidate_id": "cand_x", "sequence": "BBBB"}),
    )

    reloaded = CandidateArchive(tmp_workspace, session_store=store)

    assert reloaded.materialize(first.snapshot_id)[0]["sequence"] == "AAAA"
