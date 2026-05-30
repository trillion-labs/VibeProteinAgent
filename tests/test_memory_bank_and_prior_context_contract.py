from __future__ import annotations

import inspect

import vibeproteinagent.agents.execution.controller as controller_mod
from vibeproteinagent.agents.execution.controller import (
    AgentController,
    _candidate_pool_items_from_summary,
    _first_submit_candidate_ref,
    _merge_candidate_pool,
)
from vibeproteinagent.agents.execution.candidate_archive import CandidateArchive
from vibeproteinagent.agents.execution.memory_bank import MemoryBank
from vibeproteinagent.agents.execution.memory_summarizer import (
    MemorySummarizer,
    PromptContextBuilder,
)
from vibeproteinagent.agents.execution.round_state import PriorRoundContext
from vibeproteinagent.agents.session.store import AgentSessionStore
from vibeproteinagent.agents.planning.workflow_planner import _prior_round_context_payload
from vibeproteinagent.agents.summarization.result_summarizer import (
    ResultSummarizer,
    _compact,
    decision_safe_summary,
)
from vibeproteinagent.workflows import WorkflowStore


def _candidate(index: int) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": f"cand_{index}",
        "source": {"tool": "producer", "local_id": str(index)},
        "sequence": "ACDE",
        "capabilities": {
            "has_sequence": True,
            "has_backbone_structure": False,
            "has_full_structure": False,
            "has_candidate_scores": False,
        },
    }


def _candidate_summary(candidates: list[dict], *, limit: int = 10) -> dict:
    items = []
    for candidate in candidates[:limit]:
        scores = candidate.get("scores")
        items.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "source": candidate.get("source"),
                "capabilities": candidate.get("capabilities", {}),
                "has_sequence": bool(candidate.get("sequence")),
                "sequence_length": len(candidate["sequence"])
                if isinstance(candidate.get("sequence"), str)
                else None,
                "score_keys": sorted(scores) if isinstance(scores, dict) else [],
            }
        )
    return {"count": len(candidates), "items": items, "truncated": len(candidates) > limit}


def _record_round_for_test(
    bank: MemoryBank,
    *,
    round_index: int,
    workflow_summary: dict,
    current_round_candidates: list[dict] | None = None,
    survivor_pool: list[dict] | None = None,
    max_round_reached: bool = False,
) -> dict:
    current_round_candidates = current_round_candidates or []
    survivor_pool = survivor_pool or []
    bank.record_round_raw(
        run_id="arun_test",
        round_index=round_index,
        workflow_id=workflow_summary.get("workflow_id"),
        workflow_summary_raw=workflow_summary,
        candidate_snapshot_id="csnap_current",
        survivor_snapshot_id="csnap_survivor",
    )
    record = bank.latest_round_record()
    assert record is not None
    record["current_round_candidate_summary"] = _candidate_summary(current_round_candidates)
    record["survivor_pool_summary"] = _candidate_summary(survivor_pool)
    record["max_round_reached"] = max_round_reached
    return record


def test_memory_bank_records_empty_event_pool_state_and_branch() -> None:
    bank = MemoryBank()
    current = []
    survivors = [_candidate(1)]
    record = _record_round_for_test(
        bank,
        round_index=0,
        workflow_summary={
            "workflow_id": "wf_1",
            "state": "completed",
            "nodes": [
                {
                    "node_id": "producer",
                    "tool": "go_sequences",
                    "state": "completed",
                    "output": {
                        "candidates": [],
                        "empty_candidate_event": {"event_type": "empty_candidate_event"},
                    },
                },
                {
                    "node_id": "consumer",
                    "tool": "sequence_analysis",
                    "state": "skipped",
                    "output": {
                        "skip_kind": "skipped_no_candidates",
                        "skip_reason": "upstream candidate stream is empty",
                    },
                },
            ],
        },
        current_round_candidates=current,
        survivor_pool=survivors,
        max_round_reached=False,
    )
    bank.record_branch(0, {"action": "next_round", "reason": "empty current round"})

    assert record["empty_candidate_events"]
    assert record["skipped_nodes"][0]["node_id"] == "consumer"
    assert record["current_round_candidate_summary"]["count"] == 0
    assert record["survivor_pool_summary"]["count"] == 1
    assert bank.latest_round_record()["branch_decision"]["action"] == "next_round"


def test_memory_bank_records_lossless_raw_round_payload() -> None:
    bank = MemoryBank()
    raw = {
        "workflow_id": "wf_1",
        "state": "completed",
        "nodes": [
            {
                "node_id": "analysis",
                "tool": "sequence_analysis",
                "output": {
                    "representative_sequences": [
                        {"sequence": "ACDE", "scores": {"sequence_analysis": {"gravy": -0.1}}}
                    ]
                },
            }
        ],
    }

    ledger = bank.record_round_raw(
        run_id="arun_1",
        round_index=0,
        workflow_id="wf_1",
        workflow_summary_raw=raw,
        candidate_snapshot_id="csnap_current",
        survivor_snapshot_id="csnap_survivor",
    )

    assert ledger.workflow_summary_raw["nodes"][0]["output"]["representative_sequences"][0][
        "scores"
    ]["sequence_analysis"]["gravy"] == -0.1
    assert "<dict>" not in str(bank.raw_events[0].payload)
    assert bank.latest_round_record()["candidate_snapshot_id"] == "csnap_current"


def test_memory_summarizer_is_bounded_and_does_not_mutate_bank() -> None:
    bank = MemoryBank()
    raw_survivors = [_candidate(i) for i in range(20)]
    record = _record_round_for_test(
        bank,
        round_index=0,
        workflow_summary={"workflow_id": "wf_1", "state": "completed", "nodes": []},
        current_round_candidates=raw_survivors,
        survivor_pool=raw_survivors,
        max_round_reached=False,
    )
    digest = MemorySummarizer().summarize(
        previous_digest=None,
        latest_round_record=record,
        memory_bank=bank,
    )

    assert digest["latest_round"]["survivor_pool_summary"]["count"] == 20
    assert len(digest["latest_round"]["survivor_pool_summary"]["items"]) == 10


def test_prompt_context_builder_emits_lossy_latest_round_context() -> None:
    raw_candidate = _candidate(1)
    raw_candidate["sequence"] = "ACDEFGHIKLMNPQRSTVWY"
    raw_candidate["structure_path"] = "/tmp/full_structure.pdb"
    raw_candidate["structure_paths"] = ["/tmp/a.pdb", "/tmp/b.pdb"]
    raw_candidate["scores"] = {
        "rank_designs": {"ranking_score": 0.91, "raw_metric": 123.4}
    }
    raw_candidate["capabilities"]["has_candidate_scores"] = True
    record = {
        "round_index": 1,
        "workflow_id": "wf_2",
        "state": "completed",
        "current_round_candidate_summary": _candidate_summary([raw_candidate]),
        "survivor_pool_summary": _candidate_summary([raw_candidate]),
        "candidate_capability_filter_events": [
            {
                "node_id": "qc",
                "tool": "protein_qc",
                "status": "filtered",
                "required_capabilities": ["has_full_structure"],
                "skipped_candidate_count": 1,
            }
        ],
        "current_round_merge_summary": {
            "output_count": 1,
            "duplicate_groups": [{"versions": [{"sequence": raw_candidate["sequence"]}]}],
        },
        "survivor_merge_summary": {"output_count": 1},
        "workflow_summary": {
            "nodes": [
                {
                    "node_id": "rank",
                    "tool": "rank_designs",
                    "state": "completed",
                    "output": {"candidates": [raw_candidate]},
                }
            ]
        },
        "branch_decision": {"action": "next_round", "reason": "improve confidence"},
    }

    digest = PromptContextBuilder().build(
        previous_digest=None,
        latest_round_record=record,
    )

    latest = digest["latest_round"]
    assert digest["round_count"] == 2
    assert {
        "candidate_summary",
        "survivor_summary",
        "capability_summary",
        "merge_summary",
        "failure_cards",
        "trajectory_cards",
        "next_move_hints",
    }.issubset(latest)
    assert latest["candidate_summary"]["items"][0]["score_keys"] == ["rank_designs"]
    assert latest["candidate_summary"]["items"][0]["sequence_length"] == 20
    assert latest["merge_summary"]["current_round"]["duplicate_group_count"] == 1
    serialized = str(digest)
    assert raw_candidate["sequence"] not in serialized
    assert "/tmp/full_structure.pdb" not in serialized
    assert "/tmp/a.pdb" not in serialized
    assert "raw_metric" not in serialized
    assert "0.91" not in serialized


def test_prompt_context_uses_archive_capability_summary_beyond_card_limit() -> None:
    candidates = [_candidate(index) for index in range(12)]
    bounded_items = [
        {
            "candidate_id": candidate["candidate_id"],
            "source": candidate["source"],
            "capabilities": candidate["capabilities"],
            "has_sequence": True,
        }
        for candidate in candidates[:10]
    ]
    record = {
        "round_index": 0,
        "current_round_candidate_summary": {
            "count": 12,
            "items": bounded_items,
            "truncated": True,
        },
        "survivor_pool_summary": {
            "count": 12,
            "items": bounded_items,
            "truncated": True,
        },
        "capability_summary": {
            "capabilities": {"has_sequence": 12},
            "count": 12,
        },
    }

    digest = PromptContextBuilder().build(
        previous_digest=None,
        latest_round_record=record,
    )

    latest = digest["latest_round"]
    assert len(latest["survivor_summary"]["items"]) == 10
    assert latest["capability_summary"]["capabilities"]["has_sequence"] == 12


def test_prompt_context_failure_and_trajectory_cards_prefer_raw_summary() -> None:
    record = {
        "round_index": 0,
        "workflow_summary_raw": {
            "nodes": [
                {
                    "node_id": "raw_failed",
                    "tool": "raw_tool",
                    "state": "failed",
                    "output": {"error": {"message": "raw failure"}},
                },
                {
                    "node_id": "raw_done",
                    "tool": "raw_done_tool",
                    "state": "completed",
                    "output": {"candidate_count": 3},
                },
            ]
        },
        "workflow_summary": {
            "nodes": [
                {
                    "node_id": "legacy_failed",
                    "tool": "legacy_tool",
                    "state": "failed",
                    "output": {"error": {"message": "legacy failure"}},
                }
            ]
        },
    }

    digest = PromptContextBuilder().build(
        previous_digest=None,
        latest_round_record=record,
    )

    latest = digest["latest_round"]
    assert latest["failure_cards"][0]["node_id"] == "raw_failed"
    assert latest["failure_cards"][0]["error"]["message"] == "raw failure"
    assert [card["node_id"] for card in latest["trajectory_cards"][:2]] == [
        "raw_failed",
        "raw_done",
    ]
    assert "legacy_failed" not in str(latest["failure_cards"])
    assert "legacy_failed" not in str(latest["trajectory_cards"])


def test_prompt_context_passes_through_archive_validation_failure_cards() -> None:
    record = {
        "round_index": 0,
        "failure_cards": [
            {
                "phase": "candidate_archive_ingest",
                "error_kind": "candidate_validation_error",
                "field": "sequence",
                "message": "sequence is required",
                "repair_hint": "Select a sequence-producing node.",
                "retryable": False,
                "structure_path": "/tmp/raw.pdb",
            }
        ],
    }

    digest = PromptContextBuilder().build(
        previous_digest=None,
        latest_round_record=record,
    )

    card = digest["latest_round"]["failure_cards"][0]
    assert card == {
        "phase": "candidate_archive_ingest",
        "error_kind": "candidate_validation_error",
        "field": "sequence",
        "message": "sequence is required",
        "repair_hint": "Select a sequence-producing node.",
        "retryable": False,
    }


def test_memory_summarizer_archive_summaries_are_lossy_prompt_context() -> None:
    candidate = _candidate(1)
    candidate["sequence"] = "ACDEFGHIKLMNPQRSTVWY"
    candidate["scores"] = {"rank_designs": {"ranking_score": 0.91}}
    record = {
        "round_index": 0,
        "current_round_candidate_summary": _candidate_summary([candidate]),
        "survivor_pool_summary": _candidate_summary([candidate]),
    }

    digest = MemorySummarizer().summarize(
        previous_digest=None,
        latest_round_record=record,
    )

    latest = digest["latest_round"]
    assert latest["current_round_candidate_summary"] == latest["candidate_summary"]
    assert latest["survivor_pool_summary"] == latest["survivor_summary"]
    assert "ACDEFGHIKLMNPQRSTVWY" not in str(digest)
    assert "ranking_score" not in str(digest)


def test_controller_production_loop_does_not_use_summary_candidate_pool() -> None:
    source = inspect.getsource(AgentController._run_agent_plan_impl)

    assert "_collect_current_round_candidates" not in source
    assert "_candidate_pool_from_summary" not in source
    assert "_summary_has_candidate_pool" not in source
    assert "record_round(" not in source
    assert "record_round_raw(" in source
    assert "surviving_candidates" not in source
    assert "get_latest_candidate_record" not in source
    assert "_latest_record" not in source


def test_controller_planner_failure_candidate_presence_uses_archive_locals() -> None:
    source = inspect.getsource(AgentController._run_agent_plan_impl)

    assert "bool(current_round_candidates) or bool(survivor_pool)" in source
    assert "_summary_has_candidate_pool" not in source


def test_executor_scope_does_not_use_legacy_candidate_fallbacks() -> None:
    source = inspect.getsource(AgentController._build_executor_scope)

    assert "surviving_candidates" not in source
    assert "get_latest_candidate_record" not in source
    assert "_latest_record" not in source
    assert "materialize_executor_outputs" in source


def test_archive_materialize_does_not_use_latest_record_lookup() -> None:
    source = inspect.getsource(CandidateArchive.materialize)

    assert "_latest_record" not in source
    assert "latest_version_by_candidate_id" not in source
    assert "_record_by_version" in source


def test_cleanup_legacy_candidate_surfaces_are_removed() -> None:
    context = PriorRoundContext(
        round_number=2,
        prior_decision_reason="continue",
        survivor_snapshot_id="csnap_survivor",
    )

    assert not hasattr(MemoryBank, "record_round")
    assert not hasattr(context, "surviving_candidates")
    assert not hasattr(AgentController, "_collect_current_round_candidates")
    assert not hasattr(AgentController, "_collect_surviving_candidates")
    assert not hasattr(controller_mod, "_summary_has_candidate_pool")
    assert not hasattr(controller_mod, "_candidate_pool_from_summary")
    assert not hasattr(CandidateArchive, "_latest_record")
    assert not hasattr(AgentSessionStore, "get_latest_candidate_record")


def test_controller_survivor_pool_keeps_richer_candidate_over_prior_sparse() -> None:
    prior = _candidate(1)
    current = _candidate(1)
    current["scores"] = {"sequence_analysis": {"gravy": -0.1}}
    current["capabilities"]["has_candidate_scores"] = True

    merged = _merge_candidate_pool([prior], [current])

    assert len(merged) == 1
    assert merged[0]["scores"]["sequence_analysis"]["gravy"] == -0.1
    assert merged[0]["capabilities"]["has_candidate_scores"] is True


def test_controller_current_round_pool_keeps_enriched_duplicate_candidate() -> None:
    source = _candidate(1)
    source["source"] = {"tool": "go_sequences", "node_id": "source", "job_id": "job_source"}
    enriched = _candidate(1)
    enriched["source"] = {
        "tool": "sequence_analysis",
        "node_id": "analysis",
        "job_id": "job_analysis",
    }
    enriched["scores"] = {"sequence_analysis": {"gravy": -0.1}}
    enriched["capabilities"]["has_candidate_scores"] = True
    summary = {
        "nodes": [
            {
                "node_id": "source",
                "tool": "go_sequences",
                "job_id": "job_source",
                "output": {"candidates": [source]},
            },
            {
                "node_id": "analysis",
                "tool": "sequence_analysis",
                "job_id": "job_analysis",
                "output": {"candidates": [enriched]},
            },
        ]
    }

    items = _candidate_pool_items_from_summary(summary)
    pool = [dict(item["candidate"]) for item in items]

    assert len(pool) == 1
    assert pool[0]["scores"]["sequence_analysis"]["gravy"] == -0.1
    assert pool[0]["source"]["tool"] == "sequence_analysis"
    assert "__candidate_merge_positions" not in pool[0]
    assert "versions" not in pool[0]
    assert items[0]["job_id"] == "job_analysis"
    assert items[0]["tool"] == "sequence_analysis"


def test_controller_merges_non_candidate_v1_rows_by_explicit_candidate_id() -> None:
    summary = {
        "nodes": [
            {
                "node_id": "source",
                "tool": "go_sequences",
                "job_id": "job_source",
                "output": {
                    "candidates": [
                        {
                            "candidate_id": "cand_x",
                            "capabilities": {"has_sequence": True},
                        }
                    ]
                },
            },
            {
                "node_id": "analysis",
                "tool": "sequence_analysis",
                "job_id": "job_analysis",
                "output": {
                    "candidates": [
                        {
                            "candidate_id": "cand_x",
                            "capabilities": {"has_sequence": True},
                            "scores": {"sequence_analysis": {"gravy": -0.1}},
                        }
                    ]
                },
            },
        ]
    }

    items = _candidate_pool_items_from_summary(summary)
    pool = [dict(item["candidate"]) for item in items]
    ref = _first_submit_candidate_ref(
        summary,
        deliverable_contract={
            "required_candidate_capabilities": ["has_candidate_scores"],
        },
    )

    assert len(pool) == 1
    assert pool[0]["candidate_id"] == "cand_x"
    assert pool[0]["scores"]["sequence_analysis"]["gravy"] == -0.1
    assert items[0]["job_id"] == "job_analysis"
    assert ref is not None
    assert ref["job_id"] == "job_analysis"
    assert ref["candidate_id"] == "cand_x"
    assert "versions" not in ref
    assert "__candidate_merge_positions" not in ref


def test_controller_non_candidate_v1_richness_and_latest_tiebreaks() -> None:
    rich_first = {
        "nodes": [
            {
                "node_id": "analysis",
                "tool": "sequence_analysis",
                "job_id": "job_analysis",
                "output": {
                    "candidates": [
                        {
                            "candidate_id": "cand_x",
                            "capabilities": {"has_sequence": True},
                            "scores": {"sequence_analysis": {"gravy": -0.1}},
                        }
                    ]
                },
            },
            {
                "node_id": "source",
                "tool": "go_sequences",
                "job_id": "job_source",
                "output": {
                    "candidates": [
                        {
                            "candidate_id": "cand_x",
                            "capabilities": {"has_sequence": True},
                        }
                    ]
                },
            },
        ]
    }
    tied = {
        "nodes": [
            {
                "node_id": "source",
                "tool": "go_sequences",
                "job_id": "job_source",
                "output": {
                    "candidates": [
                        {
                            "candidate_id": "cand_y",
                            "capabilities": {"has_sequence": True},
                        }
                    ]
                },
            },
            {
                "node_id": "analysis",
                "tool": "sequence_analysis",
                "job_id": "job_analysis",
                "output": {
                    "candidates": [
                        {
                            "candidate_id": "cand_y",
                            "capabilities": {"has_sequence": True},
                        }
                    ]
                },
            },
        ]
    }

    assert _candidate_pool_items_from_summary(rich_first)[0]["job_id"] == "job_analysis"
    assert _candidate_pool_items_from_summary(tied)[0]["job_id"] == "job_analysis"


def test_result_summarizer_removes_raw_duplicate_merge_audit_from_decision_summary(
    tmp_workspace,
) -> None:
    wf_store = WorkflowStore(tmp_workspace)
    wf_id = wf_store.create(
        name="merge_audit_summary",
        version="1.0",
        descriptor_yaml="name: merge_audit_summary\nversion: '1.0'\nnodes: []\n",
        params={},
    )
    wf_store.add_node(
        workflow_id=wf_id,
        node_id="merge",
        tool_name="merge_candidates",
        tool_version="1.0.0",
        upstream=[],
        params_hash="abc",
        input_hash="def",
        cache_key="ghi",
    )
    wf_store.update_node_state(
        wf_id,
        "merge",
        "completed",
        output={
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_x",
                    "source": {"tool": "go_sequences", "local_id": "x"},
                    "__candidate_merge_positions": [0, 1],
                }
            ],
            "merge_audit": {
                "dedup_policy": "keep_all",
                "output_count": 1,
                "duplicates_removed": 1,
                "duplicate_groups": [
                    {
                        "identity_key": "candidate_id:cand_x",
                        "versions": [{"source": {"native_id": "ACC1"}}],
                    }
                ],
                "conflicts": [{"field": "scores.semantic_relevance"}],
            },
        },
    )
    wf_store.update_state(wf_id, "completed", finished=True)

    summary = ResultSummarizer(tmp_workspace).summarize_workflow(wf_id)

    audit = summary["nodes"][0]["output"]["merge_audit"]
    candidate = summary["nodes"][0]["output"]["candidates"][0]
    assert audit == {
        "dedup_policy": "keep_all",
        "output_count": 1,
        "duplicates_removed": 1,
        "duplicate_group_count": 1,
        "conflict_count": 1,
    }
    assert "duplicate_groups" not in str(summary)
    assert "versions" not in str(summary)
    assert "__candidate_merge_positions" not in candidate


def test_memory_digest_contains_only_count_level_survivor_merge_summary() -> None:
    bank = MemoryBank()
    record = _record_round_for_test(
        bank,
        round_index=0,
        workflow_summary={"workflow_id": "wf_1", "state": "completed", "nodes": []},
        current_round_candidates=[],
        survivor_pool=[_candidate(1)],
        max_round_reached=False,
    )
    record["survivor_merge_summary"] = {
        "duplicates_merged_count": 1,
        "duplicate_group_count": 1,
        "enriched_candidate_count": 1,
    }

    digest = MemorySummarizer().summarize(
        previous_digest=None,
        latest_round_record=record,
        memory_bank=bank,
    )

    merge_summary = digest["latest_round"]["survivor_merge_summary"]
    assert merge_summary["duplicates_merged_count"] == 1
    assert "versions" not in str(merge_summary)


def test_memory_digest_contains_only_count_level_current_round_merge_summary() -> None:
    bank = MemoryBank()
    record = _record_round_for_test(
        bank,
        round_index=0,
        workflow_summary={"workflow_id": "wf_1", "state": "completed", "nodes": []},
        current_round_candidates=[_candidate(1)],
        survivor_pool=[],
        max_round_reached=False,
    )
    record["current_round_merge_summary"] = {
        "duplicates_merged_count": 1,
        "duplicate_group_count": 1,
        "enriched_candidate_count": 1,
    }

    digest = MemorySummarizer().summarize(
        previous_digest=None,
        latest_round_record=record,
        memory_bank=bank,
    )

    merge_summary = digest["latest_round"]["current_round_merge_summary"]
    assert merge_summary["duplicate_group_count"] == 1
    assert "versions" not in str(merge_summary)


def test_memory_summarizer_includes_bounded_capability_filter_events() -> None:
    bank = MemoryBank()
    workflow_summary = {
        "workflow_id": "wf_1",
        "state": "completed",
        "nodes": [
            {
                "node_id": f"consumer_{index}",
                "tool": "protein_qc",
                "state": "completed",
                "output": {
                    "candidate_capability_filter_event": {
                        "status": "filtered",
                        "required_capabilities": ["has_full_structure"],
                        "compatible_candidate_count": 0,
                        "skipped_candidate_count": index + 1,
                    }
                },
            }
            for index in range(12)
        ],
    }
    record = _record_round_for_test(
        bank,
        round_index=0,
        workflow_summary=workflow_summary,
        current_round_candidates=[],
        survivor_pool=[],
        max_round_reached=False,
    )
    before = list(bank.round_records)

    digest = MemorySummarizer().summarize(
        previous_digest=None,
        latest_round_record=record,
        memory_bank=bank,
    )

    events = digest["latest_round"]["candidate_capability_filter_events"]
    assert len(events) == 10
    assert events[0]["node_id"] == "consumer_0"
    assert bank.round_records == before


def test_prior_round_prompt_payload_contains_prompt_context_only() -> None:
    context = PriorRoundContext(
        round_number=2,
        prior_decision_reason="try another producer",
        prompt_digest={"round_count": 1, "survivor_pool_summary": {"count": 20}},
    )

    payload = _prior_round_context_payload(context)

    assert "survivor_snapshot_id" not in payload
    assert payload["digest"]["survivor_pool_summary"]["count"] == 20


def test_result_summarizer_compacts_candidate_v1_to_display_card() -> None:
    candidate = _candidate(1)
    candidate["sequence"] = "ACDEFGHIKLMNPQRSTVWY"
    candidate["scores"] = {"rank_designs": {"ranking_score": 0.91}}
    candidate["artifacts"] = {"rank_designs": {"path": "/tmp/raw.json"}}
    candidate["structure_path"] = "/tmp/raw.pdb"
    compact = _compact({"output": {"candidates": [candidate]}})

    card = compact["output"]["candidates"][0]
    assert card["schema_version"] == "candidate_v1"
    assert card["candidate_id"] == "cand_1"
    assert card["source"]["tool"] == "producer"
    assert card["capabilities"]["has_sequence"] is True
    assert card["sequence_length"] == 20
    assert card["score_keys"] == ["rank_designs"]
    assert card["artifact_keys"] == ["rank_designs"]
    assert card["has_structure_path"] is True
    serialized = str(compact)
    assert "ACDEFGHIKLMNPQRSTVWY" not in serialized
    assert "ranking_score" not in serialized
    assert "/tmp/raw.pdb" not in serialized


def test_decision_safe_summary_does_not_reintroduce_raw_candidate_payload() -> None:
    candidate = _candidate(1)
    candidate["sequence"] = "ACDEFGHIKLMNPQRSTVWY"
    candidate["scores"] = {"rank_designs": {"ranking_score": 0.91}}
    candidate["full_structure_path"] = "/tmp/full.pdb"

    summary = decision_safe_summary({"candidates": [candidate]})
    serialized = str(summary)

    assert summary["candidates"][0]["sequence_length"] == 20
    assert "ACDEFGHIKLMNPQRSTVWY" not in serialized
    assert "ranking_score" not in serialized
    assert "/tmp/full.pdb" not in serialized
