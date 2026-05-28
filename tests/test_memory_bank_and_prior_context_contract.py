from __future__ import annotations

from dataclasses import dataclass

from vibeproteinagent.agents.execution.memory_bank import MemoryBank
from vibeproteinagent.agents.execution.memory_summarizer import MemorySummarizer
from vibeproteinagent.agents.execution.round_state import PriorRoundContext
from vibeproteinagent.agents.planning.workflow_planner import _prior_round_context_payload
from vibeproteinagent.agents.summarization.result_summarizer import _compact


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


def test_memory_bank_records_empty_event_pool_state_and_branch() -> None:
    bank = MemoryBank()
    current = []
    survivors = [_candidate(1)]
    record = bank.record_round(
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


def test_memory_summarizer_is_bounded_and_does_not_mutate_bank() -> None:
    bank = MemoryBank()
    raw_survivors = [_candidate(i) for i in range(20)]
    record = bank.record_round(
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

    assert len(bank.survivor_pool) == 20
    assert digest["latest_round"]["survivor_pool_summary"]["count"] == 20
    assert len(digest["latest_round"]["survivor_pool_summary"]["items"]) == 10


def test_prior_round_prompt_payload_omits_raw_surviving_candidates() -> None:
    context = PriorRoundContext(
        round_number=2,
        surviving_candidates=[_candidate(i) for i in range(20)],
        prior_decision_reason="try another producer",
        prompt_digest={"round_count": 1, "survivor_pool_summary": {"count": 20}},
    )

    payload = _prior_round_context_payload(context)

    assert "surviving_candidates" not in payload
    assert payload["digest"]["survivor_pool_summary"]["count"] == 20


def test_result_compaction_preserves_candidate_v1_records_for_memory_bank() -> None:
    compact = _compact({"output": {"candidates": [_candidate(1)]}})

    candidate = compact["output"]["candidates"][0]
    assert candidate["schema_version"] == "candidate_v1"
    assert candidate["source"]["tool"] == "producer"
    assert candidate["capabilities"]["has_sequence"] is True
