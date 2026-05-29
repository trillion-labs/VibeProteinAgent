from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.adapters.context import null_context
from vibeproteinagent.agents.planning.canonical_tools import CandidateCapabilityContract
from vibeproteinagent.workflows.descriptor import WorkflowNode
from vibeproteinagent.workflows.state import NodeRuntime
from vibeproteinagent.workflows import Workflow, WorkflowExecutor, load_workflow
import vibeproteinagent.workflows.executor as executor_mod
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _run_pending_jobs(store: JobStore, registry: ToolRegistry, stop: threading.Event) -> None:
    import typing
    import torch

    device = torch.device("cpu")
    weights_dir = Path("/tmp/vpa_test_weights")
    weights_dir.mkdir(exist_ok=True)
    while not stop.is_set():
        recs = store.list(state=JobState.QUEUED, limit=8)
        if not recs:
            time.sleep(0.02)
            continue
        for rec in recs:
            claimed = store.claim_next_queued(worker_id=0, gpu_ids=[], job_id=rec.job_id)
            if claimed is None:
                continue
            _descriptor, cls = registry.get(rec.tool_name, "*")
            adapter = cls(config={}, weights_dir=weights_dir, device=device)
            hints = typing.get_type_hints(adapter.run)
            input_model = hints.get("params")
            params = input_model(**rec.params) if input_model is not None else rec.params
            out = adapter.run(params, null_context(rec.job_id))
            payload = out.model_dump() if hasattr(out, "model_dump") else dict(out)
            store.transition(rec.job_id, JobState.COMPLETED, output=payload)


@pytest.fixture
def inline_supervisor(tmp_workspace: Workspace):
    store = JobStore(tmp_workspace)
    registry = ToolRegistry()
    registry.discover()
    stop = threading.Event()
    thread = threading.Thread(target=_run_pending_jobs, args=(store, registry, stop), daemon=True)
    thread.start()
    try:
        yield object()
    finally:
        stop.set()
        thread.join(timeout=2.0)


def _run_without_worker(
    tmp_workspace: Workspace,
    tmp_path: Path,
    wf: Workflow,
    *,
    virtual_outputs: dict[str, object],
    supervisor: object | None = None,
) -> tuple[dict[str, str], dict[str, object], str]:
    p = tmp_path / f"{wf.name}.yaml"
    wf.save(p)
    registry = ToolRegistry()
    registry.discover()
    executor = WorkflowExecutor(
        tmp_workspace,
        registry,
        supervisor=supervisor,
        virtual_outputs=virtual_outputs,
    )
    wf_id = executor.submit(load_workflow(p), {})
    rec = executor.run_sync(wf_id, max_wait_s=2, poll_s=0.01)
    rows = executor.store.list_nodes(wf_id)
    states = {row.node_id: row.state for row in rows}
    outputs = {row.node_id: row.output for row in rows}
    return states, outputs, rec.state


def test_empty_candidate_stream_skips_only_candidate_consumers(
    tmp_workspace: Workspace,
    tmp_path: Path,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="empty_candidate_contract", version="1.0.0")
    wf.node(
        "sequence_metrics",
        tool="sequence_analysis",
        inputs={"candidates": "${outputs.prior_round.candidates}"},
    )
    wf.node(
        "literature",
        tool="noop",
        inputs={"payload": "still-runs"},
    )

    states, outputs, workflow_state = _run_without_worker(
        tmp_workspace,
        tmp_path,
        wf,
        virtual_outputs={"prior_round": {"candidates": []}},
        supervisor=inline_supervisor,
    )

    assert workflow_state == "completed"
    assert states["sequence_metrics"] == "skipped"
    assert outputs["sequence_metrics"]["status"] == "skipped_no_candidates"
    assert outputs["sequence_metrics"]["technical_failure"] is False
    assert states["literature"] == "completed"


def test_runtime_capability_gate_skips_before_adapter_execution(
    tmp_workspace: Workspace,
    tmp_path: Path,
) -> None:
    wf = Workflow(name="capability_gate_contract", version="1.0.0")
    wf.node(
        "struct_metrics",
        tool="structure_analysis",
        inputs={"candidates": "${outputs.prior_round.candidates}"},
    )

    states, outputs, workflow_state = _run_without_worker(
        tmp_workspace,
        tmp_path,
        wf,
        virtual_outputs={
            "prior_round": {
                "candidates": [
                    {
                        "schema_version": "candidate_v1",
                        "candidate_id": "cand_seq",
                        "source": {"tool": "go_sequences", "local_id": "0"},
                        "sequence": "ACDE",
                        "capabilities": {
                            "has_sequence": True,
                            "has_backbone_structure": False,
                            "has_full_structure": False,
                            "has_candidate_scores": False,
                        },
                    }
                ]
            }
        },
    )

    assert workflow_state == "completed"
    assert states["struct_metrics"] == "skipped"
    assert outputs["struct_metrics"]["status"] == "skipped_missing_candidate_capability"
    assert outputs["struct_metrics"]["runtime_gate_event"]["missing_capabilities"] == [
        "has_backbone_structure",
        "has_full_structure",
    ]
    filter_event = outputs["struct_metrics"]["candidate_capability_filter_event"]
    assert filter_event["status"] == "all_incompatible"
    assert filter_event["compatible_candidate_count"] == 0
    assert filter_event["skipped_candidate_count"] == 1
    assert filter_event["filtered_candidates"][0]["candidate_id"] == "cand_seq"
    assert filter_event["filtered_candidates"][0]["source"] == {
        "tool": "go_sequences",
        "local_id": "0",
    }


def test_runtime_capability_gate_filters_mixed_candidate_stream(
    tmp_workspace: Workspace,
    tmp_path: Path,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="mixed_capability_filter_contract", version="1.0.0")
    wf.node(
        "sequence_metrics",
        tool="sequence_analysis",
        inputs={"candidates": "${outputs.prior_round.candidates}"},
    )

    states, outputs, workflow_state = _run_without_worker(
        tmp_workspace,
        tmp_path,
        wf,
        virtual_outputs={
            "prior_round": {
                "candidates": [
                    {
                        "schema_version": "candidate_v1",
                        "candidate_id": "cand_seq",
                        "source": {"tool": "mpnn", "local_id": "0"},
                        "sequence": "ACDEFGHIK",
                        "capabilities": {
                            "has_sequence": True,
                            "has_backbone_structure": False,
                            "has_full_structure": False,
                            "has_candidate_scores": False,
                        },
                    },
                    {
                        "schema_version": "candidate_v1",
                        "candidate_id": "cand_backbone",
                        "source": {"tool": "rfdiffusion", "local_id": "0"},
                        "structure_path": "/tmp/backbone_only.pdb",
                        "capabilities": {
                            "has_sequence": False,
                            "has_backbone_structure": True,
                            "has_full_structure": False,
                            "has_candidate_scores": False,
                        },
                    },
                ]
            }
        },
        supervisor=inline_supervisor,
    )

    assert workflow_state == "completed"
    assert states["sequence_metrics"] == "completed"
    event = outputs["sequence_metrics"]["candidate_capability_filter_event"]
    assert event["status"] == "partitioned"
    assert event["compatible_candidate_count"] == 1
    assert event["filtered_candidates"][0]["candidate_id"] == "cand_backbone"
    assert event["filtered_candidates"][0]["source"] == {
        "tool": "rfdiffusion",
        "local_id": "0",
    }
    assert event["filtered_candidates"][0]["capabilities"] == ["has_backbone_structure"]
    assert outputs["sequence_metrics"]["representative_sequences"][0]["sequence"] == "ACDEFGHIK"


def test_memory_bank_summarizes_empty_facets_and_capability_filter_events() -> None:
    from vibeproteinagent.agents.execution.memory_bank import MemoryBank

    bank = MemoryBank()
    record = bank.record_round(
        round_index=0,
        workflow_summary={
            "workflow_id": "wf_test",
            "state": "completed",
            "nodes": [
                {
                    "node_id": "resolver",
                    "tool": "function_term_resolver",
                    "state": "completed",
                    "output": {
                        "empty_facets": [
                            {
                                "event_type": "empty_candidate_event",
                                "intent_id": "facet_b",
                            }
                        ]
                    },
                },
                {
                    "node_id": "sequence_metrics",
                    "tool": "sequence_analysis",
                    "state": "completed",
                    "output": {
                        "candidate_capability_filter_event": {
                            "status": "partitioned",
                            "filtered_candidates": [{"candidate_id": "cand_backbone"}],
                        }
                    },
                },
                {
                    "node_id": "struct_metrics",
                    "tool": "structure_analysis",
                    "state": "skipped",
                    "output": {
                        "status": "skipped_missing_candidate_capability",
                        "candidate_capability_filter_event": {
                            "status": "all_incompatible",
                            "filtered_candidates": [{"candidate_id": "cand_seq"}],
                            "compatible_candidate_count": 0,
                            "skipped_candidate_count": 1,
                        },
                    },
                },
            ],
        },
        current_round_candidates=[],
        survivor_pool=[],
        max_round_reached=False,
    )

    assert record["empty_candidate_events"][0]["intent_id"] == "facet_b"
    assert record["candidate_capability_filter_events"][0]["node_id"] == "sequence_metrics"
    assert record["candidate_capability_filter_events"][1]["status"] == "all_incompatible"
    digest = bank.bounded_digest()
    statuses = {
        event["status"]
        for event in digest["recent_rounds"][0]["candidate_capability_filter_events"]
    }
    assert {"partitioned", "all_incompatible"} <= statuses


def test_empty_candidate_cascade_allows_non_candidate_dependents_to_consume_skip_status(
    tmp_workspace: Workspace,
    tmp_path: Path,
    inline_supervisor: object,
) -> None:
    wf = Workflow(name="empty_candidate_dependent_cleanup", version="1.0.0")
    wf.node(
        "sequence_metrics",
        tool="sequence_analysis",
        inputs={"candidates": "${outputs.prior_round.candidates}"},
    )
    wf.node(
        "dependent_status",
        tool="noop",
        inputs={"payload": "${outputs.sequence_metrics.status}"},
    )

    states, outputs, workflow_state = _run_without_worker(
        tmp_workspace,
        tmp_path,
        wf,
        virtual_outputs={"prior_round": {"candidates": []}},
        supervisor=inline_supervisor,
    )

    assert workflow_state == "completed"
    assert states["sequence_metrics"] == "skipped"
    assert states["dependent_status"] == "completed"
    assert outputs["dependent_status"]["payload"] == "skipped_no_candidates"


def test_runtime_item_count_gate_skips_single_candidate_stream(
    tmp_workspace: Workspace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(executor_mod.CANONICAL_CONSUMERS, "noop", "candidates")
    monkeypatch.setattr(
        executor_mod,
        "candidate_contract_for_inputs",
        lambda tool, inputs: CandidateCapabilityContract(min_candidate_items=2),
    )
    wf = Workflow(name="runtime_min_items_skip", version="1.0.0")
    wf.node(
        "pair_consumer",
        tool="noop",
        inputs={"candidates": "${outputs.prior_round.candidates}"},
    )

    states, outputs, workflow_state = _run_without_worker(
        tmp_workspace,
        tmp_path,
        wf,
        virtual_outputs={
            "prior_round": {
                "candidates": [
                    {
                        "schema_version": "candidate_v1",
                        "candidate_id": "cand_one",
                        "source": {"tool": "rfdiffusion", "local_id": "0"},
                        "structure_path": "/tmp/one.pdb",
                        "capabilities": {"has_backbone_structure": True},
                    }
                ]
            }
        },
    )

    assert workflow_state == "completed"
    assert states["pair_consumer"] == "skipped"
    assert outputs["pair_consumer"]["status"] == "skipped_missing_candidate_capability"
    assert outputs["pair_consumer"]["candidate_capability_filter_event"]["status"] == "all_incompatible"
    assert outputs["pair_consumer"]["candidate_capability_filter_event"][
        "missing_capabilities"
    ] == ["at_least_2_candidate_items"]
    assert outputs["pair_consumer"]["candidate_capability_filter_event"][
        "filtered_candidates"
    ][0]["source"] == {"tool": "rfdiffusion", "local_id": "0"}


def test_runtime_item_count_gate_allows_two_candidate_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(executor_mod.CANONICAL_CONSUMERS, "noop", "candidates")
    monkeypatch.setattr(
        executor_mod,
        "candidate_contract_for_inputs",
        lambda tool, inputs: CandidateCapabilityContract(min_candidate_items=2),
    )
    rt = NodeRuntime(
        desc=WorkflowNode(
            id="pair_consumer",
            tool="noop",
            inputs={"candidates": "${outputs.prior_round.candidates}"},
        ),
        runtime_id="pair_consumer",
    )
    resolved = {
        "candidates": [
            {
                "schema_version": "candidate_v1",
                "candidate_id": "cand_a",
                "source": {"tool": "rfdiffusion", "local_id": "0"},
                "structure_path": "/tmp/a.pdb",
                "capabilities": {"has_backbone_structure": True},
            },
            {
                "schema_version": "candidate_v1",
                "candidate_id": "cand_b",
                "source": {"tool": "rfdiffusion", "local_id": "1"},
                "structure_path": "/tmp/b.pdb",
                "capabilities": {"has_backbone_structure": True},
            },
        ]
    }

    gate, filter_event = executor_mod._candidate_runtime_gate(rt, resolved)

    assert gate is None
    assert filter_event is None
    assert [item["candidate_id"] for item in resolved["candidates"]] == [
        "cand_a",
        "cand_b",
    ]
