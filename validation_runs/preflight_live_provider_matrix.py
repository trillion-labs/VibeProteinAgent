#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vibeproteinagent.agents.execution.candidate_archive import CandidateArchive
from vibeproteinagent.agents.execution.prompt_context import PromptContextBuilder
from vibeproteinagent.agents.planning.compiler import compile_planner_ir
from vibeproteinagent.agents.session import AgentSessionStore
from vibeproteinagent.agents.summarization.result_summarizer import decision_safe_summary
from vibeproteinagent.core.errors import InvalidParamsError
from vibeproteinagent.workspace import Workspace
from vibeproteinagent.workflows.parameter_resolution import normalize_tool_parameters

MATRIX_SCRIPT = REPO_ROOT / "validation_runs" / "run_live_provider_matrix_vm024.sh"
TARGET_PDB = REPO_ROOT / "tests" / "fixtures" / "6MRR.pdb"
AUX_STREAM_FIELDS = {"predictions", "analyses", "qc_results"}
RAW_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"
RAW_STRUCTURE_PATH = "/tmp/preflight/private/full_structure.pdb"


def main() -> int:
    checks: list[dict[str, Any]] = []
    for name, fn in (
        ("live_matrix_shell_script", _check_shell_script),
        ("live_matrix_target_artifact", _check_target_artifact),
        ("rank_designs_aux_empty_literal_cleanup", _check_rank_aux_cleanup),
        ("candidate_archive_materialization", _check_candidate_archive),
        ("summary_and_prompt_redaction", _check_summary_prompt_redaction),
    ):
        try:
            detail = fn()
        except Exception as exc:  # noqa: BLE001 - preflight should report every boundary.
            checks.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        else:
            checks.append({"name": name, "status": "PASS", "detail": detail or {}})

    result = {
        "ok": all(check["status"] == "PASS" for check in checks),
        "repo_root": str(REPO_ROOT),
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def _check_shell_script() -> dict[str, Any]:
    proc = subprocess.run(
        ["bash", "-n", str(MATRIX_SCRIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.strip() or proc.stdout.strip())
    return {"script": str(MATRIX_SCRIPT)}


def _check_target_artifact() -> dict[str, Any]:
    if not TARGET_PDB.is_file():
        raise AssertionError(f"missing target PDB: {TARGET_PDB}")
    size = TARGET_PDB.stat().st_size
    if size <= 0:
        raise AssertionError(f"empty target PDB: {TARGET_PDB}")
    return {"target_pdb": str(TARGET_PDB), "size_bytes": size}


def _check_rank_aux_cleanup() -> dict[str, Any]:
    plan = compile_planner_ir(
        _rank_ir(
            {
                "predictions": [],
                "analyses": [],
                "qc_results": [],
            }
        )
    )
    rank = _node(plan, "rank")
    for field in AUX_STREAM_FIELDS:
        if field in rank.get("inputs", {}):
            raise AssertionError(f"{field} empty planner literal was not dropped")
        if field in rank.get("input_provenance", {}):
            raise AssertionError(f"{field} provenance was not dropped")
    if rank["inputs"].get("candidates") != "${outputs.search_sequence.candidates}":
        raise AssertionError("rank_designs.candidates was not compiler-wired")
    if rank["input_provenance"].get("candidates") != "compiled_wiring":
        raise AssertionError("rank_designs.candidates provenance is not compiled_wiring")

    normalize_tool_parameters(
        tool="rank_designs",
        inputs={"candidates": [], "top_k": rank["inputs"]["top_k"]},
        provenance=rank["input_provenance"],
        original_inputs=rank["inputs"],
    )

    non_empty = compile_planner_ir(
        _rank_ir({"qc_results": [{"id": "qc_1", "flags": []}]})
    )
    bad_rank = _node(non_empty, "rank")
    if "qc_results" not in bad_rank["inputs"]:
        raise AssertionError("non-empty planner literal qc_results was incorrectly dropped")
    try:
        normalize_tool_parameters(
            tool="rank_designs",
            inputs={
                "candidates": [],
                "top_k": bad_rank["inputs"]["top_k"],
                "qc_results": bad_rank["inputs"]["qc_results"],
            },
            provenance=bad_rank["input_provenance"],
            original_inputs=bad_rank["inputs"],
        )
    except InvalidParamsError as exc:
        event = exc.details.get("parameter_conflict_event", {})
        if event.get("field") != "qc_results":
            raise AssertionError(f"unexpected ownership event: {event}") from exc
    else:
        raise AssertionError("non-empty planner literal qc_results did not fail ownership")

    preserved = compile_planner_ir(
        _rank_ir(
            {"qc_results": []},
            input_provenance={
                "top_k": "planner_literal",
                "qc_results": "compiled_wiring",
            },
        )
    )
    preserved_rank = _node(preserved, "rank")
    if preserved_rank["inputs"].get("qc_results") != []:
        raise AssertionError("compiled_wiring empty qc_results was incorrectly dropped")
    return {
        "dropped_empty_fields": sorted(AUX_STREAM_FIELDS),
        "preserved_non_planner_empty": "qc_results",
        "non_empty_literal_rejects": True,
    }


def _check_candidate_archive() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vpa-live-preflight-") as tmp:
        os.environ["VPA_HOME"] = str(Path(tmp) / "home")
        os.environ["VPA_RUNTIME_DIR"] = str(Path(tmp) / "runtime")
        workspace = Workspace.create("preflight")
        store = AgentSessionStore(workspace)
        run_id = store.create_run("preflight live provider matrix invariants").run_id
        archive = CandidateArchive(workspace, session_store=store)
        good_snapshot = archive.ingest_workflow_round(
            run_id=run_id,
            round_index=0,
            workflow_id="wf_good",
            raw_workflow_summary=_workflow_summary(
                "wf_good",
                {
                    "candidate_id": "cand_good",
                    "sequence": RAW_SEQUENCE,
                    "scores": {"sequence_analysis": {"gravy": -0.1}},
                    "artifacts": {"sequence_analysis": {"summary_path": "/tmp/raw.json"}},
                },
            ),
        )
        bad_snapshot = archive.ingest_workflow_round(
            run_id=run_id,
            round_index=1,
            workflow_id="wf_bad",
            raw_workflow_summary=_workflow_summary(
                "wf_bad",
                {
                    "candidate_id": "cand_bad",
                    "sequence": "FGHI",
                    "scores": "<dict>",
                },
            ),
        )
        survivors = archive.merge_survivors(
            run_id=run_id,
            round_index=1,
            previous_snapshot_id=good_snapshot.snapshot_id,
            current_snapshot_id=bad_snapshot.snapshot_id,
        )

        reloaded = CandidateArchive(workspace, session_store=AgentSessionStore(workspace))
        materialized = reloaded.materialize(survivors.snapshot_id)
        if [candidate.get("candidate_id") for candidate in materialized] != ["cand_good"]:
            raise AssertionError(f"unexpected survivor materialization: {materialized}")
        scores = materialized[0].get("scores")
        if not isinstance(scores, dict) or scores["sequence_analysis"]["gravy"] != -0.1:
            raise AssertionError("nested scores were not preserved")
        executor_outputs = reloaded.materialize_executor_outputs(
            snapshot_id=survivors.snapshot_id,
            round_number=1,
            reason="preflight",
            summary={},
        )
        prior_candidates = executor_outputs["outputs"]["prior_round"]["candidates"]
        if prior_candidates != materialized:
            raise AssertionError("executor outputs did not materialize archive snapshot")
        failure_fields = {
            card.get("field") for card in reloaded.failure_cards(bad_snapshot.snapshot_id)
        }
        if "scores" not in failure_fields:
            raise AssertionError("compact scores rejection was not surfaced as a failure card")
        return {
            "survivor_snapshot_id": survivors.snapshot_id,
            "materialized_count": len(materialized),
            "failure_fields": sorted(field for field in failure_fields if field),
        }


def _check_summary_prompt_redaction() -> dict[str, Any]:
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_secret",
        "sequence": RAW_SEQUENCE,
        "structure_path": RAW_STRUCTURE_PATH,
        "scores": {"rank_designs": {"ranking_score": 0.91}},
        "source": {"tool": "rank_designs", "node_id": "rank", "job_id": "job_rank"},
        "capabilities": {"has_sequence": True, "has_full_structure": True},
    }
    summary = decision_safe_summary({"ranked_candidates": [candidate]})
    _assert_no_raw_candidate_payload(summary)
    card = summary["ranked_candidates"][0]
    if card.get("sequence_length") != len(RAW_SEQUENCE):
        raise AssertionError(f"summary card missing sequence_length: {card}")
    if card.get("score_keys") != ["rank_designs"]:
        raise AssertionError(f"summary card missing score_keys: {card}")
    if card.get("has_structure_path") is not True:
        raise AssertionError(f"summary card missing structure path flag: {card}")

    prompt_context = PromptContextBuilder().build(
        previous_digest=None,
        latest_round_record={
            "round_index": 0,
            "workflow_id": "wf_prompt",
            "state": "completed",
            "current_round_candidate_summary": {
                "count": 1,
                "items": [card],
                "truncated": False,
            },
            "survivor_pool_summary": {
                "count": 1,
                "items": [card],
                "truncated": False,
            },
            "current_round_merge_summary": {"output_count": 1},
            "survivor_merge_summary": {"output_count": 1},
            "capability_summary": {
                "current_round": {
                    "count": 1,
                    "capability_counts": {"has_sequence": 1, "has_full_structure": 1},
                },
                "survivor_pool": {
                    "count": 1,
                    "capability_counts": {"has_sequence": 1, "has_full_structure": 1},
                },
            },
            "workflow_summary_raw": {"workflow_id": "wf_prompt", "nodes": []},
            "failure_cards": [],
        },
    )
    _assert_no_raw_candidate_payload(prompt_context)
    return {"candidate_card_keys": sorted(card), "prompt_context_keys": sorted(prompt_context)}


def _rank_ir(
    aux_inputs: dict[str, Any],
    *,
    input_provenance: dict[str, str] | None = None,
) -> dict[str, Any]:
    rank_node: dict[str, Any] = {
        "id": "rank",
        "tool": "rank_designs",
        "intent": "Rank candidate stream.",
        "inputs": {"top_k": 3, **aux_inputs},
        "upstream": [{"node": "search_sequence", "relationship": "candidate_source"}],
    }
    if input_provenance is not None:
        rank_node["input_provenance"] = input_provenance
    return {
        "version": "planner_ir_v1",
        "goal": "Preflight candidate-stream ownership.",
        "assumptions": [],
        "missing_inputs": [],
        "nodes": [
            {
                "id": "search_sequence",
                "tool": "go_sequences",
                "intent": "Retrieve candidates.",
                "inputs": {"go_id": "GO:0003824"},
                "upstream": [],
            },
            rank_node,
        ],
        "finalization": {"success_criteria": {}},
    }


def _node(plan: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in plan["workflow_plan"]["nodes"]:
        if node.get("id") == node_id:
            return node
    raise AssertionError(f"node not found: {node_id}")


def _workflow_summary(workflow_id: str, candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "workflow_id": workflow_id,
        "state": "completed",
        "nodes": [
            {
                "node_id": "analysis",
                "tool": "sequence_analysis",
                "job_id": f"job_{workflow_id}",
                "state": "completed",
                "output": {"representative_sequences": [candidate]},
            }
        ],
    }


def _assert_no_raw_candidate_payload(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True)
    forbidden_values = {
        RAW_SEQUENCE,
        RAW_STRUCTURE_PATH,
        "ranking_score",
        '"scores":',
        '"sequence":',
        '"structure_path":',
        '"full_structure_path":',
        '"backbone_path":',
    }
    leaked = sorted(item for item in forbidden_values if item in serialized)
    if leaked:
        raise AssertionError(f"raw candidate payload leaked: {leaked}")


if __name__ == "__main__":
    raise SystemExit(main())
