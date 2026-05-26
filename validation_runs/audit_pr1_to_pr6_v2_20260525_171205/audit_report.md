# Independent Re-Audit Report: VPA PR #1-#6 Refactor

Audit root: `validation_runs/audit_pr1_to_pr6_v2_20260525_171205/`  
Repository: `/mnt/vast/trillion/hyunjin/VibeProteinAgent`  
Python: `/mnt/vast/trillion/env/miniconda3/envs/vpa/bin/python`  
Mode: read-only source/test audit. I did not modify source code, commit, push, or run live provider matrices.

## 1. Summary

Total findings:

- BLOCKER: 1
- CONCERN: 4
- OBSERVATION: 8
- CLEAN verification sections: 7

Recommendation: **needs follow-up before matrix cycle**.

Comparison vs first audit:

- Prior BLOCKER #1, default v1 next-round routing through legacy `draft_next_round_plan`, is **resolved for the tested automatic next-round path**. `_maybe_compile_planner_ir` is deleted and v1 next-round planning calls `_plan_v1` with `prior_round_context`.
- Prior BLOCKER #2, legacy validators load-bearing on compiled v1 plans, is **resolved for `source_kind == "planner_ir_compiled"`**.
- Prior BLOCKER #3, decision prompt/schema mismatch, is **resolved**. The prompt examples validate and strict `DecisionEngine.decide()` accepts them.
- Prior BLOCKER #4, scope creep, is **partly resolved by clarification**. Parallel-track files are not re-flagged. A smaller residual scope-hygiene concern remains for unlisted test/metadata edits.
- New BLOCKER: first-round Planner IR can still reference virtual `prior_round` even when no prior round exists; this compiles and passes DAG validation, then fails only when runtime resolution looks for missing `outputs.prior_round`.

## 2. Per-PR Re-Verification

PR #1: **CLEAN with one carried concern**.

- `src/vibeproteinagent/schemas/planner_ir_v1.json` locks `version` to `planner_ir_v1`.
- `src/vibeproteinagent/schemas/candidate_v1.json` requires `source.local_id`, rejects deferred fields via `additionalProperties: false`, and enforces producer-namespaced object values for `scores` and `artifacts`.
- `stable_candidate_id(producer_tool, local_id, *, ...)` requires both producer tool and local id; structure hashing uses `Path(structure_path).name`.
- CONCERN carried from first audit: schemas live under `src/vibeproteinagent/schemas/`, not repo-root `schemas/` as named in the contract text. Runtime helpers consistently use the package-local path.

PR #2: **CLEAN**.

- `DecisionEngine.decide()` validates `decision_v1` before normalization.
- `DecisionSchemaViolation` is caught by the controller retry boundary and re-fed as structured retry context.
- `VPA_NORMALIZE_DECISION_REASONING` is per-normalizer; no global normalization mode was found.
- PR #6 default now resolves to strict as expected.

PR #3: **CLEAN for requested rollout**.

- `go_sequences` and `sequence_analysis` emit additive `candidates[]`.
- `rank_designs` accepts all-canonical candidates and rejects mixed canonical/legacy candidate lists.
- `submit_candidate` resolves canonical candidates before legacy rows.
- The canonical rollout remains limited to `go_sequences` and `sequence_analysis`; no new producer rollout was found.

PR #4: **CLEAN with one carried concern**.

- `compile_planner_ir` emits executor IR mechanically and marks it `source_kind: "planner_ir_compiled"`.
- `merge_candidates` remains planner-invisible and audit-visible.
- `FIXUP_KINDS` is exactly `ambiguous_upstream_candidates`, `missing_canonical_input_slot`, and `upstream_does_not_produce_candidates`.
- CONCERN carried from first audit: `merge_candidates` still exposes deferred dedup policies `dedup_by_sequence` and `dedup_by_candidate_id`, beyond the contracted `keep_all` default.

PR #5: **CLEAN**.

- `PLANNER_IR_SYSTEM_PROMPT` is 742 words, within the 700-1100 budget.
- The Planner IR prompt has no `${outputs.`, `${jobs.`, `${item.`, `foreach`, `workflow_ref`, `submit_final`, `kept_sequences`, `affinity_pred_value`, `confidence_score`, or `output_fields` teaching content.
- The prompt mentions `prior_round_context` once.

PR #6: **CLEAN**.

- Defaults: `planner_ir_mode= v1`, `decision_normalization_mode= strict`.
- Deleted symbols remain deleted: `_repair_scalar_inputs_from_output_arrays`, `_fallback_workflow_plan_from_selected_tools`, `_CANDIDATE_OUTPUT_FIELDS`, and `_maybe_compile_planner_ir` have no production hits.
- Legacy prompt/path remains available with `VPA_PLANNER_IR=off`; focused legacy tests pass.

## 3. BLOCKER #2 Fix Verification

Status: **CLEAN**.

- `WorkflowPlanner.validate_plan()` skips only `_validate_static_output_refs`, `_validate_static_output_ref_types`, and `_validate_static_job_refs` when `plan.get("source_kind") == "planner_ir_compiled"`.
- Legacy `draft_plan` output does not carry that marker and still runs the validators.
- Compiler-emitted IR carries `source_kind: "planner_ir_compiled"`.
- Focused spy coverage passed: `tests/test_v1_bypasses_legacy_validators.py` and `tests/test_source_kind_preserved_on_next_round.py`.
- No other validators were silently disabled; DAG validation, internal-tool visibility checks, availability/install checks, RFdiffusion mode readiness, runtime param typing, and concrete input validation remain active.

## 4. BLOCKER #3 Fix Verification

Status: **CLEAN**.

- `DECISION_SYSTEM_PROMPT` no longer teaches per-candidate `reasoning`.
- It no longer teaches always-present branch payloads.
- It no longer teaches `next_round.plan`.
- The four JSON examples in the prompt validate against `src/vibeproteinagent/schemas/decision_v1.json`.
- `DecisionEngine.decide()` accepts each example in strict mode.
- A response with per-candidate `reasoning` or `next_round.plan` triggers `DecisionSchemaViolation` in strict mode and therefore enters structured retry rather than silent normalization.

## 5. BLOCKER #1 Fix Verification

Status: **mostly clean, with one new BLOCKER**.

What is fixed:

- `_maybe_compile_planner_ir` is deleted; grep found only tests asserting absence.
- In v1 mode, automatic next-round routing now calls `planner._plan_v1(...)`, not `draft_next_round_plan`.
- `_plan_v1` accepts keyword-only `prior_round_context: Optional[PriorRoundContext] = None`.
- When context is present, `_planner_ir_prompt()` includes a `prior_round_context` block in the user message.
- `PriorRoundContext` is a frozen dataclass with fields `round_number`, `surviving_candidates`, `prior_decision_reason`, and optional `summary`; it has no file IO.
- `controller._build_executor_scope()` pre-populates `outputs.prior_round` from in-memory state and passes it to `WorkflowExecutor(virtual_outputs=...)`.
- `compiler.py` emits `${outputs.prior_round.candidates}` directly for a single virtual upstream and emits a `merge_<consumer>_candidates` node when `prior_round` and a real producer both feed one consumer.
- `canonical_tools.py` has `prior_round` as a producer only; it is not a consumer.
- `ir.py` accepts `upstream[].node == "prior_round"` and still rejects other unknown upstream ids.
- `decision_v1.json` forbids `next_round.plan` via `additionalProperties: false`.
- No real `prior_round` adapter or registry entry was found.
- No carry-forward state file write was found near `_round_state`; state is held in memory.

**BLOCKER-1: `prior_round` is admissible in first-round Planner IR without any prior round context.**

Evidence:

- `_planner_ir_prompt()` always includes `"prior_round"` in `canonical_wiring.candidate_producers`, even when `prior_round_context` is `None`: `src/vibeproteinagent/agents/planning/workflow_planner.py:425-427`.
- The system prompt qualifies usage as "When continuing from a prior round", but the structured user payload still advertises `prior_round` on first-round planning.
- `validate_planner_ir_v1()` accepts `prior_round` unconditionally as a virtual upstream: `src/vibeproteinagent/agents/planning/ir.py:233-236`.
- `compile_planner_ir()` then emits `${outputs.prior_round.candidates}` and `Workflow.validate_dag()` allows it as a virtual output.
- Reproduction artifact: `validation_runs/audit_pr1_to_pr6_v2_20260525_171205/first_round_prior_round_hazard.log` shows this first-round IR compiles and passes DAG validation, then resolution fails with `InvalidParamsError workflow reference '${outputs.prior_round.candidates}' has no 'prior_round'`.

Impact:

This violates the virtual producer contract that `prior_round` is supplied from the immediately previous round. On the first round there is no previous round, but the Planner IR path can still accept and compile a `prior_round` reference. The failure is deferred to runtime resolution instead of being prevented at prompt construction, schema validation, or compilation.

**CONCERN-1: `_round_state` is not cleared on all round-loop exits.**

Evidence:

- `_round_state` is set before v1 next-round planning at `src/vibeproteinagent/agents/execution/controller.py:701-706`.
- It is cleared when a decision does not recommend another round at `controller.py:698-700`.
- It is not cleared on terminal exits through final submission, max-round fallback, next-round planning failure fallback, or normal return after a second-round workflow with `final_submission`.
- `tests/test_v1_next_round_routing.py:233-238` currently asserts that `controller._round_state` remains populated after a completed two-round run.

Impact:

The state is still in-memory and is reset at the start of the next submitted agent run, so I do not see cross-run execution leakage. It nevertheless falls short of the stated "duration of one agent run" cleanup requirement and should be cleaned up in a `finally` or equivalent once the report is assembled.

## 6. Cross-Coordination Verification

Status: **CLEAN except BLOCKER-1 above**.

- Multi-round v1 fixture passed: first-round `_plan_v1`, `next_round`, second-round `_plan_v1` with `prior_round_context`, `source_kind` preserved, skipped legacy validators, terminal completion.
- `tests/test_source_kind_preserved_on_next_round.py` passed.
- The decision prompt examples remain valid against post-BLOCKER-#1 `decision_v1.json`; no example contains `next_round.plan`.
- Strict default rejects misnested reasoning and `next_round.plan` with structured retry.

## 7. Architectural Integrity

Status: **one BLOCKER, otherwise clean**.

- LLM-facing Planner IR prompt avoids executor references and forbidden executor tokens.
- Compiler owns canonical candidate wiring for v1 executor IR.
- Legacy validators are demoted off compiled v1 plans.
- Strict decision validation is the primary defense before normalization.
- `merge_candidates` is planner-invisible and audit-visible.
- `prior_round` has no adapter, no registry entry, and is not in the tool catalog.
- IN-MEMORY carry-forward is respected for the stored candidate payload.
- BLOCKER: `prior_round` is still valid without a prior-round context, so a first-round plan can compile to an unresolved virtual output reference.

## 8. Scope Creep With Parallel-Track Guard

Status: **no re-flag of clarified parallel-track files; residual CONCERNS remain**.

Not re-flagged per clarification:

- `src/vibeproteinagent/adapters/literature_search/`
- `src/vibeproteinagent/adapters/_remote/europepmc.py`
- `src/vibeproteinagent/reporting/render.py`
- `src/vibeproteinagent/agents/provider/openrouter.py`
- `src/vibeproteinagent/workflows/executor.py` parallel-track portion, with the additive `virtual_outputs` portion treated as BLOCKER #1 support.

**CONCERN-2: residual unlisted scope-hygiene changes remain.**

- `tests/test_adapter_mpnn_to_protenix_input.py` changes tuple-record assertions to dict-record assertions. This is outside PR #1-#6 allowed file lists and outside the parallel-track clarification.
- `validation_runs/logs/.gitkeep` is deleted in the working tree. This is not source behavior, but it is unrelated metadata churn.
- `tests/test_agent_layer.py` contains broad legacy/off fixture and planner normalizer additions. Much of this appears related to preserving legacy behavior after the default flip, but it is outside the explicit blocker-fix file list and should be owned/acknowledged.

Other scope checks:

- No new fixup kinds beyond the three declared.
- No new normalizer shims beyond `decision_reasoning`.
- No `raw_inputs` escape hatch.
- No `wiring` override field in Planner IR.
- No new schema versions such as `candidate_v2`, `planner_ir_v1_1`, or `decision_v3`.
- No new canonical producer rollout beyond `go_sequences` and `sequence_analysis`.
- `reporting/render.py` changes are literature-search rendering changes matching the clarified parallel track.

## 9. Test Execution Results

Focused blocker/default command:

- Log: `validation_runs/audit_pr1_to_pr6_v2_20260525_171205/blocker_and_default.log`
- Result: `24 passed in 52.56s`

Defaults command:

- Log: `validation_runs/audit_pr1_to_pr6_v2_20260525_171205/defaults.log`
- Result:
  - `planner_ir_mode= v1`
  - `decision_normalization_mode= strict`

Full suite:

- Log: `validation_runs/audit_pr1_to_pr6_v2_20260525_171205/full_suite.log`
- Result: `1129 passed, 17 skipped, 90 warnings in 1745.17s (0:29:05)`
- This matches the expected baseline of at least 1129 passed and no new skips beyond the documented 17.

Live provider matrix:

- Not run, per audit instruction. Existing PR #5/#6 matrix runs remain known BLOCKED due auth/preflight and are not reinterpreted here.

## 10. Risk Classification

BLOCKER:

- `prior_round` can be emitted, validated, and compiled on first-round v1 planning without prior context, leading to a runtime missing-reference failure.

CONCERN:

- `_round_state` is not cleared on all round-loop exits; tests currently assert lingering state after a completed multi-round run.
- Residual unlisted scope-hygiene changes in `tests/test_adapter_mpnn_to_protenix_input.py`, `tests/test_agent_layer.py`, and `validation_runs/logs/.gitkeep`.
- `merge_candidates` still includes deferred dedup policies beyond `keep_all`.
- Schema path mismatch remains: implementation uses `src/vibeproteinagent/schemas/`, while the contract text names repo-root `schemas/`.

OBSERVATION:

- Prior BLOCKER #1 routing fix is effective for covered automatic v1 next-round flows.
- Prior BLOCKER #2 validator demotion is effective and narrowly scoped.
- Prior BLOCKER #3 decision prompt/schema sync is effective.
- Virtual `prior_round` is not a real adapter and not registry-visible.
- Carry-forward payload is in-memory and not serialized as carry-forward state.
- Full suite matches expected pass/skip baseline.
- Strict defaults are active.
- Legacy `VPA_PLANNER_IR=off` escape hatch still smoke-passes.

## 11. Recommendation

Recommendation: **needs follow-up before matrix cycle**.

The three prior code blockers are substantially fixed and the full suite is clean. However, the virtual `prior_round` abstraction has one remaining architectural hole: it is admissible without an actual prior round. Before spending a live provider matrix cycle, I recommend gating `prior_round` availability to calls with `prior_round_context`, or rejecting `prior_round` during validation/compilation when no context exists. The `_round_state` cleanup concern is smaller but should be cleaned up at the same time.
