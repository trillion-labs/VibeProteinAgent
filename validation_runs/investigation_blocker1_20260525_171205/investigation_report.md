# BLOCKER #1 Read-Only Investigation

Repository: `/mnt/vast/trillion/hyunjin/VibeProteinAgent`  
Audit input: `validation_runs/audit_pr1_to_pr6_20260525_171205/audit_report.md`  
Contracts input: `codex_notes/pr1_6_contracts.md`  
Mode: read-only source/log investigation. No code edits, no provider matrix runs.

## 1. Terminology resolution (Phase 1)

The audit's "next-round" refers to **(c) the multi-round agentic loop where the decision engine picks `next_round` and the controller starts a new planning cycle**, not a parser retry and not the next iteration of `_run_llm_entry`.

Evidence:

- `src/vibeproteinagent/agents/execution/controller.py:539` starts the multi-round loop: `for round_index in range(max(1, self.config.max_rounds))`.
- `src/vibeproteinagent/agents/execution/controller.py:583-599` asks the decision engine after a completed workflow round.
- `src/vibeproteinagent/agents/execution/controller.py:673-694` reads `decision["next_round"]`, applies the max-round guard, records the decision, and breaks when `recommended` is false.
- `src/vibeproteinagent/agents/execution/controller.py:695-739` is the planning branch taken when `next_round.recommended` is true. If the decision does not already include a concrete plan, lines `711-727` call `planner.draft_next_round_plan(...)`; lines `728-734` then call `_maybe_compile_planner_ir(...)`.

So the bug is in the controller's automatic round N+1 planning after a completed round N decision recommends another round.

## 2. First-round trace with file:line refs (Phase 2)

Step-by-step first-round planning path:

1. Agent-plan entry point: `src/vibeproteinagent/agents/execution/controller.py:122-138` routes non-direct/non-explicit requests into `await self._run_agent_plan(...)`.
2. Controller initializes the planner IR mode once in `AgentController.__init__`: `src/vibeproteinagent/agents/execution/controller.py:69` assigns `self.planner_ir_mode = resolve_planner_ir_mode()`.
3. The default mode is v1: `src/vibeproteinagent/agents/config.py:129-135` returns `"v1"` when `VPA_PLANNER_IR` is unset, empty, or explicitly `"v1"`.
4. `_run_agent_plan` creates `draft_initial_plan`: `src/vibeproteinagent/agents/execution/controller.py:475-509`.
5. The first-round dispatch checks the stored mode at `src/vibeproteinagent/agents/execution/controller.py:480`. With `VPA_PLANNER_IR=v1`, it calls `planner._plan_v1(...)` at `src/vibeproteinagent/agents/execution/controller.py:490-496`. With mode off, it calls the legacy `planner.draft_plan(...)` at lines `497-503`.
6. `_plan_v1` is documented to return executor IR: `src/vibeproteinagent/agents/planning/workflow_planner.py:1078-1090`.
7. `_plan_v1` builds the Planner IR request payload using `_planner_ir_prompt(...)`: `src/vibeproteinagent/agents/planning/workflow_planner.py:1101-1106`; the helper's output contract is explicitly Planner IR v1 at `src/vibeproteinagent/agents/planning/workflow_planner.py:409-456`.
8. `_plan_v1` uses the v1 system prompt, not the legacy planner prompt: `src/vibeproteinagent/agents/planning/workflow_planner.py:1107-1118` sends `ChatMessage(role="system", content=PLANNER_IR_SYSTEM_PROMPT)`.
9. `PLANNER_IR_SYSTEM_PROMPT` tells the LLM to emit `version: "planner_ir_v1"`, semantic nodes, `upstream`, and `relationship: "candidate_source"`: `src/vibeproteinagent/agents/planning/prompts.py:106-225`.
10. Schema parsing and validation happen before compilation: `src/vibeproteinagent/agents/planning/workflow_planner.py:1119-1122` calls `_load_planner_ir_json_or_raise(...)` and `_validate_planner_ir_v1_or_raise(...)`; those helpers are defined at `src/vibeproteinagent/agents/planning/workflow_planner.py:256-292`.
11. The compiler call is inside `_plan_v1`: if the controller supplied fixup transport, lines `1134-1135` call it; otherwise line `1136` directly calls `compile_planner_ir(planner_ir)`.
12. The controller-supplied fixup transport is created at `src/vibeproteinagent/agents/execution/controller.py:481-488` and calls `_compile_with_fixup(...)`. `_compile_with_fixup` calls `compile_planner_ir(current_ir)` at `src/vibeproteinagent/agents/execution/controller.py:938-950`.
13. `compile_planner_ir` validates Planner IR v1 and returns an executor agent-plan shape with `mode: "agent_plan"` and `workflow_plan.nodes`: `src/vibeproteinagent/agents/planning/compiler.py:21-54`.
14. The compiled executor plan is validated as a normal invocation at `src/vibeproteinagent/agents/execution/controller.py:504-508`, returned from `_run_llm_entry` at `src/vibeproteinagent/agents/execution/controller.py:511-515`, revalidated at `src/vibeproteinagent/agents/execution/controller.py:516-520`, and stored as `status="validated"` at `src/vibeproteinagent/agents/execution/controller.py:521`.
15. Execution receives this executor `workflow_plan`: the round loop passes `current_plan` to `_execute_validated_plan` at `src/vibeproteinagent/agents/execution/controller.py:539-547`; `_execute_validated_plan` validates and compiles it into a workflow at `src/vibeproteinagent/agents/execution/controller.py:1002-1037`.

Conclusion: first-round v1 planning is internally consistent. The LLM emits Planner IR, the schema validator validates Planner IR, the compiler compiles Planner IR, and the executor receives the compiled executor `workflow_plan`.

## 3. Next-round trace with file:line refs (Phase 3)

Step-by-step automatic next-round path:

1. Round execution begins at `src/vibeproteinagent/agents/execution/controller.py:539-547`.
2. After a completed workflow without an explicit `final_submission`, the controller asks the decision engine at `src/vibeproteinagent/agents/execution/controller.py:583-599`.
3. The next-round decision payload is read at `src/vibeproteinagent/agents/execution/controller.py:673`; if the recommendation is false, the loop exits at `src/vibeproteinagent/agents/execution/controller.py:693-694`.
4. If the decision includes a concrete plan in `next_round.plan` or `next_round.workflow_plan`, the controller validates it directly as an executor plan at `src/vibeproteinagent/agents/execution/controller.py:695-709`. This branch does not call `_maybe_compile_planner_ir`.
5. If there is no concrete decision-supplied plan, the controller calls `_run_llm_entry` for `phase=f"next_round_plan_{round_index}"` at `src/vibeproteinagent/agents/execution/controller.py:711-727`.
6. The method called inside that entry is hardcoded: `planner.draft_next_round_plan(...)` at `src/vibeproteinagent/agents/execution/controller.py:719-726`. There is no `self.planner_ir_mode == "v1"` dispatch at this call site.
7. `WorkflowPlanner.draft_next_round_plan` is the legacy next-round helper: `src/vibeproteinagent/agents/planning/workflow_planner.py:1260-1356`.
8. It performs tool selection via `_select_candidate_tools(...)` at `src/vibeproteinagent/agents/planning/workflow_planner.py:1296-1314` using `TOOL_SELECTION_SYSTEM_PROMPT`; the selection helper sends that prompt at `src/vibeproteinagent/agents/planning/workflow_planner.py:1372-1380`.
9. It then constructs an executor-plan output contract at `src/vibeproteinagent/agents/planning/workflow_planner.py:1319-1341`. The shape is `mode: "agent_plan"` with `workflow_plan.nodes[].inputs`, not Planner IR v1 with `version`, semantic `nodes[].upstream`, or `finalization`.
10. The next-round LLM call uses `NEXT_ROUND_SYSTEM_PROMPT`, not `PLANNER_IR_SYSTEM_PROMPT`: `src/vibeproteinagent/agents/planning/workflow_planner.py:1342-1350`.
11. `NEXT_ROUND_SYSTEM_PROMPT` explicitly asks for legacy executor shape: `src/vibeproteinagent/agents/planning/prompts.py:255-266` says to return an explicit `workflow_plan`, match the `agent_plan workflow plan contract`, and even mentions `foreach` and `max_parallel`.
12. The helper parses generic JSON and validates it as an executor round plan: `src/vibeproteinagent/agents/planning/workflow_planner.py:1351-1356` calls `_load_json_object_response(...)` and `self.validate_round_plan(plan)`.
13. `validate_round_plan` delegates to `validate_plan`: `src/vibeproteinagent/agents/planning/workflow_planner.py:1547-1550`. `validate_plan` requires a `workflow_plan` object and non-empty `workflow_plan.nodes`: `src/vibeproteinagent/agents/planning/workflow_planner.py:1461-1494`, then compiles it as executor workflow at lines `1524-1528`.
14. After that executor plan returns to the controller, the controller calls `_maybe_compile_planner_ir(current_plan, ...)` at `src/vibeproteinagent/agents/execution/controller.py:728-734`.
15. `_maybe_compile_planner_ir` compiles when `self.planner_ir_mode == "v1"`: `src/vibeproteinagent/agents/execution/controller.py:923-936`.
16. `_compile_with_fixup` passes that same object to `compile_planner_ir(current_ir)`: `src/vibeproteinagent/agents/execution/controller.py:938-950`.
17. Only after this attempted compile does the controller run `planner.validate_invocation(...)`: `src/vibeproteinagent/agents/execution/controller.py:735-739`.

Conclusion: the automatic next-round path emits and validates legacy executor IR, then the controller treats that same object as Planner IR v1 because the global mode is v1.

## 4. Divergence point

The first-round and next-round paths diverge in `src/vibeproteinagent/agents/execution/controller.py`.

- First round: `src/vibeproteinagent/agents/execution/controller.py:480-496` checks `self.planner_ir_mode == "v1"` before the LLM planner call and dispatches to `WorkflowPlanner._plan_v1(...)`.
- Automatic next round: `src/vibeproteinagent/agents/execution/controller.py:711-727` calls `WorkflowPlanner.draft_next_round_plan(...)` unconditionally when the decision did not include a concrete plan. The mode check happens too late, after the helper has already returned a validated legacy executor plan, at `src/vibeproteinagent/agents/execution/controller.py:728-734`.

That late compile is the point where legacy executor output is reinterpreted as Planner IR.

## 5. Compile-legacy-as-IR call sites (Phase 4)

Production `compile_planner_ir` call sites:

1. `src/vibeproteinagent/agents/planning/workflow_planner.py:1134-1136`: `_plan_v1` compiles `planner_ir` after `_load_planner_ir_json_or_raise(...)` and `_validate_planner_ir_v1_or_raise(...)` at lines `1119-1122`. Input shape is Planner IR v1. This call site is correct.
2. `src/vibeproteinagent/agents/execution/controller.py:938-950`: `_compile_with_fixup` compiles its `llm_ir`/`current_ir` argument. Input shape depends on caller:
   - Initial v1 caller: `src/vibeproteinagent/agents/execution/controller.py:481-488` supplies this as `_plan_v1`'s `compile_with_fixup`, so the input is validated Planner IR v1. Correct.
   - Automatic next-round caller: `src/vibeproteinagent/agents/execution/controller.py:728-734` passes `current_plan`, which just came from `planner.draft_next_round_plan(...)` at lines `711-727`. That helper returns `self.validate_round_plan(plan)` at `src/vibeproteinagent/agents/planning/workflow_planner.py:1351-1356`, so the input is an executor `agent_plan` with `workflow_plan`, not Planner IR v1. Incorrect.
3. `src/vibeproteinagent/agents/execution/controller.py:923-936`: `_maybe_compile_planner_ir` gates the call to `_compile_with_fixup` on `self.planner_ir_mode`. It does not inspect the shape of `llm_output`. In the next-round path, this lets a validated executor plan reach the Planner IR compiler.

Compiler shape expectation:

- `src/vibeproteinagent/agents/planning/compiler.py:21-30` immediately calls `validate_planner_ir_v1(llm_ir)`.
- `src/vibeproteinagent/agents/planning/ir.py:26-35` validates against `planner_ir_v1.json` and rejects executor-only syntax and forbidden tokens.
- The audit artifact `validation_runs/audit_pr1_to_pr6_20260525_171205/next_round_compile_mismatch_check.log` contains `ValidationError` and `'version' is a required property`, which is exactly what happens when an executor `agent_plan` lacking top-level `version: "planner_ir_v1"` is passed to this compiler.

Concrete bad call chain:

`decision.next_round.recommended == true` -> no `next_round.plan` in the decision -> `controller.py:711-727` -> `WorkflowPlanner.draft_next_round_plan` -> `workflow_planner.py:1319-1356` returns executor `workflow_plan` -> `controller.py:728-734` -> `_maybe_compile_planner_ir` -> `controller.py:923-936` -> `_compile_with_fixup` -> `controller.py:938-950` -> `compile_planner_ir(executor_agent_plan)`.

## 6. Root cause case classification (Phase 5)

Classification: **CASE E - Combination**, specifically **CASE B + CASE C**, with a small design question around next-round Planner IR context.

Evidence for CASE B - wrong dispatch/helper unaware of the flag:

- First-round planning has explicit mode routing at `src/vibeproteinagent/agents/execution/controller.py:480-503`.
- Automatic next-round planning hardcodes `planner.draft_next_round_plan(...)` at `src/vibeproteinagent/agents/execution/controller.py:719-726`.
- `draft_next_round_plan` is a legacy helper that constructs an executor `workflow_plan` output contract at `src/vibeproteinagent/agents/planning/workflow_planner.py:1319-1341` and uses `NEXT_ROUND_SYSTEM_PROMPT` at `src/vibeproteinagent/agents/planning/workflow_planner.py:1342-1350`.

Evidence for CASE C - shape confusion:

- After `draft_next_round_plan` has already parsed and validated an executor `agent_plan`, the controller calls `_maybe_compile_planner_ir` at `src/vibeproteinagent/agents/execution/controller.py:728-734`.
- `_maybe_compile_planner_ir` only checks the mode flag, not the actual shape: `src/vibeproteinagent/agents/execution/controller.py:923-936`.
- `_compile_with_fixup` then calls `compile_planner_ir(current_ir)` at `src/vibeproteinagent/agents/execution/controller.py:938-950`, and the compiler requires Planner IR v1 at `src/vibeproteinagent/agents/planning/compiler.py:21-30`.

Why not pure CASE A:

- It is not only that a first-round `if planner_ir_mode == "v1"` check was omitted. A late compile hook was added after the legacy helper, creating a second bug: even if the helper remains legacy, compiling its returned executor plan is wrong.

Why not full CASE D based on current evidence:

- The existing Planner IR v1 schema can express a new within-round topology through `nodes[].upstream`. Nothing in the traced automatic next-round path proves that round N+1 must directly reference round N executor outputs. The current legacy next-round prompt receives prior summary and decision context, but it still drafts a fresh `workflow_plan`. That suggests the immediate failure is routing/shape handling, not a proven IR expressiveness blocker.
- There is still an open design question about whether a high-quality next-round v1 planner needs structured previous-plan/summary/decision inputs. That question changes fix quality, but the observed crash does not require a new cross-round reference syntax to explain it.

## 7. Fix-scope recommendation (Phase 6)

Recommendation: **Cross-cutting fix**, not a one-line surgical patch and not a full design rework.

Likely modules:

- `src/vibeproteinagent/agents/execution/controller.py`: move next-round v1 routing before the LLM next-round planner call, and remove/avoid the late compile of already-validated executor plans.
- `src/vibeproteinagent/agents/planning/workflow_planner.py`: add or adapt a v1 next-round planner entry point that uses `PLANNER_IR_SYSTEM_PROMPT`, validates Planner IR v1, and compiles through the same fixup transport as `_plan_v1`.
- Tests: add a routing regression that drives `decision.next_round.recommended == true` with no concrete `next_round.plan` under default/v1 mode and asserts that the next-round LLM prompt is Planner IR v1, not `NEXT_ROUND_SYSTEM_PROMPT`, and that `compile_planner_ir` receives Planner IR rather than executor `workflow_plan`.

This is cross-cutting because the correct fix must preserve all three invariants at once:

- the LLM-facing next-round default path should use the PR #5 Planner IR prompt,
- the compiler should only receive Planner IR v1,
- legacy executor `draft_next_round_plan` should remain usable only in the legacy/off path or explicitly validated escape-hatch branches.

A minimal "do not compile `draft_next_round_plan` output" change would stop the immediate crash, but it would leave default v1 next-round planning on a legacy LLM-facing `workflow_plan` prompt and would still violate the PR #1-#6 contract. A minimal "compile after legacy helper" change is the current broken behavior. The fix likely fits in a few files but needs a real next-round v1 route, so I would not classify it as a purely surgical patch unless the architect accepts reusing `_plan_v1` with a synthesized next-round request string as sufficient context.

## 8. Adjacent risks (Phase 7)

- Decision-supplied concrete next-round plans bypass Planner IR v1 entirely. `src/vibeproteinagent/agents/execution/controller.py:695-709` accepts `next_round.plan` or `next_round.workflow_plan` and validates it as executor IR. This may be an intentional escape hatch, but it is another default-path route where executor `workflow_plan` can enter a v1 run without the Planner IR prompt/compiler.
- The legacy `draft_next_round_plan` method remains tested as an executor-plan producer in `tests/test_agent_layer.py:2021-2075`; this test is useful for legacy mode but does not cover default/v1 next-round routing.
- Current routing tests cover only initial planner routing. `tests/test_controller_planner_routing.py:65-158` asserts `_plan_v1` vs `draft_plan` for first-round plan-only execution, but there is no analogous default/v1 test for automatic next-round planning.
- The fixup transport from PR #4 is reachable from the broken next-round path. `src/vibeproteinagent/agents/execution/controller.py:728-734` calls `_maybe_compile_planner_ir`, which can enter `_compile_with_fixup` at `src/vibeproteinagent/agents/execution/controller.py:938-960`. In this failure mode, the bad shape usually fails schema validation before structured fixup can help, as shown by the audit log's missing `version` validation error.
- BLOCKER #2 is adjacent but separate. The same first-round and next-round v1 paths call `planner.validate_invocation(...)` after compilation at `src/vibeproteinagent/agents/execution/controller.py:504-508`, `516-520`, and `735-739`; `validate_plan` still runs `_validate_static_output_refs`, `_validate_static_output_ref_types`, and `_validate_static_job_refs` at `src/vibeproteinagent/agents/planning/workflow_planner.py:1524-1528`. That is connected to the same routing surface after compilation, but it is not the compile-legacy-as-IR cause.
- Parser/schema retry semantics differ between first-round v1 and next-round legacy. `_plan_v1` catches `PlannerIRSchemaViolation` internally and retries with Planner IR structured error context at `src/vibeproteinagent/agents/planning/workflow_planner.py:1122-1133`. `draft_next_round_plan` returns generic `InvalidParamsError` from executor validation at `src/vibeproteinagent/agents/planning/workflow_planner.py:1351-1356`, which `_run_llm_entry` can retry only as a generic VPA error at `src/vibeproteinagent/agents/execution/controller.py:897-917`.

## 9. Open questions for the user/architect

1. Should automatic v1 next-round planning reuse `PLANNER_IR_SYSTEM_PROMPT` with an enriched user payload containing `previous_plan`, `compact_summary`, and `decision`, or should it have a separate next-round Planner IR prompt constant that still emits exactly Planner IR v1?
2. Should decision-supplied `next_round.plan` remain an executor-plan escape hatch in default v1 mode, or must it also be Planner IR v1 / compiled before execution?
3. Does round N+1 ever need to reference artifacts or outputs from round N directly in the executable workflow, or is prior-round information only advisory context for drafting a fresh plan?
