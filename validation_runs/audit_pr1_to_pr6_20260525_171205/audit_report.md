# Independent Audit Report: PR #1 through PR #6

Audit root: `validation_runs/audit_pr1_to_pr6_20260525_171205/`
Repository: `/mnt/vast/trillion/hyunjin/VibeProteinAgent`
Python: `/mnt/vast/trillion/env/miniconda3/envs/vpa/bin/python`
Mode: read-only audit of source, tests, git history, and captured matrix outputs. No source files were modified.

## 1. Summary

Total findings:

- BLOCKER: 4
- CONCERN: 5
- OBSERVATION: 5

Recommendation: **needs follow-up before release**.

The unit and full test suite pass now, but the audit found contract/invariant violations in the default v1 planner path and scope boundaries. The highest-risk issue is that the initial v1 planner path uses `PLANNER_IR_SYSTEM_PROMPT`, but automatic next-round planning still calls the legacy next-round prompt and then attempts to compile that legacy executor-plan shape as Planner IR.

## Findings

### BLOCKER-1: Default v1 next-round planning bypasses the PR #5 Planner IR prompt path

Evidence:

- `src/vibeproteinagent/agents/execution/controller.py:711-728` calls `planner.draft_next_round_plan(...)`, which uses `NEXT_ROUND_SYSTEM_PROMPT`, then immediately calls `_maybe_compile_planner_ir(...)` when `planner_ir_mode == "v1"`.
- `src/vibeproteinagent/agents/planning/workflow_planner.py:255-266` shows `NEXT_ROUND_SYSTEM_PROMPT` asks for an explicit legacy `workflow_plan`, including legacy terms such as `foreach`.
- `validation_runs/audit_pr1_to_pr6_20260525_171205/next_round_compile_mismatch_check.log` confirms `compile_planner_ir()` rejects a legacy `agent_plan` shape with `ValidationError: 'version' is a required property`.

Impact: after PR #6's default flip, initial planning uses Planner IR v1, but a later next-round LLM invocation bypasses `_plan_v1` and does not use `PLANNER_IR_SYSTEM_PROMPT`. This violates the cross-PR requirement that no caller of the compiler bypasses the PR #5 prompt path after PR #6, and can break next-round workflows.

### BLOCKER-2: The v1 path still relies on legacy static-output validators after compilation

Evidence:

- `src/vibeproteinagent/agents/execution/controller.py:504-516` validates the compiled v1 initial plan through `planner.validate_invocation(...)`.
- `src/vibeproteinagent/agents/execution/controller.py:735-737` does the same for next-round plans.
- `src/vibeproteinagent/agents/planning/workflow_planner.py:1524-1528` calls `_validate_static_output_refs`, `_validate_static_output_ref_types`, and `_validate_static_job_refs` inside `validate_plan`.

Impact: this violates the architectural invariant that validators are demoted and not load-bearing in the new path. They remain on the default v1 path after PR #6.

### BLOCKER-3: The decision prompt conflicts with `decision_v1.json` under strict default mode

Evidence:

- `src/vibeproteinagent/agents/planning/prompts.py:238-252` instructs the model to include `reasoning` per candidate and to always include `submit_candidate` and `next_round`.
- `src/vibeproteinagent/schemas/decision_v1.json:6-63` requires top-level `reasoning`, forbids unknown fields in candidate refs via `additionalProperties: false`, and requires mutually exclusive branch payloads.
- PR #6 default is strict: `validation_runs/audit_pr1_to_pr6_20260525_171205/defaults_check.log` prints `OK: defaults are v1 + strict`.

Impact: schema validation is correctly inside the retry boundary, but the LLM-facing decision prompt still teaches an invalid shape for strict mode. This makes strict-mode retry exhaustion likely in live decision paths.

### BLOCKER-4: Scope boundaries were exceeded by unowned file modifications

Evidence:

- `validation_runs/audit_pr1_to_pr6_20260525_171205/git_diff_name_status.log` shows changes outside PR #1-#6 allowed lists, including:
  - `src/vibeproteinagent/adapters/_remote/europepmc.py`
  - `src/vibeproteinagent/adapters/literature_search/adapter.py`
  - `src/vibeproteinagent/adapters/literature_search/schemas/*.json`
  - `src/vibeproteinagent/agents/provider/openrouter.py`
  - `src/vibeproteinagent/reporting/render.py`
  - `src/vibeproteinagent/workflows/executor.py`
- `validation_runs/audit_pr1_to_pr6_20260525_171205/scope_creep_diff_excerpt.log` shows substantial literature-search and EuropePMC implementation changes.

Impact: the PR contracts had tight allowed-file lists and strict non-goals. These unrelated changes make the PR #1-#6 contract boundary unverifiable and violate the scope discipline requested for this series.

### CONCERN-1: `merge_candidates` implements deferred dedup policies

Evidence:

- `src/vibeproteinagent/adapters/merge_candidates/adapter.py` supports `dedup_by_sequence` and `dedup_by_candidate_id`.
- `src/vibeproteinagent/adapters/merge_candidates/schemas/merge_candidates.input.json` exposes those enum values.
- PR #4 explicitly deferred dedup policies other than `keep_all` default existence.

Impact: no test failure observed, but this is scope creep beyond PR #4.

### CONCERN-2: Captured live matrix cycles are all BLOCKED

Evidence:

- PR #5 summary: `BLOCKED=3, FAIL=0, PASS=0`.
- PR #6 summary: `BLOCKED=3, FAIL=0, PASS=0`.
- Underlying preflight logs show `codex command not found`, `claude command not found`, and `OpenRouter credential is unset`.

Impact: categorization is plausible and no FAIL appears hidden as BLOCKED, but acceptance criterion (a), "two cycles green for non-BLOCKED providers", has no real provider execution signal.

### CONCERN-3: PR #6 legacy smoke matrix was also entirely BLOCKED

Evidence:

- `validation_runs/pr6_legacy_smoke_20260525_154607/summary.json` reports `BLOCKED=3, FAIL=0, PASS=0`.

Impact: unit smoke tests verify the legacy path with fakes, but the captured PR #6 matrix smoke does not prove legacy end-to-end provider execution.

### CONCERN-4: Schema files live under `src/vibeproteinagent/schemas`, not repo-root `schemas`

Evidence:

- `validation_runs/audit_pr1_to_pr6_20260525_171205/static_checks.json` shows `candidate_src_exists=true`, `planner_src_exists=true`, but `candidate_root_exists=false`, `planner_root_exists=false`.

Impact: runtime helpers correctly load the package-local schemas, but the checklist named `schemas/planner_ir_v1.json` and `schemas/candidate_v1.json`. This is a contract-path mismatch unless the intended contract allowed package-local schema placement.

### CONCERN-5: `decision_v1.json` forbids candidate `reasoning` indirectly, not with an explicit negative schema clause

Evidence:

- Candidate refs use `additionalProperties: false`, and tests assert a candidate `reasoning` violation is reported.
- The schema does not contain a field-specific `not` clause or explicit `reasoning: false` entry.

Impact: behavior is correct, but the checklist asked that candidate refs "EXPLICITLY forbid `reasoning` field." The explicitness is implemented in the custom violation mapper rather than in JSON Schema itself.

### OBSERVATION-1: PR #1 schema contracts are largely satisfied

Planner IR v1 locks `version` to `planner_ir_v1`; candidate v1 requires `source.local_id`; candidate deferred fields `kind`, `rank`, and `metadata` are rejected; Planner IR rejects `wiring`, `produces`, `consumes`, and relationships beyond `candidate_source`; candidate scores/artifacts require producer-namespaced object values; `stable_candidate_id` requires `producer_tool` and `local_id`; the hash test verifies basename/path independence.

### OBSERVATION-2: PR #2 retry-boundary and mode plumbing are present

`DecisionEngine.decide()` validates before `_normalize_decision()`. `DecisionSchemaViolation` is caught by controller `_run_llm_entry()` and re-fed as structured retry context. The mode flag is per-normalizer: `VPA_NORMALIZE_DECISION_REASONING`. `DECISION_CONTRACT_VERSION` and version-keyed `NORMALIZATION_DEFAULTS` exist.

### OBSERVATION-3: PR #3 canonical candidate rollout is validated on actual adapter output

The audit ran actual mocked `go_sequences` and local `sequence_analysis` adapter executions. Their emitted `candidates[]` validate against `candidate_v1.json`, local IDs are deterministic (`seq_0000`, `representative_0000`), source indexes are 0-based, and legacy representative indexes remain 1-based.

### OBSERVATION-4: PR #4 compiler/fixup core is present

`compile_planner_ir` handles single-upstream direct wiring and multi-upstream `merge_candidates`. `FIXUP_KINDS` contains exactly the three declared kinds. Fixup payloads use `choice_id` options. `merge_candidates` is planner-invisible and audit-visible.

### OBSERVATION-5: PR #5 Planner IR prompt passes the static prompt checks

`PLANNER_IR_SYSTEM_PROMPT` exists, is 736 words, contains no forbidden executor tokens from the checklist, and examples use `relationship: "candidate_source"`.

## 2. Per-PR Contract Verification

### PR #1: Planner IR v1 + Candidate v1 contracts

- PASS: `src/vibeproteinagent/schemas/planner_ir_v1.json` exists and locks `version` to `"planner_ir_v1"`.
- CONCERN: repo-root `schemas/planner_ir_v1.json` does not exist; implementation uses package-local `src/vibeproteinagent/schemas`.
- PASS: `src/vibeproteinagent/schemas/candidate_v1.json` exists and requires `source.local_id`.
- PASS: candidate schema rejects `kind`, `rank`, and `metadata`.
- PASS: Planner IR rejects `wiring`, `produces`, `consumes`, and relationship enum values beyond `candidate_source`.
- PASS: `scores` and `artifacts` enforce producer-namespaced top-level object values.
- PASS: `stable_candidate_id(producer_tool, local_id, *, ...)` requires `producer_tool` and `local_id`.
- PASS: hash helper test verifies absolute path independence by hashing only basename for `structure_path`.
- CONCERN: no per-PR commit boundary is available in git for isolating whether runtime files were modified during PR #1; current diff has many runtime modifications from the whole series.
- PASS: PR #1 tests are covered by the full suite, which passed.

### PR #2: Decision schema validation + normalizer mode flag

- PASS: `src/vibeproteinagent/schemas/decision_v1.json` exists with top-level `additionalProperties: false` and candidate refs with `additionalProperties: false`.
- CONCERN: candidate `reasoning` is forbidden indirectly through `additionalProperties: false`; custom violation mapping makes it explicit at runtime.
- PASS: validation runs inside `DecisionEngine.decide()` before `_normalize_decision()` and schema violations are re-raisable into `_run_llm_entry()`.
- PASS: mode flag is per-normalizer: `VPA_NORMALIZE_DECISION_REASONING`.
- PASS: `DECISION_CONTRACT_VERSION` exists and `NORMALIZATION_DEFAULTS` is keyed by contract version.
- HISTORICAL NOTE: current default is strict due PR #6; PR #2's original compat default cannot be verified from current HEAD alone except by the preserved v1 entry.
- PASS: `_submit_candidate_params_from_decision` was gated, not deleted.
- PASS: PR #2 tests are covered by the full suite, which passed.

### PR #3: Canonical Candidate rollout

- PASS: `go_sequences` and `sequence_analysis` outputs gained `candidates[]` additively.
- PASS: legacy `sequences[]`, `representative_sequences[]`, and `candidate_id` fields remain.
- PASS: `representative_sequences[].index` remains 1-based; new `source.index` is 0-based.
- PASS: actual emitted candidates validate against `candidate_v1.json`.
- PASS: `source.local_id` matches documented formats.
- PASS: `candidate_id` is deterministic across re-runs on same input.
- PASS: `rank_designs` rejects mixed canonical and legacy `candidates`.
- PASS: `submit_candidate` prioritizes canonical `candidates` over legacy lists.
- BLOCKER: current diff includes unrelated adapter modifications outside the PR #3 allowed set, including `literature_search` and remote EuropePMC code.
- PASS: PR #3 tests are covered by the full suite, which passed.

### PR #4: Compiler + merge_candidates + fixup transport

- PASS: `merge_candidates` adapter exists with `planner_visible: false` and `audit_visible: true`.
- PASS: `dedup_policy` default is `keep_all`.
- PASS: `keep_all` concatenates in upstream order; test covers multi-input order.
- PASS: `compile_planner_ir` handles single-upstream direct wire, multi-upstream merge, and structured fixup.
- PASS: `FIXUP_KINDS` is limited to the three declared kinds.
- PASS: fixup payloads use enum-style `choice_id` options.
- PASS: compiler validates Planner IR v1 through `validate_planner_ir_v1`.
- HISTORICAL NOTE: current `VPA_PLANNER_IR` default is `v1` due PR #6; PR #4's default-off state cannot be verified from current HEAD except by explicit `VPA_PLANNER_IR=off` tests.
- PASS: PR #4 unit tests are covered by the full suite, which passed.
- CONCERN: additional dedup policies were implemented despite being deferred.

### PR #5: Planner prompt emits new IR + first live cycle

- PASS: `PLANNER_IR_SYSTEM_PROMPT` exists and is 736 words.
- PASS: new prompt contains none of the forbidden executor terms.
- HISTORICAL NOTE: old `PLANNER_SYSTEM_PROMPT` is trimmed now per PR #6; PR #5's "unchanged" intermediate state cannot be verified from current HEAD alone.
- PASS: `_plan_v1` method exists.
- PASS: `PlannerIRSchemaViolation` mirrors the structured-error pattern.
- PASS: fixup transport is reused through `compile_with_fixup`.
- HISTORICAL NOTE: current `VPA_PLANNER_IR` default is `v1` due PR #6.
- PASS: PR #5 live matrix output exists at `validation_runs/pr5_first_live_cycle_20260525_152000/`.
- PASS: `summary.json` BLOCKED/FAIL/PASS categorization is plausible for the captured logs.
- PASS: no FAIL was found misclassified as BLOCKED.
- PASS: PR #5 unit tests are covered by the full suite, which passed.

### PR #6: Default flip + obsolete deletion + strict normalizer + acceptance cycle

- PASS: default planner IR mode resolves to `v1`.
- PASS: `DECISION_CONTRACT_VERSION == "vpa-agent-decision-v2"`.
- PASS: `NORMALIZATION_DEFAULTS["vpa-agent-decision-v2"] == {"decision_reasoning": "strict"}`.
- PASS: default normalizer mode resolves to `strict`.
- PASS: `_repair_scalar_inputs_from_output_arrays` is deleted.
- PASS: `_fallback_workflow_plan_from_selected_tools` is deleted.
- PASS: `_CANDIDATE_OUTPUT_FIELDS` is deleted.
- PASS: `compact_tool_schema` no longer exposes `output_fields`.
- PASS: legacy `PLANNER_SYSTEM_PROMPT` is trimmed and has the `# LEGACY:` comment.
- PASS: trimmed legacy prompt lacks forbidden teaching terms.
- PASS: `_submit_candidate_params_from_decision` compat lift code remains reachable under env override.
- PASS: validators are not deleted.
- BLOCKER: validators still run in the v1 path, violating the demotion invariant.
- BLOCKER: next-round default v1 path uses legacy next-round prompt and then compiles legacy output as Planner IR.
- CONCERN: captured PR #6 default matrix and legacy smoke matrix are entirely BLOCKED.
- PASS: all PR #6 unit/default tests passed now.

## 3. Cross-PR Consistency Findings

- PASS: actual PR #3 adapter outputs validate against PR #1 `candidate_v1.json`.
- PASS: PR #4 compiler accepts real PR #1 Planner IR v1 fixtures.
- PASS: PR #5 prompt examples use `upstream[].relationship: "candidate_source"`.
- PASS: initial `_plan_v1` path calls controller `_compile_with_fixup` when supplied.
- PASS: PR #6 default flip activates `_plan_v1` for initial planning via `resolve_planner_ir_mode() == "v1"`.
- FAIL: after PR #6, next-round planning bypasses `_plan_v1`/`PLANNER_IR_SYSTEM_PROMPT` and sends legacy prompt output to `_maybe_compile_planner_ir`.

## 4. Architectural Integrity Findings

- PASS: LLM-facing `PLANNER_IR_SYSTEM_PROMPT` has no `${...}` executor syntax or forbidden executor tokens.
- PASS: initial v1 executor IR generation is mechanical through `compile_planner_ir`.
- FAIL: validators are not demoted in the v1 path; `validate_invocation` still calls static output/job validators after compilation.
- CONCERN: normalizer is no longer the primary defense, but the decision prompt still teaches schema-invalid strict-mode output.
- PASS: `merge_candidates` is planner-invisible via descriptor and visibility filtering.
- PASS: `merge_candidates` is audit-visible via descriptor (`audit_visible: true`).
- PASS: there is no global `NORMALIZATION_MODE`; only `VPA_NORMALIZE_DECISION_REASONING` was found.
- PASS: mode defaults are resolved from `NORMALIZATION_DEFAULTS[DECISION_CONTRACT_VERSION]`.
- PASS: flag-off legacy path is covered by unit smoke tests; captured provider smoke matrix was blocked.

## 5. Matrix Data Findings

PR #5 cycle:

- Summary path: `validation_runs/pr5_first_live_cycle_20260525_152000/summary.json`
- Counts: `BLOCKED=3`, `FAIL=0`, `PASS=0`.
- Spot checks: `preflight_codex.log`, `preflight_claude.log`, and `preflight_qwen.log` match the provider records.
- No FAIL was silently reclassified as BLOCKED.
- All BLOCKED provider records include `reason: "preflight_failed"`.

PR #6 cycle:

- Summary path: `validation_runs/pr6_default_flip_cycle_20260525_154607/summary.json`
- Counts: `BLOCKED=3`, `FAIL=0`, `PASS=0`.
- Spot checks match preflight logs.
- No FAIL was silently reclassified as BLOCKED.
- All BLOCKED provider records include `reason: "preflight_failed"`.
- `defaults.log` confirms `planner_ir_mode= v1` and `decision_normalization_mode= strict`.

Acceptance gate evaluation:

- (a) Two cycles green for non-BLOCKED providers: **NOT EVIDENCED**. There were zero non-BLOCKED providers in both cycles.
- (b) Strict normalizer ran/configured for PR #6: **PASS** by `defaults.log` and inline defaults check.
- (c) Escape hatch pattern count / `raw_inputs`: **PASS / trivially true**. No `raw_inputs` code was found.
- (d) No new prompt rules added since PR #4: **PARTIAL**. Granular PR4-to-PR6 commit boundaries are unavailable in the current git history; current IR prompt has no forbidden executor rules.
- (e) All wiring through compiler: **FAIL for full live path** because next-round planning bypasses the PR #5 prompt path and sends legacy output to the compiler.

## 6. Scope Creep Findings

Git state note: PR #1-#6 changes are present as a dirty working tree, not as six auditable PR commits. `git log --oneline 9773c6e..HEAD` lists older commits whose messages do not map cleanly to PR #1-#6. Therefore, per-commit PR mapping could not be verified from git history.

Full current diff shows modifications outside the PR contracts, including:

- Literature search and EuropePMC remote implementation.
- `reporting/render.py`.
- `workflows/executor.py`.
- `agents/provider/openrouter.py`.
- Additional adapter/schema changes not owned by PR #1-#6.

Search results:

- No `raw_inputs` escape hatch found.
- No `candidate_v2` or `planner_ir_v1_1` found.
- No extra fixup kinds beyond the declared three.
- Extra `merge_candidates` dedup policies were found.
- No new canonical producer rollouts beyond `go_sequences`, `sequence_analysis`, and internal `merge_candidates`.
- `compact_tool_schema` no longer exposes `output_fields`.

## 7. Test Execution Results

Full suite:

```text
/mnt/vast/trillion/env/miniconda3/envs/vpa/bin/python -m pytest -q
1114 passed, 17 skipped, 90 warnings in 1794.14s (0:29:54)
```

Artifact: `validation_runs/audit_pr1_to_pr6_20260525_171205/full_suite.log`

Default behavior regression:

```text
/mnt/vast/trillion/env/miniconda3/envs/vpa/bin/python -m pytest \
  tests/test_default_flag_is_v1.py \
  tests/test_decision_contract_v2_default_strict.py \
  tests/test_legacy_planner_still_runs_when_opted_in.py \
  tests/test_legacy_prompt_is_trimmed.py -v
12 passed in 2.88s
```

Artifact: `validation_runs/audit_pr1_to_pr6_20260525_171205/default_behavior.log`

Defaults inline check:

```text
OK: defaults are v1 + strict
```

Artifact: `validation_runs/audit_pr1_to_pr6_20260525_171205/defaults_check.log`

## 8. Recommendation

Recommendation: **needs follow-up before release**.

Rationale: tests pass, and many contract pieces are clean, but the default v1 live path still has architectural violations: next-round planning bypasses the Planner IR prompt path, validators remain load-bearing in v1 validation, and the strict decision prompt/schema mismatch can drive invalid LLM output. Scope creep outside the PR contracts should also be separated or explicitly accepted before release.

