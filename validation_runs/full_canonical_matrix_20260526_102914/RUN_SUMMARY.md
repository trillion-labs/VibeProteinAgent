# Live Provider Matrix on vm-024

Run root: `validation_runs/full_canonical_matrix_20260526_102914`
Cases: 1 passed / 17 failed / 0 blocked / 18 total

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `plan_semantic_codex` | plan | codex | `gpt-5.4` | **FAIL** | 138 |
| `plan_binder_codex` | plan | codex | `gpt-5.4` | **FAIL** | 130 |
| `plan_denovo_codex` | plan | codex | `gpt-5.4` | **FAIL** | 156 |
| `plan_semantic_claude` | plan | claude-code | `sonnet` | **FAIL** | 190 |
| `plan_binder_claude` | plan | claude-code | `sonnet` | **FAIL** | 155 |
| `plan_denovo_claude` | plan | claude-code | `sonnet` | **FAIL** | 196 |
| `plan_semantic_qwen` | plan | openrouter | `qwen/qwen3.5-9b` | **PASS** | 53 |
| `plan_binder_qwen` | plan | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 122 |
| `plan_denovo_qwen` | plan | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 102 |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **FAIL** | 148 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **FAIL** | 167 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **FAIL** | 166 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **FAIL** | 259 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **FAIL** | 180 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **FAIL** | 423 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 65 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 105 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 149 |

Artifacts:
- Status TSV: `validation_runs/full_canonical_matrix_20260526_102914/status.tsv`
- Logs: `validation_runs/full_canonical_matrix_20260526_102914/logs`
- JSON outputs: `validation_runs/full_canonical_matrix_20260526_102914/outputs`
