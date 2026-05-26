# Live Provider Matrix on vm-024

Run root: `validation_runs/full_canonical_matrix_v3_20260526_190721`
Cases: 2 passed / 16 failed / 0 blocked / 18 total

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `plan_semantic_codex` | plan | codex | `gpt-5.4` | **FAIL** | 258 |
| `plan_binder_codex` | plan | codex | `gpt-5.4` | **FAIL** | 280 |
| `plan_denovo_codex` | plan | codex | `gpt-5.4` | **FAIL** | 272 |
| `plan_semantic_claude` | plan | claude-code | `sonnet` | **FAIL** | 210 |
| `plan_binder_claude` | plan | claude-code | `sonnet` | **FAIL** | 149 |
| `plan_denovo_claude` | plan | claude-code | `sonnet` | **PASS** | 127 |
| `plan_semantic_qwen` | plan | openrouter | `qwen/qwen3.5-9b` | **PASS** | 54 |
| `plan_binder_qwen` | plan | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 118 |
| `plan_denovo_qwen` | plan | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 94 |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **FAIL** | 296 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **FAIL** | 160 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **FAIL** | 526 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **FAIL** | 196 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **FAIL** | 166 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **FAIL** | 240 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 290 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 324 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 224 |

Artifacts:
- Status TSV: `validation_runs/full_canonical_matrix_v3_20260526_190721/status.tsv`
- Logs: `validation_runs/full_canonical_matrix_v3_20260526_190721/logs`
- JSON outputs: `validation_runs/full_canonical_matrix_v3_20260526_190721/outputs`
