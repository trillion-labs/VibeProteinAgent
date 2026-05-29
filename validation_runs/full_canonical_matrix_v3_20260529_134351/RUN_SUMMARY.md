# Live Provider Matrix on shared-cluster-vm-024

Run root: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_134351`
Cases: 3 passed / 6 failed / 0 blocked / 0 missing / 9 recorded

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **PASS** | 386 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **FAIL** | 469 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **FAIL** | 368 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **FAIL** | 282 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **FAIL** | 297 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **FAIL** | 312 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **PASS** | 1085 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 508 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **PASS** | 873 |

Artifacts:
- Status TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_134351/status.tsv`
- Case events TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_134351/case_events.tsv`
- Expected cases TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_134351/expected_cases.tsv`
- Logs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_134351/logs`
- Case outputs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_134351/{codex,claude,qwen}/*.json`
