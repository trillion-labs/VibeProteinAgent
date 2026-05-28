# Live Provider Matrix on shared-cluster-vm-007

Run root: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_083409`
Cases: 6 passed / 3 failed / 0 blocked / 0 missing / 9 recorded

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **PASS** | 368 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **PASS** | 1055 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **PASS** | 397 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **PASS** | 353 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **FAIL** | 269 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **PASS** | 626 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **PASS** | 168 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 406 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 312 |

Artifacts:
- Status TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_083409/status.tsv`
- Case events TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_083409/case_events.tsv`
- Expected cases TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_083409/expected_cases.tsv`
- Logs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_083409/logs`
- Case outputs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_083409/{codex,claude,qwen}/*.json`
