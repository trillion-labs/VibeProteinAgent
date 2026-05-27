# Live Provider Matrix on shared-cluster-vm-008

Run root: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_082416`
Cases: 0 passed / 9 failed / 0 blocked / 9 total

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **FAIL** | 397 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **FAIL** | 1971 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **FAIL** | 304 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **FAIL** | 153 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **FAIL** | 888 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **FAIL** | 405 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 58 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 555 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 458 |

Artifacts:
- Status TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_082416/status.tsv`
- Logs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_082416/logs`
- Case outputs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_082416/{codex,claude,qwen}/*.json`
