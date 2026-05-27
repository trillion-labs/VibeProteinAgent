# Live Provider Matrix on shared-cluster-vm-008

Run root: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_151735`
Cases: 0 passed / 9 failed / 0 blocked / 9 total

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **FAIL** | 195 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **FAIL** | 1313 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **FAIL** | 580 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **FAIL** | 379 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **FAIL** | 1273 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **FAIL** | 533 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 174 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 208 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 135 |

Artifacts:
- Status TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_151735/status.tsv`
- Logs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_151735/logs`
- Case outputs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260527_151735/{codex,claude,qwen}/*.json`
