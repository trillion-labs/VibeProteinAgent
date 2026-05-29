# Live Provider Matrix on shared-cluster-vm-007

Run root: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_105828`
Cases: 0 passed / 0 failed / 9 blocked / 0 missing / 9 recorded

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **BLOCKED** | 0 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **BLOCKED** | 0 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **BLOCKED** | 0 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **BLOCKED** | 0 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **BLOCKED** | 0 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **BLOCKED** | 0 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **BLOCKED** | 0 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **BLOCKED** | 0 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **BLOCKED** | 0 |

Artifacts:
- Status TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_105828/status.tsv`
- Case events TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_105828/case_events.tsv`
- Expected cases TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_105828/expected_cases.tsv`
- Logs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_105828/logs`
- Case outputs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260529_105828/{codex,claude,qwen}/*.json`
