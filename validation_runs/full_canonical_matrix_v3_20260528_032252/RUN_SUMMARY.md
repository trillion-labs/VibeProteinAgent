# Live Provider Matrix on shared-cluster-vm-007

Run root: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_032252`
Cases: 1 passed / 8 failed / 0 blocked / 9 total

| Case | Mode | Provider | Model | Status | Seconds |
| --- | --- | --- | --- | --- | ---: |
| `actual_semantic_codex` | actual | codex | `gpt-5.4` | **FAIL** | 424 |
| `actual_binder_codex` | actual | codex | `gpt-5.4` | **FAIL** | 822 |
| `actual_denovo_codex` | actual | codex | `gpt-5.4` | **FAIL** | 362 |
| `actual_semantic_claude` | actual | claude-code | `sonnet` | **FAIL** | 282 |
| `actual_binder_claude` | actual | claude-code | `sonnet` | **FAIL** | 299 |
| `actual_denovo_claude` | actual | claude-code | `sonnet` | **FAIL** | 592 |
| `actual_semantic_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **PASS** | 236 |
| `actual_binder_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 245 |
| `actual_denovo_qwen` | actual | openrouter | `qwen/qwen3.5-9b` | **FAIL** | 281 |

Artifacts:
- Status TSV: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_032252/status.tsv`
- Logs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_032252/logs`
- Case outputs: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/full_canonical_matrix_v3_20260528_032252/{codex,claude,qwen}/*.json`
