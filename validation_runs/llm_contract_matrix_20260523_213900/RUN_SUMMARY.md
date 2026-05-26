# LLM Contract Matrix Validation

Date: 2026-05-23

Validation root: `validation_runs/llm_contract_matrix_20260523_213900`

Authoritative outputs:

| Query type | Provider | Mode | Status | Workflow |
| --- | --- | --- | --- | --- |
| Semantic functional generation | Codex gpt-5.4 | plan-only | pass | `semantic_constraint_to_family_sequences` |
| Semantic functional generation | OpenRouter/Qwen | plan-only | pass | `semantic_constraint_to_family_sequences` |
| Semantic functional generation | Claude Code/Sonnet | plan-only | pass | `semantic_constraint_to_family_sequences` |
| Binder generation | Codex gpt-5.4 | plan-only | pass | `target_sequence_binder_design_round` |
| Binder generation | OpenRouter/Qwen | plan-only | pass | `target_sequence_binder_design_round` |
| Binder generation | Claude Code/Sonnet | plan-only | pass | `target_sequence_binder_design_round` |
| De novo explicit tool-chain | Codex gpt-5.4 | plan-only | pass | `rfdiffusion_backbone_design_round` |
| De novo explicit tool-chain | OpenRouter/Qwen | plan-only | pass | `rfdiffusion_backbone_design_round` |
| De novo explicit tool-chain | Claude Code/Sonnet | plan-only | pass | `rfdiffusion_backbone_design_round` |
| Semantic functional generation | Codex gpt-5.4 | actual | pass | `wf_995d85cdf58747408b9b1467b133acfe` |
| Semantic functional generation | OpenRouter/Qwen | actual | pass | `wf_34c2f7c16cae4d618e0e807e8118f142` |
| Semantic functional generation | Claude Code/Sonnet | actual | pass | `wf_fc89643db7594d2ea4485ba0c60d9257` |
| De novo explicit tool-chain | OpenRouter/Qwen | actual GPU | pass | `wf_99e95af56e084d0a8ecfaa35a21b21a9` |

Actual output files:

- `actual/semantic_codex_fixed.json`
- `actual/semantic_openrouter_fixed.json`
- `actual/semantic_claude_fixed.json`
- `actual/denovo_openrouter_actual.json`
- `actual/semantic_codex_fixed_report.html`
- `actual/semantic_openrouter_fixed_report.html`
- `actual/semantic_claude_fixed_report.html`
- `actual/denovo_openrouter_actual_report.html`

Notes:

- Early `actual/semantic_*.json` files without the `_fixed` suffix are retained as setup/debug evidence. They were caused by a daemon workspace mismatch and one missing environment key injection, then rerun successfully.
- The semantic actual runs exposed a product issue where completed workflows with `representative_sequences` could be sent back to the LLM critic for another round. The code now treats representative/top/ranked candidate outputs as terminal final candidates.
- The OpenRouter/Qwen de novo actual run completed RFdiffusion, ProteinMPNN, Protenix, structure analysis, protein QC, and ranking on the GPU daemon workspace.
- Artifact paths are rooted under `/mnt/vast/trillion/hyunjin/.vibeproteinagent`.
- Secret and developer-home path scans passed for this validation root plus repo code/docs/demo/workflows.

Verification:

- Ruff passed on touched agent/planner/summarizer/test files.
- `tests/test_agent_layer.py`: 52 passed.
- `tests/test_agent_layer.py tests/test_remote_helpers.py tests/test_adapter_literature_search.py tests/test_adapter_sequence_analysis.py`: 70 passed.
