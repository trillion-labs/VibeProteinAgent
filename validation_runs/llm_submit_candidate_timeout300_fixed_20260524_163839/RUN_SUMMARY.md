# LLM Submit Candidate Timeout 300 Validation

Date: 2026-05-24

Validation root: `/mnt/vast/trillion/hyunjin/VibeProteinAgent/validation_runs/llm_submit_candidate_timeout300_fixed_20260524_163839`

## Results

- Plan-only contract: 9/9 passed across Codex gpt-5.4, Claude Code/Sonnet, OpenRouter Qwen 3.5 9B.
- Actual semantic end-to-end contract: 3/3 passed with `submitted_final_candidate`.
- Artifact/path/secret scan: clean for this validation root.
- Shared env: `/mnt/vast/trillion/hyunjin/.vibeproteinagent/env.sh`.

## Plan Contract

```tsv
case	ok	workflow_ref	submit_final	has_submit_candidate
binder_claude	True	target_sequence_binder_design_round	True	True
binder_codex	True	target_sequence_binder_design_round	True	True
binder_qwen	True	target_sequence_binder_design_round	True	True
denovo_claude	True	rfdiffusion_backbone_design_round	True	True
denovo_codex	True	rfdiffusion_backbone_design_round	True	True
denovo_qwen	True	rfdiffusion_backbone_design_round	True	True
semantic_claude	True	semantic_constraint_to_family_sequences	True	True
semantic_codex	True	semantic_constraint_to_family_sequences	True	True
semantic_qwen	True	semantic_constraint_to_family_sequences	True	True
```

## Actual Contract

```tsv
case	ok	state	decision	has_final_submission
semantic_claude	True	completed	submitted_final_candidate	True
semantic_codex	True	completed	submitted_final_candidate	True
semantic_qwen	True	completed	submitted_final_candidate	True
```
