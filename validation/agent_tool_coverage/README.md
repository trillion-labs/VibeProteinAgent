# Agent Tool Coverage Runs

These plans are hand-authored `vpa agent submit-plan` inputs that mimic the
validated JSON an LLM planner should return. Together they cover every
agent-visible first-class tool while excluding internal glue tools.

Run from the repository root so the relative fixture paths in
`params.local.json` resolve correctly.

## Covered Tools

- Run 01: `colabdesign_utils`, `dssp`, `foldseek`, `go_sequences`,
  `interpro_sequences`, `ipsae`, `literature_search`, `mmseqs2_gpu`, `prodigy`,
  `protein_qc`, `pxmeter`, `rank_designs`, `sequence_analysis`,
  `sequence_edit`, `structure_analysis`, `template_select`, `tmalign`, `vina`
- Run 02: `bindcraft`, `esm2`, `esm3`, `ligandmpnn`, `proteinmpnn`,
  `pxdesign`, `pyrosetta`, `rfdiffusion`, `rfdiffusion3`, `rfpeptides`
- Run 03: `boltz2_affinity`, `boltzgen`, `proteina_complexa`, `protenix_v2`

## Environment

If the host has no system Python, use the installed VPA runtime directly:

```bash
export VPA_HOME="${VPA_HOME:-/shared/projects/vpa/.vibeproteinagent}"
source "$VPA_HOME/env.sh"
export PATH="$VPA_HOME/envs/vpa/bin:$PATH"
vpa --help
```

On a multi-node cluster, `VPA_HOME` should be on shared storage. Do not let each
node create its own local `~/.vibeproteinagent` cache.

For full-profile sidecars, these exports are often written by install. If
validation reports a sidecar env is missing even though install completed, set
the standard locations explicitly:

```bash
export VPA_BINDCRAFT_ENV_PATH="$VPA_HOME/sidecars/bindcraft"
export VPA_PROTEINA_COMPLEXA_VENV="$VPA_HOME/sidecars/proteina-complexa/.venv"
```

Run 02 includes ESM3, so gated Hugging Face access must be available before
`validate-plan` or `submit-plan`:

```bash
export VPA_HF_CREDENTIAL="<huggingface-token>"
```

If you prefer a real micromamba activation:

```bash
export VPA_HOME="${VPA_HOME:-/shared/projects/vpa/.vibeproteinagent}"
source "$VPA_HOME/env.sh"
eval "$("$VPA_HOME/bootstrap/bin/micromamba" shell hook -s bash)"
micromamba activate "$VPA_HOME/envs/vpa"
vpa --help
```

## Run Order

Start workers once for the shared workspace:

```bash
vpa daemon start --workers 1 --workspace agent_tool_coverage
vpa daemon status --workspace agent_tool_coverage
```

Validate all three plans before spending GPU time:

```bash
vpa agent validate-plan validation/agent_tool_coverage/01_catalog_qc_report_plan.json --workspace agent_tool_coverage --json
vpa agent validate-plan validation/agent_tool_coverage/02_design_engines_plan.json --workspace agent_tool_coverage --json
vpa agent validate-plan validation/agent_tool_coverage/03_foundation_models_plan.json --workspace agent_tool_coverage --json
```

Submit each plan. Copy the `workflow_id` from each JSON response:

```bash
vpa agent submit-plan validation/agent_tool_coverage/01_catalog_qc_report_plan.json \
  --params-file validation/agent_tool_coverage/params.local.json \
  --workspace agent_tool_coverage \
  --max-wait-s 7200 \
  --json | tee validation/agent_tool_coverage/01_result.json

vpa agent submit-plan validation/agent_tool_coverage/02_design_engines_plan.json \
  --params-file validation/agent_tool_coverage/params.local.json \
  --workspace agent_tool_coverage \
  --max-wait-s 14400 \
  --json | tee validation/agent_tool_coverage/02_result.json

vpa agent submit-plan validation/agent_tool_coverage/03_foundation_models_plan.json \
  --params-file validation/agent_tool_coverage/params.local.json \
  --workspace agent_tool_coverage \
  --max-wait-s 21600 \
  --json | tee validation/agent_tool_coverage/03_result.json
```

Render reports after each run:

```bash
vpa report <workflow_id_from_run_01> --workspace agent_tool_coverage \
  --out validation/agent_tool_coverage/01_report.html

vpa report <workflow_id_from_run_02> --workspace agent_tool_coverage \
  --out validation/agent_tool_coverage/02_report.html

vpa report <workflow_id_from_run_03> --workspace agent_tool_coverage \
  --out validation/agent_tool_coverage/03_report.html
```

Inspect stored agent runs:

```bash
vpa agent runs --workspace agent_tool_coverage --json
vpa workflow status <workflow_id> --workspace agent_tool_coverage
```

Notes:

- These are broad coverage runs, not benchmark-quality scientific protocols.
- Most nodes are independent and use `on_failure=continue`, so a missing
  optional binary or sidecar should still leave the rest of the report useful.
- If `validate-plan` fails, install or repair the named tool first; submission
  intentionally refuses unavailable first-class tools.
