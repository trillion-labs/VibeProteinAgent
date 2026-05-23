# VPA validation runs

These scripts are intentionally outside the test suite. They exercise the
installed `vpa` CLI, worker daemon, reports, visualizations, workflows, and
optional OpenRouter-backed agent mode against real model/tool paths.

No credential values belong in this directory. Put local secrets in
`validation_runs/local.env` or export them in the submit shell. `local.env` is
ignored by git.

## Setup

```bash
cp validation_runs/env.example validation_runs/local.env
# edit local.env with local paths and credential values
```

Required only for scenarios that use gated/first-run Hugging Face access:

```bash
export HF_TOKEN="<huggingface-token>"
# or
export VPA_HF_CREDENTIAL="<huggingface-token>"
```

Required only for OpenRouter agent scenarios. The default OpenRouter validation
model is `qwen/qwen3-8b`.

```bash
export OPENROUTER_API_KEY="<openrouter-api-key>"
# or
export VPA_OPENROUTER_CREDENTIAL="<openrouter-api-key>"
```

Codex and Claude Code scenarios use local authenticated CLIs. Override their
commands or models if your cluster image uses wrappers:

```bash
export VPA_CODEX_COMMAND=codex
export VPA_CODEX_ARGS="exec --model {model} --skip-git-repo-check --sandbox read-only -"
export VPA_VALIDATION_CODEX_MODEL=gpt-5.2

export VPA_CLAUDE_CODE_COMMAND=claude
export VPA_CLAUDE_CODE_ARGS="-p --model {model} --output-format text"
export VPA_VALIDATION_CLAUDE_MODEL=sonnet

export VPA_VALIDATION_OPENROUTER_MODEL=qwen/qwen3-8b
```

On a multi-node cluster, point VPA at one shared install root before running
local or Slurm validation. This keeps envs, sidecars, workspaces, and model
caches reusable across nodes:

```bash
export VPA_SHARED_HOME=/shared/projects/vpa/.vibeproteinagent
export VPA_HOME="$VPA_SHARED_HOME"
source "$VPA_HOME/env.sh"
```

Scenarios use the active/shared `VPA_HOME` by default. Set
`VPA_VALIDATION_ISOLATED=1` only when you intentionally want a scratch
`VPA_HOME` under `output/validation_runs/...`.

## Local runs

```bash
bash validation_runs/00_preflight_cli_install.sh
bash validation_runs/01_report_visualize_cli.sh
bash validation_runs/02_gpu_tool_matrix.sh
bash validation_runs/03_workflow_mpnn_refold.sh
bash validation_runs/04_workflow_rfdiffusion_round.sh
bash validation_runs/05_esm3_hf_access.sh
bash validation_runs/06_agent_plan_only_codex.sh
bash validation_runs/07_agent_plan_only_claude.sh
bash validation_runs/08_agent_plan_only_openrouter_qwen3_8b.sh
bash validation_runs/09_agent_e2e_codex_gpu.sh
bash validation_runs/10_agent_e2e_claude_gpu.sh
bash validation_runs/11_agent_e2e_openrouter_qwen3_8b_gpu.sh
bash validation_runs/99_secret_guard.sh
```

## Slurm runs

Each Slurm script targets `<gpu-node-name>`, requests all visible GPUs,
and then calls the matching scenario script.

```bash
sbatch validation_runs/slurm_00_preflight_cli_install.sbatch
sbatch validation_runs/slurm_01_report_visualize_cli.sbatch
sbatch validation_runs/slurm_02_gpu_tool_matrix.sbatch
sbatch validation_runs/slurm_03_workflow_mpnn_refold.sbatch
sbatch validation_runs/slurm_04_workflow_rfdiffusion_round.sbatch
sbatch validation_runs/slurm_05_esm3_hf_access.sbatch
sbatch validation_runs/slurm_06_agent_plan_only_codex.sbatch
sbatch validation_runs/slurm_07_agent_plan_only_claude.sbatch
sbatch validation_runs/slurm_08_agent_plan_only_openrouter_qwen3_8b.sbatch
sbatch validation_runs/slurm_09_agent_e2e_codex_gpu.sbatch
sbatch validation_runs/slurm_10_agent_e2e_claude_gpu.sbatch
sbatch validation_runs/slurm_11_agent_e2e_openrouter_qwen3_8b_gpu.sbatch
```

Submit all scenarios:

```bash
bash validation_runs/submit_all_slurm.sh
```

Use `VPA_VALIDATION_RUN_ROOT` to redirect runtime artifacts:

```bash
export VPA_VALIDATION_RUN_ROOT=/scratch/${USER}/vpa_validation
```
