#!/usr/bin/env bash
set -Eeuo pipefail

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${dir}/logs"

scripts=(
  slurm_00_preflight_cli_install.sbatch
  slurm_01_report_visualize_cli.sbatch
  slurm_02_gpu_tool_matrix.sbatch
  slurm_03_workflow_mpnn_refold.sbatch
  slurm_04_workflow_rfdiffusion_round.sbatch
  slurm_05_esm3_hf_access.sbatch
  slurm_06_agent_plan_only_codex.sbatch
  slurm_07_agent_plan_only_claude.sbatch
  slurm_08_agent_plan_only_openrouter_qwen3_8b.sbatch
  slurm_09_agent_e2e_codex_gpu.sbatch
  slurm_10_agent_e2e_claude_gpu.sbatch
  slurm_11_agent_e2e_openrouter_qwen3_8b_gpu.sbatch
)

for script in "${scripts[@]}"; do
  echo "+ sbatch ${dir}/${script}"
  sbatch "${dir}/${script}"
done
