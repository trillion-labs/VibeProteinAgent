#!/usr/bin/env bash
#SBATCH --job-name=vpa-val-08-qwen-plan
#SBATCH --nodelist=<gpu-node-name>
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=validation_runs/logs/slurm_08_qwen_plan_%j.out
#SBATCH --error=validation_runs/logs/slurm_08_qwen_plan_%j.err

export VPA_REPO_ROOT="${VPA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${VPA_REPO_ROOT}/validation_runs/slurm_common.sh"
bash "${VPA_REPO_ROOT}/validation_runs/08_agent_plan_only_openrouter_qwen3_8b.sh"
