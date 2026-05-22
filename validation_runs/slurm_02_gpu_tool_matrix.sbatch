#!/usr/bin/env bash
#SBATCH --job-name=vpa-val-02-gpu-tools
#SBATCH --nodelist=<gpu-node-name>
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=validation_runs/logs/slurm_02_gpu_tool_matrix_%j.out
#SBATCH --error=validation_runs/logs/slurm_02_gpu_tool_matrix_%j.err

export VPA_REPO_ROOT="${VPA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${VPA_REPO_ROOT}/validation_runs/slurm_common.sh"
bash "${VPA_REPO_ROOT}/validation_runs/02_gpu_tool_matrix.sh"
