#!/usr/bin/env bash
#SBATCH --job-name=vpa-val-00-preflight
#SBATCH --nodelist=<gpu-node-name>
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=validation_runs/logs/slurm_00_preflight_%j.out
#SBATCH --error=validation_runs/logs/slurm_00_preflight_%j.err

export VPA_REPO_ROOT="${VPA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${VPA_REPO_ROOT}/validation_runs/slurm_common.sh"
bash "${VPA_REPO_ROOT}/validation_runs/00_preflight_cli_install.sh"
