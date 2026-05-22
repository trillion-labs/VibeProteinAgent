#!/usr/bin/env bash
#SBATCH --job-name=vpa-val-05-esm3
#SBATCH --nodelist=<gpu-node-name>
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=validation_runs/logs/slurm_05_esm3_%j.out
#SBATCH --error=validation_runs/logs/slurm_05_esm3_%j.err

export VPA_REPO_ROOT="${VPA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${VPA_REPO_ROOT}/validation_runs/slurm_common.sh"
bash "${VPA_REPO_ROOT}/validation_runs/05_esm3_hf_access.sh"
