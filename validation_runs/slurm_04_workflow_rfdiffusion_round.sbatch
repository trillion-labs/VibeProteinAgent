#!/usr/bin/env bash
#SBATCH --job-name=vpa-val-04-rfdiff
#SBATCH --nodelist=<gpu-node-name>
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=validation_runs/logs/slurm_04_rfdiffusion_%j.out
#SBATCH --error=validation_runs/logs/slurm_04_rfdiffusion_%j.err

export VPA_REPO_ROOT="${VPA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${VPA_REPO_ROOT}/validation_runs/slurm_common.sh"
bash "${VPA_REPO_ROOT}/validation_runs/04_workflow_rfdiffusion_round.sh"
