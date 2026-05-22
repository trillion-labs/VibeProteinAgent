#!/usr/bin/env bash
#SBATCH --job-name=vpa-val-01-report-vis
#SBATCH --nodelist=<gpu-node-name>
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=validation_runs/logs/slurm_01_report_visualize_%j.out
#SBATCH --error=validation_runs/logs/slurm_01_report_visualize_%j.err

export VPA_REPO_ROOT="${VPA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${VPA_REPO_ROOT}/validation_runs/slurm_common.sh"
bash "${VPA_REPO_ROOT}/validation_runs/01_report_visualize_cli.sh"
