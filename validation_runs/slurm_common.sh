#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -z "${VPA_REPO_ROOT:-}" ]]; then
  VPA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
export VPA_REPO_ROOT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${VPA_VALIDATION_GPUS:-0,1,2,3,4,5,6,7}}"
export VPA_VALIDATION_GPUS="${VPA_VALIDATION_GPUS:-${CUDA_VISIBLE_DEVICES}}"
export VPA_VALIDATION_WORKERS="${VPA_VALIDATION_WORKERS:-8}"
export VPA_VALIDATION_RUN_ROOT="${VPA_VALIDATION_RUN_ROOT:-${VPA_REPO_ROOT}/output/validation_runs}"

mkdir -p "${VPA_VALIDATION_RUN_ROOT}/slurm_logs"

if [[ -f "${VPA_REPO_ROOT}/validation_runs/local.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${VPA_REPO_ROOT}/validation_runs/local.env"
  set +a
fi
