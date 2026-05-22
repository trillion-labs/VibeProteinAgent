#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "05_esm3_hf_access"
require_hf_credential
trap stop_daemon EXIT

ensure_daemon 1

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/esm3_structure_submit.json" \
  run esm3 sequence=ACDEFGHIKLMNPQRSTVWY "model_name=${VPA_ESM3_MODEL:-esm3_sm_open_v1}" 'tracks=["sequence","structure"]' num_steps=4 temperature=0.1 \
  --workspace "${VPA_WORKSPACE}" --force-new --json
job_id="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/esm3_structure_submit.json" data.job_id)"
wait_job_completed "${job_id}" "${VPA_VALIDATION_RUN_DIR}/logs/esm3_structure_wait.json"

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/esm3_report.json" \
  report "${job_id}" --workspace "${VPA_WORKSPACE}" --viewer offline --out "${VPA_VALIDATION_RUN_DIR}/${job_id}_report.html" --json
assert_file_nonempty "${VPA_VALIDATION_RUN_DIR}/${job_id}_report.html"

predicted_pdb="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/esm3_structure_wait.json" data.jobs.0.output.predicted_pdb)"
assert_file_nonempty "${predicted_pdb}"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/esm3_visualize.json" \
  visualize "${predicted_pdb}" --viewer offline --title "ESM3 predicted structure" --out "${VPA_VALIDATION_RUN_DIR}/${job_id}_viewer.html" --json
assert_file_nonempty "${VPA_VALIDATION_RUN_DIR}/${job_id}_viewer.html"

log "ESM3 HF access scenario complete: ${VPA_VALIDATION_RUN_DIR}"
