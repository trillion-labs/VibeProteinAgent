#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "04_workflow_rfdiffusion_round"
trap stop_daemon EXIT

ensure_daemon "${VPA_VALIDATION_WORKERS}"

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_rfdiffusion_round.json" \
  workflow run rfdiffusion_backbone_design_round \
  --param length=60 \
  --param mpnn_num_designs=1 \
  --param top_k=1 \
  --workspace "${VPA_WORKSPACE}" \
  --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
  --json
assert_workflow_completed "${VPA_VALIDATION_RUN_DIR}/logs/workflow_rfdiffusion_round.json"

workflow_id="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/workflow_rfdiffusion_round.json" data.workflow_id)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_status.json" workflow status "${workflow_id}" --workspace "${VPA_WORKSPACE}" --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_report.json" \
  report "${workflow_id}" --workspace "${VPA_WORKSPACE}" --viewer offline --format bundle --out "${VPA_VALIDATION_RUN_DIR}/${workflow_id}_report_bundle" --json
assert_file_nonempty "${VPA_VALIDATION_RUN_DIR}/${workflow_id}_report_bundle/report.html"

log "rfdiffusion workflow complete: ${VPA_VALIDATION_RUN_DIR}"
