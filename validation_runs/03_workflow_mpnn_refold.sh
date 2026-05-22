#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "03_workflow_mpnn_refold"
trap stop_daemon EXIT

ensure_daemon "${VPA_VALIDATION_WORKERS}"

pdb="$(fixture_path tests/fixtures/6MRR.pdb)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_mpnn_refold.json" \
  workflow run mpnn_then_refold \
  --param "pdb_path=${pdb}" \
  --param num_designs=2 \
  --param top_k=1 \
  --workspace "${VPA_WORKSPACE}" \
  --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
  --json
assert_workflow_completed "${VPA_VALIDATION_RUN_DIR}/logs/workflow_mpnn_refold.json"

workflow_id="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/workflow_mpnn_refold.json" data.workflow_id)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_status.json" workflow status "${workflow_id}" --workspace "${VPA_WORKSPACE}" --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_report.json" \
  report "${workflow_id}" --workspace "${VPA_WORKSPACE}" --viewer offline --format bundle --out "${VPA_VALIDATION_RUN_DIR}/${workflow_id}_report_bundle" --json
assert_file_nonempty "${VPA_VALIDATION_RUN_DIR}/${workflow_id}_report_bundle/report.html"

log "mpnn_then_refold workflow complete: ${VPA_VALIDATION_RUN_DIR}"
