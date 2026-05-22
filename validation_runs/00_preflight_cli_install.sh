#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "00_preflight_cli_install"

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/version.json" version --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/tools.json" tools --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/tool_proteinmpnn.json" tool info proteinmpnn --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/tool_esm3.json" tool info esm3 --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_list.json" workflow list --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/workflow_inspect_mpnn.json" workflow inspect mpnn_then_refold --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/install_agent_dry_run.json" install --profile agent --dry-run --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/install_full_dry_run.json" install --profile full --dry-run --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/install_doctor_selected.json" install doctor --tools proteinmpnn,protenix_v2,esm3,rfdiffusion --json

log "preflight complete: ${VPA_VALIDATION_RUN_DIR}"
