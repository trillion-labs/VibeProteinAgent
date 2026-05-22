#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "09_agent_e2e_codex_gpu"
require_codex_cli
trap stop_daemon EXIT

ensure_daemon "${VPA_VALIDATION_WORKERS}"

pdb="$(fixture_path tests/fixtures/6MRR.pdb)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/agent_e2e_codex.json" \
  agent run \
  "Run a minimal GPU validation workflow for this backbone using VPA tools: inverse-fold one or two sequences, refold the best candidate, then summarize outputs suitable for report generation." \
  --target "${pdb}" \
  --provider codex \
  --model "${VPA_VALIDATION_CODEX_MODEL}" \
  --max-rounds "${VPA_VALIDATION_AGENT_MAX_ROUNDS:-2}" \
  --max-entries 3 \
  --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
  --workspace "${VPA_WORKSPACE}" \
  --json
assert_agent_completed "${VPA_VALIDATION_RUN_DIR}/logs/agent_e2e_codex.json"

run_id="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/agent_e2e_codex.json" data.run_id)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/agent_status_codex.json" agent show "${run_id}" --workspace "${VPA_WORKSPACE}" --json
report_agent_outputs "${VPA_VALIDATION_RUN_DIR}/logs/agent_e2e_codex.json"

log "Codex agent E2E scenario complete: ${VPA_VALIDATION_RUN_DIR}"
