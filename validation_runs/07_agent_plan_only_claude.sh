#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "07_agent_plan_only_claude"
require_claude_cli

pdb="$(fixture_path tests/fixtures/6MRR.pdb)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/agent_plan_only_claude.json" \
  agent run \
  "Plan a compact VPA validation: design two sequences for the provided backbone, refold the best one, and describe report/visualization checks." \
  --target "${pdb}" \
  --provider claude-code \
  --model "${VPA_VALIDATION_CLAUDE_MODEL}" \
  --plan-only \
  --max-rounds 8 \
  --max-entries 3 \
  --workspace "${VPA_WORKSPACE}" \
  --json

run_id="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/agent_plan_only_claude.json" data.run_id)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/agent_status_claude.json" agent show "${run_id}" --workspace "${VPA_WORKSPACE}" --json

log "Claude Code agent plan-only scenario complete: ${VPA_VALIDATION_RUN_DIR}"
