#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "08_agent_plan_only_openrouter_qwen3_8b"
require_openrouter_credential

pdb="$(fixture_path tests/fixtures/6MRR.pdb)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/agent_plan_only_openrouter_qwen3_8b.json" \
  agent run \
  "Plan a compact VPA validation: design two sequences for the provided backbone, refold the best one, and describe report/visualization checks." \
  --target "${pdb}" \
  --provider openrouter \
  --model "${VPA_VALIDATION_OPENROUTER_MODEL}" \
  --plan-only \
  --max-rounds 8 \
  --max-entries 3 \
  --workspace "${VPA_WORKSPACE}" \
  --json

run_id="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/agent_plan_only_openrouter_qwen3_8b.json" data.run_id)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/agent_status_openrouter_qwen3_8b.json" agent show "${run_id}" --workspace "${VPA_WORKSPACE}" --json

log "OpenRouter Qwen3-8B agent plan-only scenario complete: ${VPA_VALIDATION_RUN_DIR}"
