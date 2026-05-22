#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "02_gpu_tool_matrix"
require_hf_credential
trap stop_daemon EXIT

ensure_daemon "${VPA_VALIDATION_WORKERS}"

pdb="$(fixture_path tests/fixtures/6MRR.pdb)"
protenix_input="$(fixture_path tests/fixtures/protenix_v2_input_tiny.json)"
batch="${VPA_VALIDATION_RUN_DIR}/tmp/proteinmpnn_batch.jsonl"
make_proteinmpnn_batch "${batch}" "${pdb}"

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/proteinmpnn_batch_submit.json" \
  run proteinmpnn --batch "${batch}" --workspace "${VPA_WORKSPACE}" --force-new --json
mapfile -t mpnn_jobs < <(json_list "${VPA_VALIDATION_RUN_DIR}/logs/proteinmpnn_batch_submit.json" data.jobs | run_py -c 'import json,sys; [print(json.loads(line)["job_id"]) for line in sys.stdin]')
wait_jobs_completed "${VPA_VALIDATION_RUN_DIR}/logs/proteinmpnn_batch_wait.json" "${mpnn_jobs[@]}"

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/protenix_submit.json" \
  run protenix_v2 "input_json=${protenix_input}" 'seeds=[101]' use_msa=false n_sample=1 n_step=20 n_cycle=1 \
  --workspace "${VPA_WORKSPACE}" --force-new --json
protenix_job="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/protenix_submit.json" data.job_id)"
wait_job_completed "${protenix_job}" "${VPA_VALIDATION_RUN_DIR}/logs/protenix_wait.json"

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/esm3_submit.json" \
  run esm3 sequence=ACDEFGHIKLMNPQRSTVWY "model_name=${VPA_ESM3_MODEL:-esm3_sm_open_v1}" 'tracks=["sequence"]' num_steps=1 temperature=0.1 \
  --workspace "${VPA_WORKSPACE}" --force-new --json
esm3_job="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/esm3_submit.json" data.job_id)"
wait_job_completed "${esm3_job}" "${VPA_VALIDATION_RUN_DIR}/logs/esm3_wait.json"

capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/rfdiffusion_submit.json" \
  run rfdiffusion mode=unconditional length_min=60 length_max=60 num_designs=1 num_diffusion_steps=15 seed=1701 \
  --workspace "${VPA_WORKSPACE}" --force-new --json
rfdiffusion_job="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/rfdiffusion_submit.json" data.job_id)"
wait_job_completed "${rfdiffusion_job}" "${VPA_VALIDATION_RUN_DIR}/logs/rfdiffusion_wait.json"

for job in "${mpnn_jobs[@]}" "${protenix_job}" "${esm3_job}" "${rfdiffusion_job}"; do
  capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/report_${job}.json" \
    report "${job}" --workspace "${VPA_WORKSPACE}" --viewer offline --out "${VPA_VALIDATION_RUN_DIR}/${job}_report.html" --json
  assert_file_nonempty "${VPA_VALIDATION_RUN_DIR}/${job}_report.html"
done

log "GPU tool matrix complete: ${VPA_VALIDATION_RUN_DIR}"
