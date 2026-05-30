#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${VPA_REPO_ROOT:-/mnt/vast/trillion/hyunjin/VibeProteinAgent}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
STAMP="${VPA_VALIDATION_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${VPA_VALIDATION_RUN_DIR:-${REPO_ROOT}/validation_runs/direct_pxdesign_protenix_chain_vm024_${STAMP}}"
RUN_ROOT="$(mkdir -p "${RUN_ROOT}" && cd "${RUN_ROOT}" && pwd)"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${LOG_DIR}"

exec > >(tee -a "${RUN_ROOT}/direct_chain.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_vpa() {
  (
    cd "${REPO_ROOT}"
    PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${VPA_PYTHON}" -m vibeproteinagent.cli.main "$@"
  )
}

run_py() {
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${VPA_PYTHON}" "$@"
}

export VPA_HOME="${VPA_HOME:-/mnt/vast/trillion/hyunjin/.vibeproteinagent}"
if [[ -f "${VPA_HOME}/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${VPA_HOME}/env.sh"
fi
export VPA_PYTHON="${VPA_PYTHON:-/mnt/vast/trillion/env/miniconda3/envs/vpa/bin/python}"
export VPA_RUNTIME_DIR="${VPA_RUNTIME_DIR:-${RUN_ROOT}/runtime}"
export VPA_MODEL_CACHE="${VPA_MODEL_CACHE:-/mnt/vast/trillion/hyunjin/.cache/vibeproteinagent/models}"
export HF_HOME="${HF_HOME:-/mnt/vast/trillion/hyunjin/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/vast/trillion/hyunjin/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VPA_WORKSPACE="${VPA_WORKSPACE:-direct_pxdesign_protenix_chain_${STAMP}}"
export VPA_VALIDATION_WORKERS="${VPA_VALIDATION_WORKERS:-8}"
export VPA_VALIDATION_MAX_WAIT_S="${VPA_VALIDATION_MAX_WAIT_S:-7200}"
mkdir -p "${VPA_HOME}" "${VPA_RUNTIME_DIR}" "${VPA_MODEL_CACHE}" "${HF_HOME}" "${TORCH_HOME}"

if [[ -z "${HF_TOKEN:-}" && -n "${VPA_HF_CREDENTIAL:-}" ]]; then
  export HF_TOKEN="${VPA_HF_CREDENTIAL}"
fi
if [[ -z "${VPA_HF_CREDENTIAL:-}" && -n "${HF_TOKEN:-}" ]]; then
  export VPA_HF_CREDENTIAL="${HF_TOKEN}"
fi

cleanup() {
  log "+ stopping validation daemon"
  run_vpa daemon stop --json >"${LOG_DIR}/daemon_stop.json" 2>&1 || true
}
trap cleanup EXIT

assert_json_ok() {
  local file="$1"
  run_py - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    envelope = json.load(handle)
if envelope.get("ok") is not True:
    raise SystemExit(f"{path} is not ok: {envelope.get('error')}")
PY
}

WORKFLOW="${RUN_ROOT}/direct_pxdesign_protenix_chain.yaml"
TARGET_PDB="${REPO_ROOT}/tests/fixtures/5o45.cif"
cat >"${WORKFLOW}" <<YAML
name: direct_pxdesign_protenix_chain
version: "1.0.0"
description: "Deterministic no-LLM GPU chain for pxdesign/proteinmpnn/protenix/protein_qc/structure_analysis/rank_designs."
nodes:
  - id: design
    tool: pxdesign
    inputs:
      target_pdb: "${TARGET_PDB}"
      target_chain: "A"
      binder_length: 50
      num_samples: 1
      num_steps: 5
      top_k_designs: 1
      mpnn_temperature: 0.1
      seed: 101
    input_provenance:
      target_pdb: known_input
      target_chain: known_input
      binder_length: known_input
      num_samples: known_input
      num_steps: known_input
      top_k_designs: known_input
      mpnn_temperature: known_input
      seed: known_input
    cache: false
    on_failure: stop
  - id: sequence_design
    tool: proteinmpnn
    inputs:
      candidates: "\${outputs.design.candidates}"
      num_designs: 1
      temperature: 0.1
      seed: 101
    input_provenance:
      candidates: compiled_wiring
      num_designs: known_input
      temperature: known_input
      seed: known_input
    cache: false
    on_failure: stop
  - id: structure_pred
    tool: protenix_v2
    inputs:
      candidates: "\${outputs.sequence_design.candidates}"
      seeds: [101]
      use_msa: false
      use_template: false
      use_rna_msa: false
      n_sample: 1
      n_step: 20
      n_cycle: 1
    input_provenance:
      candidates: compiled_wiring
      seeds: known_input
      use_msa: known_input
      use_template: known_input
      use_rna_msa: known_input
      n_sample: known_input
      n_step: known_input
      n_cycle: known_input
    cache: false
    on_failure: stop
  - id: qc
    tool: protein_qc
    inputs:
      candidates: "\${outputs.structure_pred.candidates}"
      metrics: ["plddt", "polybasic", "gravy", "instability_index", "pi"]
    input_provenance:
      candidates: compiled_wiring
      metrics: known_input
    cache: false
    on_failure: stop
  - id: structure_summary
    tool: structure_analysis
    inputs:
      candidates: "\${outputs.qc.candidates}"
      max_annotations: 50
      run_dssp: false
    input_provenance:
      candidates: compiled_wiring
      max_annotations: known_input
      run_dssp: known_input
    cache: false
    on_failure: stop
  - id: rank
    tool: rank_designs
    inputs:
      candidates: "\${outputs.structure_summary.candidates}"
      top_k: 1
    input_provenance:
      candidates: compiled_wiring
      top_k: known_input
    cache: false
    on_failure: stop
YAML

log "repo=${REPO_ROOT}"
log "run_root=${RUN_ROOT}"
log "workspace=${VPA_WORKSPACE}"
log "runtime=${VPA_RUNTIME_DIR}"
log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
log "workflow=${WORKFLOW}"
log "target=${TARGET_PDB}"

log "+ daemon start"
run_vpa daemon start --workers "${VPA_VALIDATION_WORKERS}" --workspace "${VPA_WORKSPACE}" --json \
  | tee "${LOG_DIR}/daemon_start.json"
assert_json_ok "${LOG_DIR}/daemon_start.json"

log "+ workflow run"
run_vpa workflow run "${WORKFLOW}" \
  --workspace "${VPA_WORKSPACE}" \
  --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
  --json \
  | tee "${LOG_DIR}/workflow_run.json"
assert_json_ok "${LOG_DIR}/workflow_run.json"

run_py - "${LOG_DIR}/workflow_run.json" "${RUN_ROOT}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

run_json, summary_json = map(Path, sys.argv[1:3])
with run_json.open(encoding="utf-8") as handle:
    envelope = json.load(handle)
data = envelope.get("data") or {}
expected = [
    "design",
    "sequence_design",
    "structure_pred",
    "qc",
    "structure_summary",
    "rank",
]
nodes = {node.get("node_id"): node for node in data.get("nodes") or []}
missing = [node_id for node_id in expected if node_id not in nodes]
if missing:
    raise SystemExit(f"missing workflow nodes: {missing}")
bad = {
    node_id: {
        "state": nodes[node_id].get("state"),
        "error_code": nodes[node_id].get("error_code"),
        "error_message": nodes[node_id].get("error_message"),
        "error_details": nodes[node_id].get("error_details"),
    }
    for node_id in expected
    if nodes[node_id].get("state") != "completed"
}
if data.get("state") != "completed" or bad:
    summary_json.write_text(json.dumps({"workflow_state": data.get("state"), "bad": bad}, indent=2))
    raise SystemExit(f"workflow did not complete: state={data.get('state')} bad={bad}")

checks = {}
for node_id in expected:
    output = nodes[node_id].get("output") or {}
    candidates = output.get("candidates")
    checks[node_id] = {
        "tool": nodes[node_id].get("tool"),
        "job_id": nodes[node_id].get("job_id"),
        "candidate_count": len(candidates) if isinstance(candidates, list) else None,
        "has_candidates": isinstance(candidates, list) and bool(candidates),
    }
    if node_id != "rank" and not checks[node_id]["has_candidates"]:
        raise SystemExit(f"{node_id} did not emit candidates")
rank_output = nodes["rank"].get("output") or {}
ranked = rank_output.get("ranked_candidates") or rank_output.get("candidates")
if not isinstance(ranked, list) or not ranked:
    raise SystemExit("rank did not emit ranked candidates/candidates")
summary = {
    "workflow_id": data.get("workflow_id"),
    "workflow_state": data.get("state"),
    "nodes": checks,
    "ranked_count": len(ranked),
    "top_rank": ranked[0] if ranked else None,
}
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
PY

workflow_id="$(run_py - "${LOG_DIR}/workflow_run.json" <<'PY'
import json, sys
print((json.load(open(sys.argv[1])).get("data") or {}).get("workflow_id") or "")
PY
)"
if [[ -n "${workflow_id}" ]]; then
  log "+ workflow status ${workflow_id}"
  run_vpa workflow status "${workflow_id}" --workspace "${VPA_WORKSPACE}" --json \
    | tee "${LOG_DIR}/workflow_status.json"
fi

log "direct chain validation complete: ${RUN_ROOT}"
