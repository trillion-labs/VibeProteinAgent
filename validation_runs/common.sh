#!/usr/bin/env bash
set -Eeuo pipefail

umask 077

VALIDATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${VALIDATION_DIR}/local.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${VALIDATION_DIR}/local.env"
  set +a
fi

if [[ -z "${VPA_REPO_ROOT:-}" ]]; then
  VPA_REPO_ROOT="$(cd "${VALIDATION_DIR}/.." && pwd)"
fi
export VPA_REPO_ROOT

VPA_VALIDATION_GPUS="${VPA_VALIDATION_GPUS:-0,1,2,3,4,5,6,7}"
VPA_VALIDATION_WORKERS="${VPA_VALIDATION_WORKERS:-8}"
VPA_VALIDATION_TIMEOUT_MS="${VPA_VALIDATION_TIMEOUT_MS:-7200000}"
VPA_VALIDATION_MAX_WAIT_S="${VPA_VALIDATION_MAX_WAIT_S:-7200}"
export VPA_VALIDATION_GPUS VPA_VALIDATION_WORKERS
export VPA_VALIDATION_TIMEOUT_MS VPA_VALIDATION_MAX_WAIT_S

if [[ -z "${VPA_BIN:-}" && -x "${VPA_REPO_ROOT}/.venv/bin/vpa" ]]; then
  VPA_BIN="${VPA_REPO_ROOT}/.venv/bin/vpa"
fi

if [[ -z "${VPA_PYTHON:-}" && -x "${VPA_REPO_ROOT}/.venv/bin/python" ]]; then
  VPA_PYTHON="${VPA_REPO_ROOT}/.venv/bin/python"
elif [[ -z "${VPA_PYTHON:-}" ]] && command -v python3 >/dev/null 2>&1; then
  VPA_PYTHON="$(command -v python3)"
fi

run_vpa() {
  (
    cd "${VPA_REPO_ROOT}"
    if [[ -n "${VPA_BIN:-}" ]]; then
      "${VPA_BIN}" "$@"
    elif command -v vpa >/dev/null 2>&1; then
      vpa "$@"
    elif [[ -n "${VPA_PYTHON:-}" ]]; then
      "${VPA_PYTHON}" -m vibeproteinagent.cli.main "$@"
    else
      echo "No vpa runner found. Set VPA_BIN or VPA_PYTHON." >&2
      return 127
    fi
  )
}

run_py() {
  if [[ -n "${VPA_PYTHON:-}" ]]; then
    "${VPA_PYTHON}" "$@"
  elif command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  else
    echo "No Python runner found. Set VPA_PYTHON." >&2
    return 127
  fi
}

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

init_scenario() {
  local scenario="$1"
  local stamp
  stamp="${VPA_VALIDATION_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
  export VPA_VALIDATION_RUN_ROOT="${VPA_VALIDATION_RUN_ROOT:-${VPA_REPO_ROOT}/output/validation_runs}"
  export VPA_VALIDATION_SCENARIO="${scenario}"
  export VPA_VALIDATION_RUN_DIR="${VPA_VALIDATION_RUN_ROOT}/${stamp}_${scenario}"
  mkdir -p "${VPA_VALIDATION_RUN_DIR}/logs" "${VPA_VALIDATION_RUN_DIR}/tmp"

  if [[ "${VPA_VALIDATION_ISOLATED:-1}" == "1" ]]; then
    export VPA_HOME="${VPA_VALIDATION_RUN_DIR}/vpa_home"
    export VPA_RUNTIME_DIR="${VPA_VALIDATION_RUN_DIR}/runtime"
    export VPA_WORKSPACE="validation_${scenario}_${stamp}"
  else
    export VPA_HOME="${VPA_HOME:-${HOME}/.vibeproteinagent}"
    export VPA_RUNTIME_DIR="${VPA_RUNTIME_DIR:-${VPA_VALIDATION_RUN_DIR}/runtime}"
    export VPA_WORKSPACE="${VPA_WORKSPACE:-validation_${scenario}_${stamp}}"
  fi

  export VPA_MODEL_CACHE="${VPA_MODEL_CACHE:-${HOME}/.cache/vibeproteinagent/models}"
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${VPA_VALIDATION_GPUS}}"
  mkdir -p "${VPA_HOME}" "${VPA_RUNTIME_DIR}" "${VPA_MODEL_CACHE}" "${HF_HOME}" "${TORCH_HOME}"

  normalize_credentials
  log "scenario=${scenario}"
  log "run_dir=${VPA_VALIDATION_RUN_DIR}"
  log "workspace=${VPA_WORKSPACE}"
  log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  credential_state
}

normalize_credentials() {
  if [[ -z "${HF_TOKEN:-}" && -n "${VPA_HF_CREDENTIAL:-}" ]]; then
    export HF_TOKEN="${VPA_HF_CREDENTIAL}"
  fi
  if [[ -z "${VPA_HF_CREDENTIAL:-}" && -n "${HF_TOKEN:-}" ]]; then
    export VPA_HF_CREDENTIAL="${HF_TOKEN}"
  fi
  if [[ -z "${OPENROUTER_API_KEY:-}" && -n "${VPA_OPENROUTER_CREDENTIAL:-}" ]]; then
    export OPENROUTER_API_KEY="${VPA_OPENROUTER_CREDENTIAL}"
  fi
  if [[ -z "${VPA_OPENROUTER_CREDENTIAL:-}" && -n "${OPENROUTER_API_KEY:-}" ]]; then
    export VPA_OPENROUTER_CREDENTIAL="${OPENROUTER_API_KEY}"
  fi
}

credential_state() {
  [[ -n "${HF_TOKEN:-}" ]] && log "HF credential: set" || log "HF credential: unset"
  [[ -n "${OPENROUTER_API_KEY:-}" ]] && log "OpenRouter credential: set" || log "OpenRouter credential: unset"
}

require_hf_credential() {
  [[ -n "${HF_TOKEN:-}${VPA_HF_CREDENTIAL:-}" ]] || die "Set HF_TOKEN or VPA_HF_CREDENTIAL for this scenario."
}

require_openrouter_credential() {
  [[ -n "${OPENROUTER_API_KEY:-}${VPA_OPENROUTER_CREDENTIAL:-}" ]] || die "Set OPENROUTER_API_KEY or VPA_OPENROUTER_CREDENTIAL for this scenario."
  export VPA_VALIDATION_OPENROUTER_MODEL="${VPA_VALIDATION_OPENROUTER_MODEL:-qwen/qwen3-8b}"
}

require_cli_provider_command() {
  local env_name="$1"
  local default_command="$2"
  local provider="$3"
  local command_name="${!env_name:-${default_command}}"
  command -v "${command_name}" >/dev/null 2>&1 || die "${provider} command ${command_name} not found. Set ${env_name} to the cluster wrapper."
}

require_codex_cli() {
  require_cli_provider_command VPA_CODEX_COMMAND codex codex
  export VPA_VALIDATION_CODEX_MODEL="${VPA_VALIDATION_CODEX_MODEL:-gpt-5.2}"
  export VPA_AGENT_TIMEOUT_S="${VPA_AGENT_TIMEOUT_S:-300}"
}

require_claude_cli() {
  require_cli_provider_command VPA_CLAUDE_CODE_COMMAND claude claude-code
  export VPA_VALIDATION_CLAUDE_MODEL="${VPA_VALIDATION_CLAUDE_MODEL:-sonnet}"
  export VPA_AGENT_TIMEOUT_S="${VPA_AGENT_TIMEOUT_S:-300}"
}

capture_vpa_json() {
  local out="$1"
  shift
  log "+ vpa $*"
  run_vpa "$@" | tee "${out}"
  assert_json_ok "${out}"
}

capture_vpa_json_allow_failure() {
  local out="$1"
  shift
  log "+ vpa $*"
  set +e
  run_vpa "$@" | tee "${out}"
  local rc=${PIPESTATUS[0]}
  set -e
  log "exit_code=${rc}"
  return 0
}

assert_json_ok() {
  local file="$1"
  run_py - "$file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
if data.get("ok") is not True:
    raise SystemExit(f"{path} is not ok: {data.get('error')}")
PY
}

json_get() {
  local file="$1"
  local path="$2"
  run_py - "$file" "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cur = json.load(handle)
for part in sys.argv[2].split("."):
    if part == "":
        continue
    if isinstance(cur, list):
        cur = cur[int(part)]
    else:
        cur = cur[part]
if isinstance(cur, (dict, list)):
    print(json.dumps(cur, separators=(",", ":")))
elif cur is None:
    print("")
else:
    print(cur)
PY
}

json_list() {
  local file="$1"
  local path="$2"
  run_py - "$file" "$path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cur = json.load(handle)
for part in sys.argv[2].split("."):
    if part == "":
        continue
    cur = cur[int(part)] if isinstance(cur, list) else cur[part]
for item in cur:
    if isinstance(item, (dict, list)):
        print(json.dumps(item, separators=(",", ":")))
    elif item is None:
        print("")
    else:
        print(item)
PY
}

ensure_daemon() {
  local workers="${1:-${VPA_VALIDATION_WORKERS}}"
  capture_vpa_json_allow_failure "${VPA_VALIDATION_RUN_DIR}/logs/daemon_status_before.json" daemon status --json
  if json_get "${VPA_VALIDATION_RUN_DIR}/logs/daemon_status_before.json" data.running | grep -qx "True"; then
    log "daemon already running in this runtime dir"
    return 0
  fi
  capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/daemon_start.json" daemon start --workers "${workers}" --workspace "${VPA_WORKSPACE}" --json
}

stop_daemon() {
  log "+ vpa daemon stop"
  run_vpa daemon stop --json >"${VPA_VALIDATION_RUN_DIR}/logs/daemon_stop.json" 2>&1 || true
}

wait_job_completed() {
  local job_id="$1"
  local out="${2:-${VPA_VALIDATION_RUN_DIR}/logs/wait_${job_id}.json}"
  capture_vpa_json "${out}" wait "${job_id}" --workspace "${VPA_WORKSPACE}" --timeout-ms "${VPA_VALIDATION_TIMEOUT_MS}" --json
  local state
  state="$(json_get "${out}" data.jobs.0.state)"
  if [[ "${state}" != "completed" ]]; then
    run_vpa status "${job_id}" --workspace "${VPA_WORKSPACE}" --json >"${VPA_VALIDATION_RUN_DIR}/logs/status_${job_id}.json" 2>&1 || true
    run_vpa logs "${job_id}" --workspace "${VPA_WORKSPACE}" --tail 200 --json >"${VPA_VALIDATION_RUN_DIR}/logs/stdout_${job_id}.json" 2>&1 || true
    run_vpa logs "${job_id}" --workspace "${VPA_WORKSPACE}" --stream stderr --tail 200 --json >"${VPA_VALIDATION_RUN_DIR}/logs/stderr_${job_id}.json" 2>&1 || true
    die "job ${job_id} ended with state=${state}; see ${VPA_VALIDATION_RUN_DIR}/logs"
  fi
}

wait_jobs_completed() {
  local out="$1"
  shift
  capture_vpa_json "${out}" wait "$@" --workspace "${VPA_WORKSPACE}" --timeout-ms "${VPA_VALIDATION_TIMEOUT_MS}" --json
  run_py - "${out}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    data = json.load(handle)
bad = [
    (job.get("job_id"), job.get("state"), job.get("error_code"), job.get("error_message"))
    for job in data["data"]["jobs"]
    if job.get("state") != "completed"
]
if bad:
    raise SystemExit(f"non-completed jobs: {bad}")
PY
}

assert_workflow_completed() {
  local file="$1"
  local state
  state="$(json_get "${file}" data.state)"
  [[ "${state}" == "completed" ]] || die "workflow ended with state=${state}; see ${file}"
}

assert_agent_completed() {
  local file="$1"
  local state
  state="$(json_get "${file}" data.state)"
  [[ "${state}" == "completed" ]] || die "agent run ended with state=${state}; see ${file}"
}

assert_file_nonempty() {
  local path="$1"
  [[ -s "${path}" ]] || die "expected non-empty file: ${path}"
}

fixture_path() {
  local rel="$1"
  local path="${VPA_REPO_ROOT}/${rel}"
  [[ -e "${path}" ]] || die "fixture missing: ${path}"
  printf '%s\n' "${path}"
}

make_proteinmpnn_batch() {
  local out="$1"
  local pdb="$2"
  run_py - "$out" "$pdb" <<'PY'
import json
import sys

out, pdb = sys.argv[1], sys.argv[2]
with open(out, "w", encoding="utf-8") as handle:
    for idx in range(8):
        record = {
            "pdb_path": pdb,
            "num_designs": 1,
            "temperature": 0.1,
            "seed": 1000 + idx,
        }
        handle.write(json.dumps(record) + "\n")
PY
}

report_agent_outputs() {
  local agent_json="$1"
  mapfile -t workflow_ids < <(json_list "${agent_json}" data.workflow_ids || true)
  for workflow_id in "${workflow_ids[@]}"; do
    [[ -n "${workflow_id}" ]] || continue
    capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/report_${workflow_id}.json" \
      report "${workflow_id}" --workspace "${VPA_WORKSPACE}" --viewer offline --format bundle --out "${VPA_VALIDATION_RUN_DIR}/${workflow_id}_report_bundle" --json
    assert_file_nonempty "${VPA_VALIDATION_RUN_DIR}/${workflow_id}_report_bundle/report.html"
  done

  mapfile -t job_ids < <(json_list "${agent_json}" data.job_ids || true)
  for job_id in "${job_ids[@]}"; do
    [[ -n "${job_id}" ]] || continue
    capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/status_${job_id}.json" status "${job_id}" --workspace "${VPA_WORKSPACE}" --json
  done
}
