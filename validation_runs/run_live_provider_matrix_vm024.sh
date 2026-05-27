#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${VPA_REPO_ROOT:-/mnt/vast/trillion/hyunjin/VibeProteinAgent}"
REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
STAMP="${VPA_VALIDATION_STAMP:-$(date '+%Y%m%d_%H%M%S')}"
HOST_LABEL="$(hostname 2>/dev/null || printf 'unknown-host')"
SAFE_HOST_LABEL="${HOST_LABEL//[^A-Za-z0-9_.-]/_}"
RUN_BASE="${VPA_VALIDATION_RUN_BASE:-${REPO_ROOT}/validation_runs}"
RUN_ROOT="${VPA_VALIDATION_RUN_DIR:-${RUN_BASE}/live_provider_matrix_${SAFE_HOST_LABEL}_${STAMP}}"
PROVIDERS_ARG="${VPA_VALIDATION_PROVIDERS:-codex,claude,qwen}"
SMOKE="${VPA_VALIDATION_SMOKE:-0}"
INCLUDE_PLAN_ONLY="${VPA_VALIDATION_INCLUDE_PLAN_ONLY:-0}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --providers)
      PROVIDERS_ARG="$2"
      shift 2
      ;;
    --output-dir)
      RUN_ROOT="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --include-plan-only)
      INCLUDE_PLAN_ONLY=1
      shift
      ;;
    --actual-only)
      INCLUDE_PLAN_ONLY=0
      shift
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

LOG_DIR="${RUN_ROOT}/logs"
WORKSPACE_ROOT="${RUN_ROOT}/workspaces"
DAEMON_WORKSPACE="matrix_daemon_${STAMP}"
mkdir -p "${LOG_DIR}" "${WORKSPACE_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
LOG_DIR="${RUN_ROOT}/logs"
WORKSPACE_ROOT="${RUN_ROOT}/workspaces"
PLANNER_LABEL="two_stage_tool_mode_selection_ir"

exec > >(tee -a "${RUN_ROOT}/matrix.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_vpa() {
  (
    cd "${REPO_ROOT}"
    if [[ -n "${VPA_BIN:-}" ]]; then
      "${VPA_BIN}" "$@"
    elif [[ -n "${VPA_PYTHON:-}" ]]; then
      PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${VPA_PYTHON}" -m vibeproteinagent.cli.main "$@"
    elif [[ -x "${REPO_ROOT}/.venv/bin/vpa" ]]; then
      "${REPO_ROOT}/.venv/bin/vpa" "$@"
    else
      PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        python3 -m vibeproteinagent.cli.main "$@"
    fi
  )
}

run_py() {
  PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${VPA_PYTHON:-${REPO_ROOT}/.venv/bin/python}" "$@"
}

export VPA_HOME="${VPA_HOME:-/mnt/vast/trillion/hyunjin/.vibeproteinagent}"
if [[ -f "${VPA_HOME}/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${VPA_HOME}/env.sh"
fi
export VPA_RUNTIME_DIR="${VPA_RUNTIME_DIR:-${RUN_ROOT}/runtime}"
if [[ -z "${VPA_PYTHON:-}" && -x /mnt/vast/trillion/env/miniconda3/envs/vpa/bin/python ]]; then
  export VPA_PYTHON=/mnt/vast/trillion/env/miniconda3/envs/vpa/bin/python
fi
export VPA_MODEL_CACHE="${VPA_MODEL_CACHE:-/mnt/vast/trillion/hyunjin/.cache/vibeproteinagent/models}"
export HF_HOME="${HF_HOME:-/mnt/vast/trillion/hyunjin/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/mnt/vast/trillion/hyunjin/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VPA_VALIDATION_WORKERS="${VPA_VALIDATION_WORKERS:-8}"
export VPA_VALIDATION_TIMEOUT_MS="${VPA_VALIDATION_TIMEOUT_MS:-7200000}"
export VPA_VALIDATION_MAX_WAIT_S="${VPA_VALIDATION_MAX_WAIT_S:-7200}"
export VPA_VALIDATION_BINDER_MAX_ATTEMPTED_TRAJECTORIES="${VPA_VALIDATION_BINDER_MAX_ATTEMPTED_TRAJECTORIES:-4}"
export VPA_AGENT_TIMEOUT_S="${VPA_AGENT_TIMEOUT_S:-300}"
export VPA_AGENT_MAX_ENTRIES="${VPA_AGENT_MAX_ENTRIES:-3}"
export VPA_AGENT_MAX_TOKENS="${VPA_AGENT_MAX_TOKENS:-2048}"
export VPA_AGENT_DISABLE_REASONING="${VPA_AGENT_DISABLE_REASONING:-1}"
if [[ -z "${VPA_CLAUDE_CODE_ARGS:-}" ]]; then
  export VPA_CLAUDE_CODE_ARGS='-p --model {model} --output-format text --permission-mode dontAsk --tools "" --no-session-persistence'
fi
export VPA_VALIDATION_CODEX_MODEL="${VPA_VALIDATION_CODEX_MODEL:-gpt-5.4}"
export VPA_VALIDATION_CLAUDE_MODEL="${VPA_VALIDATION_CLAUDE_MODEL:-sonnet}"
export VPA_VALIDATION_OPENROUTER_MODEL="${VPA_VALIDATION_OPENROUTER_MODEL:-qwen/qwen3.5-9b}"
mkdir -p "${VPA_HOME}" "${VPA_RUNTIME_DIR}" "${VPA_MODEL_CACHE}" "${HF_HOME}" "${TORCH_HOME}"

STATUS_TSV="${RUN_ROOT}/status.tsv"
SUMMARY_MD="${RUN_ROOT}/RUN_SUMMARY.md"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
printf 'case\tmode\tprovider\tmodel\tstatus\texit_code\tseconds\toutput\n' > "${STATUS_TSV}"

TARGET_PDB="${REPO_ROOT}/tests/fixtures/6MRR.pdb"

semantic_query="Plan a VPA workflow for a protein-understanding task: find and analyze representative sequences for NAD(P)HX repair proteins with ADP-dependent NAD(P)H-hydrate dehydratase and NADPHX epimerase activity, then rank representative sequence candidates using only available VPA tools."
binder_query="Plan a compact VPA binder-design workflow against the provided target structure. Design at most one or two candidate binders, evaluate basic structural quality, rank candidates, and keep all runtime choices minimal."
denovo_query="Plan a compact de novo protein-design workflow using available VPA tools. Generate at most one small soluble candidate, create or evaluate sequence/structure evidence, perform basic QC, and rank candidates."

validate_case() {
  local mode="$1"
  local output="$2"
  run_py - "${mode}" "${output}" <<'PY'
import json
import sys

mode, path = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    envelope = json.load(handle)
if envelope.get("ok") is not True:
    raise SystemExit(f"CLI envelope not ok: {envelope.get('error')}")
data = envelope.get("data") or {}
plan = data.get("plan") or (data.get("result") or {}).get("plan")
if not isinstance(plan, dict):
    raise SystemExit("missing agent plan")
if "workflow_ref" in plan or "workflow" in plan:
    raise SystemExit("planner returned workflow_ref/workflow instead of workflow_plan")
workflow_plan = plan.get("workflow_plan")
if not isinstance(workflow_plan, dict):
    raise SystemExit("planner returned no workflow_plan object")
nodes = workflow_plan.get("nodes")
if not isinstance(nodes, list) or not nodes:
    raise SystemExit("workflow_plan.nodes is missing or empty")
bad_submit_nodes = [n.get("id") for n in nodes if n.get("tool") == "submit_candidate"]
if bad_submit_nodes:
    raise SystemExit(f"planner scheduled submit_candidate node(s): {bad_submit_nodes}")
serialized_plan = json.dumps(plan, sort_keys=True)
if "submit_final" in serialized_plan:
    raise SystemExit("planner leaked submit_final")

if mode == "plan":
    if data.get("state") != "planned":
        raise SystemExit(f"plan-only state is not planned: {data.get('state')}")
    raise SystemExit(0)

if data.get("state") != "completed":
    raise SystemExit(f"actual run state is not completed: {data.get('state')}")
summary = data.get("summary") or {}
latest = summary.get("latest") if isinstance(summary, dict) else None
if not isinstance(latest, dict):
    latest = summary if isinstance(summary, dict) else {}
final_submission = latest.get("final_submission")
if not isinstance(final_submission, dict):
    raise SystemExit("actual run completed without explicit final_submission")
if final_submission.get("terminal") is not True:
    raise SystemExit("final_submission is not terminal")
if final_submission.get("outcome") not in {"accepted", "no_acceptable_candidate"}:
    raise SystemExit(f"unexpected final_submission outcome: {final_submission.get('outcome')}")
decision = data.get("decision") or {}
if decision.get("decision") not in {"submit_candidate", "no_acceptable_candidate", "submitted_final_candidate"}:
    raise SystemExit(f"unexpected final decision: {decision.get('decision')}")
submit_job_id = decision.get("submit_job_id")
if decision.get("decision") in {"submit_candidate", "no_acceptable_candidate"} and not submit_job_id:
    raise SystemExit("standalone submit_candidate decision did not expose submit_job_id")
PY
}

case_output_path() {
  local case_name="$1"
  local label="${case_name##*_}"
  mkdir -p "${RUN_ROOT}/${label}"
  printf '%s/%s/%s.json\n' "${RUN_ROOT}" "${label}" "${case_name}"
}

run_case() {
  local case_name="$1"
  local mode="$2"
  local provider="$3"
  local model="$4"
  local query="$5"
  local target_arg="${6:-}"
  local extra_param="${7:-}"
  local workspace="matrix_${case_name}_${STAMP}"
  if [[ "${mode}" == "actual" ]]; then
    workspace="${DAEMON_WORKSPACE}"
  fi
  local output
  output="$(case_output_path "${case_name}")"
  local label="${case_name##*_}"
  local case_dir="${RUN_ROOT}/${label}"
  mkdir -p "${case_dir}"
  local stderr="${case_dir}/${case_name}.stderr.log"
  local start elapsed rc status
  local extra_args=()
  if [[ -n "${extra_param}" ]]; then
    extra_args+=(--param "${extra_param}")
  fi
  start="$(date +%s)"
  log "START ${case_name} (${mode}, ${provider}, ${model})"
  set +e
  if [[ "${mode}" == "plan" ]]; then
    if [[ -n "${target_arg}" ]]; then
      run_vpa agent run "${query}" --target "${target_arg}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --plan-only --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" \
        --workspace "${workspace}" --json > "${output}" 2> "${stderr}"
    else
      run_vpa agent run "${query}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --plan-only --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" \
        --workspace "${workspace}" --json > "${output}" 2> "${stderr}"
    fi
  else
    if [[ -n "${target_arg}" ]]; then
      run_vpa agent run "${query}" --target "${target_arg}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
        --workspace "${workspace}" --json > "${output}" 2> "${stderr}"
    else
      run_vpa agent run "${query}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
        --workspace "${workspace}" --json > "${output}" 2> "${stderr}"
    fi
  fi
  rc=$?
  set -e
  elapsed=$(( $(date +%s) - start ))
  if [[ "${rc}" -eq 0 ]] && validate_case "${mode}" "${output}" >> "${case_dir}/${case_name}.validate.log" 2>&1; then
    status="PASS"
  else
    status="FAIL"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${mode}" "${provider}" "${model}" "${status}" "${rc}" "${elapsed}" "${output}" >> "${STATUS_TSV}"
  log "END ${case_name}: ${status} (exit=${rc}, ${elapsed}s)"
}

record_blocked_case() {
  local case_name="$1"
  local mode="$2"
  local provider="$3"
  local model="$4"
  local reason="$5"
  local label="${case_name##*_}"
  mkdir -p "${RUN_ROOT}/${label}"
  local output="${RUN_ROOT}/${label}/${case_name}.blocked.txt"
  printf '%s\n' "${reason}" > "${output}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${mode}" "${provider}" "${model}" "BLOCKED" "127" "0" "${output}" >> "${STATUS_TSV}"
  log "SKIP ${case_name}: ${reason}"
}

provider_preflight() {
  local label="$1"
  local provider="$2"
  local model="$3"
  local out="${LOG_DIR}/preflight_${label}.log"
  case "${provider}" in
    codex)
      if ! command -v codex >/dev/null 2>&1; then
        printf 'codex command not found\n' > "${out}"
        return 1
      fi
      timeout 45s codex exec --model "${model}" --skip-git-repo-check --sandbox read-only - \
        > "${out}" 2>&1 <<'EOF'
Return exactly this JSON object and nothing else: {"ok":true}
EOF
      ;;
    claude-code)
      if ! command -v claude >/dev/null 2>&1; then
        printf 'claude command not found\n' > "${out}"
        return 1
      fi
      timeout 60s claude -p --model "${model}" --output-format text \
        --permission-mode dontAsk --tools "" --no-session-persistence \
        'Return exactly this JSON object and nothing else: {"ok":true}' \
        > "${out}" 2>&1
      ;;
    openrouter)
      if [[ -z "${OPENROUTER_API_KEY:-}${VPA_OPENROUTER_CREDENTIAL:-}" ]]; then
        printf 'OpenRouter credential is unset\n' > "${out}"
        return 1
      fi
      printf 'OpenRouter credential is set; API is exercised by matrix cases.\n' > "${out}"
      ;;
  esac
}

runtime_preflight() {
  local out="${LOG_DIR}/preflight_runtime.log"
  run_py - > "${out}" 2>&1 <<'PY'
import json
import os
import sys

from vibeproteinagent.install.installer import install_readiness

required_tools = ("boltzgen",)
report = install_readiness(required_tools, runtime_probe=True)
print(json.dumps(report, indent=2, sort_keys=True))
if report.get("install_ready") is not True:
    raise SystemExit("runtime preflight failed: required tools are not install-ready")

expected_planner = "two_stage_tool_mode_selection_ir"
observed = os.environ.get("VPA_VALIDATION_PLANNER_LABEL", expected_planner)
if observed != expected_planner:
    raise SystemExit(f"runtime preflight failed: planner={observed!r}")
PY
}

run_provider_plan_cases() {
  local label="$1"
  local provider="$2"
  local model="$3"
  if provider_preflight "${label}" "${provider}" "${model}"; then
    run_case "plan_semantic_${label}" "plan" "${provider}" "${model}" "${semantic_query}"
    run_case "plan_binder_${label}" "plan" "${provider}" "${model}" "${binder_query}" "${TARGET_PDB}" "max_attempted_trajectories=${VPA_VALIDATION_BINDER_MAX_ATTEMPTED_TRAJECTORIES}"
    run_case "plan_denovo_${label}" "plan" "${provider}" "${model}" "${denovo_query}"
  else
    local reason="provider preflight failed; see ${LOG_DIR}/preflight_${label}.log"
    record_blocked_case "plan_semantic_${label}" "plan" "${provider}" "${model}" "${reason}"
    record_blocked_case "plan_binder_${label}" "plan" "${provider}" "${model}" "${reason}"
    record_blocked_case "plan_denovo_${label}" "plan" "${provider}" "${model}" "${reason}"
  fi
}

run_provider_actual_cases() {
  local label="$1"
  local provider="$2"
  local model="$3"
  if provider_preflight "${label}" "${provider}" "${model}"; then
    run_case "actual_semantic_${label}" "actual" "${provider}" "${model}" "${semantic_query}"
    run_case "actual_binder_${label}" "actual" "${provider}" "${model}" "${binder_query}" "${TARGET_PDB}" "max_attempted_trajectories=${VPA_VALIDATION_BINDER_MAX_ATTEMPTED_TRAJECTORIES}"
    run_case "actual_denovo_${label}" "actual" "${provider}" "${model}" "${denovo_query}"
  else
    local reason="provider preflight failed; see ${LOG_DIR}/preflight_${label}.log"
    record_blocked_case "actual_semantic_${label}" "actual" "${provider}" "${model}" "${reason}"
    record_blocked_case "actual_binder_${label}" "actual" "${provider}" "${model}" "${reason}"
    record_blocked_case "actual_denovo_${label}" "actual" "${provider}" "${model}" "${reason}"
  fi
}

run_provider_smoke_cases() {
  local label="$1"
  local provider="$2"
  local model="$3"
  if provider_preflight "${label}" "${provider}" "${model}"; then
    run_case "actual_semantic_${label}" "actual" "${provider}" "${model}" "${semantic_query}"
  else
    local reason="provider preflight failed; see ${LOG_DIR}/preflight_${label}.log"
    record_blocked_case "actual_semantic_${label}" "actual" "${provider}" "${model}" "${reason}"
  fi
}

start_daemon() {
  log "Stopping any daemon in this validation runtime, then starting ${VPA_VALIDATION_WORKERS} workers."
  run_vpa daemon stop --json > "${LOG_DIR}/daemon_stop_before.json" 2>&1 || true
  run_vpa daemon start --workers "${VPA_VALIDATION_WORKERS}" --workspace "${DAEMON_WORKSPACE}" --json \
    > "${LOG_DIR}/daemon_start.json"
}

stop_daemon() {
  log "Stopping validation daemon."
  run_vpa daemon stop --json > "${LOG_DIR}/daemon_stop_after.json" 2>&1 || true
}

write_summary() {
  run_py - "${STATUS_TSV}" "${SUMMARY_MD}" "${SUMMARY_JSON}" "${RUN_ROOT}" "${SMOKE}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

status_tsv, summary_md, summary_json, run_root = map(Path, sys.argv[1:5])
smoke = sys.argv[5] == "1"
rows = list(csv.DictReader(status_tsv.open(encoding="utf-8"), delimiter="\t"))
passed = sum(1 for r in rows if r["status"] == "PASS")
blocked = sum(1 for r in rows if r["status"] == "BLOCKED")
failed = sum(1 for r in rows if r["status"] == "FAIL")
lines = [
    f"# Live Provider Matrix on {os.environ.get('VPA_VALIDATION_HOST_LABEL', 'unknown-host')}",
    "",
    f"Run root: `{run_root}`",
    f"Cases: {passed} passed / {failed} failed / {blocked} blocked / {len(rows)} total",
    "",
    "| Case | Mode | Provider | Model | Status | Seconds |",
    "| --- | --- | --- | --- | --- | ---: |",
]
for row in rows:
    lines.append(
        f"| `{row['case']}` | {row['mode']} | {row['provider']} | "
        f"`{row['model']}` | **{row['status']}** | {row['seconds']} |"
    )
lines.extend([
    "",
    "Artifacts:",
    f"- Status TSV: `{status_tsv}`",
    f"- Logs: `{run_root / 'logs'}`",
    f"- Case outputs: `{run_root}/{{codex,claude,qwen}}/*.json`",
])
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

def provider_label(row):
    case = row["case"]
    suffix = case.rsplit("_", 1)[-1]
    if suffix in {"codex", "claude", "qwen"}:
        return suffix
    provider = row["provider"]
    if provider == "claude-code":
        return "claude"
    if provider == "openrouter":
        return "qwen"
    return provider

def first_error(row):
    paths = [Path(row["output"])]
    stem = Path(row["output"]).with_suffix("")
    paths.extend([
        stem.with_suffix(".stderr.log"),
        stem.with_suffix(".validate.log"),
    ])
    for path in paths:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                if "see " in text:
                    hinted = text.rsplit("see ", 1)[-1].strip()
                    hinted_path = Path(hinted)
                    if hinted_path.is_file():
                        hinted_text = hinted_path.read_text(
                            encoding="utf-8",
                            errors="replace",
                        ).strip()
                        if hinted_text:
                            return hinted_text[:500]
                return text[:500]
    return ""

def reason_for(rows_for_provider, status):
    if status == "BLOCKED":
        return "preflight_failed"
    failed_rows = [row for row in rows_for_provider if row["status"] == "FAIL"]
    if failed_rows:
        err = first_error(failed_rows[0]).lower()
        if "planner_ir_schema_violation" in err:
            return "planner_ir_schema_violation_exhausted"
        if "non-json" in err or "json" in err:
            return "planner_json_parse_or_validation_failed"
        if "compile" in err or "fixup" in err:
            return "compiler_or_fixup_failed"
        if "final_submission" in err or "decision" in err:
            return "decision_or_submission_failed"
        return "workflow_failed"
    return "workflow_completed"

grouped = defaultdict(list)
for row in rows:
    grouped[provider_label(row)].append(row)

providers = []
for provider, provider_rows in sorted(grouped.items()):
    status_values = {row["status"] for row in provider_rows}
    if status_values <= {"BLOCKED"}:
        status = "BLOCKED"
    elif "FAIL" in status_values:
        status = "FAIL"
    else:
        status = "PASS"
    first_problem = next((row for row in provider_rows if row["status"] != "PASS"), None)
    duration = sum(float(row.get("seconds") or 0) for row in provider_rows)
    provider_dir = run_root / provider
    provider_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "provider": provider,
        "status": status,
        "reason": reason_for(provider_rows, status),
        "first_error": first_error(first_problem) if first_problem else "",
        "duration_seconds": duration,
        "artifacts_path": str(provider_dir),
    }
    (provider_dir / "provider_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    providers.append(record)

counts = Counter(record["status"] for record in providers)
pr_name = run_root.name
if not pr_name.startswith("pr"):
    pr_name = "live_provider_matrix"
summary_json.write_text(
    json.dumps(
        {
            "pr": pr_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "planner": os.environ.get("VPA_VALIDATION_PLANNER_LABEL", "two_stage_ir"),
            "providers": providers,
            "counts": {
                "PASS": counts.get("PASS", 0),
                "FAIL": counts.get("FAIL", 0),
                "BLOCKED": counts.get("BLOCKED", 0),
            },
            "notes": (
                "live provider matrix smoke cycle."
                if smoke
                else "live provider matrix cycle with current defaults."
            ),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

provider_specs() {
  local raw item
  IFS=',' read -ra raw <<< "${PROVIDERS_ARG}"
  for item in "${raw[@]}"; do
    case "${item}" in
      codex)
        printf 'codex|codex|%s\n' "${VPA_VALIDATION_CODEX_MODEL}"
        ;;
      claude|claude-code)
        printf 'claude|claude-code|%s\n' "${VPA_VALIDATION_CLAUDE_MODEL}"
        ;;
      qwen|openrouter)
        printf 'qwen|openrouter|%s\n' "${VPA_VALIDATION_OPENROUTER_MODEL}"
        ;;
      *)
        printf 'unknown provider selector: %s\n' "${item}" >&2
        return 2
        ;;
    esac
  done
}

main() {
  export VPA_VALIDATION_HOST_LABEL="${HOST_LABEL}"
  export VPA_VALIDATION_PLANNER_LABEL="${PLANNER_LABEL}"
  log "host=${HOST_LABEL}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || \
      log "nvidia-smi query failed; continuing because provider/plan preflight can still run."
  else
    log "nvidia-smi not found; continuing because provider/plan preflight can still run."
  fi
  log "run_root=${RUN_ROOT}"
  log "repo=${REPO_ROOT}"
  log "vpa_home=${VPA_HOME}"
  log "runtime=${VPA_RUNTIME_DIR}"
  log "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  log "smoke=${SMOKE}"
  log "include_plan_only=${INCLUDE_PLAN_ONLY}"
  log "binder_max_attempted_trajectories=${VPA_VALIDATION_BINDER_MAX_ATTEMPTED_TRAJECTORIES}"
  log "planner=${PLANNER_LABEL}"
  [[ -n "${OPENROUTER_API_KEY:-}${VPA_OPENROUTER_CREDENTIAL:-}" ]] && log "OpenRouter credential: set" || log "OpenRouter credential: unset"
  runtime_preflight || {
    log "Runtime preflight failed; see ${LOG_DIR}/preflight_runtime.log"
    exit 1
  }

  mapfile -t providers < <(provider_specs)

  if [[ "${SMOKE}" == "1" ]]; then
    start_daemon
    trap stop_daemon EXIT
    for spec in "${providers[@]}"; do
      IFS='|' read -r label provider model <<< "${spec}"
      run_provider_smoke_cases "${label}" "${provider}" "${model}"
    done
    stop_daemon
    trap - EXIT
    write_summary
    log "summary=${SUMMARY_MD}"
    return
  fi

  if [[ "${INCLUDE_PLAN_ONLY}" == "1" ]]; then
    for spec in "${providers[@]}"; do
      IFS='|' read -r label provider model <<< "${spec}"
      run_provider_plan_cases "${label}" "${provider}" "${model}"
    done
  else
    log "Skipping plan-only matrix cases; actual cases will plan and execute in one agent run."
  fi

  start_daemon
  trap stop_daemon EXIT
  for spec in "${providers[@]}"; do
    IFS='|' read -r label provider model <<< "${spec}"
    run_provider_actual_cases "${label}" "${provider}" "${model}"
  done

  stop_daemon
  trap - EXIT
  write_summary
  log "summary=${SUMMARY_MD}"
}

main "$@"
