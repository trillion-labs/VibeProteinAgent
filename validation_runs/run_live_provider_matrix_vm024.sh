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

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if ! bash -n "${SCRIPT_PATH}"; then
  printf 'matrix script syntax preflight failed: %s\n' "${SCRIPT_PATH}" >&2
  exit 2
fi

LOG_DIR="${RUN_ROOT}/logs"
WORKSPACE_ROOT="${RUN_ROOT}/workspaces"
DAEMON_WORKSPACE="matrix_daemon_${STAMP}"
if [[ -d "${RUN_ROOT}" ]] && find "${RUN_ROOT}" -mindepth 1 -print -quit | grep -q .; then
  printf 'output directory is not empty; use a clean run directory for release evidence: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi
mkdir -p "${LOG_DIR}" "${WORKSPACE_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
LOG_DIR="${RUN_ROOT}/logs"
WORKSPACE_ROOT="${RUN_ROOT}/workspaces"
PLANNER_LABEL="two_stage_tool_mode_selection_ir"
SCRIPT_SHA_START="$(sha256sum "${SCRIPT_PATH}" | awk '{print $1}')"
if [[ -e "${RUN_ROOT}/status.tsv" || -e "${RUN_ROOT}/summary.json" || -e "${RUN_ROOT}/RUN_SUMMARY.md" ]]; then
  printf 'output directory already contains matrix result files; use a clean run directory: %s\n' "${RUN_ROOT}" >&2
  exit 2
fi

exec > >(tee -a "${RUN_ROOT}/matrix.log") 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

case_progress_monitor() {
  local workspace="$1"
  local case_name="$2"
  local provider="$3"
  local model="$4"
  local start_us="$5"
  local stop_file="$6"
  run_py - "${VPA_HOME}" "${workspace}" "${case_name}" "${provider}" "${model}" \
    "${start_us}" "${stop_file}" <<'PY'
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional in shell environments
    yaml = None

vpa_home, workspace, case_name, provider, model, start_us_raw, stop_file_raw = sys.argv[1:8]
start_us = int(start_us_raw)
stop_file = Path(stop_file_raw)
db_path = Path(vpa_home) / "workspaces" / workspace / "workspace.db"
use_color = os.environ.get("VPA_VALIDATION_COLOR", "1") != "0"


def style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if use_color else text


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def emit(line: str = "") -> None:
    print(line, flush=True)


def title(text: str) -> None:
    emit(f"[{stamp()}] {style('◆', '36;1')} {style(text, '1')}")


def info(text: str) -> None:
    emit(f"[{stamp()}]   {style('·', '36')} {text}")


def ok(text: str) -> None:
    emit(f"[{stamp()}]   {style('✓', '32;1')} {text}")


def warn(text: str) -> None:
    emit(f"[{stamp()}]   {style('!', '33;1')} {text}")


def fail(text: str) -> None:
    emit(f"[{stamp()}]   {style('✗', '31;1')} {text}")


def short(value: Any, n: int = 12) -> str:
    text = str(value or "")
    return text if len(text) <= n else text[:n]


def connect() -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    uri = f"file:{db_path}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=1.0)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error:
        return None


def safe_json(value: Any, default: Any) -> Any:
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (table,),
    ).fetchone()
    return row is not None


def latest_run(con: sqlite3.Connection) -> sqlite3.Row | None:
    if not table_exists(con, "agent_runs"):
        return None
    return con.execute(
        """
        select * from agent_runs
        where created_at >= ?
        order by created_at desc
        limit 1
        """,
        (start_us,),
    ).fetchone()


def rows_for_current_workflows(
    con: sqlite3.Connection,
    run: sqlite3.Row | None,
) -> list[sqlite3.Row]:
    if not table_exists(con, "workflows"):
        return []
    workflow_ids = safe_json(run["workflow_ids_json"], []) if run is not None else []
    if workflow_ids:
        placeholders = ",".join("?" for _ in workflow_ids)
        return con.execute(
            f"""
            select * from workflows
            where workflow_id in ({placeholders})
            order by submitted_at
            """,
            tuple(workflow_ids),
        ).fetchall()
    return con.execute(
        """
        select * from workflows
        where submitted_at >= ?
        order by submitted_at
        """,
        (start_us,),
    ).fetchall()


def nodes_for_workflow(con: sqlite3.Connection, workflow_id: str) -> list[sqlite3.Row]:
    if not table_exists(con, "workflow_nodes"):
        return []
    return con.execute(
        """
        select * from workflow_nodes
        where workflow_id = ?
        order by rowid
        """,
        (workflow_id,),
    ).fetchall()


def decisions(con: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    if not table_exists(con, "agent_decisions"):
        return []
    return con.execute(
        """
        select * from agent_decisions
        where run_id = ?
        order by created_at
        """,
        (run_id,),
    ).fetchall()


def plan_shape_from_descriptor(descriptor: str) -> list[dict[str, Any]]:
    if yaml is None:
        return []
    try:
        payload = yaml.safe_load(descriptor) or {}
    except Exception:
        return []
    nodes = payload.get("nodes")
    return nodes if isinstance(nodes, list) else []


def node_inputs(node: dict[str, Any]) -> list[str]:
    upstream: list[str] = []
    raw = node.get("upstream")
    if isinstance(raw, list):
        upstream.extend(str(v) for v in raw)
    inputs = node.get("inputs")
    if isinstance(inputs, dict):
        for value in inputs.values():
            if isinstance(value, str) and value.startswith("${outputs."):
                token = value.removeprefix("${outputs.").split(".", 1)[0]
                if token and token not in upstream:
                    upstream.append(token)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.startswith("${outputs."):
                        token = item.removeprefix("${outputs.").split(".", 1)[0]
                        if token and token not in upstream:
                            upstream.append(token)
    return upstream


def print_trajectory(round_no: int, workflow: sqlite3.Row) -> None:
    workflow_id = workflow["workflow_id"]
    title(
        f"Round {round_no} trajectory · workflow {short(workflow_id, 16)}"
    )
    nodes = plan_shape_from_descriptor(workflow["descriptor"] or "")
    if not nodes:
        info("trajectory descriptor unavailable; watching workflow node states")
        return
    for idx, node in enumerate(nodes, start=1):
        node_id = str(node.get("id") or f"node_{idx}")
        tool = str(node.get("tool") or "?")
        upstream = node_inputs(node)
        edge = f" ← {', '.join(upstream)}" if upstream else ""
        emit(
            f"[{stamp()}]     {style(str(idx).rjust(2), '36')}  "
            f"{style(tool, '1')}  {node_id}{edge}"
        )


def decision_label(decision_json: str) -> tuple[str, str]:
    payload = safe_json(decision_json, {})
    decision = str(payload.get("decision") or "?")
    reason = str(payload.get("reason") or payload.get("reasoning") or "")
    if len(reason) > 160:
        reason = reason[:157] + "..."
    return decision, reason


printed_run = False
printed_wait = False
seen_workflows: set[str] = set()
seen_node_state: dict[tuple[str, str], str] = {}
seen_decisions: set[tuple[int, int]] = set()
seen_run_state: str | None = None
round_by_workflow: dict[str, int] = {}

title(f"Live progress · {case_name} · {provider}/{model}")

while True:
    con = connect()
    if con is None:
        if not printed_wait:
            info(f"waiting for workspace database: {db_path}")
            printed_wait = True
        time.sleep(2)
        if stop_file.exists():
            break
        continue
    try:
        run = latest_run(con)
        if run is None:
            if not printed_wait:
                info("thinking / provider planning has started; waiting for agent run record")
                printed_wait = True
        else:
            run_state = str(run["state"])
            if not printed_run:
                title(f"Agent run {short(run['run_id'], 16)} started")
                info(f"state={run_state}  workspace={workspace}")
                printed_run = True
            if run_state != seen_run_state:
                if seen_run_state is not None:
                    info(f"agent state → {run_state}")
                seen_run_state = run_state

            workflows = rows_for_current_workflows(con, run)
            for index, workflow in enumerate(workflows, start=1):
                workflow_id = str(workflow["workflow_id"])
                round_by_workflow.setdefault(workflow_id, index)
                if workflow_id not in seen_workflows:
                    print_trajectory(round_by_workflow[workflow_id], workflow)
                    seen_workflows.add(workflow_id)
                wf_state = str(workflow["state"])
                wf_key = (workflow_id, "__workflow__")
                if seen_node_state.get(wf_key) != wf_state:
                    seen_node_state[wf_key] = wf_state
                    if wf_state == "completed":
                        ok(
                            f"Round {round_by_workflow[workflow_id]} workflow completed "
                            f"({short(workflow_id, 16)})"
                        )
                    elif wf_state == "failed":
                        fail(
                            f"Round {round_by_workflow[workflow_id]} workflow failed "
                            f"({short(workflow_id, 16)})"
                        )
                    elif wf_state == "running":
                        info(
                            f"Round {round_by_workflow[workflow_id]} workflow running "
                            f"({short(workflow_id, 16)})"
                        )

                for node in nodes_for_workflow(con, workflow_id):
                    key = (workflow_id, str(node["node_id"]))
                    node_state = str(node["state"])
                    if seen_node_state.get(key) == node_state:
                        continue
                    seen_node_state[key] = node_state
                    label = (
                        f"R{round_by_workflow[workflow_id]} "
                        f"{node['tool_name']}:{node['node_id']}"
                    )
                    job = f" job={short(node['job_id'])}" if node["job_id"] else ""
                    if node_state == "completed":
                        ok(f"{label} passed{job}")
                    elif node_state == "failed":
                        reason = " ".join(
                            str(v)
                            for v in (node["error_code"], node["error_message"])
                            if v
                        )
                        fail(f"{label} failed{job}  {reason}".rstrip())
                    elif node_state == "skipped":
                        warn(f"{label} skipped")
                    elif node_state == "running":
                        info(f"{label} running{job}")

            for row in decisions(con, str(run["run_id"])):
                key = (int(row["round_index"]), int(row["created_at"]))
                if key in seen_decisions:
                    continue
                seen_decisions.add(key)
                decision, reason = decision_label(row["decision_json"])
                msg = f"Round {int(row['round_index']) + 1} decision: {decision}"
                if reason:
                    msg += f" · {reason}"
                if decision in {"submit_candidate", "submitted_final_candidate"}:
                    ok(msg)
                elif decision.startswith("next_round"):
                    info(msg)
                else:
                    warn(msg)

            if run_state == "completed":
                ok(f"Agent run completed · {case_name}")
            elif run_state == "failed":
                fail(f"Agent run failed · {case_name}")
    except sqlite3.Error as exc:
        warn(f"progress monitor skipped one poll: {exc}")
    finally:
        con.close()

    if stop_file.exists():
        break
    time.sleep(3)

title(f"Progress monitor closed · {case_name}")
PY
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

if [[ "${VPA_VALIDATION_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  log "running deterministic preflight before live provider matrix"
  run_py "${REPO_ROOT}/validation_runs/preflight_live_provider_matrix.py" \
    | tee "${RUN_ROOT}/preflight_live_provider_matrix.json"
fi

STATUS_TSV="${RUN_ROOT}/status.tsv"
CASE_EVENTS_TSV="${RUN_ROOT}/case_events.tsv"
EXPECTED_CASES_TSV="${RUN_ROOT}/expected_cases.tsv"
SUMMARY_MD="${RUN_ROOT}/RUN_SUMMARY.md"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
RELEASE_GUARDRAIL_JSON="${RUN_ROOT}/release_evidence_guardrail.json"
printf 'case\tmode\tprovider\tmodel\tstatus\tcase_status\tprocess_exit_code\tapi_ok\trun_state\tvalidation_status\tseconds\toutput\n' > "${STATUS_TSV}"
printf 'timestamp\tcase\tmode\tprovider\tmodel\tevent\tseconds\tpath\n' > "${CASE_EVENTS_TSV}"
printf 'case\tmode\tprovider\tmodel\n' > "${EXPECTED_CASES_TSV}"

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
decision = data.get("decision") or {}
result = data.get("result") or {}
if (
    decision.get("decision") == "no_available_candidate"
    or result.get("status") == "no_available_candidate"
):
    if result.get("technical_failure") is not False:
        raise SystemExit("no_available_candidate did not report technical_failure=false")
    raise SystemExit(0)
final_submission = latest.get("final_submission")
if not isinstance(final_submission, dict):
    raise SystemExit("actual run completed without explicit final_submission")
if final_submission.get("terminal") is not True:
    raise SystemExit("final_submission is not terminal")
if final_submission.get("outcome") not in {"accepted", "no_acceptable_candidate"}:
    raise SystemExit(f"unexpected final_submission outcome: {final_submission.get('outcome')}")
if decision.get("decision") not in {"submit_candidate", "no_acceptable_candidate", "submitted_final_candidate"}:
    raise SystemExit(f"unexpected final decision: {decision.get('decision')}")
submit_job_id = decision.get("submit_job_id")
if decision.get("decision") in {"submit_candidate", "no_acceptable_candidate"} and not submit_job_id:
    raise SystemExit("standalone submit_candidate decision did not expose submit_job_id")
PY
}

inspect_case_json() {
  local output="$1"
  run_py - "${output}" <<'PY'
import json
import sys

path = sys.argv[1]
api_ok = "unknown"
run_state = "unknown"
try:
    with open(path, encoding="utf-8") as handle:
        envelope = json.load(handle)
    api_ok = "true" if envelope.get("ok") is True else "false"
    data = envelope.get("data")
    if isinstance(data, dict):
        state = data.get("state")
        if state is not None:
            run_state = str(state)
except Exception:
    pass
print(f"{api_ok}\t{run_state}")
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
  local output_tmp="${output}.tmp"
  local label="${case_name##*_}"
  local case_dir="${RUN_ROOT}/${label}"
  mkdir -p "${case_dir}"
  local stderr="${case_dir}/${case_name}.stderr.log"
  local stderr_tmp="${stderr}.tmp"
  local start elapsed rc status api_ok run_state validation_status
  local progress_stop="${case_dir}/${case_name}.progress.stop"
  local progress_pid=""
  local case_start_us
  local extra_args=()
  if [[ -n "${extra_param}" ]]; then
    extra_args+=(--param "${extra_param}")
  fi
  case_start_us="$(date '+%s%6N')"
  start="$(date +%s)"
  log "START ${case_name} (${mode}, ${provider}, ${model})"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${case_name}" "${mode}" "${provider}" "${model}" "STARTED" "0" "${output}" >> "${CASE_EVENTS_TSV}"
  rm -f "${output_tmp}" "${stderr_tmp}" "${progress_stop}"
  if [[ "${VPA_VALIDATION_PROGRESS:-1}" != "0" ]]; then
    case_progress_monitor \
      "${workspace}" "${case_name}" "${provider}" "${model}" \
      "${case_start_us}" "${progress_stop}" &
    progress_pid="$!"
  fi
  set +e
  if [[ "${mode}" == "plan" ]]; then
    if [[ -n "${target_arg}" ]]; then
      run_vpa agent run "${query}" --target "${target_arg}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --plan-only --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" \
        --workspace "${workspace}" --json > "${output_tmp}" 2> "${stderr_tmp}"
    else
      run_vpa agent run "${query}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --plan-only --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" \
        --workspace "${workspace}" --json > "${output_tmp}" 2> "${stderr_tmp}"
    fi
  else
    if [[ -n "${target_arg}" ]]; then
      run_vpa agent run "${query}" --target "${target_arg}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
        --workspace "${workspace}" --json > "${output_tmp}" 2> "${stderr_tmp}"
    else
      run_vpa agent run "${query}" "${extra_args[@]}" --provider "${provider}" --model "${model}" \
        --max-rounds 2 --max-entries "${VPA_AGENT_MAX_ENTRIES}" --max-wait-s "${VPA_VALIDATION_MAX_WAIT_S}" \
        --workspace "${workspace}" --json > "${output_tmp}" 2> "${stderr_tmp}"
    fi
  fi
  rc=$?
  set -e
  if [[ -n "${progress_pid}" ]]; then
    touch "${progress_stop}"
    wait "${progress_pid}" 2>/dev/null || true
  fi
  elapsed=$(( $(date +%s) - start ))
  mv -f "${output_tmp}" "${output}" 2>/dev/null || true
  mv -f "${stderr_tmp}" "${stderr}" 2>/dev/null || true
  IFS=$'\t' read -r api_ok run_state < <(inspect_case_json "${output}")
  if [[ "${rc}" -eq 0 ]] && validate_case "${mode}" "${output}" >> "${case_dir}/${case_name}.validate.log" 2>&1; then
    status="PASS"
    validation_status="PASS"
  else
    status="FAIL"
    validation_status="FAIL"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${mode}" "${provider}" "${model}" "${status}" "${status}" "${rc}" "${api_ok}" "${run_state}" "${validation_status}" "${elapsed}" "${output}" >> "${STATUS_TSV}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" "${case_name}" "${mode}" "${provider}" "${model}" "${status}" "${elapsed}" "${output}" >> "${CASE_EVENTS_TSV}"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_name}" "${mode}" "${provider}" "${model}" "BLOCKED" "BLOCKED" "127" "unknown" "blocked" "BLOCKED" "0" "${output}" >> "${STATUS_TSV}"
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

write_expected_cases() {
  : > "${EXPECTED_CASES_TSV}"
  printf 'case\tmode\tprovider\tmodel\n' >> "${EXPECTED_CASES_TSV}"
  for spec in "$@"; do
    IFS='|' read -r label provider model <<< "${spec}"
    if [[ "${SMOKE}" == "1" ]]; then
      printf 'actual_semantic_%s\tactual\t%s\t%s\n' "${label}" "${provider}" "${model}" >> "${EXPECTED_CASES_TSV}"
      continue
    fi
    if [[ "${INCLUDE_PLAN_ONLY}" == "1" ]]; then
      printf 'plan_semantic_%s\tplan\t%s\t%s\n' "${label}" "${provider}" "${model}" >> "${EXPECTED_CASES_TSV}"
      printf 'plan_binder_%s\tplan\t%s\t%s\n' "${label}" "${provider}" "${model}" >> "${EXPECTED_CASES_TSV}"
      printf 'plan_denovo_%s\tplan\t%s\t%s\n' "${label}" "${provider}" "${model}" >> "${EXPECTED_CASES_TSV}"
    fi
    printf 'actual_semantic_%s\tactual\t%s\t%s\n' "${label}" "${provider}" "${model}" >> "${EXPECTED_CASES_TSV}"
    printf 'actual_binder_%s\tactual\t%s\t%s\n' "${label}" "${provider}" "${model}" >> "${EXPECTED_CASES_TSV}"
    printf 'actual_denovo_%s\tactual\t%s\t%s\n' "${label}" "${provider}" "${model}" >> "${EXPECTED_CASES_TSV}"
  done
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

finalize_incomplete_run() {
  local rc=$?
  set +e
  stop_daemon
  write_summary
  log "summary=${SUMMARY_MD} (incomplete, exit=${rc})"
  exit "${rc}"
}

write_summary() {
  local script_sha_end release_evidence_valid
  script_sha_end="$(sha256sum "${SCRIPT_PATH}" | awk '{print $1}')"
  release_evidence_valid="true"
  if [[ "${script_sha_end}" != "${SCRIPT_SHA_START}" ]]; then
    release_evidence_valid="false"
    log "Matrix script changed during run; this run is not release evidence."
  fi
  run_py - "${STATUS_TSV}" "${SUMMARY_MD}" "${SUMMARY_JSON}" "${RUN_ROOT}" "${SMOKE}" "${SCRIPT_PATH}" "${SCRIPT_SHA_START}" "${script_sha_end}" "${release_evidence_valid}" "${RELEASE_GUARDRAIL_JSON}" <<'PY'
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

status_tsv, summary_md, summary_json, run_root = map(Path, sys.argv[1:5])
smoke = sys.argv[5] == "1"
script_path, script_sha_start, script_sha_end, release_evidence_valid = sys.argv[6:10]
release_guardrail_json = Path(sys.argv[10])
rows = list(csv.DictReader(status_tsv.open(encoding="utf-8"), delimiter="\t"))
expected_path = run_root / "expected_cases.tsv"
expected_rows = (
    list(csv.DictReader(expected_path.open(encoding="utf-8"), delimiter="\t"))
    if expected_path.is_file()
    else []
)
observed_cases = {row["case"] for row in rows}
missing_cases = [row for row in expected_rows if row["case"] not in observed_cases]
def case_status(row):
    return row.get("case_status") or row.get("status") or "MISSING"

passed = sum(1 for r in rows if case_status(r) == "PASS")
blocked = sum(1 for r in rows if case_status(r) == "BLOCKED")
failed = sum(1 for r in rows if case_status(r) == "FAIL")
lines = [
    f"# Live Provider Matrix on {os.environ.get('VPA_VALIDATION_HOST_LABEL', 'unknown-host')}",
    "",
    f"Run root: `{run_root}`",
    (
        f"Cases: {passed} passed / {failed} failed / {blocked} blocked / "
        f"{len(missing_cases)} missing / {len(rows)} recorded"
    ),
    "",
    "| Case | Mode | Provider | Model | Status | Seconds |",
    "| --- | --- | --- | --- | --- | ---: |",
]
for row in rows:
    lines.append(
        f"| `{row['case']}` | {row['mode']} | {row['provider']} | "
        f"`{row['model']}` | **{case_status(row)}** | {row['seconds']} |"
    )
for row in missing_cases:
    lines.append(
        f"| `{row['case']}` | {row['mode']} | {row['provider']} | "
        f"`{row['model']}` | **MISSING** | 0 |"
    )
lines.extend([
    "",
    "Artifacts:",
    f"- Status TSV: `{status_tsv}`",
    f"- Case events TSV: `{run_root / 'case_events.tsv'}`",
    f"- Expected cases TSV: `{expected_path}`",
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
    failed_rows = [row for row in rows_for_provider if case_status(row) == "FAIL"]
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
    status_values = {case_status(row) for row in provider_rows}
    if status_values <= {"BLOCKED"}:
        status = "BLOCKED"
    elif "FAIL" in status_values:
        status = "FAIL"
    else:
        status = "PASS"
    first_problem = next((row for row in provider_rows if case_status(row) != "PASS"), None)
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

case_counts = {
    "PASS": passed,
    "FAIL": failed,
    "BLOCKED": blocked,
    "MISSING": len(missing_cases),
}
provider_counts = Counter(record["status"] for record in providers)
pr_name = run_root.name
if not pr_name.startswith("pr"):
    pr_name = "live_provider_matrix"
def log_guardrail_violations(run_root):
    patterns = (
        "command not found",
        "syntax error",
        "unexpected eof",
        "unbound variable",
    )
    violations = []
    for path in (run_root / "matrix.log", run_root / "runner.log"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[-1000000:]
        for line in text.splitlines():
            lowered = line.lower()
            matched = next((pattern for pattern in patterns if pattern in lowered), None)
            if matched is None:
                continue
            violations.append(
                {
                    "path": str(path),
                    "pattern": matched,
                    "line": line[-500:],
                }
            )
            if len(violations) >= 20:
                return violations
    return violations
log_violations = log_guardrail_violations(run_root)
release_evidence = {
    "clean_run_directory_required": True,
    "script_path": script_path,
    "script_sha_start": script_sha_start,
    "script_sha_end": script_sha_end,
    "script_unchanged_during_run": script_sha_start == script_sha_end,
    "log_guardrail_violations": log_violations,
    "release_evidence_valid": release_evidence_valid == "true" and not log_violations,
}
release_guardrail_json.write_text(
    json.dumps(release_evidence, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
summary_json.write_text(
    json.dumps(
        {
            "pr": pr_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "planner": os.environ.get("VPA_VALIDATION_PLANNER_LABEL", "two_stage_ir"),
            "providers": providers,
            "counts": case_counts,
            "case_counts": case_counts,
            "provider_counts": {
                "PASS": provider_counts.get("PASS", 0),
                "FAIL": provider_counts.get("FAIL", 0),
                "BLOCKED": provider_counts.get("BLOCKED", 0),
                "MISSING": 0,
            },
            "expected_case_count": len(expected_rows),
            "recorded_case_count": len(rows),
            "missing_cases": missing_cases,
            "release_evidence": release_evidence,
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
  write_expected_cases "${providers[@]}"

  if [[ "${SMOKE}" == "1" ]]; then
    start_daemon
    trap finalize_incomplete_run EXIT
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
  trap finalize_incomplete_run EXIT
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
