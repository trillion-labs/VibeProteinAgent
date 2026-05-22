#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

WORKSPACE="${VPA_SMOKE_WORKSPACE:-sharded_gpu_smoke}"
GPU_WORKERS="${VPA_SMOKE_GPU_WORKERS:-2}"
CPU_WORKERS="${VPA_SMOKE_CPU_WORKERS:-2}"
MAX_WAIT_S="${VPA_SMOKE_MAX_WAIT_S:-21600}"
PLAN="validation/sharded_gpu_smoke/sharded_heavy_plan.json"
PARAMS="validation/agent_tool_coverage/params.local.json"
OUT_DIR="${VPA_SMOKE_OUT_DIR:-validation/sharded_gpu_smoke/out}"

mkdir -p "$OUT_DIR"

if [[ -f "${VPA_HOME:-$HOME/.vibeproteinagent}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${VPA_HOME:-$HOME/.vibeproteinagent}/env.sh"
fi
if [[ -n "${VPA_SMOKE_BOLTZ_CACHE:-}" ]]; then
  export VPA_BOLTZ_CACHE="$VPA_SMOKE_BOLTZ_CACHE"
fi
if [[ -n "${VPA_SMOKE_BOLTZGEN_CACHE:-}" ]]; then
  export VPA_BOLTZGEN_CACHE="$VPA_SMOKE_BOLTZGEN_CACHE"
fi

VPA_BIN="${VPA_BIN:-$(command -v vpa || true)}"
if [[ -z "$VPA_BIN" && -x "${VPA_HOME:-$HOME/.vibeproteinagent}/envs/vpa/bin/vpa" ]]; then
  VPA_BIN="${VPA_HOME:-$HOME/.vibeproteinagent}/envs/vpa/bin/vpa"
fi
if [[ -z "$VPA_BIN" ]]; then
  echo "vpa executable not found; set VPA_BIN or activate the VPA environment" >&2
  exit 127
fi
VPA_PYTHON_BIN="${VPA_PYTHON_BIN:-$(dirname "$VPA_BIN")/python}"
if [[ ! -x "$VPA_PYTHON_BIN" ]]; then
  echo "VPA python executable not found at $VPA_PYTHON_BIN; set VPA_PYTHON_BIN" >&2
  exit 127
fi

"$VPA_BIN" daemon start \
  --workers "$GPU_WORKERS" \
  --cpu-workers "$CPU_WORKERS" \
  --workspace "$WORKSPACE" \
  --json | tee "$OUT_DIR/daemon_start.json"

"$VPA_BIN" agent validate-plan "$PLAN" \
  --workspace "$WORKSPACE" \
  --json | tee "$OUT_DIR/validate.json"

"$VPA_BIN" agent submit-plan "$PLAN" \
  --params-file "$PARAMS" \
  --workspace "$WORKSPACE" \
  --max-wait-s "$MAX_WAIT_S" \
  --json | tee "$OUT_DIR/submit.json"

WORKFLOW_ID="$(
  "$VPA_PYTHON_BIN" - "$OUT_DIR/submit.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
data = payload.get("data", payload)
workflow_ids = data.get("workflow_ids") or []
result = data.get("result") or {}
print((workflow_ids or [result.get("workflow_id")])[0])
PY
)"

"$VPA_BIN" report "$WORKFLOW_ID" \
  --workspace "$WORKSPACE" \
  --out "$OUT_DIR/report.html"

"$VPA_PYTHON_BIN" - "$WORKSPACE" "$WORKFLOW_ID" "$OUT_DIR/gpu_ids.json" <<'PY'
import json
import sys
from vibeproteinagent.workspace import JobStore, Workspace

workspace, workflow_id, out_path = sys.argv[1:4]
store = JobStore(Workspace.resolve(workspace))
jobs = [
    {
        "job_id": job.job_id,
        "node": job.workflow_node_id,
        "tool": job.tool_name,
        "state": job.state.value,
        "worker_id": job.worker_id,
        "gpu_ids": job.gpu_ids,
    }
    for job in store.list(workflow_id=workflow_id, limit=1000)
]
gpu_jobs = [job for job in jobs if job["gpu_ids"]]
Path = __import__("pathlib").Path
Path(out_path).write_text(json.dumps({"workflow_id": workflow_id, "jobs": jobs}, indent=2))
if len({tuple(job["gpu_ids"]) for job in gpu_jobs}) < 2:
    raise SystemExit("expected shard jobs to record at least two distinct gpu_ids")
PY

"$VPA_PYTHON_BIN" - "$OUT_DIR/submit.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
data = payload.get("data", payload)
state = data.get("state") or (data.get("result") or {}).get("state")
if state != "completed":
    raise SystemExit(f"expected completed workflow, got {state!r}")
PY

printf 'Sharded GPU smoke complete: %s\n' "$WORKFLOW_ID"
