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
OUT_DIR="validation/sharded_gpu_smoke/out"

mkdir -p "$OUT_DIR"

if [[ -f "${VPA_HOME:-$HOME/.vibeproteinagent}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${VPA_HOME:-$HOME/.vibeproteinagent}/env.sh"
fi

vpa daemon start \
  --workers "$GPU_WORKERS" \
  --cpu-workers "$CPU_WORKERS" \
  --workspace "$WORKSPACE" \
  --json | tee "$OUT_DIR/daemon_start.json"

vpa agent validate-plan "$PLAN" \
  --params-file "$PARAMS" \
  --workspace "$WORKSPACE" \
  --json | tee "$OUT_DIR/validate.json"

vpa agent submit-plan "$PLAN" \
  --params-file "$PARAMS" \
  --workspace "$WORKSPACE" \
  --max-wait-s "$MAX_WAIT_S" \
  --json | tee "$OUT_DIR/submit.json"

WORKFLOW_ID="$(
  python - "$OUT_DIR/submit.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
data = payload.get("data", payload)
workflow_ids = data.get("workflow_ids") or []
result = data.get("result") or {}
print((workflow_ids or [result.get("workflow_id")])[0])
PY
)"

vpa report "$WORKFLOW_ID" \
  --workspace "$WORKSPACE" \
  --out "$OUT_DIR/report.html"

python - "$WORKSPACE" "$WORKFLOW_ID" "$OUT_DIR/gpu_ids.json" <<'PY'
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

printf 'Sharded GPU smoke complete: %s\n' "$WORKFLOW_ID"
