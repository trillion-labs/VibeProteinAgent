#!/usr/bin/env bash
set -euo pipefail

cd /mnt/vast/trillion/hyunjin/VibeProteinAgent

export VPA_HOME=/mnt/vast/trillion/hyunjin/.vibeproteinagent
source "$VPA_HOME/env.sh"
export PATH="$VPA_HOME/envs/vpa/bin:$PATH"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

OUT="$PWD/validation_runs/agent_tool_coverage_vm007_direct_retry_optree_20260523_121447"
WS="agent_tool_coverage_vm007_direct_retry_optree_20260523_121447"
export VPA_RUNTIME_DIR="$OUT/runtime"
mkdir -p "$OUT" "$VPA_RUNTIME_DIR"
echo "$WS" > "$OUT/workspace.txt"

VPA="$VPA_HOME/envs/vpa/bin/vpa"
PY="$VPA_HOME/envs/vpa/bin/python"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$OUT/run.log"
}

extract_wf() {
  "$PY" - "$1" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
print(data.get("data", {}).get("workflow_id", ""))
PY
}

cleanup() {
  log "stopping daemon"
  "$VPA" daemon stop --workspace "$WS" --json > "$OUT/daemon_stop.json" 2>&1 || true
}
trap cleanup EXIT

log "host=$(hostname) workspace=$WS VPA_HOME=$VPA_HOME VPA_RUNTIME_DIR=$VPA_RUNTIME_DIR"
head -1 "$VPA" | tee "$OUT/vpa_shebang.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee "$OUT/gpus_before.txt"

log "starting daemon"
"$VPA" daemon start --workers 8 --cpu-workers 4 --workspace "$WS" --json | tee "$OUT/daemon_start.json"
"$VPA" daemon status --workspace "$WS" --json | tee "$OUT/daemon_status.json"

log "validating plans"
"$VPA" agent validate-plan validation/agent_tool_coverage/02_design_engines_plan.json --workspace "$WS" --json | tee "$OUT/02_validate.json"
"$VPA" agent validate-plan validation/agent_tool_coverage/03_foundation_models_plan.json --workspace "$WS" --json | tee "$OUT/03_validate.json"

log "submitting run 02"
"$VPA" agent submit-plan validation/agent_tool_coverage/02_design_engines_plan.json \
  --params-file validation/agent_tool_coverage/params.local.json \
  --workspace "$WS" \
  --max-wait-s 14400 \
  --json | tee "$OUT/02_result.json"
WF2=$(extract_wf "$OUT/02_result.json")
log "run 02 workflow=$WF2"
"$VPA" report "$WF2" --workspace "$WS" --out "$OUT/02_report.html" --json | tee "$OUT/02_report.json"

log "submitting run 03"
"$VPA" agent submit-plan validation/agent_tool_coverage/03_foundation_models_plan.json \
  --params-file validation/agent_tool_coverage/params.local.json \
  --workspace "$WS" \
  --max-wait-s 21600 \
  --json | tee "$OUT/03_result.json"
WF3=$(extract_wf "$OUT/03_result.json")
log "run 03 workflow=$WF3"
"$VPA" report "$WF3" --workspace "$WS" --out "$OUT/03_report.html" --json | tee "$OUT/03_report.json"

log "final daemon status"
"$VPA" daemon status --workspace "$WS" --json | tee "$OUT/daemon_status_final.json"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | tee "$OUT/gpus_after.txt"
log "completed retry runs"
