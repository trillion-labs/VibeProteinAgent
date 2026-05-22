#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "99_secret_guard"

log "Scanning source-like files for committed credential-looking tokens"

set +e
rg -n -I --hidden \
  -g '!.git/**' \
  -g '!output/**' \
  -g '!validation_runs/local.env' \
  -g '!validation_runs/artifacts/**' \
  -g '!validation_runs/logs/**' \
  -e 'hf_[A-Za-z0-9]{20,}' \
  -e 'sk-or-v1-[A-Za-z0-9_-]{20,}' \
  "${VPA_REPO_ROOT}" >"${VPA_VALIDATION_RUN_DIR}/logs/secret_scan.txt"
scan_rc=$?
set -e

if [[ ${scan_rc} -eq 0 ]]; then
  cat "${VPA_VALIDATION_RUN_DIR}/logs/secret_scan.txt"
  die "credential-looking token found in source-like files"
elif [[ ${scan_rc} -eq 1 ]]; then
  log "secret scan clean"
else
  die "rg failed during secret scan"
fi

log "secret guard complete: ${VPA_VALIDATION_RUN_DIR}"
