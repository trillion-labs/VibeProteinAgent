#!/usr/bin/env bash
set -Eeuo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
init_scenario "01_report_visualize_cli"
trap stop_daemon EXIT

ensure_daemon 1

pdb="$(fixture_path tests/fixtures/6MRR.pdb)"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/structure_submit.json" \
  run structure_analysis "structure_path=${pdb}" max_annotations=40 \
  --workspace "${VPA_WORKSPACE}" --force-new --json

job_id="$(json_get "${VPA_VALIDATION_RUN_DIR}/logs/structure_submit.json" data.job_id)"
wait_job_completed "${job_id}" "${VPA_VALIDATION_RUN_DIR}/logs/structure_wait.json"

report_html="${VPA_VALIDATION_RUN_DIR}/structure_analysis_report.html"
report_bundle="${VPA_VALIDATION_RUN_DIR}/structure_analysis_report_bundle"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/report_html.json" \
  report "${job_id}" --workspace "${VPA_WORKSPACE}" --viewer offline --out "${report_html}" --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/report_bundle.json" \
  report "${job_id}" --workspace "${VPA_WORKSPACE}" --viewer offline --format bundle --out "${report_bundle}" --json
assert_file_nonempty "${report_html}"
assert_file_nonempty "${report_bundle}/report.html"

offline_html="${VPA_VALIDATION_RUN_DIR}/6mrr_offline_viewer.html"
online_html="${VPA_VALIDATION_RUN_DIR}/6mrr_online_viewer.html"
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/visualize_offline.json" \
  visualize "${pdb}" --viewer offline --title "6MRR offline validation" --out "${offline_html}" --json
capture_vpa_json "${VPA_VALIDATION_RUN_DIR}/logs/visualize_online.json" \
  visualize "${pdb}" --viewer online --title "6MRR online validation" --out "${online_html}" --json
assert_file_nonempty "${offline_html}"
assert_file_nonempty "${online_html}"

grep -q "Loading offline viewer" "${offline_html}" || die "offline visualization is missing offline viewer shell"
grep -q "https://3Dmol.org/build/3Dmol-min.js" "${online_html}" || die "online visualization is missing 3Dmol CDN URL"
grep -Fq '$3Dmol' "${online_html}" || die "online visualization is missing 3Dmol entry point"

log "report + visualize complete: ${VPA_VALIDATION_RUN_DIR}"
