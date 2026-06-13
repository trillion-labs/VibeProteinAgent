---
name: vpa-state-reporting
description: Inspect existing VPA traj_/job_/wf_ state and render evidence-centered HTML reports. Use for reading finished state and reports; to launch jobs or workflows, use vpa-run-and-workspace.
---

# VPA State Reporting

Use this for `traj_`, `job_`, or `wf_` inspection and HTML reports.

## Inspect

```text
vpa trajectory status <traj_id> --workspace <name> --json
vpa jobs list --workspace <name> --json   # enumerate this workspace's jobs
vpa status <job_id> --workspace <name>
vpa results <job_id> --workspace <name>
vpa workflow status <wf_id> --workspace <name>
vpa artifacts <job_id> --workspace <name>
```

`vpa trajectory status` is the query-level completion view. `vpa jobs list` is
workspace-scoped; filter with `--state`, `--tool`, `--workflow`, `--trajectory`,
or `--since` when a workspace is shared.

## Report

```text
vpa report <traj_id> --workspace <name>
```

Each query gets its own uniquely-named workspace; write its report to a unique
path. Omitting `--out` writes `<id>_report.html` in the resolved workspace root
(`$VPA_HOME/workspaces/<workspace>/`). Do not hardcode user-specific absolute
paths; use the `report_html` value returned by `vpa report --json` or the
printed "Report written" path. Don't override it with a shared `report.html`
that overwrites a prior query's report.

`traj_` reports cover the query trajectory: tagged direct jobs plus tagged
workflows. `job_` reports cover one job unless the job is a legacy
`submit_candidate` handle; `wf_` reports cover one workflow only. A final
objective report should target the completed trajectory after `submit_candidate`.
Final selection comes from explicit `submit_candidate` output; reports are
read-only deterministic evidence rendering and do not auto-select candidates.
Do not hand-edit generated HTML to add missing artifacts or rationale. If the
report is missing evidence, record that evidence through VPA jobs or structured
artifacts, then render the report again.
If final candidate dossiers lack validation evidence, fix the upstream VPA
candidate/rank/submit flow and re-render; do not patch the report narrative.
Prefer final submissions that reference enriched/ranked candidates carrying the
selected evidence. Submitting an earlier sequence-only candidate can make the
final dossier look incomplete even when downstream structure or scoring jobs
exist elsewhere in the trajectory.
If `vpa report` fails preflight because an accepted submission cannot be joined
to `objective_rank` evidence, re-submit using the ranked candidate row,
`ranked_candidate_id`, or another recorded stable identity, then render again.

## Evidence

Summaries should include objective, criteria, workspace, source id, candidate
ids, ranking metrics, submitted outcome, report path, and caveats. Keep
uncertainty tied to measured evidence rather than narrative confidence.
When adaptive subset validation was used, summarize generated, evaluated,
ranked, and submitted populations separately.
