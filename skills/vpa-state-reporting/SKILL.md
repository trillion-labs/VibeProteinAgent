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
vpa report <traj_id> --workspace <name> --out report.html
```

`traj_` reports cover the query trajectory: tagged direct jobs plus tagged
workflows. `job_` reports cover one job unless the job is a legacy
`submit_candidate` handle; `wf_` reports cover one workflow only. A final
objective report should target the completed trajectory after `submit_candidate`.
Final selection comes from explicit `submit_candidate` output; reports are
read-only evidence rendering and do not auto-select candidates.

## Evidence

Summaries should include objective, criteria, workspace, source id, candidate
ids, ranking metrics, submitted outcome, report path, and caveats. Keep
uncertainty tied to measured evidence rather than narrative confidence.
