---
name: vpa-run-and-workspace
description: Run direct VPA tools and explicit YAML workflows; manage workspaces, daemons, workers, and trajectory waits. Use when launching or executing jobs. To inspect finished traj_/job_/wf_ state or render reports, use vpa-state-reporting.
---

# VPA Run And Workspace

Use this for active execution: direct tool runs, explicit YAML workflows,
workspace setup, daemons, workers, waits, and artifacts produced by the jobs and
workflows you launch. For retrospective `traj_`/`job_`/`wf_` inspection or report rendering, use
vpa-state-reporting.

## Workspace

```text
vpa workspace create <name>
vpa workspace use <name>
vpa daemon start --workspace <name>
vpa daemon status --workspace <name>
```

The daemon is a precondition for execution. `vpa run` and `vpa workflow run`
only enqueue a job and return — it sits in `queued` and never progresses unless
the supervisor is running. Start the daemon and confirm `vpa daemon status`
before launching anything; otherwise the agent waits forever on a job that will
never run.

Prefer one workspace per objective/session, then start one trajectory for the
user query:

```text
vpa trajectory start --workspace <name> --user-query "<original request>" --json
vpa trajectory status <traj_id> --workspace <name> --json
```

The workspace owns execution; the trajectory owns query accounting. Reuse an
existing workspace only when the user points to its `traj_`/`job_`/`wf_`/artifacts;
otherwise create a fresh one.

## Warm pool

High-startup tools (model-weight load) can be preloaded so repeated jobs skip
the cold start:

```text
vpa daemon warm <tool>                 # warm every worker
vpa daemon warm <tool> --workers 0,1   # warm specific GPUs
```

Only helps `warm_load_capable` tools (check `vpa tool info <tool>`), and it pins
the model in worker VRAM — warm the few tools you'll reuse across many jobs
(e.g. a scorer re-run on every candidate), not everything. The daemon must be
running first; `vpa daemon warm` has no `--workspace` flag (it targets the
running supervisor).

## Active Job State

Don't rely on memory for jobs you just launched. While executing, use the job
records to keep the foreground objective moving:

```text
vpa wait <job_id> --workspace <name> --timeout-ms <ms>
vpa wait --trajectory <traj_id> --workspace <name> --timeout-ms <ms>
vpa status <job_id> --workspace <name> --json
vpa results <job_id> --workspace <name> --json
```

For retrospective workspace reconstruction, broad `traj_`/`job_`/`wf_` inspection, or
report rendering, use vpa-state-reporting.

`vpa wait --trajectory` accounts for both tagged direct jobs and tagged
workflows. It waits for launched execution to become terminal; use
`vpa trajectory status` to check final `submit_candidate` and report completion.

## Direct Tool

Use direct tools for known, narrow actions:

```text
vpa run <tool> key=value ... --workspace <name> --trajectory <traj_id>
vpa wait <job_id> --workspace <name>
vpa results <job_id> --workspace <name>
```

Prefer `vpa wait` over hand-written shell polling. If you must inspect state
directly, the terminal job states are exactly `completed`, `failed`, and
`cancelled`; `succeeded` is not a VPA state. Do not leave background pollers
running after the foreground task has enough state to continue.

Prefer `vpa wait --timeout-ms ...` over background pollers; use
`vpa wait --trajectory <traj_id> --timeout-ms ...` when the query has several
launched jobs/workflows.

Independent jobs run in parallel via the worker pool, so submit several at once
and wait on them together rather than serializing the exploration.

## Explicit Workflow

Use workflows when multiple deterministic tool calls need declared dataflow:

```text
vpa workflow run workflow.yaml --workspace <name> --trajectory <traj_id>
vpa workflow status <wf_id> --workspace <name>
```

Rough trade-off: prefer an explicit workflow for a long, fixed multi-step run
that benefits from declared dataflow, caching, and resume; prefer direct tools
when an external agent is steering a flexible, branching loop and deciding the
next call from each result.

`vpa workflow run` is an execution boundary, not the whole protein-design
strategy. Use it only when you decide the block's dataflow is fixed enough;
otherwise keep choosing direct `vpa run` calls from the accumulated evidence.

Keep workflow dataflow explicit. Runtime references such as
`${outputs.node.field}` belong in YAML workflows, not in natural language.

For Protenix -> ipSAE, wire structure candidates and PAE as separate inputs.
The candidate stream supplies the predicted structure (often mmCIF); ipSAE also
requires the matching Protenix PAE/confidence output:

```yaml
nodes:
  - id: protenix
    tool: protenix_v2
    inputs:
      input_json: ${params.protenix_input_json}

  - id: ipsae
    tool: ipsae
    inputs:
      candidates: ${outputs.protenix.candidates}
      pae_path: ${outputs.protenix.pae_path}
      chain_a: A
      chain_b: B
```

## Failure

Jobs fail (OOM, CUDA, bad params) — expect it during unattended runs. Read
`vpa logs <job_id>` for the cause, then fix the params and `vpa retry <job_id>`
(or `vpa run ... --retry N` for retryable errors), or route around the failure
with a different tool. Don't abandon the whole objective on a single failed job.

## Finalization

By default, run `objective_rank` once before `submit_candidate` and before
rendering the report — make it the standard last step of any run that produced
scoring/structure evidence, not just an option for close calls. Pass explicit
objective gates and a score formula, then prefer `submit_candidate` with
`rank=1` from that rank job. Run it even when a single candidate survives: the
ranked row carries the canonical id, gate verdicts, and score that the report's
ranking and "Why this candidate" cards render from tool output instead of prose,
which is exactly what makes the HTML report clean. Skip it only when there is
genuinely nothing to rank (e.g. one throwaway diagnostic), and say so in the
`submit_candidate` reasoning.
When the selected design has been enriched by
structure prediction or scoring, submit the enriched/ranked candidate rather
than an earlier sequence-only upstream candidate so the final dossier retains
the evidence. For accepted submissions, include `selection_summary` with a
short headline, one concise explanation, and 2-4 evidence bullets; this feeds
the report's "Why this candidate" card. Then record accepted or no acceptable candidate. If
`submit_candidate` returns `next_report_command`, run that command from any
directory; by default the report is written to the workspace root. Otherwise
render `vpa report <traj_id>`. One user query should end with one trajectory
report that covers the jobs/workflows used for that query.

Prefer canonical `candidates=[...]` wiring for evaluator tools so evidence
stays attached to the logical design. Use raw `pdb_path`/`structure_path`
diagnostics only when candidate identity is unavailable or intentionally
ambiguous, and note that such evidence may render as unmatched or missing from
final candidate dossiers.
If `objective_rank` collapses distinct logical designs into fewer rows than the
evaluated population, treat it as a candidate-identity contract failure. Do not
silently rely on the collapsed rank as final truth; inspect the per-candidate
tool outputs, record the caveat in `submit_candidate` reasoning if proceeding,
and prefer fixing the upstream candidate wiring before rendering the final
report.
