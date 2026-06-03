# VPA Execution Environment Runbook

This runbook is for agents creating workspaces and executing VPA workflows for
external users after installation is ready. Keep it portable: never hard-code a
developer name, home directory, cluster host, or one-off local path. Use
user-provided paths and repository-relative paths.

For installation, repair, binary setup, sidecars, or model weights, read
[vpa_installation.md](vpa_installation.md) first.

## Portable Variables

Use these names in commands and explanations:

- `PROJECT_ROOT`: the VPA repository checkout.
- `VPA_WORKSPACE`: the logical workspace name for this run.
- `VPA_HOME`: optional VPA state root. If unset, VPA uses its default home.
- `VPA_PYTHON`: optional Python executable for the VPA environment.
- `TARGET_PDB`: user-provided target structure path when a workflow needs one.
- `PYTHONPATH`: include `$PROJECT_ROOT/src` when running from a source checkout.

Prefer the installed `vpa` executable when available. From a source checkout, a
portable fallback is:

```bash
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
${VPA_PYTHON:-python} -m vibeproteinagent.cli.main version
```

If an environment activation script is supplied by the user or deployment, source
it before running VPA commands. Do not invent activation paths.

## Command Surface

The main CLI groups are `tools`, `tool`, `workspace`, `workflow`, `daemon`,
`mcp`, `agent`, and `install`. Top-level operational commands include `run`,
`status`, `wait`, `logs`, `results`, `cancel`, `retry`, `artifacts`,
`download`, `export`, `doctor`, `visualize`, and `report`.

Use `vpa tools --json` and `vpa tool info <tool> --json` to inspect the current
installed tool surface before assuming a tool or mode is available. Use
`vpa mcp serve` only when the user explicitly asks to expose VPA tools through
MCP.

## Workspace Setup

Create and select a workspace before live execution. Workspace names should be
short, descriptive, and free of personal identifiers.

```bash
export VPA_WORKSPACE="${VPA_WORKSPACE:-design_round}"

vpa workspace create "$VPA_WORKSPACE"
vpa workspace use "$VPA_WORKSPACE"
vpa workspace status
```

If `vpa` is not on `PATH`, use:

```bash
${VPA_PYTHON:-python} -m vibeproteinagent.cli.main workspace create "$VPA_WORKSPACE"
${VPA_PYTHON:-python} -m vibeproteinagent.cli.main workspace use "$VPA_WORKSPACE"
```

## Readiness Check

Before live execution, confirm that the requested profile or tools are ready. If
they are not ready, switch to [vpa_installation.md](vpa_installation.md) instead
of guessing paths or patching workflow parameters.

```bash
vpa install doctor --profile full
vpa install doctor --tools foldseek,mmseqs2_gpu,template_select
```

## Worker Lifecycle

Start workers for the selected workspace before live tool, workflow, or agent
execution. Use CPU workers for light local adapters and GPU workers for GPU
design or prediction tools.

```bash
vpa daemon start --workers 8 --cpu-workers 8 --workspace "$VPA_WORKSPACE"
vpa daemon status --workspace "$VPA_WORKSPACE"
```

If a run should not use GPUs, start only CPU workers:

```bash
vpa daemon start --workers 0 --cpu-workers 8 --workspace "$VPA_WORKSPACE"
```

## Warm Tool Pools

After the daemon is running, preload tools that the next workflow is likely to
use. `vpa daemon warm` talks to the currently running daemon through
`VPA_RUNTIME_DIR`; it does not take a `--workspace` flag, so keep the same
runtime environment that was used for `daemon start`.

```bash
vpa daemon warm sequence_analysis --json
vpa daemon warm protein_qc --json
```

For GPU design or prediction tools, warm only after `vpa install doctor` reports
the tool ready and GPU workers are running:

```bash
vpa daemon warm proteinmpnn --workers all --json
vpa daemon warm protenix_v2 --workers all --json
```

Do not treat warm failures as scientific results. They are environment/readiness
signals: inspect the `warmed[]` entries, repair installation or worker setup if
needed, then run the workflow.

## Direct Tool Runs

Use direct tool runs for targeted debugging or when the user explicitly asks to
run one adapter. Use `KEY=VALUE` positional parameters for simple values and a
params file when parameters are complex.

```bash
vpa run sequence_analysis \
  --workspace "$VPA_WORKSPACE" \
  sequence=MKTIIALSYIFCLVFADYKDDDDK

cat > params.json <<'JSON'
{"sequence": "MKTIIALSYIFCLVFADYKDDDDK"}
JSON
vpa run sequence_analysis --workspace "$VPA_WORKSPACE" --params-file params.json

vpa status <job_id> --workspace "$VPA_WORKSPACE"
vpa wait <job_id> --workspace "$VPA_WORKSPACE"
vpa results <job_id> --workspace "$VPA_WORKSPACE" --json
```

## Agent Runs From Natural Language

For user-facing workflows, prefer `vpa agent run`. The user request can be plain
natural language; the agent should handle routing, planning, validation,
execution, decision, and reporting through VPA's deterministic contracts.

Do not invent run-boundary options. Before live execution, resolve these fields
from the user's request, explicit environment/configuration, or a direct user
confirmation:

- provider and model;
- workspace name;
- target or other required user assets;
- max rounds;
- max wait time;
- plan-only vs live execution;
- GPU/CPU worker budget when the daemon is not already running.

If any required live-run value is absent from the user request and not already
configured, ask the user once before running. Use the user's answer verbatim in
the command. If the user explicitly says to choose defaults, use conservative
defaults and report them before execution.

```bash
export VPA_PROVIDER="<user-confirmed-provider>"
export VPA_MODEL="<user-confirmed-model>"
export VPA_MAX_ROUNDS="<user-confirmed-max-rounds>"
export VPA_MAX_WAIT_S="<user-confirmed-max-wait-seconds>"

vpa agent run \
  "Design protein binders for the supplied target structure and report the top candidates." \
  --workspace "$VPA_WORKSPACE" \
  --provider "$VPA_PROVIDER" \
  --model "$VPA_MODEL" \
  --target "$TARGET_PDB" \
  --max-rounds "$VPA_MAX_ROUNDS" \
  --max-wait-s "$VPA_MAX_WAIT_S"
```

For expensive or uncertain runs, validate plan shape first:

```bash
vpa agent run \
  "Retrieve representative sequences for this protein family and rank candidates." \
  --workspace "$VPA_WORKSPACE" \
  --provider "$VPA_PROVIDER" \
  --model "$VPA_MODEL" \
  --plan-only
```

Track persisted agent runs:

```bash
vpa agent runs --workspace "$VPA_WORKSPACE"
vpa agent show <agent_run_id> --workspace "$VPA_WORKSPACE"
```

For non-executing classification/planning or saved plans:

```bash
vpa agent ask "Explain which VPA tools would answer this request." \
  --workspace "$VPA_WORKSPACE"
vpa agent validate-plan plan.json --workspace "$VPA_WORKSPACE"
vpa agent submit-plan plan.json --workspace "$VPA_WORKSPACE" --max-wait-s 7200
```

## Workflow Runs

Use `vpa workflow run` when the user provides or requests an explicit workflow
descriptor rather than an agent-authored plan.

```bash
vpa workflow run path/to/workflow.yaml --workspace "$VPA_WORKSPACE"
vpa workflow status <workflow_id> --workspace "$VPA_WORKSPACE"
```

## Reporting

Reports are read-only over workspace state and artifacts. For LLM-agent runs,
use the `arun_...` agent run id by default so the HTML includes the agent
timeline, workflow rounds, decisions, final response, and candidate
archive-derived candidates/snapshots where available for the user's query.
Use a `wf_...` workflow id or `job_...` job id only when the user explicitly asks
for a round-specific, workflow-specific, or direct-tool report.

```bash
vpa report arun_<id> \
  --workspace "$VPA_WORKSPACE" \
  --out report.html
```

For explicit workflow or direct tool debugging only:

```bash
vpa report <workflow_id> \
  --workspace "$VPA_WORKSPACE" \
  --out workflow_report.html

vpa report <job_id> \
  --workspace "$VPA_WORKSPACE" \
  --out job_report.html
```

When reporting success, record the provider/model, workspace, agent run id,
workflow ids, final decision, submitted candidates or answer, report path, and
any environment caveats.

## Failure Handling

Classify failures before proposing fixes:

- `install/preflight`: missing binary, missing model weights, CUDA mismatch, or
  unavailable credentials.
- `planning`: invalid selected tools, missing user assets, or schema violations.
- `execution`: adapter runtime failure, worker crash, timeout, or empty candidate
  stream.
- `decision/reporting`: no acceptable candidate, alias resolution failure, or
  report artifact issue.

Do not patch code or mutate workflow artifacts during a user run unless the user
explicitly asks for implementation work. For operational failures, prefer
install repair, environment correction, plan-only validation, or a new workspace
run with a clear audit trail.
