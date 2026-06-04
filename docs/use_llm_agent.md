# Use The LLM Agent

Use the agent when you want to describe a scientific goal in natural language
and let an LLM choose tools, build a VPA plan, execute it, and make a final
decision from the stored outputs.

`vpa agent run` is only for LLM-authored workflow planning. It is not a
free-text router for direct tool runs, existing job status/results, reports, or
explicit workflow descriptors. Use `vpa run`, `vpa status`, `vpa results`,
`vpa report`, or `vpa workflow run` for those explicit operations.

Demo notebook: [`demo/use_llm_agent_demo.ipynb`](../demo/use_llm_agent_demo.ipynb)
is safe by default. It inspects the local tool surface and prints execution
commands unless `VPA_DEMO_RUN_LLM=1` is set.

## 1. Install And Activate

```bash
export VPA_HOME="${VPA_HOME:-$HOME/.vibeproteinagent}"
npx vibeproteinagent install --profile full --cuda-profile auto --yes
source "$VPA_HOME/env.sh"
vpa install doctor --profile full
```

Use the lightweight `agent` profile only when the machine cannot support a full
install or you explicitly want planning/retrieval without heavyweight design
tools:

```bash
npx vibeproteinagent install --profile agent --cuda-profile auto --yes
vpa install doctor --profile agent
```

## 2. Configure A Provider

VPA currently supports these agent providers:

| Provider | Setup |
|---|---|
| `codex` | Install/login to Codex CLI once with `codex login`. |
| `claude-code` | Install/login to Claude Code once with `claude /login`. |
| `openrouter` | Export `OPENROUTER_API_KEY` or `VPA_OPENROUTER_CREDENTIAL`. |

Examples:

```bash
codex login
claude /login
export OPENROUTER_API_KEY=OPENROUTER_API_KEY_VALUE
```

Choose the provider and a model available in that provider environment:

```bash
vpa agent run "Plan a binder design round for target.pdb" \
  --provider codex \
  --model MODEL_ID \
  --plan-only

vpa agent run "Plan a binder design round for target.pdb" \
  --provider claude-code \
  --model MODEL_ID \
  --plan-only

vpa agent run "Plan a binder design round for target.pdb" \
  --provider openrouter \
  --model PROVIDER/MODEL_ID \
  --plan-only
```

Do not commit provider tokens or CLI session files to the repository.

## 3. Plan First

`--plan-only` asks the provider for a semantic plan, validates it locally, and
compiles it into VPA workflow intent. It does not submit jobs, warm tools, or
run models.

```bash
vpa agent run \
  "Design binder candidates for target.pdb, run QC, rank them, and report the best designs." \
  --provider codex \
  --model MODEL_ID \
  --target target.pdb \
  --workspace agent_round \
  --plan-only
```

This step is the best place to catch missing inputs, unavailable tools,
credential gaps, or an unsuitable provider/model before launching expensive
jobs.

## 4. Execute

Start workers for the workspace before live execution:

```bash
vpa daemon start --workers 1 --cpu-workers 2 --workspace agent_round
```

By default, live `vpa agent run` prewarms tools from the validated plan before
submitting workflow work. You can also warm known tools manually after the daemon
starts:

```bash
vpa daemon warm proteinmpnn --workers all --json
vpa daemon warm protein_qc --json
vpa daemon warm rank_designs --json
```

Then run the request:

```bash
vpa agent run \
  "Design binder candidates for target.pdb, run QC, rank them, and report the best designs." \
  --provider codex \
  --model MODEL_ID \
  --target target.pdb \
  --workspace agent_round \
  --max-rounds 8 \
  --max-entries 3 \
  --max-wait-s 7200
```

Important execution controls:

- `--max-rounds` caps optimization rounds after the initial plan.
- `--max-entries` fails fast after repeated invalid, timed-out, or unusable LLM
  planning/decision entries.
- `--max-wait-s` caps how long VPA waits for workflow execution.
- `--no-execute` submits validated jobs/workflows but does not drive execution.
- `--no-warm` skips tool pre-warming.

You can also validate and submit a saved plan:

```bash
vpa agent validate-plan plan.json --workspace agent_round
vpa agent submit-plan plan.json --workspace agent_round --max-wait-s 7200
```

## 5. Output Modes

The same `vpa agent run` command supports two scientific result shapes:

| Mode | Typical request | Final result |
|---|---|---|
| Design / retrieval | "design", "generate", "find binders", "retrieve candidates" | Candidate submission or no acceptable candidate. |
| Understanding / Q&A | "what is", "compare", "explain", "how does" | Text answer with optional workflow-node citations. |

Example Q&A request:

```bash
vpa agent run \
  "What is the pI of UniProt P44612, and does it carry common sequence liabilities?" \
  --provider codex \
  --model MODEL_ID \
  --workspace qa_round
```

## 6. Inspect And Report

```bash
vpa agent runs --workspace agent_round
vpa agent show arun_ID --workspace agent_round
vpa workflow status WORKFLOW_ID --workspace agent_round
vpa report arun_ID --workspace agent_round --out agent_report.html --open
```

Reports can target `arun_...`, `wf_...`, or `job_...` ids. Agent reports should
use the `arun_...` id by default and include the workflow rounds tied to the
user query, plus the agent timeline, validated plans, decisions, final response,
candidate context, and replay metadata when available.

For requests that include local files, pass paths explicitly in the prompt or
through `--target`, `--param`, or `--params-file`. For example, use
`--target target.pdb` for binder design and `--param fasta_path=/abs/path/input.fasta`
for a sequence-analysis request.

If the user names a tool in the agent request, that name is planner context, not
direct adapter dispatch. For example, "include rfdiffusion" means the planner
should consider that tool in a workflow; direct RFdiffusion debugging should use
`vpa run rfdiffusion ...`.
