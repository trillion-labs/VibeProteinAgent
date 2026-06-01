# Use The LLM Agent

Use the agent when you want to describe a scientific goal in natural language
and let an LLM choose tools, build a VPA plan, execute it, and make a final
decision from the stored outputs.

Demo notebook: [`demo/use_llm_agent_demo.ipynb`](../demo/use_llm_agent_demo.ipynb)
is safe by default. It inspects the local tool surface and prints execution
commands unless `VPA_DEMO_RUN_LLM=1` is set.

## 1. Install Agent Mode

```bash
npx vibeproteinagent install --profile agent --cuda-profile auto --yes
source "${VPA_HOME:-$HOME/.vibeproteinagent}/env.sh"
vpa install doctor --profile agent
```

For broad design-agent runs across heavyweight tools:

```bash
export HF_TOKEN=<huggingface-access-token>  # only when selected tools need it
npx vibeproteinagent install --profile full --cuda-profile auto --yes
vpa install doctor --profile full
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
export OPENROUTER_API_KEY=<openrouter-api-key>
```

Choose provider and model per run:

```bash
vpa agent run "Plan a binder design round for target.pdb" \
  --provider codex \
  --model gpt-5.2 \
  --plan-only

vpa agent run "Plan a binder design round for target.pdb" \
  --provider claude-code \
  --model sonnet \
  --plan-only

vpa agent run "Plan a binder design round for target.pdb" \
  --provider openrouter \
  --model openai/gpt-5.1 \
  --plan-only
```

Do not commit provider tokens or CLI session files to the repository.

## 3. Plan First

`--plan-only` classifies the request, asks the provider for a semantic plan,
validates it locally, and compiles it into VPA workflow intent. It does not
submit jobs, warm tools, or run models.

```bash
vpa agent run \
  "Design binder candidates for target.pdb, run QC, rank them, and report the best designs." \
  --provider codex \
  --model gpt-5.2 \
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

Then run the request:

```bash
vpa agent run \
  "Design binder candidates for target.pdb, run QC, rank them, and report the best designs." \
  --provider codex \
  --model gpt-5.2 \
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
  --model gpt-5.2 \
  --workspace qa_round
```

## 6. Inspect And Report

```bash
vpa agent runs --workspace agent_round
vpa agent show <agent_run_id> --workspace agent_round
vpa workflow status <workflow_id> --workspace agent_round
vpa report <workflow_id> --workspace agent_round --out agent_report.html --open
```

Agent reports include the stored workflow evidence plus agent timeline,
validated plan, decisions, final response, candidate context, and replay
metadata when available.

## Agent Workflow Skills

The agent-facing workflow instructions live outside the public user guides:

- [`../skills.md`](../skills.md) defines when protein conversation becomes an
  executable workflow.
- [`agent_workflows/protein_design_skill.md`](agent_workflows/protein_design_skill.md)
  describes canonical retrieval, binder design, de novo design, ranking,
  submission, and report trajectories.
- [`agent_workflows/vpa_contracts.md`](agent_workflows/vpa_contracts.md)
  records the candidate archive, prompt context, planner/compiler, and decision
  contracts agents must preserve.
- [`agent_workflows/live_matrix_validation.md`](agent_workflows/live_matrix_validation.md)
  is for live provider validation.

Do not edit those files for ordinary user documentation. Codex, Claude Code,
and similar agents read them as operating protocol so they can use VPA skills
without treating LLM summaries as executable candidate sources.
