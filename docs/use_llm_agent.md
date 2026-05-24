# Use The LLM Agent

Use this path when you want to describe the goal in natural language and let the agent choose tools or workflows.

Demo notebook: [`demo/use_llm_agent_demo.ipynb`](../demo/use_llm_agent_demo.ipynb)
shows how to inspect the public tool surface, draft an agent plan for a
generation/prediction/QC loop, and run it once a provider is configured.
It is smoke-safe by default: it validates GPU/CLI access and prints agent
commands unless you set `VPA_DEMO_RUN_LLM=1`; model-backed execution also needs
`VPA_DEMO_RUN_HEAVY=1`.

## 1. Install Agent Mode

Agent mode should have the planner, runtime, and tool metadata ready:

```bash
# `--yes` skips confirmation prompts and starts the install right away.
npx vibeproteinagent install preflight --profile agent --cuda-profile auto
npx vibeproteinagent install --profile agent --yes
```

This install does not require an OpenRouter key. The agent profile installs the
VPA runtime, local planner surface, and lightweight retrieval/template tools.
Provider credentials are only needed when you run a provider that needs them.

For broad tool selection across heavyweight design tools:

```bash
# Optional if ESM3 first-run downloads or gated Hugging Face variants are needed.
export HF_TOKEN=<huggingface-access-token>

npx vibeproteinagent install preflight --profile full --cuda-profile auto
npx vibeproteinagent install --profile full --cuda-profile auto --yes
```

The full profile installs all non-gated first-class tools and sidecars; gated
or first-run Hugging Face tools become ready when credentials, accepted model
terms, or a populated cache are available.

## 2. Configure The LLM Provider

VPA currently supports three agent providers:

- `codex`: default local Codex CLI provider.
- `claude-code`: local Claude Code CLI provider.
- `openrouter`: hosted API provider through OpenRouter.

For Codex or Claude Code, make sure the matching CLI is installed and logged in
on your machine. For OpenRouter, set an API key:

```bash
# Codex CLI provider: run once in the shell/account that will launch VPA.
codex login

# Claude Code CLI provider: run once in the shell/account that will launch VPA.
claude /login

# OpenRouter API provider.
export OPENROUTER_API_KEY=<openrouter-api-key>
```

`VPA_OPENROUTER_CREDENTIAL` is also accepted if you prefer a VPA-specific env
var name. Do not commit provider tokens or CLI session material into the repo;
keep them in your local shell, OS keychain, or CI/cluster secret store.

Choose a provider per command:

```bash
# Default: Codex CLI
vpa agent run "Plan a binder design round for target.pdb" \
  --provider codex \
  --model gpt-5.2 \
  --plan-only

# Claude Code CLI
vpa agent run "Plan a binder design round for target.pdb" \
  --provider claude-code \
  --model sonnet \
  --plan-only

# OpenRouter API
vpa agent run "Plan a binder design round for target.pdb" \
  --provider openrouter \
  --model openai/gpt-5.1 \
  --plan-only
```

You can also use a compact provider/model form, such as
`--model codex/gpt-5.2`, `--model claude-code/sonnet`, or
`--model openrouter/openai/gpt-5.1`.

To check which models you can use:

```bash
# Codex CLI models depend on your Codex install/account.
codex --help

# Claude Code accepts aliases such as sonnet or opus, plus full model names.
claude --help

# OpenRouter model ids are listed by OpenRouter.
curl https://openrouter.ai/api/v1/models | python -m json.tool
```

## 3. Ask For A Plan First

`--plan-only` uses the same VPA-level planning flow for every provider. VPA sends
the planner system prompt, your request, the available VPA tools, and a strict
JSON output contract. The planner must return an explicit `workflow_plan.nodes[]`
tool chain. VPA validates the returned plan locally, but does not submit jobs,
start tool execution, or warm models.

For tools that advertise sharding metadata, the planner should prefer `foreach`
fan-out with small seed/candidate shards and `max_parallel` set to the available
GPU budget. ProteinMPNN/LigandMPNN keep warm batches together unless the design
count is above their sharding threshold.

NOTE: You can skip this preview and run everything at once by omitting `--plan-only`.

```bash
vpa agent run \
  "Design binder backbones for target.pdb, refold the best candidates, run QC, and rank them." \
  --provider openrouter \
  --model openrouter/openai/gpt-5.1 \
  --target target.pdb \
  --workspace agent_round \
  --plan-only
```

## 4. Execute The Agent

Before execution, start workers in the workspace you will use for the agent run.
The workspace stores the agent run, jobs, logs, artifacts, and reports; the
workers execute the jobs submitted by the agent.

```bash
vpa daemon start --workers 1 --cpu-workers 2 --workspace agent_round
```

`--workers` counts GPU workers. `--cpu-workers` keeps CPU-only retrieval,
analysis, validation, and aggregation jobs from occupying GPU workers. The
planner should express GPU fan-out with workflow `foreach` plus `max_parallel`
rather than assuming one model process spans multiple GPUs.

Now you can execute the agent request in one of two ways:

```bash
# Option A: run end-to-end in one command. This creates a fresh plan, validates it, submits jobs, and waits. 
# It does not reuse the earlier --plan-only output.
vpa agent run \
  "Design binder backbones for target.pdb, refold candidates, run QC, and rank the top designs." \
  --target target.pdb \
  --workspace agent_round \
  --max-rounds 8 \
  --max-entries 3 \
  --max-wait-s 7200
```

`--max-rounds` caps optimization rounds after the initial plan. `--max-entries`
caps failed LLM planning/decision entries across the run; timeout, non-JSON, or
locally invalid LLM responses are counted and the agent fails fast when the
budget is exhausted.

```bash
# Option B: execute a plan you already saved to a file.
vpa agent submit-plan plan.json \
  --workspace agent_round \
  --max-wait-s 7200
```

## 5. Inspect The Agent Run

```bash
# `run` starts a new agent request; `runs` lists previous agent requests in this workspace.
vpa agent runs --workspace agent_round
# `show` displays the details of one saved agent request from the `runs` list.
vpa agent show <agent_run_id> --workspace agent_round
```

For a human-readable report, use the workflow or job id produced by the agent:

```bash
vpa report <workflow_id> --workspace agent_round --open
```

Tip: `agent ask` is a lightweight no-execution shortcut. Use it for quick tool
advice or analysis of existing `job_*` / `wf_*` results. It is similar to
`agent run --plan-only`, but for a serious run preview that you may later execute,
prefer `agent run --plan-only`.

```bash
vpa agent ask "Which VPA tools should I use to evaluate this design?"
```
