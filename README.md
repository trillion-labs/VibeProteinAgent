# VibeProteinAgent

[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm--NC--1.0.0-6b1ea8.svg)](LICENSE)
[![Use: Non-commercial](https://img.shields.io/badge/use-non--commercial-cc0980.svg)](#license)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-1a1535.svg)](pyproject.toml)

**VibeProteinAgent (VPA)** is a local-first, GPU-ready protein design
framework. It gives researchers one command surface for protein design tools,
workflow orchestration, LLM-assisted planning, artifact tracking, and
human-readable reports.

![VPA report snapshot](assets/readme/vpa-report-snapshot.svg)

VPA is built for practical design rounds: generate or retrieve candidates,
score and filter them, keep every intermediate artifact in a durable workspace,
then produce a report that another scientist can inspect.

## Why Use VPA?

- **One interface for many tools.** Run protein design, structure prediction, analysis, docking, and sequence retrieval from
  the same `vpa` CLI.
- **Free, local-first execution.** Workspaces, logs, candidate records, artifacts,
  caches, and reports stay on your machine or cluster storage. You can run VPA
  for free without a separate model API subscription.
- **Heavy-tool friendly.** Persistent workers and sidecars reduce cold starts
  for GPU-backed protein models.
- **Run tools and get structured reports.** VPA executes the underlying protein
  tools, tracks the resulting artifacts, and turns completed runs into
  inspection-ready HTML reports.
- **Provider-flexible.** Agent mode can use Codex, Claude Code, or OpenRouter
  without changing the underlying VPA runtime.

> [!IMPORTANT]
> **Non-commercial use only.** VPA first-party code is licensed under the
> [PolyForm Noncommercial License 1.0.0](LICENSE). Third-party tools, model
> weights, sidecars, and vendored or pointer-vendored upstream components keep
> their own licenses and terms. Review [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
> and the relevant `vendor/*/LICENSE` / `vendor/*/UPSTREAM.md` files before
> redistribution or production use.

## Install

The simplest setup is a full native install. It creates one VPA root,
`~/.vibeproteinagent` by default, with the Python environment, sidecars, tool
checkouts, model caches, workspaces, and `env.sh`. On a cluster, set
`VPA_HOME` to shared storage before installing.

```bash
export VPA_HOME="$HOME/.vibeproteinagent"

# Optional: needed for gated Hugging Face downloads such as ESM3.
export HF_TOKEN=<huggingface-access-token>

npx vibeproteinagent install --profile full --cuda-profile auto --yes
source "$VPA_HOME/env.sh"
vpa install doctor --profile full
```

Docker is optional. Use it when you want the same CUDA runtime everywhere or the
host Python/CUDA stack is fragile. The large install still lives in the mounted
`VPA_HOME`.

```bash
export VPA_HOME="$HOME/.vibeproteinagent"

docker build -f containers/Dockerfile.cu128 -t vibeproteinagent:cu128 .

docker run --rm --gpus all \
  -e VPA_HOME=/vpa \
  -e HF_TOKEN \
  -v "${VPA_HOME:-$HOME/.vibeproteinagent}:/vpa" \
  vibeproteinagent:cu128 \
  vpa install --profile full --cuda-profile cu128 --yes

docker run --rm --gpus all \
  -e VPA_HOME=/vpa \
  -v "${VPA_HOME:-$HOME/.vibeproteinagent}:/vpa" \
  vibeproteinagent:cu128 \
  vpa install doctor --profile full
```

See [`containers/README.md`](containers/README.md) for container-specific
notes.

## Quick Start

VPA's primary workflow is an LLM-agent-driven, end-to-end protein design run.
The repository-level instructions in [`skills.md`](skills.md) and
[`docs/agent_workflows/`](docs/agent_workflows/) tell Codex, Claude Code, and
other coding agents how to turn a design request into canonical VPA planning,
execution, ranking, submission, and report generation.

Start workers for the design workspace, then ask the agent for a full binder
round:

```bash
# One GPU worker and 2 CPU workers.
vpa daemon start --workers 1 --cpu-workers 2 --workspace binder_round

# Execute the end-to-end run.
vpa agent run \
  "Design 50-80 aa soluble mini-protein binders to chain A of target.pdb. Avoid cysteines and long polybasic patches. Return the top 5 designs with structures, QC, ranking, and a report." \
  --provider codex \
  --model gpt-5.2 \
  --target target.pdb \
  --workspace binder_round \
  --max-rounds 8 \
  --max-entries 3 \
  --max-wait-s 7200

# Inspect the agent run.
vpa agent runs --workspace binder_round
vpa agent show <agent_run_id> --workspace binder_round
# Render the workflow report.
vpa report <workflow_id> --workspace binder_round --out binder_round_report.html --open
```

For Claude Code or OpenRouter, switch only the provider/model:

```bash
vpa agent run "Design 50-80 aa binders to chain A of target.pdb, avoid cysteines, and report the top 5." \
  --provider claude-code --model sonnet --target target.pdb --workspace binder_round --plan-only

vpa agent run "Design 50-80 aa binders to chain A of target.pdb, avoid cysteines, and report the top 5." \
  --provider openrouter --model openai/gpt-5.1 --target target.pdb --workspace binder_round --plan-only
```

If you are driving VPA from Codex or Claude Code directly, give the assistant
the same request in this repository. `AGENTS.md` points protein workflow tasks
to `skills.md`, which in turn routes executable design/ranking/reporting work
through the VPA operating protocol.

For a lightweight CPU-only sanity check before launching model-backed design,
see [`docs/use_tools.md`](docs/use_tools.md).

## High-Level Structure

VPA gives every run a durable workspace. The CLI submits tool jobs, workflows,
or agent plans into that workspace; workers execute jobs; artifacts and cache
metadata are recorded; reports turn stored results into inspectable HTML.

![VPA framework overview](assets/readme/vpa-framework.svg)

| Layer | Role |
|---|---|
| `vpa run` | Submit one explicit tool job with typed parameters. |
| `vpa workflow run` | Execute a YAML graph that wires tool outputs into later tool inputs. |
| `vpa agent run` | Ask an LLM provider to plan, validate, execute, and decide from the VPA tool surface. |
| Workspace | Durable jobs, workflow state, logs, artifacts, candidate archives, and reports. |
| Workers | Local persistent CPU/GPU processes that run adapters and sidecars. |
| Reports | Read-only HTML or bundle output for review, sharing, and replay. |

## Tools

List the tool surface available in your install:

```bash
vpa tools
vpa tools --json
vpa tool info proteinmpnn
vpa tool info protenix_v2 --json
```

Common public tools include:

| Area | Examples |
|---|---|
| Backbone generation | `rfdiffusion`, `rfdiffusion3`, `rfpeptides` |
| Inverse folding | `proteinmpnn`, `ligandmpnn` |
| Structure prediction | `boltz2_affinity`, `protenix_v2` |
| Binder design | `pxdesign`, `bindcraft`, `boltzgen`, `proteina_complexa` |
| Docking | `vina` |
| Protein language models | `esm2`, `esm3` |
| Scoring and QC | `protein_qc`, `rank_designs`, `ipsae`, `prodigy`, `pxmeter`, `pyrosetta`, `tmalign` |
| Retrieval | `search_sequence`, `literature_search`, `template_select`, `foldseek`, `mmseqs2_gpu` |
| Analysis | `sequence_analysis`, `structure_analysis` |
| Sequence engineering | `sequence_edit` |

VPA can also discover custom tools from `~/.vibeproteinagent/tools/<tool>/` or
Python package entry points under `vibeproteinagent.tools`.

## How To Use VPA

### 1. Use An LLM Agent With VPA Skills

This is the main VPA experience. Describe the scientific goal and constraints;
the agent uses [`skills.md`](skills.md) and [`docs/agent_workflows/`](docs/agent_workflows/)
to choose a canonical design/retrieval/evaluation workflow, run tools, rank
outputs, submit a final answer or candidate, and generate report-ready evidence.

```bash
# Run the request end to end:
vpa agent run \
  "Design 50-80 aa soluble binders to chain A of target.pdb. Avoid cysteines and long polybasic patches. Return the top 5 with structures, QC, ranking, and a report." \
  --provider codex \
  --model gpt-5.2 \
  --target target.pdb \
  --workspace binder_round \
  --max-rounds 8 \
  --max-entries 3 \
  --max-wait-s 7200
```
Add `--plan-only` when you want to inspect the proposed workflow before launching expensive jobs.

The same agent interface also handles retrieval and Q&A style requests:

```bash
vpa agent run \
  "Find proteins related to this GO term, screen for liabilities, and report the strongest candidates." \
  --provider codex \
  --model gpt-5.2 \
  --workspace retrieval_round

vpa agent run \
  "What are the likely developability risks in this sequence?" \
  --provider codex \
  --model gpt-5.2 \
  --param sequence=ACDEFGHIKLMNPQRSTVWY \
  --workspace qa_round
```

Guide: [`docs/use_llm_agent.md`](docs/use_llm_agent.md)
Notebook: [`demo/use_llm_agent_demo.ipynb`](demo/use_llm_agent_demo.ipynb)

### 2. Run A YAML Workflow Manually

Use YAML workflows when you want deterministic orchestration without an LLM, or
when you want to save a reusable protocol for a known tool chain.

```yaml
name: sequence_triage
params:
  sequence: {type: str, required: true}
  mapping: {type: dict, default: {"3": "A"}}
nodes:
  - id: mutate
    tool: sequence_edit
    inputs:
      operation: mutate_residues
      sequence: ${params.sequence}
      mapping: ${params.mapping}

  - id: qc
    tool: protein_qc
    inputs:
      sequence: ${outputs.mutate.mutant_sequence}
      metrics: ["gravy", "instability_index", "pi", "polybasic"]

  - id: rank
    tool: rank_designs
    inputs:
      candidates: ${outputs.qc.candidates}
      qc_results: ${outputs.qc}
      top_k: 1
```

```bash
vpa workflow inspect sequence_triage.yaml
vpa workflow run sequence_triage.yaml \
  --workspace workflow_demo \
  --param sequence=ACDEFGHIKLMNPQRSTVWY \
  --follow

vpa report <workflow_id> --workspace workflow_demo --open
```

Guide: [`docs/use_workflows.md`](docs/use_workflows.md)
Notebook: [`demo/use_workflows_demo.ipynb`](demo/use_workflows_demo.ipynb)

### 3. Run Tools Directly

Use direct runs when you know the exact tool and parameters, or when you want a
quick sanity check before launching an agent workflow.

```bash
vpa run protein_qc \
  sequence=ACDEFGHIKLMNPQRSTVWY \
  metrics='["gravy","instability_index","pi","polybasic"]' \
  --workspace quickstart \
  --follow

vpa report <job_id> --workspace quickstart --open
```

Guide: [`docs/use_tools.md`](docs/use_tools.md)
Notebook: [`demo/use_tools_demo.ipynb`](demo/use_tools_demo.ipynb)

## Agent Skills

This repository includes agent-facing operating instructions:

- [`skills.md`](skills.md) defines when ordinary protein discussion becomes an
  executable VPA workflow.
- [`docs/agent_workflows/protein_design_skill.md`](docs/agent_workflows/protein_design_skill.md)
  gives Codex, Claude Code, and similar agents canonical retrieval, binder
  design, de novo design, ranking, submission, and reporting trajectories.
- [`docs/agent_workflows/vpa_contracts.md`](docs/agent_workflows/vpa_contracts.md)
  documents candidate archive, prompt-context, planner/compiler, and decision
  contracts.
- [`docs/agent_workflows/live_matrix_validation.md`](docs/agent_workflows/live_matrix_validation.md)
  is the live provider validation runbook.

These files are for agents and operators. External users normally read the
public guides in `README.md` and `docs/use_*.md`; agents use
`docs/agent_workflows/` to run VPA without relying on brittle prompt-only
candidate wiring.

## Reports

`vpa report` creates a read-only human report from a completed job or workflow.

```bash
# Single self-contained HTML file.
vpa report <job_or_workflow_id> \
  --workspace design_round \
  --viewer offline \
  --out report.html \
  --open

# Bundle mode for larger structure/artifact sets.
vpa report <job_or_workflow_id> \
  --workspace design_round \
  --format bundle \
  --out report_bundle
```

Reports can include:

| Section | What it shows |
|---|---|
| Summary | Source id, state, top-level outcome, warnings, and replay hints. |
| Workflow DAG | Node/job state, cached/skipped/failed status, and execution provenance. |
| Candidates | Ranked candidates, sequences, structures, metrics, QC flags, and merge audit context. |
| Plots and viewers | Score plots, sequence/QC plots, MSA/FASTA context, and offline structure viewers when artifacts exist. |
| Reproducibility | Tool versions, parameters, artifact paths/hashes, workspace id, and report command. |

You can also generate standalone structure viewers:

```bash
vpa visualize design.pdb --viewer offline --out design_viewer.html
vpa visualize --job-id <job_id> --workspace design_round --viewer offline --out job_viewer.html
```

## Demo Notebooks

The notebooks are safe by default. They run lightweight CLI checks and CPU-safe
examples, then print heavyweight or LLM commands unless you opt in.

| Intent | Guide | Notebook |
|---|---|---|
| Direct tool runs | [`docs/use_tools.md`](docs/use_tools.md) | [`demo/use_tools_demo.ipynb`](demo/use_tools_demo.ipynb) |
| YAML workflows | [`docs/use_workflows.md`](docs/use_workflows.md) | [`demo/use_workflows_demo.ipynb`](demo/use_workflows_demo.ipynb) |
| LLM agent | [`docs/use_llm_agent.md`](docs/use_llm_agent.md) | [`demo/use_llm_agent_demo.ipynb`](demo/use_llm_agent_demo.ipynb) |

Environment flags:

```bash
export VPA_DEMO_RUN_INSTALL=1  # run optional install cells
export VPA_DEMO_RUN_HEAVY=1    # run heavyweight GPU/model examples
export VPA_DEMO_RUN_LLM=1      # run live LLM agent execution
```

## Cite

If you use VPA in research, cite:

```bibtex
@software{vpa2026vibeproteinagent,
  author       = {VibeProteinAgent contributors},
  title        = {VibeProteinAgent},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/<your-org>/VibeProteinAgent}
}
```

Citation metadata is also available in [`CITATION.cff`](CITATION.cff).

## License

VPA first-party code is released for non-commercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a
separate license. Third-party components remain under their own upstream terms.
