# Use A Workflow

Use this path when you want VPA to run several tools in order, pass outputs between them, and produce one workflow record.

Demo notebook: [`demo/use_workflows_demo.ipynb`](../demo/use_workflows_demo.ipynb)
builds a generation-to-ranking workflow with RFdiffusion, ProteinMPNN,
Protenix-v2, structure analysis, protein QC, and ranking.
It is smoke-safe by default: it validates GPU/CLI access and prints the heavy
workflow command unless you set `VPA_DEMO_RUN_HEAVY=1`.

## 1. Write A Simple Workflow YAML

A workflow YAML has two main parts:

- `params`: values you provide when you run the workflow.
- `nodes`: tool steps VPA runs in order.

Inside a node, use `${params.name}` to read a workflow parameter, and
`${outputs.step_id.field}` to use a result from an earlier step.
For expensive candidate generators, prefer `foreach` fan-out over one large
single-GPU node. Add `max_parallel` to cap active shards so the plan matches
your GPU budget.

Example `my_sequence_workflow.yaml`:

```yaml
name: my_sequence_workflow
description: "Mutate a sequence, then analyze the mutated result."

params:
  sequence: {type: str, required: true}
  mapping: {type: dict, required: true}

nodes:
  - id: mutate
    tool: sequence_edit
    inputs:
      operation: mutate_residues
      sequence: ${params.sequence}
      mapping: ${params.mapping}
      output_name: repaired
    cache: true

  - id: analyze_mutant
    tool: sequence_analysis
    inputs:
      sequence: ${outputs.mutate.mutant_sequence}
    cache: true
```

Run it like this:

```bash
vpa workflow run my_sequence_workflow.yaml \
  --workspace design_round \
  --param sequence=ACDEFGHIKLMNPQRSTVWY \
  --param mapping='{"3":"A"}' \
  --follow
```

Use `vpa tool <tool_name>` to see which input fields each tool accepts.
Shardable heavy tools advertise planner metadata in `vpa tool <tool_name>`:
`shardable_fields`, `recommended_shard_size`, and `default_max_parallel`.

Cf. You do not need to write `version` in a simple workflow; VPA uses the installed
tool version by default. Add a version only when you intentionally want to pin a
specific tool range.

## 2. Install The Tools Used By The Workflow

For full workflow capability:

```bash
# `--yes` skips confirmation prompts and starts the install right away.
npx vibeproteinagent install preflight --profile full --cuda-profile auto
npx vibeproteinagent install --profile full --cuda-profile auto --yes
```

`--profile full` includes ESM3. If the open ESM3 cache is not already present,
or if you plan to use gated Hugging Face variants, export a credential first:

```bash
export HF_TOKEN=<huggingface-access-token>
```

Or install only the tools you need:

```bash
npx vibeproteinagent install preflight --tools rfdiffusion,proteinmpnn,protenix_v2,structure_analysis,protein_qc,rank_designs --cuda-profile auto
npx vibeproteinagent install --tools rfdiffusion,proteinmpnn,protenix_v2,structure_analysis,protein_qc,rank_designs --cuda-profile auto --yes
```

## 3. Create A Workspace And Start Workers

A workspace keeps related jobs, logs, artifacts, and reports together. Use the
same workspace name for commands that belong to the same run. Starting the daemon
creates the workspace if needed and launches local workers that execute queued
tool jobs.

```bash
vpa daemon start --workers 1 --cpu-workers 2 --workspace design_round
vpa daemon status --workspace design_round
```

`--workers` counts GPU workers. `--cpu-workers` adds separate CPU-only workers
for retrieval, sequence analysis, QC, and other jobs that do not need a GPU.
GPU-heavy tools stay single-GPU per job by default; use workflow `foreach`
with `max_parallel` to fan out independent seed/candidate shards across GPU
workers.

## 4. See Available Workflows

```bash
vpa workflow list
vpa workflow inspect workflows/examples/mutation_repair_round.yaml
```

## 5. Run A Workflow

Run with inline parameters:

```bash
# This workflow mutates the 1-indexed sequence position 3 to alanine (A).
# For dictionary params, pass JSON like '{"3":"A"}'.
vpa workflow run workflows/examples/mutation_repair_round.yaml \
  --workspace design_round \
  --param structure_path=target.pdb \
  --param wild_type_sequence=ACDEFGHIKLMNPQRSTVWY \
  --param mapping='{"3":"A"}' \
  --follow
```

Run with a YAML params file:

```bash
# Wait up to 7200 seconds (2 hours) for the workflow to finish before the CLI gives up waiting.
vpa workflow run workflows/examples/rfdiffusion_backbone_design_round.yaml \
  --workspace design_round \
  --params-file workflow_params.yaml \
  --max-wait-s 7200 
```

## 6. Monitor And Share Results

```bash
vpa workflow status <workflow_id> --workspace design_round
vpa report <workflow_id> --workspace design_round --out report.html --open
```

If a run is interrupted, resume it:

```bash
vpa workflow run workflows/examples/mutation_repair_round.yaml \
  --workspace design_round \
  --resume <workflow_id>
```
