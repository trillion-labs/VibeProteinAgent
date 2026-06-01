# Use YAML Workflows

Use a workflow when outputs from one tool should feed later tools automatically.
A workflow creates one durable workflow record, submits the node jobs in order,
tracks cache keys, and lets you report the full run.

Demo notebook: [`demo/use_workflows_demo.ipynb`](../demo/use_workflows_demo.ipynb)
writes and runs the lightweight workflow below.

## 1. Start Workers

This guide uses CPU-safe tools:

```bash
vpa daemon start --cpu-workers 1 --workspace workflow_demo
```

For mixed GPU/CPU design workflows, start both:

```bash
vpa daemon start --workers 1 --cpu-workers 2 --workspace design_round
```

## 2. Write A Minimal Workflow

Save this as `sequence_triage.yaml`:

```yaml
name: sequence_triage
version: "1.0.0"
description: "Mutate a sequence, analyze the mutant, run QC, and rank it."

params:
  sequence:
    type: str
    required: true
  mapping:
    type: dict
    default: {"3": "A"}

nodes:
  - id: mutate
    tool: sequence_edit
    inputs:
      operation: mutate_residues
      sequence: ${params.sequence}
      mapping: ${params.mapping}
      output_name: demo_mutant
    cache: true

  - id: analyze
    tool: sequence_analysis
    inputs:
      sequence: ${outputs.mutate.mutant_sequence}
    cache: true

  - id: qc
    tool: protein_qc
    inputs:
      sequence: ${outputs.mutate.mutant_sequence}
      metrics: ["gravy", "instability_index", "pi", "polybasic"]
    cache: true

  - id: rank
    tool: rank_designs
    inputs:
      candidates: ${outputs.qc.candidates}
      qc_results: ${outputs.qc}
      top_k: 1
      output_prefix: sequence_triage_ranked
    cache: true
```

Workflow basics:

- `params` declares user-supplied values.
- `${params.sequence}` reads a workflow parameter.
- `${outputs.mutate.mutant_sequence}` reads a field from an earlier node.
- `cache: true` lets VPA reuse matching node outputs in the same environment.
- `rank_designs` can consume canonical candidates emitted by upstream tools.

## 3. Inspect And Run

```bash
vpa workflow inspect sequence_triage.yaml

vpa workflow run sequence_triage.yaml \
  --workspace workflow_demo \
  --param sequence=ACDEFGHIKLMNPQRSTVWY \
  --param mapping='{"3":"A","9":"Q"}' \
  --follow
```

The command prints a `workflow_id` and each node state. If the daemon is not
running, VPA records the workflow but cannot execute the nodes.

## 4. Monitor, Resume, And Report

```bash
vpa workflow status <workflow_id> --workspace workflow_demo

vpa workflow run sequence_triage.yaml \
  --workspace workflow_demo \
  --resume <workflow_id> \
  --max-wait-s 3600

vpa report <workflow_id> \
  --workspace workflow_demo \
  --out workflow_report.html \
  --open
```

Use `--params-file params.yaml` when parameters are easier to maintain in a
file:

```yaml
sequence: ACDEFGHIKLMNPQRSTVWY
mapping:
  "3": A
  "9": Q
```

```bash
vpa workflow run sequence_triage.yaml \
  --workspace workflow_demo \
  --params-file params.yaml \
  --follow
```

## Optional Heavy Design Pattern

A heavier de novo or binder workflow follows the same shape:

```text
rfdiffusion or pxdesign
-> proteinmpnn or ligandmpnn
-> protenix_v2 or boltz2_affinity
-> protein_qc and structure_analysis
-> rank_designs
-> report
```

PXDesign already assigns binder sequence and emits binder backbone coordinates.
Add a structure-prediction/refinement step such as `protenix_v2` when the final
deliverable needs full-atom binder sidechain coordinates.

Before running heavyweight workflows:

```bash
vpa install doctor --tools rfdiffusion,proteinmpnn,protenix_v2,protein_qc,rank_designs
vpa daemon start --workers 1 --cpu-workers 2 --workspace design_round
```

For broad candidate generation, prefer workflow fan-out with `foreach` and
`max_parallel` so independent seeds/candidates use your GPU budget explicitly.
User-authored YAML is executed as written; VPA does not silently rewrite it.
