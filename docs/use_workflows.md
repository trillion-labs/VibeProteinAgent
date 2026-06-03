# Use YAML Workflows

Use a workflow when outputs from one tool should feed later tools automatically.
A workflow creates one durable workflow record, submits the node jobs in order,
tracks cache keys, and lets you report the full run.

Demo notebook: [`demo/use_workflows_demo.ipynb`](../demo/use_workflows_demo.ipynb)
writes and runs the lightweight workflow below.

## 1. Install And Activate

```bash
export VPA_HOME="${VPA_HOME:-$HOME/.vibeproteinagent}"
npx vibeproteinagent install --profile base --cuda-profile auto --yes
source "$VPA_HOME/env.sh"
vpa install doctor --profile base
```

Use `full` or targeted `--tools` installs before running heavyweight design
workflows.

## 2. Start Workers

This guide uses CPU-safe tools:

```bash
vpa daemon start --cpu-workers 1 --workspace workflow_demo
```

For mixed GPU/CPU design workflows, start both:

```bash
vpa daemon start --workers 1 --cpu-workers 2 --workspace design_round
```

## 3. Optional Prewarm

Prewarm the tools that the workflow will call after the daemon is running.
`vpa daemon warm` uses the currently running daemon socket, so run it from the
same shell/runtime environment as `daemon start`.

```bash
vpa daemon warm sequence_edit --json
vpa daemon warm sequence_analysis --json
vpa daemon warm protein_qc --json
vpa daemon warm rank_designs --json
```

For heavyweight workflows, prewarm only the installed tools you actually plan to
use:

```bash
vpa daemon warm proteinmpnn --workers all --json
vpa daemon warm protenix_v2 --workers all --json
```

## 4. Write A Minimal Workflow

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

## 5. Inspect And Run

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

## 6. Monitor, Resume, And Report

```bash
vpa workflow status WORKFLOW_ID --workspace workflow_demo

vpa workflow run sequence_triage.yaml \
  --workspace workflow_demo \
  --resume WORKFLOW_ID \
  --max-wait-s 3600

vpa report WORKFLOW_ID \
  --workspace workflow_demo \
  --out workflow_report.html \
  --open
```

Use `--params-file params.yaml` when parameters are easier to maintain in a
file. This is the recommended path for long sequences, FASTA paths, or nested
parameters:

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
rfdiffusion, rfdiffusion3, rfpeptides, pxdesign, boltzgen, bindcraft, or proteina_complexa
-> proteinmpnn, ligandmpnn, esm2, or esm3 when sequence design/refinement is needed
-> protenix_v2, boltz2_affinity, vina, or another requested prediction/affinity tool
-> protein_qc, sequence_analysis, structure_analysis, and requested scoring tools
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
