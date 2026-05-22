# Use A Single Tool Or Manual Separate Runs

Use this path when you want to run one tool directly, or when you prefer to run
several tools yourself as separate commands. `vpa run` submits one tool at a
time; if you want VPA to connect tools automatically, use a workflow instead.

Demo notebook: [`demo/use_tools_demo.ipynb`](../demo/use_tools_demo.ipynb)
walks through direct ProteinMPNN, Protenix-v2, sequence analysis, and QC runs.
It is smoke-safe by default: it validates GPU/CLI access and prints long model
commands unless you set `VPA_DEMO_RUN_HEAVY=1`.

## 1. Install What You Need

For a small setup:

```bash
# `--yes` skips confirmation prompts and starts the install right away.
npx vibeproteinagent install --profile base --yes
```

For selected heavy tools and their weights (e.g., Boltz2, RFDiffusion, ProteinMPNN):

```bash
npx vibeproteinagent install preflight --tools boltz2_affinity,rfdiffusion,proteinmpnn --cuda-profile auto
npx vibeproteinagent install --tools boltz2_affinity,rfdiffusion,proteinmpnn --cuda-profile auto --yes
```

For ESM3 or other gated Hugging Face model downloads, export a credential before
installing or before the first run:

```bash
export HF_TOKEN=<huggingface-access-token>
npx vibeproteinagent install --tools esm3 --cuda-profile auto --yes
```

`VPA_HF_CREDENTIAL` is also accepted as a VPA-specific alias.

Check readiness:

```bash
vpa install doctor --tools boltz2_affinity,rfdiffusion,proteinmpnn
```

## 2. Create A Workspace And Start Workers

A workspace keeps related jobs, logs, artifacts, and reports together. Use the
same workspace name for commands that belong to the same run. Starting the daemon
creates the workspace if needed and launches local workers that execute queued
tool jobs.

```bash
vpa daemon start --workers 1 --workspace my_run
vpa daemon status --workspace my_run
```

`--workers` is the GPU worker count. For mixed workflows, add
`--cpu-workers M` so CPU-only retrieval, analysis, and reporting-adjacent jobs
do not occupy GPU workers.

## 3. Find And Inspect Tools

```bash
vpa tools
vpa tool boltz2_affinity
```

## 4. Run One Tool

Run a lightweight analysis tool:

```bash
vpa run sequence_analysis sequence=ACDEFGHIKLMNPQRSTVWY --workspace my_run --json
```

Run a structure prediction tool with a params file:

```bash
# --follow shows live progress in this terminal.
vpa run boltz2_affinity --params-file boltz_params.yaml --workspace my_run --follow
```

For manual chaining, inspect the completed job's output and pass the needed
path into the next command yourself. If you do not want to copy job ids or file
paths by hand, use a workflow instead.

```bash
# Print only the structure path from the first job.
vpa results <boltz_job_id> --workspace my_run --fields predicted_cif

# Copy that printed path into the next command.
vpa run protein_qc pdb_path=<predicted_cif path> --workspace my_run --follow
```

Example `boltz_params.yaml`:

```yaml
input_yaml: target_with_ligand.yaml
use_msa: true
msa_backend: remote
recycling_steps: 1
sampling_steps: 20
diffusion_samples: 1
sampling_steps_affinity: 20
diffusion_samples_affinity: 1
```

## 5. Read Results

```bash
vpa status <job_id> --workspace my_run --json
vpa results <job_id> --workspace my_run --json
vpa artifacts <job_id> --workspace my_run
vpa download <job_id> --workspace my_run --out ./outputs
```

For a human-readable result page:

```bash
vpa report <job_id> --workspace my_run --open
```
