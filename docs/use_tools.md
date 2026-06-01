# Use Direct Tool Runs

Use direct runs when you know which tool to call and want to inspect each result
yourself. This is the fastest way to learn VPA because it exercises the same
workspace, worker, artifact, and report system used by workflows and the agent.

Demo notebook: [`demo/use_tools_demo.ipynb`](../demo/use_tools_demo.ipynb)
runs this guide with CPU-safe defaults.

## 1. Install And Verify

For lightweight direct runs:

```bash
npx vibeproteinagent install --profile base --cuda-profile auto --yes
source "${VPA_HOME:-$HOME/.vibeproteinagent}/env.sh"
vpa version
vpa install doctor --profile base
```

For selected GPU tools, preview and install only what you need:

```bash
vpa install preflight --tools proteinmpnn,protenix_v2 --cuda-profile auto --json
npx vibeproteinagent install --tools proteinmpnn,protenix_v2 --cuda-profile auto --yes
vpa install doctor --tools proteinmpnn,protenix_v2
```

Export `HF_TOKEN` only for tools or model variants that need Hugging Face
credentials.

## 2. Start A Worker

A workspace stores jobs, logs, artifacts, cache records, and reports. The daemon
starts workers that execute queued jobs.

```bash
vpa daemon start --cpu-workers 1 --workspace direct_demo
vpa daemon status --workspace direct_demo
```

For GPU tools:

```bash
vpa daemon start --workers 1 --cpu-workers 2 --workspace design_round
```

## 3. Inspect The Tool Surface

```bash
vpa tools
vpa tools --json
vpa tool info sequence_analysis
vpa tool info protein_qc --json
```

Tool descriptors show the accepted inputs, output schema, GPU requirement, and
availability state.

## 4. Run Lightweight Tools

Run sequence analysis:

```bash
vpa run sequence_analysis \
  sequence=ACDEFGHIKLMNPQRSTVWY \
  --workspace direct_demo \
  --follow
```

Run QC on the same sequence:

```bash
vpa run protein_qc \
  sequence=ACDEFGHIKLMNPQRSTVWY \
  metrics='["gravy","instability_index","pi","polybasic"]' \
  --workspace direct_demo \
  --follow
```

`--follow` streams job events until the job reaches a terminal state. Without
`--follow`, VPA returns the `job_id` immediately and the worker continues in
the background.

## 5. Inspect Outputs

```bash
vpa status <job_id> --workspace direct_demo
vpa wait <job_id> --workspace direct_demo
vpa results <job_id> --workspace direct_demo --json
vpa artifacts <job_id> --workspace direct_demo
vpa download <job_id> --workspace direct_demo --out ./outputs
```

Use `results` for structured adapter output and `artifacts` / `download` for
files written by the job.

## 6. Generate A Report

```bash
vpa report <job_id> \
  --workspace direct_demo \
  --viewer offline \
  --out direct_tool_report.html \
  --open
```

Reports summarize the stored job output and artifacts. They do not rerun the
tool.

## Optional Heavy Tool Runs

After `vpa install doctor` reports the tools ready and a GPU worker is running,
you can run heavier jobs the same way:

```bash
vpa run proteinmpnn \
  pdb_path=tests/fixtures/6MRR.pdb \
  num_designs=2 \
  seed=17 \
  --workspace design_round \
  --follow

vpa run protenix_v2 \
  input_json=tests/fixtures/protenix_v2_input_tiny.json \
  --workspace design_round \
  --follow
```

If a tool is unavailable, start with:

```bash
vpa tool info <tool>
vpa install doctor --tools <tool> --json
```
