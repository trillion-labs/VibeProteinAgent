<h1><img src="assets/vpa_logo_inline.png" alt="" height="36" align="top" style="margin-right:5px">VibeProteinAgent</h1>

[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/license-PolyForm--NC--1.0.0-6b1ea8.svg)](LICENSE)
[![Use: Non-commercial](https://img.shields.io/badge/use-non--commercial-cc0980.svg)](#license)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-1a1535.svg)](pyproject.toml)

**VibeProteinAgent (VPA)** helps coding agents run local protein-design tools
for you. You describe the installation task or protein-design goal in natural
language; Codex or Claude Code agents choose the VPA tools,
runs them, checks the evidence, and gives you an inspectable HTML report.

![VPA report snapshot](assets/readme/vpa-report-snapshot.svg)

## Why Use VPA?

- **One interface for many tools.** Retrieval, sequence analysis, backbone and
  binder design, structure prediction, docking, QC, ranking, and reporting all
  sit behind a single `vpa` command surface.
- **Agent-first.** Codex or Claude Code reads the repo skills,
  picks the tools that fit your objective, runs them, and analyzes the evidence.
- **No design-API bills.** Protein design models run on your own GPUs; once you
  have a coding agent you can run autonomous protein design without subscribing
  to a hosted protein-design service. (First-party code is source-available for
  non-commercial use — see [License](#license).)
- **Built for GPU tools.** Persistent workers and warm sidecars keep heavyweight
  models resident, cutting cold starts across repeated jobs.
- **Inspectable reports.** Every run produces a self-contained HTML report with
  interactive 3D structure rendering, per-candidate statistics, comparison
  tables, and rankings.

## Requirements

- **Hugging Face token.** ESM3 requires gated downloads; export
  `HF_TOKEN` before installing so VPA can fetch them when needed.
- **OS / GPU.** Linux with NVIDIA GPU(s). Each GPU tool runs one model per GPU;
  the heaviest single tool needs ~24 GiB VRAM (most need 1–20 GiB), so any modern
  ≥24 GiB card runs the full tool set (Reference install: 8× NVIDIA B200).
- **CUDA / Python.** Driver supporting CUDA 12.8 (the env ships
  `torch 2.7.1+cu128`); Python 3.11+.
- **Disk.** The Python/PyTorch environment is about 14 GB. Protein tools also
  download their own sidecars and model checkpoints/weights as needed, and
  those are usually the large part. For the full GPU profile, plan for well over
  100 GB and leave room for growth as you enable more tools (the reference
  weight/checkpoint cache is ~256 GB). A focused subset of tools needs much
  less.

## Activate The Installed Runtime

Full GPU tool readiness is defined by the installed VPA runtime. After
installation, activate the runtime before asking Codex or Claude Code to run
protein-design jobs:

```bash
export VPA_HOME=/path/to/your/.vibeproteinagent
source "$VPA_HOME/env.sh"
```

Then verify the same shell sees the expected command and tools:

```bash
vpa version
vpa tools --json
```

For a full-profile install, heavyweight tools such as `boltzgen`, `bindcraft`,
`pxdesign`, `protenix_v2`, and `boltz2_affinity` should report
`available=true`. If they do not, run `vpa install doctor --profile full` from
that activated shell and repair the reported tool/runtime gap before launching
long jobs.

## How It Feels To Use

You do not need to memorize VPA commands for normal use. Ask your coding agent
for the result you want:

```text
Install VPA into /vpa and check that the full GPU profile is ready.
```

or:

```text
Design 50-80 aa soluble mini-protein binders to chain A of `/path/to/target.pdb`.
Return the strongest candidates with structures, quality check, ranking, a final submission decision, and an HTML report.
```

The external agent reads the skills in this repo, decides the command boundary,
derives objective thresholds, runs the tools/workflows, analyzes the outputs, and generates the HTML summary report.

## Prepare Your Agent Shell

First, activate VPA in the shell where you plan to run jobs, then add any
credentials and GPU settings your run needs:

```bash
export VPA_HOME=/path/to/your/.vibeproteinagent
source "$VPA_HOME/env.sh"

export HF_TOKEN="your_huggingface_token"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## Use With Codex

Open this repository in Codex and ask in natural language. For example:

```text
Install VibeProteinAgent into /vpa.
Use the full profile with CUDA auto-detection, run doctor, and tell me whether
VPA is ready.
```

Then send a protein request:

```text
Design 50-80 aa soluble mini-protein binders to chain A of `/path/to/target.pdb`.
Return the strongest candidates with structures, quality check, ranking, a final submission decision, and an HTML report.
```

Codex uses [AGENTS.md](AGENTS.md) and the [`skills/`](skills) directory to
decide what to run. The portable interface is plain language.

## Use With Claude Code

In Claude Code, use the project command included in this repo:

```text
/vpa Install VibeProteinAgent into /vpa. Use the full profile with CUDA
auto-detection, run doctor, and report readiness.
```

Then run a protein request through the same command:

```text
/vpa Design 50-80 aa soluble mini-protein binders to chain A of `/path/to/target.pdb`.
Return the strongest candidates with structures, quality check, ranking, a final submission decision, and an HTML report.
```

The `/vpa` command is defined at [.claude/commands/vpa.md](.claude/commands/vpa.md).
It is a convenience wrapper around the same repo skills used by Codex.

## How Results Are Decided

For protein-design work, the external agent makes the objective and acceptance
thresholds explicit, then *explores*: it tries different tools and approaches,
compares the evidence, and iterates to push the objective as high as it can
before finishing with a recorded submission. The aim is the strongest candidate;
let the evidence decide when the returns plateau.

```text
objective -> thresholds -> explore tools/workflows -> evidence -> rank
   ^                                                                |
   +------------------ iterate to maximize the objective -----------+
-> submit -> report
```

VPA itself stays a stateless, deterministic substrate; the external agent owns
the exploration loop and runs as many jobs as the objective warrants.

## A Note on Report Rendering

The HTML report is generated automatically from the run's artifacts. Because the
external agent's behavior is dynamic, the renderer can't always anticipate every
shape the results take, so sometimes a piece of a report (e.g., structure view) 
may come out missing or blank.

If that happens, just tell the LLM agent about the rendering issue and ask it 
to re-link the affected artifact or adjust the values that feed the HTML and 
render the report again.

## Cite

```bibtex
@software{vpa2026vibeproteinagent,
  author       = {Hyunjin Seo},
  title        = {VibeProteinAgent},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/trillion-labs/VibeProteinAgent}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## License

VPA first-party code is released for non-commercial use under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use requires a
separate license. Third-party tools, model weights, sidecars, and vendored or
pointer-vendored upstream components keep their own licenses and terms. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or
production use.
