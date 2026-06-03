# VPA Installation Runbook

This runbook is for agents handling user requests such as "install VPA",
"repair VPA", "make this tool available", or "set up VPA so I can run protein
workflows." Keep it portable for external users: do not hard-code developer
names, personal home directories, cluster hosts, or one-off local paths.

## Install Request Boundary

When the user asks to install, repair, or prepare VPA, read this file before
running commands. Installation work is distinct from workflow execution:

- Installation creates or repairs environments, sidecars, binaries, weights, and
  cache state.
- Execution creates workspaces, starts workers, submits runs, tracks status, and
  renders reports. Use [vpa_execution_environment.md](vpa_execution_environment.md)
  after installation is ready.

## Portable Variables

Use these names in commands and summaries:

- `PROJECT_ROOT`: the VPA repository checkout, when running from source.
- `VPA_HOME`: optional install and state root. If unset, VPA uses its default.
- `VPA_PYTHON`: optional Python executable for source-checkout fallback commands.
- `PYTHONPATH`: include `$PROJECT_ROOT/src` when running from source.

From a source checkout:

```bash
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
${VPA_PYTHON:-python} -m vibeproteinagent.cli.main version
```

Prefer `vpa` when it is already on `PATH`. Use the Python module fallback only
when the executable is unavailable.

## Installation Profile Policy

Default to `full` for ordinary external-user setup so the agent can expose and
execute the complete supported VPA tool surface without discovering missing
tools mid-workflow.

- `full`: all supported first-class in-tree tools, sidecars, binaries, and
  required model weights. Use this for normal VPA installation.
- `agent`: agent orchestration plus lightweight planning/retrieval tools. Use
  only when the user explicitly requests a lightweight planning/retrieval setup
  or the machine cannot support full installation.
- `base`: core runtime plus lightweight local tools. Use only for minimal local
  smoke tests or constrained environments.

If the user asks "install VPA" without narrowing the scope, install or repair
`full`. Use targeted `--tools` installs only when the user asks for a specific
tool/capability or when a full install is blocked and the user accepts a narrower
setup.

## Preflight First

Start with non-mutating checks when the machine is unknown:

```bash
vpa install preflight --cuda-profile auto
vpa install plan --profile full --cuda-profile auto
vpa install doctor --profile full
```

For a source-checkout fallback:

```bash
${VPA_PYTHON:-python} -m vibeproteinagent.cli.main install preflight --cuda-profile auto
${VPA_PYTHON:-python} -m vibeproteinagent.cli.main install plan --profile full --cuda-profile auto
```

Record CUDA profile, missing binaries, missing credentials, required local
artifacts, and expected disk usage before mutating a shared system.

## Install Or Repair

Install the selected profile:

```bash
vpa install run --profile full --cuda-profile auto --yes
vpa install doctor --profile full
```

Repair an existing install:

```bash
vpa install run --profile full --cuda-profile auto --yes --repair
vpa install doctor --profile full
```

Use `--dry-run` before a destructive or expensive install:

```bash
vpa install run --profile full --cuda-profile auto --dry-run
```

## Targeted Tool Installs

When the user asks for a specific capability, install only the necessary tools.

Sequence and template retrieval:

```bash
vpa install run --tools search_sequence,mmseqs2_gpu,template_select,foldseek --yes --repair
vpa install doctor --tools search_sequence,mmseqs2_gpu,template_select,foldseek
```

PXDesign full-structure binder workflow:

```bash
vpa install run --tools pxdesign,protenix_v2,protein_qc,rank_designs --yes --repair
vpa install doctor --tools pxdesign,protenix_v2,protein_qc,rank_designs
```

De novo design:

```bash
vpa install run --tools rfdiffusion,proteinmpnn,protenix_v2,protein_qc,rank_designs --yes --repair
vpa install doctor --tools rfdiffusion,proteinmpnn,protenix_v2,protein_qc,rank_designs
```

Semantic de novo or family-guided sequence design:

```bash
vpa install run --tools search_sequence,family_profile,esm2,esm3,sequence_analysis,rank_designs --yes --repair
vpa install doctor --tools search_sequence,family_profile,esm2,esm3,sequence_analysis,rank_designs
```

Peptide, motif, or alternate backbone generation:

```bash
vpa install run --tools rfpeptides,rfdiffusion3,proteinmpnn,protenix_v2,protein_qc,rank_designs --yes --repair
vpa install doctor --tools rfpeptides,rfdiffusion3,proteinmpnn,protenix_v2,protein_qc,rank_designs
```

Ligand-aware or small-molecule binding workflows:

```bash
vpa install run --tools ligandmpnn,proteina_complexa,vina,boltz2_affinity,protein_qc,rank_designs --yes --repair
vpa install doctor --tools ligandmpnn,proteina_complexa,vina,boltz2_affinity,protein_qc,rank_designs
```

Structure scoring, refinement, and comparison:

```bash
vpa install run --tools pyrosetta,ipsae,prodigy,pxmeter,tmalign,structure_analysis,protein_qc,rank_designs --yes --repair
vpa install doctor --tools pyrosetta,ipsae,prodigy,pxmeter,tmalign,structure_analysis,protein_qc,rank_designs
```

Do not install unrelated heavyweight tools just because they exist in the
registry. Match installation scope to the user's requested workflow.

## External Binaries

External binary adapters must be resolved by VPA probes. Do not invent binary
paths. For `foldseek`, resolution order is:

1. `$VPA_FOLDSEEK_BINARY`, if it points to an executable file;
2. the active Python environment's `bin/foldseek`;
3. `foldseek` on `PATH`.

Only set an override when the user provided a path or an install command produced
a real executable:

```bash
export VPA_FOLDSEEK_BINARY="/path/to/foldseek"
test -x "$VPA_FOLDSEEK_BINARY"
vpa install doctor --tools foldseek
```

For other binary-backed tools, inspect the descriptor or install manifest, then
verify with `vpa install doctor --tools <tool>`.

## Credentials And Acceptance

If `doctor` reports missing credentials or required acceptance, do not guess
secrets. Ask the user to provide the required token, license acceptance, or local
artifact. After the user provides it, export the documented environment variable
and rerun `doctor`.

## Completion Criteria

An installation task is complete only when:

- the requested profile or tool set has been installed or repaired;
- `vpa install doctor` reports the requested profile or tools as ready;
- required external binaries are discoverable by VPA;
- any missing credentials, license acceptance, or local artifacts are reported
  clearly if they block readiness;
- the final response tells the user which profile/tools were installed, where
  VPA state lives if known, and what execution command family to use next.
