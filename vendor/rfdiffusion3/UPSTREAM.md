# RFdiffusion3 (RFD3) — Foundry vendor pointer

## Provenance

- **Distribution**: RosettaCommons Foundry Python package, `rc-foundry[rfd3]`
- **CLI**: `rfd3 design`
- **Docs**: `https://rosettacommons.github.io/foundry/`
- **Local env used by VPA**: `vendor/rfdiffusion3/.venv` unless
  `$VPA_RFDIFFUSION3_ENV` points elsewhere

Earlier VPA phases treated RFD3 as `BLOCKED-UNAVAILABLE` because no standalone
`RFdiffusion3` GitHub repository existed. That is stale: RFD3 is public through
Foundry, so the adapter is now a Foundry-backed local compatibility island.

## Why a compatibility env?

Installing `rc-foundry[rfd3]` directly into the shared `vpa` conda env would
pull a large atomworks/wandb/biotite stack and can downgrade packages such as
`pandas`. Keep the shared env stable and run RFD3 through a local env:

```bash
cd vendor/rfdiffusion3
uv venv --python /opt/miniconda3/envs/vpa/bin/python .venv
source .venv/bin/activate
uv pip install torch==2.7.1+cu128 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install 'rc-foundry[rfd3]'
```

Cluster images may set `DEBUG=release`, but Foundry parses `DEBUG` as a
boolean. Use `DEBUG=false` for Foundry commands if needed.

## Checkpoints

Default checkpoint directory:

```bash
vendor/rfdiffusion3/checkpoints/
```

Install:

```bash
cd vendor/rfdiffusion3
mkdir -p checkpoints
DEBUG=false .venv/bin/foundry install rfd3 --checkpoint-dir checkpoints
```

The adapter also honors:

| variable | purpose |
| --- | --- |
| `VPA_RFDIFFUSION3_ENV` | env containing `bin/rfd3` |
| `VPA_RFDIFFUSION3_CKPT` | exact checkpoint file |
| `VPA_RFDIFFUSION3_WEIGHTS` | checkpoint directory |

## Status

`rfdiffusion3` is no longer upstream-unavailable. It should be reported as
`blocked_dep` until the Foundry env and checkpoint are present, then the
availability probe returns healthy.
