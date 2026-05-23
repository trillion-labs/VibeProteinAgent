# BoltzGen — pointer-only vendor

The BoltzGen Python tree and model cache are too large to copy under
``vendor/``. The installer fetches the source checkout into
``$VPA_HOME/sidecars/boltzgen`` and its HuggingFace-style model + dataset cache
under ``$VPA_HOME/cache/boltzgen``. The adapter resolves both via env vars.

## Upstream pin

| field | value |
| --- | --- |
| Upstream URL | https://github.com/HannesStark/boltzgen |
| Pinned commit | `31d9d9b9c72245b4ed6fe8742d6fbf4e1a3552a0` ("Version bump to 0.3.2") |
| Upstream version | `boltzgen` 0.3.2 (per `pyproject.toml`) |
| License | MIT (`./LICENSE`) |
| Default installed source tree | `$VPA_HOME/sidecars/boltzgen/` |
| Default installed model + dataset cache | `$VPA_HOME/cache/boltzgen/` |

## Required cache contents

The adapter expects the standard HuggingFace cache layout produced by
`huggingface_hub.hf_hub_download(..., cache_dir=<VPA_BOLTZGEN_CACHE>)`:

```
$VPA_BOLTZGEN_CACHE/
  models--boltzgen--boltzgen-1/snapshots/<rev>/boltzgen1_diverse.ckpt
  models--boltzgen--boltzgen-1/snapshots/<rev>/boltzgen1_adherence.ckpt
  models--boltzgen--boltzgen-1/snapshots/<rev>/boltzgen1_ifold.ckpt
  models--boltzgen--boltzgen-1/snapshots/<rev>/boltz2_conf_final.ckpt
  datasets--boltzgen--inference-data/snapshots/<rev>/mols.zip
```

The `boltz2_aff.ckpt` (used only by the `protein-small_molecule` protocol) is
not strictly required for `protein-anything` / `peptide-anything`; if you
plan to use the small-molecule protocol, also place it under
`models--boltzgen--boltzgen-1/snapshots/<rev>/`.

## Env vars consumed by the adapter

- `VPA_BOLTZGEN_TREE` — path to the BoltzGen source checkout (must contain
  `src/boltzgen/`). Adapter inserts `<tree>/src` onto `sys.path`.
- `VPA_BOLTZGEN_CACHE` — HuggingFace-style cache directory (no trailing
  `hub/`). Passed to BoltzGen as `--cache` so all artifact lookups
  short-circuit to local files.

## Why pointer-only

- The on-disk source tree is ~50 MB of Python.
- The cache directory is ~6 GB of model checkpoints.
- Both are installed into `$VPA_HOME` by `vpa install`. Copying them under
  `vendor/` would bloat the repo and the cache layout depends on the HF rev
  hash.

## Deliberate-skip notes

- We force `--use_kernels false`. The `cuequivariance_*` wheels (>600 MiB
  of CUDA binaries) are not installed in the `vpa` env and pulling them
  risks disturbing the pinned torch/numpy stack. Pure-PyTorch kernels run
  slower but produce identical outputs.
- We force `--no_subprocess`. The CLI's default is to fork each pipeline
  step into a fresh Python subprocess; inside a worker process that's
  redundant and complicates output streaming. With `--no_subprocess` all
  steps run in-process for one BoltzGen invocation.
- The adapter launches `boltzgen run` itself as a subprocess (one per
  `run()`) — this is the safe outermost isolation boundary, even though
  internally everything runs in one Python process.
