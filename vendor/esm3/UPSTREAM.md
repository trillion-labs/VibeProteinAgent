# ESM3 — pointer vendor (pip-installed, no source copy)

| field | value |
| --- | --- |
| Upstream URL | https://github.com/evolutionaryscale/esm |
| Pinned commit | `453b2e2e713ee35a1dcbc684b11d03303a59294e` |
| Installed version | `esm` 3.2.4a1 |
| License | Cambrian Open License Agreement (see `./LICENSE`) |
| Weights license (open variant) | Cambrian Open License (open access; no HF token needed) |
| Weights license (medium/large) | Cambrian Non-Commercial License (HF-gated) |
| Vendored on | 2026-05-17 |
| Vendored by | VibeProteinAgent Phase 14 |

## What is vendored

**Nothing under source control beyond this README + LICENSE.** The `esm`
package is pip-installed into the `vpa` env via `--no-deps` (see Phase 1.7,
note `2eff00ef-…`). All adapter code in
`src/vibeproteinagent/adapters/esm3/adapter.py` imports the installed
package directly (`from esm.models.esm3 import ESM3`).

This mirrors the Phase 4.2 Protenix-v2 *pointer-vendor* pattern: tools too
large to copy under `vendor/` and whose canonical install path is "pip into
the worker env" record only provenance + license here.

## Bootstrap recipe

The ESM3 install is part of Phase 1.7 VPA env augmentation v1:

```bash
# In the vpa env:
python -m pip install --no-user --no-deps \
    anarcii zstandard cloudpathlib msgpack-numpy pygtrie \
    tenacity pydssp dna_features_viewer py3dmol zstd
python -m pip install --no-user --no-deps /path/to/esm-checkout-or-tarball
```

`--no-deps` is **load-bearing** for the final ESM3 install: ESM3's
pyproject pins `transformers<4.53.0`, but vpa runs `transformers 5.8.1`
(needed for B200 / cu128). With `--no-deps`, pip skips the pin and we run
ESM3's source against the newer transformers — confirmed working for
sequence embeddings, per-residue logits, and structure generation on
`esm3_sm_open_v1` in Phase 1.7 + Phase 14.

## Weights

`esm3_sm_open_v1` (1.4 B params; ~5.7 GiB on disk) is **openly accessible
via HF without a token** — confirmed during Phase 14 by issuing a fresh
`ESM3.from_pretrained("esm3_sm_open_v1", device=...)` on a machine with
`HF_TOKEN` unset; the SDK downloaded 22 files from
`EvolutionaryScale/esm3-sm-open-v1` in ~35 s. Weights cache under
`~/.cache/huggingface/hub/models--EvolutionaryScale--esm3-sm-open-v1`.

Larger variants (`esm3-medium-2024-08`, `esm3-large-2024-12`, etc.) are
gated behind the Cambrian Non-Commercial License + an HF token. To use
them via the adapter, set `HF_TOKEN` or `VPA_HF_CREDENTIAL` in the worker env and accept the
EvolutionaryScale terms at
`https://huggingface.co/EvolutionaryScale/<model>`. The adapter raises
`VPAError(E_DEPENDENCY, …)` with a clear pointer to that URL if a gated
weight download fails.

## Modifications

None. The pip-installed `esm` package is used as-is.
