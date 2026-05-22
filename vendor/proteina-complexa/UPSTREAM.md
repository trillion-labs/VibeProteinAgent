# Proteina-Complexa — vendor pointer

## Provenance

- **URL**: https://github.com/NVIDIA-Digital-Bio/Proteina-Complexa
- **Default branch**: `dev`
- **Pinned commit** (Phase 17 vendor sweep, 2026-05-17): `5ae24b055d828918296f2aad63616b1f4cc0e491`
- **Paper**: Didi, Zhang, Zhou, Reidenbach, Cao, Cha, Geffner, Dallago, Tang,
  Bronstein, Steinegger, Kucukbenli, Vahdat, Kreis — *Scaling Atomistic
  Protein Binder Design with Generative Pretraining and Test-Time Compute*.
  ICLR 2026 Oral. [OpenReview](https://openreview.net/forum?id=qmCpJtFZra).
- **Project page**: https://research.nvidia.com/labs/genair/proteina-complexa/

## Licence

This repository ships **multiple components covered by different licences** —
see the `licenses/` directory for the canonical text. Summary:

- Code → Apache 2.0 (`licenses/license_code.txt`).
- Datasets → NVIDIA's dataset licence (`licenses/license_datasets.txt`).
- Weights (the NGC checkpoints listed below) → NVIDIA Open Model Licence
  (`licenses/license_weights.txt`). Commercially usable; derivative
  models allowed.
- Third-party components → mixed (see `licenses/license_third_party.txt`).

## Vendored tree layout

```
vendor/proteina-complexa/
├── README.md
├── LICENSE
├── pyproject.toml
├── community_models/   # openfold, colabdesign, ProteinMPNN, LigandMPNN copies
├── configs/            # hydra configs for binder / ligand_binder / ame pipelines
├── docs/               # CONFIGURATION_GUIDE.md, INFERENCE.md, EVALUATION_METRICS.md, …
├── env/                # build_uv_env.sh + Dockerfile
├── licenses/
├── script_utils/
└── src/proteinfoundation/   # complexa CLI + generation/evaluation/analysis modules
```

`assets/` (the 114 MiB figures + model card images) is **not** vendored
to stay well under the per-phase weight cap; nothing the adapter needs
lives under `assets/`.

## Pipeline variants (one adapter each)

| Variant adapter ID | Pipeline config | NGC checkpoint pair |
| --- | --- | --- |
| `proteina_complexa_protein_binder` | `configs/search_binder_local_pipeline.yaml` | `complexa.ckpt` + `complexa_ae.ckpt` |
| `proteina_complexa_ligand_binder` | `configs/search_ligand_binder_local_pipeline.yaml` | `complexa_ligand.ckpt` + `complexa_ligand_ae.ckpt` |
| `proteina_complexa_motif` (AME) | `configs/search_ame_local_pipeline.yaml` | `complexa_ame.ckpt` + `complexa_ame_ae.ckpt` |

Each adapter shells out to `<venv>/bin/complexa design <config_path> ++…`
with cwd at `$VPA_PROTEINA_COMPLEXA_TREE` (defaults to this directory).

## Status: **PARTIAL / BLOCKED-DEP**

The vpa env *cannot* run Proteina-Complexa in-process. The Phase 12-20
hard-rules forbid modifying torch/numpy/transformers/pandas/scipy.
Upstream's `pyproject.toml` requires:

| dep | upstream pin | vpa pinned | conflict |
| --- | --- | --- | --- |
| `scipy` | `==1.12.0` | `1.17.1` | **hard conflict** — explicit `==` pin |
| `lightning` | `>=2.5.0,<2.6` | (not pinned) | tolerable, but upstream documents a known checkpoint-loading bug at 2.6+ |
| PyTorch | `cu126` extra index | `2.7.1+cu128` | wheel index mismatch |
| `tmol` | git URL, requires custom `numba`/`llvmlite` on Python 3.12 | — | environment-specific install footgun |
| `pyrosetta` | required by `tmol` | — | separate distribution channel |
| AF2/RF3/ESM2 weights | required by reward + refold | — | gated downloads, gigabytes |
| Foldseek/MMseqs2/DSSP/SC | binaries on PATH | (vpa has its own copies under `vpa/bin/`) | path harmonisation required at runtime |

So this adapter is structured exactly like BindCraft (Phase 16b): the
adapter shell registers in the vpa registry, accepts canonical input
schemas, builds correct hydra overrides, parses standard
`inference/<run_name>/…` output layouts. Runtime is deferred to a
**separate uv venv** built by upstream's installer:

```bash
cd vendor/proteina-complexa
./env/build_uv_env.sh              # creates .venv/ with all deps
source .venv/bin/activate
complexa init                       # writes .env scaffolding
complexa download --complexa-all    # fetches the three NGC ckpt pairs (≈ tens of GiB)
# Optionally also fetch AF2 / ESM2 / RF3 weights for reward & refold —
# see docs/EVALUATION_METRICS.md for the environment-variable map.
```

After the venv is built and the NGC checkpoints have landed under
`$VPA_PROTEINA_COMPLEXA_CKPTS` (default `vendor/proteina-complexa/ckpts/`)
the adapter's `healthcheck()` will flip to True, the slow regression
tests will run end-to-end, and the variant can be promoted to
**VERIFIED** without code changes.

## Environment variables

| variable | default | purpose |
| --- | --- | --- |
| `VPA_PROTEINA_COMPLEXA_TREE` | `vendor/proteina-complexa` (this dir) | upstream source tree root |
| `VPA_PROTEINA_COMPLEXA_VENV` | `<tree>/.venv` | uv venv built by `env/build_uv_env.sh` |
| `VPA_PROTEINA_COMPLEXA_CKPTS` | `<tree>/ckpts` | directory holding the three NGC checkpoint pairs |

## Promotion to VERIFIED (Phase 20 punch-list)

1. Provision the uv venv via `vendor/proteina-complexa/env/build_uv_env.sh`
   on Ubuntu 22.04+ (older releases throw GLIBC errors per the upstream
   README; use the Dockerfile under `env/docker/` as a fallback).
2. Download all three NGC checkpoint pairs into `$VPA_PROTEINA_COMPLEXA_CKPTS`.
3. Optionally stage AF2 (`AF2_DIR`), ESM2 (`ESM_DIR`), RF3 (`RF3_CKPT_PATH`,
   `RF3_EXEC_PATH`) — required for the refold/reward stages.
4. Run the slow regression tests:
   ```
   pytest -q -m slow tests/test_adapter_proteina_complexa_protein_binder_real.py
   pytest -q -m slow tests/test_adapter_proteina_complexa_ligand_binder_real.py
   pytest -q -m slow tests/test_adapter_proteina_complexa_motif_real.py
   ```
   Each should produce ≥ 1 design PDB. No code changes needed — the
   adapter routes through `<venv>/bin/complexa design`.
