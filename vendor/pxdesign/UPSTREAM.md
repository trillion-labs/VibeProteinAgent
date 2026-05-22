# PXDesign — vendored source

ByteDance's open-source de novo protein-binder design suite. Vendored under
its original package name (`pxdesign/`) so that internal `from pxdesign.X
import Y` and inter-package `from pxdbench.X import Y` imports resolve
without a global rename. The Phase 5.2 adapter will load this tree via
`sys.path.insert(0, <vendor>/pxdesign)`.

| field | value |
| --- | --- |
| Upstream URL | https://github.com/bytedance/PXDesign |
| Pinned commit | `f788441313c84c3074fe9596ac2433f96b15c763` (HEAD on `main`, vendored 2026-05-16) |
| Tag at clone time | none (no published tag yet — using HEAD) |
| License | Apache 2.0 (see `./LICENSE`) |
| Vendored on | 2026-05-16 |
| Vendored by | VibeProteinAgent Phase 5.1 |
| Layout under here | `pxdesign/` (the Python package, ~1.4 MB) |
| Skipped | `assets/` (17 MB, images/binaries), `examples/`, `Dockerfile`, `download_tool_weights.sh`, `install.sh`, `setup.py`, `requirements.txt`, `README.md` |

## What this depends on (peer vendors)

- `vendor/protenix-pxd/` — PXDesign's hard-coded structure backbone is the
  `Protenix v0.5.0+pxd` fork; both `pxdesign/configs/configs_infer.py:18`
  (`from protenix.config.extend_types import ListValue, RequiredValue`) and
  `pxdesign/utils/infer.py:27` (`from protenix.config import parse_configs`)
  pull in protenix at import time. The Phase 5.2 adapter must add both
  `vendor/protenix-pxd` AND `vendor/pxdesign` to `sys.path` before importing
  `pxdesign`.
- `vendor/pxdbench/` — `pxdesign/utils/infer.py:30` and
  `pxdesign/runner/helpers.py:12-14` import `pxdbench.run`,
  `pxdbench.utils`, `pxdbench.tools.ptx.ptx_utils`,
  `pxdbench.pxd_configs.eval`. This is the PXDesignBench package
  (`bytedance/PXDesignBench` at tag `v0.1.2`).

## Patches applied to upstream

None to PXDesign source. The B200 (Blackwell sm_100) patch lives entirely in
`vendor/protenix-pxd/.patches/01-b200-arch.patch` — PXDesign itself doesn't
build custom CUDA extensions.

## Environment variables (consumed by Phase 5.2 adapter)

| var | default | purpose |
| --- | --- | --- |
| `VPA_PXDESIGN_VENDOR` | `<repo>/vendor/pxdesign` | path to this vendored copy |
| `VPA_PXDBENCH_VENDOR` | `<repo>/vendor/pxdbench` | path to vendored pxdbench |
| `VPA_PROTENIX_PXD_VENDOR` | `<repo>/vendor/protenix-pxd` | path to vendored protenix v0.5.0+pxd |
| `VPA_PXDESIGN_CKPT_DIR` | `<repo>/release_data/checkpoint` (default per upstream `configs_infer.py`) | directory containing `pxdesign_v0.1.0.pt` and the Protenix `base`/`mini` checkpoints; first run will populate it on cache-miss |

## License

Apache 2.0 (full text in `./LICENSE`, copied verbatim from upstream).
Downstream filter weights it expects to load — AlphaFold2 params from Google
and ProteinMPNN weights from Dauparas — carry their own licenses (CC-BY-4.0
derivative and MIT respectively). Acquiring those is out of scope for this
phase.

## Pipeline mode (real-sequence design)

The Phase 19 adapter extension adds a second runtime `mode` to the existing
`pxdesign` adapter:

| mode | runtime location | outputs |
| --- | --- | --- |
| `infer` (default) | in-process inside the `vpa` env (diffusion only) | backbone-only CIFs, sequence carries `X` placeholders for residues the CIF parser can't resolve |
| `pipeline` | subprocess into the dedicated **`pxdesign-pipeline-vpa`** conda env | full pipeline — diffusion + AF2-IG filter + ProteinMPNN sequence design + top-K selection; CIFs carry real designed amino-acid sequences |

### Why a dedicated conda env (not vpa)

`pipeline` mode needs JAX + dm-haiku + optax + ColabDesign, none of which
can co-exist in vpa:

- `jaxlib==*cuda*` bundles its own `cudnn-cu12` / `nccl-cu12` / `cublas-cu12`
  wheels that would override torch's bundled CUDA libs and break every
  other adapter that depends on torch 2.7.1+cu128 (see Phase 16b note
  `094ffe75-…` — same blocker BindCraft hit).
- ColabDesign pins `numpy<2`; vpa is pinned to `numpy 2.3.5`.
- AF2 weights are ~5 GiB on disk + carry their own (CC-BY-4.0 derivative)
  redistribution constraint.

So `pipeline` mode subprocesses out into a dedicated env. The vpa env stays
clean; pinned scientific packages remain untouched.

### Bootstrap

```
conda create -n pxdesign-pipeline-vpa python=3.11
conda activate pxdesign-pipeline-vpa
pip install "jax[cuda12]==0.4.29" jaxlib==0.4.29
pip install "optax==0.2.5" "dm-haiku==0.0.13" dm-tree \
            "transformers==4.51.3" msgpack torch==2.7.1+cu128 \
            --extra-index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/sokrypton/ColabDesign.git
pip install aiosqlite pydantic rich typer jsonschema mcp ninja \
            biotite rdkit optree fair-esm scikit-learn
# The vendored PXDesign/PXDBench/Protenix trees in this checkout are
# source-only snapshots (setup.py/pyproject.toml were intentionally skipped).
# The VPA adapter injects these roots into the sidecar PYTHONPATH:
#   <repo>/vendor/pxdesign
#   <repo>/vendor/pxdbench
#   <repo>/vendor/protenix-pxd
# Stage AF2 + ProteinMPNN tool weights:
bash <upstream>/download_tool_weights.sh          # populates TOOL_WEIGHTS_ROOT
export TOOL_WEIGHTS_ROOT=/path/to/tool_weights    # consumed by pxdesign.utils.pipeline.check_tool_weights
```

After provisioning, surface the env to the adapter:

```
export VPA_PXDESIGN_PIPELINE_ENV=$(conda info --envs | grep pxdesign-pipeline-vpa | awk '{print $NF}')
```

### Adapter dispatch

When `mode=pipeline`, the vpa-side adapter:

1. Probes `availability_skip_reason()` (resolves
   `$VPA_PXDESIGN_PIPELINE_ENV` → `<env>/bin/python`).
   - The probe imports the persistent sidecar, JAX/Haiku/Optax/ColabDesign,
     `pxdesign.runner.pipeline`, and runs `check_tool_weights()`.
   - If the env, imports, or `$TOOL_WEIGHTS_ROOT` AF2 params are missing,
     raises `DependencyError` (`E_DEPENDENCY`) with the bootstrap pointer
     above — no half-baked partial outputs.
2. Stages the target structure + writes a single-task PXDesign input JSON
   into `<job_dir>/inputs/`.
3. Calls the persistent sidecar in `<env>/bin/python`, which executes:

       <env>/bin/python -m pxdesign.runner.pipeline \
         --dump_dir <job>/outputs \
         --load_checkpoint_dir $VPA_PXDESIGN_CKPT_DIR \
         --input_json_path <job>/inputs/<task>.json \
         --N_sample <num_samples> --N_step <num_steps> \
         --eta_type const --eta_min 2.5 --eta_max 2.5 \
         --dtype bf16 --seeds <seed> \
         --preset preview --N_max_runs 1 \
         --min_total_return <top_k> --max_success_return <top_k>

   The subprocess env strips `PYTHONPATH`, sets `PYTHONNOUSERSITE=1`, and
   forces `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX shares VRAM cooperatively.
4. Parses `<job>/outputs/design_outputs/<task>/summary.csv` and follows
   each row's `chosen_struct_path` to the chosen CIF. The CSV's
   `sequence` column carries the designed AA string (the upstream pipeline
   injects it into the CIF via `replace_last_xpb_chain_sequence`).

### Promotion to VERIFIED (Phase 20)

No code changes — the operator only needs to provision the env above.
Re-run `pytest -q -m slow tests/test_adapter_pxdesign_pipeline_real.py` to
produce the first real-sequence sample output.
