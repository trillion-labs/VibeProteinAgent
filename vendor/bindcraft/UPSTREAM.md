# BindCraft — vendored copy

BindCraft (Pacesa et al., ETH Zurich) is the canonical AF2-backprop
binder design pipeline. The full source is vendored here (~3 MiB) so
the adapter can reference ``bindcraft.py`` / ``settings_filters/`` /
``functions/`` without an out-of-tree path.

## Upstream pin

| field | value |
| --- | --- |
| Upstream URL | https://github.com/martinpacesa/BindCraft |
| Pinned commit | `b971db42ba6e091afab63ccb30ae02215150a990` (HEAD of `main` on 2026-05-17) |
| Cloned via | `git clone --depth 1 https://github.com/martinpacesa/BindCraft.git vendor/bindcraft` |
| License | MIT (`./LICENSE`) |
| Vendored on | 2026-05-17 |
| Cite | Pacesa, Nickel, Schellhaas, et al., *Nature* 2025 |

## Required weights (NOT shipped — too large)

| file/dir | size | purpose | URL |
| --- | --- | --- | --- |
| `params/params_model_*_ptm.npz` | ~5.3 GiB total | AlphaFold2 / multimer | https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar |

Standard install path::

```bash
mkdir -p vendor/bindcraft/params
wget -O /tmp/af2.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xvf /tmp/af2.tar -C vendor/bindcraft/params/
```

Override the location via `$VPA_BINDCRAFT_PARAMS` if the params live
elsewhere on disk. The brief's 2 GiB cap covers the compressed
tarball partway — operators decide whether to populate the cache.

ProteinMPNN weights re-use whatever is already at
`vendor/proteinmpnn/weights/v_48_020.pt`; BindCraft loads them via
ColabDesign which looks them up from `model_path: "v_48_020"` against
the in-env ``colabdesign`` package's shipped weight dir.

## Required separate conda env

BindCraft cannot share a process with the ``vpa`` env:

* `numpy < 2.0` (BindCraft installer) vs our pinned `numpy 2.3.5`
* `jaxlib==*cuda*` bundles a cuDNN/CUBLAS that would override torch's
  bundled CUDA libs (Phase 12-20 hard rule: no edits to the pinned ML
  stack).
* `pyrosetta` is installed from its own distribution channel and should stay
  isolated from the shared VPA runtime.

Set up the dedicated env via the upstream installer::

```bash
cd vendor/bindcraft
bash install_bindcraft.sh --pkg_manager conda --cuda 12.8
```

This creates a ``BindCraft`` conda env with the right JAX + ColabDesign
+ PyRosetta combo. Our adapter shells out via ``conda run -n BindCraft
python bindcraft.py …``.

Override the env name via `$VPA_BINDCRAFT_CONDA_ENV` if you renamed it.

## Repairing an existing sidecar env

The installer is idempotent for packages that are already present. If a
previously-created VPA BindCraft sidecar is missing a newly-added lightweight
runtime dependency, operators can repair it without rebuilding the heavy
BindCraft stack:

```bash
source "$VPA_HOME/env.sh"
bash vendor/bindcraft/install_bindcraft.sh --pkg_manager micromamba \
    --env-prefix "$VPA_BINDCRAFT_ENV_PATH" --cuda "${VPA_BINDCRAFT_CUDA:-12.8}"
```

The smoke-checked VPA sidecar dependencies currently include `pydantic`,
`msgpack`, and `aiosqlite`; the latter is imported transitively by
`vibeproteinagent.workspace.store` before BindCraft starts.

## Env vars consumed by the adapter

- `VPA_BINDCRAFT_TREE` — root of the BindCraft checkout (must contain
  `bindcraft.py`). Defaults to the in-tree `vendor/bindcraft/`
  directory.
- `VPA_BINDCRAFT_PARAMS` — directory containing
  `params_model_*_ptm.npz`. Defaults to `<VPA_BINDCRAFT_TREE>/params/`.
- `VPA_BINDCRAFT_CONDA_ENV` — name of the conda env holding the
  BindCraft Python stack. Default `BindCraft`.

## Verdict for Phase 16b — PARTIAL / BLOCKED-DEP

The adapter shell + fast unit tests + import-path verification land in
Phase 16b. The slow test skips cleanly with a clear blocker message
when:

* the dedicated conda env is not present, **or**
* AF2 params are not on disk under `$VPA_BINDCRAFT_PARAMS`, **or**
* PyRosetta is not installed in the dedicated env.

Phase 20 punch-list: with the BindCraft env set up, the slow test should
produce ≥1 accepted binder PDB. Expect
multi-hour runtime per real trajectory; the smallest viable test uses
``binder_length=30, num_designs=1`` and is best run as a dedicated
overnight job.
