# Boltz-2 + affinity — pointer-only vendor entry

This directory is **not a vendored source copy.** The installer fetches the
Boltz source checkout into `$VPA_HOME/sidecars/boltz` and stores model
checkpoints plus CCD molecules under `$VPA_HOME/cache/boltz`. The adapter at
`src/vibeproteinagent/adapters/boltz2_affinity/` resolves both via env vars.

| field | value |
| --- | --- |
| Upstream URL | https://github.com/jwohlwend/boltz |
| Pinned commit | `cb04aeccdd480fd4db707f0bbafde538397fa2ac` (HEAD message: "Merge branch 'main' of github.com:jwohlwend/boltz") |
| Upstream version | `boltz` 2.2.1 (per `pyproject.toml`) |
| License | MIT (see `./LICENSE`) |
| Vendored on | 2026-05-17 |
| Vendored by | VibeProteinAgent Phase 15 |
| Default installed source tree | `$VPA_HOME/sidecars/boltz/` |
| Default installed weight + CCD cache | `$VPA_HOME/cache/boltz/` |

## What is *not* in this directory

The Boltz Python package (`boltz.main`, `boltz.model.models.boltz2`, etc.),
the CCD molecule cache (`mols/` and `mols.tar`, ~3.5 GiB total), and the two
Boltz-2 checkpoints (`boltz2_conf.ckpt` 2.3 GiB + `boltz2_aff.ckpt` 2.0 GiB)
are not committed to this repository. `vpa install --tools boltz2_affinity`
clones/downloads them into the local install root. The adapter validates each
expected file at construction time.

## Environment variables (consumed by the adapter)

| var | default | purpose |
| --- | --- | --- |
| `VPA_BOLTZ_TREE`  | `$VPA_HOME/sidecars/boltz` | absolute path to the Boltz source root (must contain `src/boltz/`) |
| `VPA_BOLTZ_CACHE` | `$VPA_HOME/cache/boltz` | absolute path to the directory holding `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, and the extracted `mols/` dir |

The adapter inserts `<VPA_BOLTZ_TREE>/src` onto `sys.path` and exports
`BOLTZ_CACHE` so Boltz's own helpers (`get_cache_path()`, `download_boltz2`)
short-circuit to the existing files instead of attempting a download.

## Relocating

To run on a different machine without the VPA installer:

1. Point `VPA_BOLTZ_TREE` at a Boltz source checkout (commit
   `cb04aec…` or compatible; the upstream API the adapter consumes is
   `boltz.main.predict` — stable across recent 2.x).
2. Point `VPA_BOLTZ_CACHE` at a directory containing both checkpoints and
   the extracted `mols/` directory (untar `mols.tar` from the Boltz release).

No further setup is required from the adapter side. The adapter does not
attempt network access — if any of the cache files is missing, it raises
`FileNotFoundError` with the offending path.

## Deliberately skipped upstream pins

`boltz`'s `pyproject.toml` hard-pins several scientific packages that
collide with the parent `vpa` environment (numpy `<2.0`, pytorch-lightning
`==2.5.0`, scipy `==1.13.1`, etc.). Per Phase 14 lesson (project note
`c9210586-…`) the adapter does **not** install `boltz` as a pip package;
it consumes the local source tree via `sys.path` and lets the inference
hot path run against the vpa stack's versions (numpy 2.3.x,
pytorch-lightning 2.6.x, scipy 1.17.x). This is verified end-to-end by
the slow regression test.

The runtime path required these `--no-deps` installs in the vpa env (none
of which override a pinned scientific package): `pytorch-lightning`,
`lightning-utilities`, `torchmetrics`, `einx`, `mashumaro`, `numba`,
`llvmlite`, `fairscale`, `chembl_structure_pipeline`, `ihm`, `frozendict`.

## License

`./LICENSE` is a verbatim copy of the upstream Boltz MIT license.
The adapter code in `src/vibeproteinagent/adapters/boltz2_affinity/` is
licensed under VibeProteinAgent's first-party PolyForm Noncommercial License
1.0.0.
