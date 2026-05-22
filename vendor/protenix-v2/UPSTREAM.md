# Protenix v2 — pointer-only vendor entry

This directory is **not a vendored source copy.** Unlike `vendor/proteinmpnn/`,
the Protenix source tree is too large (hundreds of MB) and already lives on
disk at a stable path with the user's B200 patches applied. The adapter at
`src/vibeproteinagent/adapters/protenix_v2/` consumes that on-disk tree
read-only.

| field | value |
| --- | --- |
| Upstream URL | https://github.com/bytedance/Protenix |
| Pinned commit | `4b5c7fadb2eac2f7d70546a97998ccd9d9d4d8b7` (HEAD message: "update README.md", 2026-02-11) |
| License | Apache 2.0 (see `./LICENSE`) |
| Vendored on | 2026-05-16 |
| Vendored by | VibeProteinAgent Phase 4.2 |
| Local source tree | `/opt/vpa-data/Protenix/` |
| Local checkpoint dir | `/opt/vpa-data/protenix_ckpt/` |
| Local patches | B200 (Blackwell sm_100) arch list + cached-extension cache-bust per project note `2a82eaba-…` (B200 Port Recipe); already applied to the local tree. |

## What is *not* in this directory

The Protenix Python package, its CCD cache, and the model checkpoint are all
referenced by path; nothing is copied. The adapter resolves them dynamically.

## Environment variables (consumed by the adapter)

| var | default | purpose |
| --- | --- | --- |
| `VPA_PROTENIX_TREE` | `/opt/vpa-data/Protenix` | absolute path to the Protenix source root (must contain `runner/` and `protenix/`) |
| `VPA_PROTENIX_CKPT` | `/opt/vpa-data/protenix_ckpt` | absolute path to the directory holding `protenix_base_default_v1.0.0.pt` and the `common/` CCD cache |

The adapter exports the second value to Protenix via `PROTENIX_ROOT_DIR`
*before* importing the Protenix modules, so Protenix's data-config paths
(`<root>/common/components.cif`, `<root>/common/clusters-by-entity-40.txt`,
etc.) resolve correctly without copying anything.

## Relocating

To run on a different machine:

1. Point `VPA_PROTENIX_TREE` at a Protenix source checkout (apply the B200
   patches from project note `2a82eaba-…` if running on Blackwell).
2. Point `VPA_PROTENIX_CKPT` at a directory containing the checkpoint and the
   four CCD cache files (`common/components.cif`,
   `common/components.cif.rdkit_mol.pkl`,
   `common/clusters-by-entity-40.txt`, `common/obsolete_release_date.csv`).

No further setup is required from the adapter side.

## License

`./LICENSE` is a verbatim copy of the upstream Protenix Apache 2.0 license.
The adapter code in `src/vibeproteinagent/adapters/protenix_v2/` is licensed
under VibeProteinAgent's first-party PolyForm Noncommercial License 1.0.0;
Apache 2.0 applies to the Protenix tree it consumes.
