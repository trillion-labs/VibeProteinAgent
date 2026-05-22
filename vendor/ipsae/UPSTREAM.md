# IPSAE — vendored source

| field | value |
| --- | --- |
| Upstream URL | https://github.com/DunbrackLab/IPSAE |
| Commit | `6174cf9e71cb1bd660cc805856a18c4871a6dec3` |
| License | MIT (see `./LICENSE`) |
| Vendored on | 2026-05-17 |
| Vendored by | VibeProteinAgent Phase 12 |

## What is vendored

- `ipsae_vendor/ipsae.py` — copied verbatim from upstream. Pure-numpy
  re-scoring script that consumes an AF2/AF3/Boltz prediction (PDB or
  mmCIF) plus its PAE matrix (JSON or NPZ) and writes a row-per-chain-pair
  scoreboard plus a per-residue file.
- `LICENSE` — copy of the upstream MIT license.

## Modifications

None. `ipsae.py` is byte-identical to upstream at the pinned commit.

## Notes

- Brief originally said `dwhgit/ipsae`; the canonical Dunbrack Lab repo
  resolved during Phase 1.6 is `DunbrackLab/IPSAE` and is the one we vendor.
- Pure numpy; CPU-only; no weights.
