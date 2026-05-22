# PRODIGY — vendored source

| field | value |
| --- | --- |
| Upstream URL | https://github.com/haddocking/prodigy |
| Commit | `6c459597af54329b65b039ad2458c0976e09c43c` |
| License | Apache 2.0 (see `./LICENSE`) |
| Vendored on | 2026-05-17 |
| Vendored by | VibeProteinAgent Phase 12 |
| Upstream package version | `prodigy-prot==2.4.0` |

## What is vendored

- `prodigy_prot/` — copied verbatim from `<upstream>/src/prodigy_prot/`.
  Pure-Python knowledge-based binding-affinity (ΔG / Kd) regressor with no
  trained weights. Dependencies: biopython, numpy, freesasa — all present in
  the `vpa` env after Phase 1.7 env augmentation.
- `LICENSE` — copy of the upstream Apache-2.0 license.

## Modifications

None. The `prodigy_prot/` subtree is byte-identical to upstream at the
pinned commit.
