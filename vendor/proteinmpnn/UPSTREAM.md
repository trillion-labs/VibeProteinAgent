# ProteinMPNN — vendored source

| field | value |
| --- | --- |
| Upstream URL | https://github.com/dauparas/ProteinMPNN |
| Commit | `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` |
| Commit date | 2023-06-27 |
| License | MIT (see `./LICENSE`) |
| Vendored on | 2026-05-16 |
| Vendored by | VibeProteinAgent Phase 4.1 |

## What is vendored

- `proteinmpnn_vendor/protein_mpnn_utils.py` — copied verbatim from the upstream root. Contains the `ProteinMPNN` `nn.Module`, `parse_PDB`, `tied_featurize`, `_scores`, `_S_to_seq`, and the structure dataset wrappers. This is the only file imported by the inference path.
- `proteinmpnn_vendor/__init__.py` — thin re-export of the public symbols above; turns the flat upstream layout into a proper Python package.
- `LICENSE` — copy of the upstream MIT license.

The upstream training script (`training/`), Colab notebooks, helper scripts, and pre-bundled `inputs/` examples are **not** vendored; the agent uses its own fixtures.

## Weights

- Default checkpoint: `weights/v_48_020.pt` (≈ 6.4 MiB; sha256 `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd`).
- Source: `<upstream>/vanilla_model_weights/v_48_020.pt` (bundled in the upstream repo).
- The file is **not** tracked in git (> 5 MiB cap). Reproduce on a fresh checkout with:

```bash
git clone --depth=1 https://github.com/dauparas/ProteinMPNN /tmp/proteinmpnn_upstream
cp /tmp/proteinmpnn_upstream/vanilla_model_weights/v_48_020.pt vendor/proteinmpnn/weights/v_48_020.pt
sha256sum vendor/proteinmpnn/weights/v_48_020.pt
# expected: c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd
```

`ProteinMPNNAdapter.healthcheck()` returns `False` until the file is present.

## Modifications

None. `protein_mpnn_utils.py` is byte-identical to upstream at the pinned commit.
