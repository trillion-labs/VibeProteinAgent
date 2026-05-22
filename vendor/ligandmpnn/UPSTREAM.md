# LigandMPNN — vendored source

| field | value |
| --- | --- |
| Upstream URL | https://github.com/dauparas/LigandMPNN |
| Commit | `26ec57ac976ade5379920dbd43c7f97a91cf82de` |
| License | MIT (see `./LICENSE`) |
| Vendored on | 2026-05-17 |
| Vendored by | VibeProteinAgent Phase 12 |

## What is vendored

- `ligandmpnn_vendor/data_utils.py` — copied verbatim from upstream root. Contains `parse_PDB`, `featurize`, `get_score`, `alphabet`, `restype_int_to_str`, etc.
- `ligandmpnn_vendor/model_utils.py` — copied verbatim from upstream root. Contains the `ProteinMPNN` `nn.Module` (variant supporting `protein_mpnn`, `ligand_mpnn`, `soluble_mpnn`, membrane-MPNN model types).
- `ligandmpnn_vendor/openfold/` — minimal openfold helper subtree required by `data_utils.py` (residue constants + rigid-frame utilities).
- `LICENSE` — copy of the upstream MIT license.

The upstream training script, packing utilities (`sc_utils.py`), Colab notebooks, helper scripts, and the bundled `inputs/` examples are **not** vendored; the agent uses its own fixtures. Side-chain packing is therefore not exposed by the adapter — sequence design is what the adapter ships.

## Weights

- Default checkpoint: `weights/ligandmpnn_v_32_020_25.pt` (≈ 10.0 MiB; sha256 `42caaa5cdb380867d2b31c30de4ab53ea89171279b3a0732ddf94bc4d3cb6981`).
- Source: `get_model_params.sh` (downloads from `https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt`).
- The file is **not** tracked in git (over the 5 MiB cap; under the 50 MiB Phase 12 ceiling). Reproduce on a fresh checkout with:

```bash
wget -O vendor/ligandmpnn/weights/ligandmpnn_v_32_020_25.pt \
  https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt
sha256sum vendor/ligandmpnn/weights/ligandmpnn_v_32_020_25.pt
# expected: 42caaa5cdb380867d2b31c30de4ab53ea89171279b3a0732ddf94bc4d3cb6981
```

`LigandMPNNAdapter.healthcheck()` returns `False` until the file is present.

Only one checkpoint is bundled. The adapter's `model_type` parameter accepts
`protein_mpnn` / `ligand_mpnn` / `soluble_mpnn` / `ca_mpnn` for descriptor
parity with the LigandMPNN suite; loading a non-`ligand_mpnn` model requires
the matching checkpoint to be staged under `vendor/ligandmpnn/weights/`.

## Modifications

None. `data_utils.py`, `model_utils.py`, and the `openfold/` subtree are
byte-identical to upstream at the pinned commit.
