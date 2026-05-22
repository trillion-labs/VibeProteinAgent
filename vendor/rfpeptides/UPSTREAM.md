# RFpeptides — local checkout of RFdiffusion `rfpeptides` branch

The Baker lab implements macrocyclic peptide design as a branch of
RFdiffusion (`rfpeptides`) rather than a standalone repo. The branch
adds `inference.cyclic=True` and `inference.cyc_chains` flags that
close the peptide chain head-to-tail.

## Upstream pin

| field | value |
| --- | --- |
| Upstream URL | https://github.com/RosettaCommons/RFdiffusion |
| Branch | `rfpeptides` |
| Pinned commit | `3ecb01f2423e90ed524a70a1bcf02a2a249a64ac` ("Update README.md") |
| Cloned via | `git clone --branch rfpeptides --depth 1 ...` |
| License | BSD (`./LICENSE`) |
| Vendored on | 2026-05-17 |
| Cite | Rettie, Juergens, Adebomi, et al., *Nature* 2025 |

## Required weight checkpoints

| file | size | required for | URL |
| --- | --- | --- | --- |
| `Base_ckpt.pt` | ~462 MiB | monomer macrocycle design | https://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt |
| `Complex_base_ckpt.pt` | ~462 MiB | macrocyclic binder design | https://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt |
| `Complex_Fold_base_ckpt.pt` | ~462 MiB | fold-conditioned binder design | https://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt |

Place weights under `vendor/rfpeptides/weights/` (or override the
location via `$VPA_RFPEPTIDES_WEIGHTS`). The `weights/.gitignore` is
configured so weights are never committed.

To download all canonical RFdiffusion weights at once::

```bash
bash vendor/rfpeptides/scripts/download_models.sh vendor/rfpeptides/weights/
```

Phase 16a's quota (≤500 MiB) ships only `Base_ckpt.pt`, which suffices
for the monomer macrocycle slow test. Binder design (full
`target_pdb` + hotspots flow) requires `Complex_base_ckpt.pt`; users
download it themselves.

## Env vars consumed by the adapter

- `VPA_RFPEPTIDES_TREE` — root of the rfpeptides checkout (must contain
  `rfdiffusion/` and `scripts/run_inference.py`). Defaults to the
  in-tree `vendor/rfpeptides/` directory.
- `VPA_RFPEPTIDES_WEIGHTS` — directory containing
  `{Base_ckpt,Complex_base_ckpt,...}.pt`. Defaults to
  `<VPA_RFPEPTIDES_TREE>/weights/`.

## Known compatibility caveats

1. **DGL + torch ABI gap.** The plain PyPI wheel `dgl==2.1.0` is
   CPU-only, and DGL 2.1.0 ships ``libgraphbolt_pytorch_X.Y.Z.so`` only
   for torch up to 2.2.1. The shared ``vpa`` env currently uses
   `dgl==2.1.0+cu121` from `https://data.dgl.ai/wheels/cu121/repo.html`
   with torch 2.7.1+cu128 on B200. There is no official `cu128` wheel,
   but the `cu121` wheel passes the RFpeptides-required CUDA graph smoke
   (`DGLGraph.to("cuda")`, `all_edges`, and `dgl.ops.edge_softmax`) on
   this host. The adapter works around the GraphBolt ABI gap by stubbing
   ``dgl.graphbolt`` before importing DGL in both the availability probe
   and subprocess. ``rfdiffusion`` itself does not use graphbolt. **If a
   future RFdiffusion update starts using graphbolt features, this
   workaround will need revisiting.**

2. **SE3 Transformer.** Vendored as a sibling tree under
   ``env/SE3Transformer``. The adapter inserts both
   ``<tree>`` and ``<tree>/env/SE3Transformer`` onto the subprocess
   ``PYTHONPATH``.

3. **B200 (sm_100) port recipe.** The codebase has **no custom CUDA
   kernels** (`grep TORCH_CUDA_ARCH_LIST vendor/rfpeptides/` returns
   nothing). Patches from the Protenix B200 port (project note
   `2a82eaba-...`) are not needed. If the CUDA DGL wheel stops working
   for this environment, rebuild DGL from source with CUDA enabled
   against the active torch/CUDA stack instead of patching RFpeptides.

4. **DGLBACKEND.** The DGL pytorch backend is selected by default; no
   `~/.dgl/config.json` setup is required.

## Deliberate-skip notes

- The conda env file (`env/SE3nv.yml`) targets python=3.9 + torch=1.9 +
  cudatoolkit=11.1 — **none** of those values apply on B200. We run
  RFpeptides against the modern `vpa` env (python 3.12, torch
  2.7.1+cu128) using ``pip install --no-deps`` for the small Python
  helpers. The DGL install command that passed on vm-009/B200 was:
  ``python -m pip install --no-deps --force-reinstall 'dgl==2.1.0+cu121' -f https://data.dgl.ai/wheels/cu121/repo.html``.
- The original RFdiffusion `setup.py` is not run. ``scripts/run_inference.py``
  is invoked directly via ``python scripts/run_inference.py`` after
  PYTHONPATH manipulation.
