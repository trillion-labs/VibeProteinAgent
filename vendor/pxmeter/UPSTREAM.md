# PXMeter — vendored source

| field | value |
| --- | --- |
| Upstream URL | https://github.com/bytedance/PXMeter |
| Commit | `f777e780427a01f969b007071b905480a55b796a` |
| License | Apache 2.0 (see `./LICENSE`) |
| Vendored on | 2026-05-17 |
| Vendored by | VibeProteinAgent Phase 14 |

## What is vendored

- `pxmeter/` — the upstream package, byte-identical to commit `f777e78` minus
  `pxmeter/test/` (unit-test fixtures, ~MB). All absolute imports of the form
  `from pxmeter.X.Y import …` resolve via `sys.path.insert(0, vendor/pxmeter/)`.
- `examples/7rss.cif` and `examples/7rss_protenix_pred.cif` — bundled
  reference + Protenix-predicted CIF used by the adapter's slow regression
  test. Reproduces the Phase 1.7 verified call (`lddt=0.947`, DockQ=0.84–0.92).
- `LICENSE` — copy of the upstream Apache 2.0 license.

## Modifications

None. Pure vendor of upstream sources. The `pxmeter/test/` directory is
omitted because it pulls multiple MB of binary fixtures that we don't need.

## Runtime dependencies

PXMeter resolves at import-time once these conda/pip packages are present:

| package | source |
| --- | --- |
| `rdkit` 2025.09.6 | conda-forge / pkgs/main |
| `gemmi` 0.7.5 | conda-forge |
| `anarcii` 2.0.6 | pip --no-deps |
| `pdbeccdutils` 1.0.3 | conda-forge |
| `tabulate` 0.10.0 | conda-forge |
| `biotite` ≥ 1.2 | already in vpa |

The conda `rdkit` package from `pkgs/main` **ships without a `dist-info/METADATA`
file**, which causes `biotite.interface.rdkit.@requires_version("rdkit", …)`
to raise a misleading `ImportError`. A one-file shim at

```
$CONDA_PREFIX/lib/python3.12/site-packages/rdkit-2025.09.6.dist-info/METADATA
```

with `Metadata-Version: 2.1 / Name: rdkit / Version: 2025.09.6` fixes the
metadata lookup. The shim is currently present in vpa. **Any future
`conda update rdkit` silently removes it** — re-create it from this stanza
if PXMeter import starts failing with the rdkit error.

## Notes

- The adapter calls `pxmeter.cli.run_eval_cif(ref_cif, model_cif, output_json)`
  — the verbatim Phase 1.7 entry point. PXMeter writes a single JSON with the
  full per-chain / per-interface metric set; the adapter surfaces the
  top-level `complex.lddt` and per-interface `dockq` values, and returns
  the full JSON path for downstream consumers.
