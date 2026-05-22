# Protenix v0.5.0+pxd — vendored source (B200 patched)

The PXDesign-flavored fork of Protenix. PXDesign hard-depends on this exact
tag's API surface, which differs subtly from upstream Protenix v1.0.4 (the
fork includes pxdesign-specific configs, dumper helpers, and `pxd_server/`).
Vendored under its original package name (`protenix/` + top-level `runner/`)
to preserve internal imports.

| field | value |
| --- | --- |
| Upstream URL | https://github.com/bytedance/Protenix |
| Pinned tag | `v0.5.0+pxd` |
| Pinned commit | `d18aa1daadd02a001b32bc7fa2278fc8a2f8f025` |
| License | Apache 2.0 (see `./LICENSE`) |
| Vendored on | 2026-05-16 |
| Vendored by | VibeProteinAgent Phase 5.1 |
| Layout under here | `protenix/` (the Python package, ~1.9 MB) + `runner/` (top-level runner module, 84 KB) + `configs/` (YAML configs used at runtime, 48 KB) |
| Skipped | `assets/` (27 MB), `scripts/` (25 MB), `examples/` (18 MB), `docs/`, `tests/`, `Protenix_Technical_Report.pdf`, `Dockerfile`, `*demo.sh`, `setup.py`, `requirements.txt`, `README.md` |

## Patches applied to upstream

CUDA arch compatibility patch. Captured as a git-format diff at
`./.patches/01-b200-arch.patch`.

- `protenix/model/layer_norm/torch_ext_compile.py` —
  - Hard-coded `TORCH_CUDA_ARCH_LIST = "7.0;8.0"` replaced with an
    env-driven default covering common VPA targets: sm_86, sm_89, sm_90,
    and sm_100.
  - `-gencode` flags are derived from `VPA_PROTENIX_CUDA_ARCH_LIST` or
    `TORCH_CUDA_ARCH_LIST`, so older CUDA toolkits can opt out of unsupported
    architectures.
- `protenix/model/layer_norm/layer_norm.py` —
  - The cached extension name includes a sanitized arch suffix so switching
    CUDA arch profiles does not reuse a stale compiled extension.

This is the same recipe the user pre-applied to the v1.0.4 Protenix tree at
`/opt/vpa-data/Protenix/` (consumed by the Phase 4.2 adapter).

## Environment variables (consumed by Phase 5.2 adapter)

The on-disk Protenix v1.0.4 stays at `/opt/vpa-data/Protenix/`
and is consumed by the **`protenix_v2`** adapter via `VPA_PROTENIX_TREE`.
The vendored Protenix v0.5.0+pxd here is a *different* tree at a different
tag, and is consumed only by the **`pxdesign`** adapter. They must never
both be on `sys.path` in the same process — pick one per worker.

| var | default | purpose |
| --- | --- | --- |
| `VPA_PROTENIX_PXD_VENDOR` | `<repo>/vendor/protenix-pxd` | path to this vendored copy |
| `PROTENIX_DATA_ROOT_DIR` | `<some-ckpt-dir>/common` (CCD cache) | must be exported before importing `protenix` modules (Protenix's `configs/configs_data.py` reads it at module top) |

## License

Apache 2.0 (full text in `./LICENSE`, copied verbatim from upstream).
