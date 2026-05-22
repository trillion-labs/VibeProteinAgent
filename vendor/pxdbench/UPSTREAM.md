# PXDesignBench (pxdbench) — vendored source

Companion benchmark + utility package for PXDesign. PXDesign imports it at
module-load time (`pxdesign/utils/infer.py:30 → pxdbench.pxd_configs.eval`),
so it must be on `sys.path` before any `import pxdesign` statement.

| field | value |
| --- | --- |
| Upstream URL | https://github.com/bytedance/PXDesignBench |
| Pinned tag | `v0.1.2` |
| Pinned commit | `f6d0d72496c23ca1374df5943754f3454ae8c549` |
| License | Apache 2.0 (see `./LICENSE`) |
| Vendored on | 2026-05-16 |
| Vendored by | VibeProteinAgent Phase 5.1 |
| Layout under here | `pxdbench/` (the Python package, ~3.3 MB) |
| Skipped | `examples/`, `Dockerfile`, `download_tool_weights.sh`, `install.sh`, `setup.py`, `*demo.sh`, `README.md` |

## Patches applied to upstream

None. pxdbench is pure-Python; no custom CUDA, no B200 work needed.

## License

Apache 2.0 (full text in `./LICENSE`, copied verbatim from upstream).
