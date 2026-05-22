# Sharded GPU Smoke

This directory contains a tiny, hand-authored agent plan for the sharded GPU
execution contract. It keeps heavy adapters single-GPU, fans out independent
seed/sample jobs with `foreach`, caps each fan-out with `max_parallel: 2`, and
then renders a report.

Run from the repository root after full-profile sidecars are installed:

```bash
bash validation/sharded_gpu_smoke/run_local.sh
```

Useful overrides:

```bash
VPA_SMOKE_GPU_WORKERS=2 VPA_SMOKE_CPU_WORKERS=2 \
VPA_SMOKE_WORKSPACE=sharded_gpu_smoke \
bash validation/sharded_gpu_smoke/run_local.sh
```

Outputs are written under `validation/sharded_gpu_smoke/out/`, including the
submit JSON, rendered `report.html`, and `gpu_ids.json` summarizing shard
placement.
