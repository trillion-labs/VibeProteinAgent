# VPA CUDA Containers

The primary install path is still the micromamba/conda environment created by
`vpa install`. These container definitions are the fallback path for clusters
where Python/CUDA dependency resolution is fragile. The image contains the VPA
CLI/runtime and CUDA 12.8 userspace; mount the repository or a cache volume when
you want to install large sidecars or weights.

## Docker

```bash
docker build -f containers/Dockerfile.cu128 -t vibeproteinagent:cu128 .
docker run --rm --gpus all -e VPA_HOME=/workspace/.vpa -v "$PWD:/workspace" \
  vibeproteinagent:cu128 vpa install preflight --profile full --cuda-profile cu128
```

The host still needs a compatible NVIDIA driver. The image provides the CUDA
12.8 userspace/toolkit and VPA conda env; model weights and large sidecars can
be installed or mounted under `VPA_HOME`. For sidecar installs from a mounted
checkout, pass `--source-root /workspace`.
