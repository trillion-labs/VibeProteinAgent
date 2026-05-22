#!/usr/bin/env bash
set -euo pipefail

PREFIX=""
CUDA_PROFILE="${VPA_CUDA_PROFILE:-cu128}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --cuda-profile)
            CUDA_PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --prefix PATH [--cuda-profile auto|cu128|cu126|cpu]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "$PREFIX" ]]; then
    echo "--prefix is required" >&2
    exit 2
fi

if [[ "$CUDA_PROFILE" == "auto" ]]; then
    CUDA_PROFILE="${VPA_CUDA_TAG:-cu128}"
fi
TORCH_SPEC="${VPA_TORCH_SPEC:-torch==2.7.1+${CUDA_PROFILE}}"
TORCH_INDEX_URL="${VPA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_PROFILE}}"
JAX_CUDA_EXTRA="${VPA_JAX_CUDA_EXTRA:-jax[cuda12]}"
if [[ "$CUDA_PROFILE" == "cpu" ]]; then
    TORCH_SPEC="${VPA_TORCH_SPEC:-torch==2.7.1}"
    TORCH_INDEX_URL="${VPA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    JAX_CUDA_EXTRA="jax"
fi

if [[ -x "$PREFIX/bin/python" ]]; then
    echo "PXDesign pipeline sidecar already present at $PREFIX"
    exit 0
fi

for candidate in \
    "${VPA_PXDESIGN_PIPELINE_ENV:-}" \
    "$HOME/.conda/envs/pxdesign-pipeline-vpa" \
    "/opt/miniconda3/envs/pxdesign-pipeline-vpa" \
    "/opt/conda/envs/pxdesign-pipeline-vpa"
do
    if [[ -n "$candidate" && -x "$candidate/bin/python" ]]; then
        mkdir -p "$(dirname "$PREFIX")"
        ln -sfn "$candidate" "$PREFIX"
        echo "Linked existing PXDesign pipeline sidecar: $PREFIX -> $candidate"
        exit 0
    fi
done

if [[ -n "${VPA_MICROMAMBA:-}" && -x "${VPA_MICROMAMBA:-}" ]]; then
    MAMBA="$VPA_MICROMAMBA"
elif command -v micromamba >/dev/null 2>&1; then
    MAMBA="$(command -v micromamba)"
else
    echo "micromamba is required to create $PREFIX from scratch" >&2
    exit 2
fi

"$MAMBA" create -y -p "$PREFIX" -c conda-forge python=3.11 pip
PY="$PREFIX/bin/python"

"$PY" -m pip install --no-user \
    "${JAX_CUDA_EXTRA}==0.4.29" jaxlib==0.4.29 \
    "optax==0.2.5" "dm-haiku==0.0.13" dm-tree \
    "transformers==4.51.3" msgpack \
    "$TORCH_SPEC" \
    --extra-index-url "$TORCH_INDEX_URL"

"$PY" -m pip install --no-user \
    git+https://github.com/sokrypton/ColabDesign.git \
    aiosqlite pydantic rich typer jsonschema mcp ninja \
    biotite rdkit optree fair-esm scikit-learn

"$PY" - <<'PY'
import importlib

for name in ("msgpack", "jax", "haiku", "optax", "colabdesign"):
    importlib.import_module(name)
print("PXDesign pipeline sidecar import probe ok")
PY
