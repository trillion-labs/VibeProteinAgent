#!/usr/bin/env bash
set -euo pipefail

PREFIX=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --prefix PATH"
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
    "jax[cuda12]==0.4.29" jaxlib==0.4.29 \
    "optax==0.2.5" "dm-haiku==0.0.13" dm-tree \
    "transformers==4.51.3" msgpack \
    torch==2.7.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

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
