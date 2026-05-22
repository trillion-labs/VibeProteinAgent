#!/usr/bin/env bash
set -euo pipefail

PREFIX=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_PROFILE="${VPA_CUDA_PROFILE:-cu128}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --repo-root)
            REPO_ROOT="$2"
            shift 2
            ;;
        --cuda-profile)
            CUDA_PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --prefix PATH [--repo-root PATH] [--cuda-profile auto|cu128|cu126|cpu]"
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
if [[ "$CUDA_PROFILE" == "cpu" ]]; then
    TORCH_SPEC="${VPA_TORCH_SPEC:-torch==2.7.1}"
    TORCH_INDEX_URL="${VPA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
fi

if [[ -x "$PREFIX/bin/rfd3" ]]; then
    echo "RFdiffusion3 sidecar already present at $PREFIX"
    exit 0
fi

for candidate in \
    "${VPA_RFDIFFUSION3_ENV:-}" \
    "$REPO_ROOT/vendor/rfdiffusion3/.venv" \
    "$HOME/.conda/envs/rfdiffusion3-vpa" \
    "/opt/miniconda3/envs/rfdiffusion3-vpa" \
    "/opt/conda/envs/rfdiffusion3-vpa"
do
    if [[ -n "$candidate" && -x "$candidate/bin/rfd3" ]]; then
        mkdir -p "$(dirname "$PREFIX")"
        ln -sfn "$candidate" "$PREFIX"
        echo "Linked existing RFdiffusion3 sidecar: $PREFIX -> $candidate"
        exit 0
    fi
done

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to create $PREFIX from scratch" >&2
    exit 2
fi

PYTHON_BIN="${VPA_PYTHON:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="${HOME}/.vibeproteinagent/envs/vpa/bin/python"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Could not locate a Python interpreter to seed RFdiffusion3 sidecar" >&2
    exit 2
fi

uv venv --python "$PYTHON_BIN" "$PREFIX"
"$PREFIX/bin/python" -m pip install --no-user \
    "$TORCH_SPEC" torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL"
"$PREFIX/bin/python" -m pip install --no-user 'rc-foundry[rfd3]'
"$PREFIX/bin/rfd3" --help >/dev/null
echo "RFdiffusion3 sidecar installed at $PREFIX"
