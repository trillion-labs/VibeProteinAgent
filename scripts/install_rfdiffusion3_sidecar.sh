#!/usr/bin/env bash
set -euo pipefail

PREFIX=""
REPO_ROOT=""
CUDA_PROFILE="cu128"

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
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PREFIX" ]]; then
  echo "--prefix is required" >&2
  exit 2
fi
if [[ -z "$REPO_ROOT" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

PYTHON_BIN="${PYTHON:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON:-python3}"
fi

case "$CUDA_PROFILE" in
  cu128)
    TORCH_SPEC="torch==2.7.1+cu128"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
    ;;
  cu126)
    TORCH_SPEC="torch==2.7.1+cu126"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
    ;;
  cpu)
    TORCH_SPEC="torch==2.7.1"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "unsupported cuda profile: $CUDA_PROFILE" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$PREFIX")"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if command -v uv >/dev/null 2>&1; then
  uv venv --clear --python "$PYTHON_BIN" "$PREFIX"
  PIP=(uv pip install --python "$PREFIX/bin/python")
else
  "$PYTHON_BIN" -m venv --clear "$PREFIX"
  "$PREFIX/bin/python" -m pip install --upgrade pip
  PIP=("$PREFIX/bin/python" -m pip install)
fi

"${PIP[@]}" "$TORCH_SPEC" torchvision torchaudio --index-url "$TORCH_INDEX_URL"
"${PIP[@]}" "rc-foundry[rfd3]"

if [[ -x "$PREFIX/bin/rfd3" ]]; then
  "$PREFIX/bin/rfd3" --help >/dev/null 2>&1 || true
elif [[ -x "$PREFIX/bin/foundry" ]]; then
  DEBUG=false "$PREFIX/bin/foundry" --help >/dev/null 2>&1 || true
else
  echo "RFdiffusion3 sidecar did not install rfd3 or foundry executable" >&2
  exit 1
fi
