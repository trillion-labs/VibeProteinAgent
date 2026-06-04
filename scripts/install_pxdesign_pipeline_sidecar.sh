#!/usr/bin/env bash
set -euo pipefail

PREFIX=""
CUDA_PROFILE="cu128"

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
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

"${PIP[@]}" "$TORCH_SPEC" --index-url "$TORCH_INDEX_URL"
"${PIP[@]}" "jax[cuda12]==0.4.29" "jaxlib==0.4.29" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
"${PIP[@]}" \
  "chex==0.1.87" \
  "optax==0.2.5" \
  "dm-haiku==0.0.13" \
  dm-tree \
  "transformers==4.51.3" \
  msgpack \
  git+https://github.com/sokrypton/ColabDesign.git \
  aiosqlite \
  pydantic \
  rich \
  typer \
  jsonschema \
  mcp \
  ninja \
  natsort \
  biotite \
  rdkit \
  optree \
  fair-esm \
  scikit-learn
"${PIP[@]}" --reinstall "jax[cuda12]==0.4.29" "jaxlib==0.4.29" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

"$PREFIX/bin/python" - <<PY
from pathlib import Path
import sys

repo = Path("$REPO_ROOT")
sys.path[:0] = [
    str(repo / "src"),
    str(repo / "vendor" / "pxdesign"),
    str(repo / "vendor" / "pxdbench"),
    str(repo / "vendor" / "protenix-pxd"),
]
import jax  # noqa: F401
import haiku  # noqa: F401
import optax  # noqa: F401
import colabdesign  # noqa: F401
import pxdesign.runner.pipeline  # noqa: F401
print("pxdesign sidecar ready")
PY
