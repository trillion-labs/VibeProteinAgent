#!/bin/bash
set -e

# ==============================================================================
# Protein Foundation Models - Public UV Environment Installation Script
# ==============================================================================
# Public release variant: installs all Python dependencies (--full) by default.
# Does NOT install Foldseek, MMseqs2, Foldcomp, or any NVIDIA-internal binaries.
#
# Usage: ./build_public_uv_env.sh [OPTIONS] [INSTALL_ROOT]
#
# Options:
#   --clean         Remove existing .venv and UV cache before building (fresh start)
#   --minimal       Skip optional dependencies (JAX, ColabFold, tmol)
#   --python VER    Python version: 3.11 or 3.12 (default: 3.12)
#   --python-bin PATH
#                   Exact Python interpreter to seed the uv venv. Prefer the
#                   vpa conda Python on clusters without system python-dev.
#   --name NAME     Custom prompt name shown when venv is activated (default: complexa)
#   --root PATH     Specify installation root directory (where .venv will be created)
#   --cuda-profile PROFILE
#                   CUDA stack: auto, cu128, cu126, or cpu (default: VPA_CUDA_PROFILE or cu128)
#   -h, --help      Show this help message
#
# Examples:
#   ./build_public_uv_env.sh                      # Full install (Python 3.12)
#   ./build_public_uv_env.sh --minimal            # Base dependencies only
#   ./build_public_uv_env.sh --python 3.11        # Full install with Python 3.11
#   ./build_public_uv_env.sh --name myenv         # Custom prompt: (myenv)
#   ./build_public_uv_env.sh --root /path/to/dir  # Create .venv in custom directory
# ==============================================================================

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INSTALL_ROOT=""
FULL_INSTALL=true
CLEAN=false
PYTHON_VERSION="3.12"
PYTHON_BIN="${VPA_PROTEINA_COMPLEXA_PYTHON:-}"
VENV_NAME="complexa"
CUDA_PROFILE="${VPA_CUDA_PROFILE:-cu128}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --minimal)
            FULL_INSTALL=false
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            if [[ "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12" ]]; then
                echo "Error: Python version must be 3.11 or 3.12"
                exit 1
            fi
            shift 2
            ;;
        --python-bin)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --name)
            VENV_NAME="$2"
            shift 2
            ;;
        --root)
            INSTALL_ROOT="$2"
            shift 2
            ;;
        --cuda-profile)
            CUDA_PROFILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./build_public_uv_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --clean         Remove existing .venv and UV cache before building"
            echo "  --minimal       Skip optional dependencies (JAX, ColabFold, tmol)"
            echo "  --python VER    Python version: 3.11 or 3.12 (default: 3.12)"
            echo "  --python-bin PATH"
            echo "                  Exact Python interpreter to seed the uv venv"
            echo "  --name NAME     Custom prompt name (default: complexa)"
            echo "  --root PATH     Specify where to create .venv (default: project dir)"
            echo "  --cuda-profile PROFILE"
            echo "                  CUDA stack: auto, cu128, cu126, or cpu"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./build_public_uv_env.sh                # Full install (Python 3.12)"
            echo "  ./build_public_uv_env.sh --minimal      # Base dependencies only"
            echo "  ./build_public_uv_env.sh --python 3.11  # Full install with Python 3.11"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
        *)
            if [[ -z "$INSTALL_ROOT" ]]; then
                INSTALL_ROOT="$1"
            else
                echo "Error: Multiple paths specified"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -n "$INSTALL_ROOT" ]]; then
    VENV_DIR="$INSTALL_ROOT"
    mkdir -p "$VENV_DIR"
else
    VENV_DIR="$PROJECT_DIR"
fi

cd "$VENV_DIR"

export UV_CACHE_DIR="$VENV_DIR/.uv-cache"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"

if [[ "$CLEAN" == "true" ]]; then
    echo "Cleaning existing environment and cache..."
    rm -rf "$VENV_DIR/.venv"
    rm -rf "$UV_CACHE_DIR"
    uv cache clean 2>/dev/null || true
    echo "Clean complete."
fi

echo "=============================================="
echo "  Proteina-Complexa - Public Environment Builder"
echo "=============================================="
echo "Project directory: $PROJECT_DIR"
echo "Install directory: $VENV_DIR"
echo "Python version: $PYTHON_VERSION"
if [[ -n "$PYTHON_BIN" ]]; then
    echo "Python binary: $PYTHON_BIN"
fi
echo "Prompt name: $VENV_NAME"
echo "Full install: $FULL_INSTALL"
echo "CUDA profile: $CUDA_PROFILE"
echo ""

if [[ "$CUDA_PROFILE" == "auto" ]]; then
    CUDA_PROFILE="${VPA_CUDA_TAG:-cu128}"
fi
TORCH_SPEC="${VPA_TORCH_SPEC:-torch==2.7.1+${CUDA_PROFILE}}"
TORCH_INDEX_URL="${VPA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/${CUDA_PROFILE}}"
PYG_WHEEL_URL="${VPA_PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.7.0+${CUDA_PROFILE}.html}"
JAX_CUDA_EXTRA="${VPA_JAX_CUDA_EXTRA:-jax[cuda12]}"
if [[ "$CUDA_PROFILE" == "cpu" ]]; then
    TORCH_SPEC="${VPA_TORCH_SPEC:-torch==2.7.1}"
    TORCH_INDEX_URL="${VPA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
    PYG_WHEEL_URL="${VPA_PYG_WHEEL_URL:-https://data.pyg.org/whl/torch-2.7.0+cpu.html}"
    JAX_CUDA_EXTRA="jax"
fi

# ------------------------------------------------------------------------------
# 1. Check/Install UV
# ------------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "[1/8] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/8] uv already installed: $(uv --version)"
fi

# ------------------------------------------------------------------------------
# 2. Create virtual environment
# ------------------------------------------------------------------------------
echo "[2/8] Creating virtual environment with Python $PYTHON_VERSION..."
if [[ -x "$VENV_DIR/.venv/bin/python" ]]; then
    echo "      Reusing existing virtual environment at $VENV_DIR/.venv"
elif [[ -n "$PYTHON_BIN" ]]; then
    uv venv --python "$PYTHON_BIN" --prompt "$VENV_NAME" "$VENV_DIR/.venv"
else
    uv venv --python "$PYTHON_VERSION" --prompt "$VENV_NAME" "$VENV_DIR/.venv"
fi

source "$VENV_DIR/.venv/bin/activate"
echo "      Python: $(which python)"

# ------------------------------------------------------------------------------
# 3. Install PyTorch for the selected CUDA profile
# ------------------------------------------------------------------------------
echo "[3/8] Installing PyTorch for CUDA profile $CUDA_PROFILE..."
uv pip install --index-strategy unsafe-best-match \
    "$TORCH_SPEC" torchvision torchaudio \
    --index-url "$TORCH_INDEX_URL"
if [[ "$CUDA_PROFILE" != "cpu" ]]; then
    # Triton 3.3.1, pulled by torch 2.7.1 wheels, can fail on Python 3.12 at
    # driver.active.utils.load_binary with:
    #   SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
    # RF3/cuequivariance evaluation reaches this path. Pin the patched Triton
    # release and verify it below with an actual kernel launch.
    uv pip install "triton==3.4.0"
fi

# ------------------------------------------------------------------------------
# 4. Install base dependencies from pyproject.toml
# ------------------------------------------------------------------------------
echo "[4/8] Installing base dependencies from pyproject.toml..."
uv pip install --index-strategy unsafe-best-match -e "$PROJECT_DIR"
# The VPA persistent sidecar imports VPA source via PYTHONPATH, not by
# installing VPA into this island env. Keep the one sidecar-only dependency
# explicit so `availability_skip_reason()` is reproducible after a clean build.
# The B200-compatible JAX stack resolves to numpy 2.x; this is the stack
# validated by VPA's real Proteina-Complexa smoke tests.
uv pip install "numpy==2.4.5" "aiosqlite>=0.20" "toolz>=1.0"

# ------------------------------------------------------------------------------
# 5. Install PyTorch Geometric packages
# ------------------------------------------------------------------------------
echo "[5/7] Installing PyTorch Geometric packages..."
uv pip install torch_geometric torch_sparse torch_cluster \
    -f "$PYG_WHEEL_URL"
uv pip install torch-scatter \
    -f "$PYG_WHEEL_URL"

# ------------------------------------------------------------------------------
# 6. Install Graphein and Atomworks
# ------------------------------------------------------------------------------
echo "[6/8] Installing Graphein and Atomworks..."
echo "      -> Graphein..."
uv pip install graphein==1.7.7 --no-deps

echo "      -> Atomworks..."
uv pip install "atomworks[ml,openbabel,dev]" || echo "Warning: atomworks install failed"

# ------------------------------------------------------------------------------
# 7. Install optional/full dependencies
# ------------------------------------------------------------------------------
if [ "$FULL_INSTALL" = true ]; then
    echo "[7/8] Installing full dependencies (ColabFold, JAX, tmol)..."

    echo "      -> ColabDesign & AlphaFold-ColabFold..."
    uv pip install colabdesign==1.1.1 alphafold-colabfold==2.3.7

    echo "      -> Installing local colabdesign (community_models/colabdesign)..."
    uv pip install -e "$PROJECT_DIR/community_models/colabdesign"

    echo "      -> JAX with CUDA..."
    uv pip install "${JAX_CUDA_EXTRA}==0.10.0" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    if [[ "$CUDA_PROFILE" != "cpu" ]]; then
        # JAX CUDA 12 wheels require cuDNN >=9.8. PyTorch cu128 can leave an
        # older cuDNN dependency in the env, which imports but fails at XLA
        # compile time with `dnn_support != nullptr`.
        uv pip install "nvidia-cudnn-cu12>=9.8,<10" --upgrade
    fi
    uv pip install flax==0.9.0 --no-deps
    uv pip install dm-haiku==0.0.16 --no-deps
    uv pip install optax==0.2.2 --no-deps
    uv pip install chex==0.1.86 --no-deps

    echo "      -> Tmol..."
    uv pip install "git+https://github.com/uw-ipd/tmol.git@d8a6f7f9649d36e74440bca25246ee7c467ce490" || echo "Warning: tmol install failed"
else
    echo "[7/8] Skipping optional dependencies (omit --minimal to install)"
fi

# ------------------------------------------------------------------------------
# 8. Install Foundry (Python version dependent)
# ------------------------------------------------------------------------------
if [[ "$PYTHON_VERSION" == "3.12" ]]; then
    echo "[8/8] Installing Foundry (rc-foundry)..."
    uv pip install "rc-foundry[all]"
else
    echo "[8/8] Not able to install RF# via rc-foundary. Please try python 3.12."
fi

RFD3_ENV="${VPA_RFDIFFUSION3_ENV:-$PROJECT_DIR/../rfdiffusion3/.venv}"
if [ -x "$RFD3_ENV/bin/rf3" ] && [ ! -x "$VENV_DIR/.venv/bin/rf3" ]; then
    ln -s "$RFD3_ENV/bin/rf3" "$VENV_DIR/.venv/bin/rf3"
    echo "Linked RF3 executable from $RFD3_ENV for Proteina-Complexa evaluation"
fi

uv pip install biotite==1.6.0
echo "Updated biotite to 1.6.0 for ligand compatibility"

if [[ "$FULL_INSTALL" == "true" && "$CUDA_PROFILE" != "cpu" ]]; then
    echo "[verify] Checking JAX GPU DNN backend..."
    NVIDIA_LIB_PATHS="$(python - <<'PY'
from pathlib import Path
import site

priority = {
    "cudnn": 0,
    "cublas": 1,
    "cuda_runtime": 2,
    "cuda_nvrtc": 3,
}
paths = []
for base in site.getsitepackages():
    root = Path(base) / "nvidia"
    if root.is_dir():
        paths.extend(path for path in root.glob("*/lib") if path.is_dir())
paths.sort(key=lambda path: (priority.get(path.parent.name, 100), path.parent.name))
print(":".join(str(path) for path in paths))
PY
)"
    if [[ -n "$NVIDIA_LIB_PATHS" ]]; then
        export LD_LIBRARY_PATH="$NVIDIA_LIB_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    JAX_TRACEBACK_FILTERING=off python - <<'PY'
import jax
import jax.numpy as jnp
from jax import lax

gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
if not gpu_devices:
    raise RuntimeError("no JAX GPU devices are visible")

device = gpu_devices[0]
x = jax.device_put(jnp.ones((1, 16, 16, 1), dtype=jnp.float32), device)
w = jax.device_put(jnp.ones((3, 3, 1, 4), dtype=jnp.float32), device)
y = lax.conv_general_dilated(
    x,
    w,
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)
float(jnp.sum(y).block_until_ready())
print("JAX GPU DNN backend: ok")
PY

    echo "[verify] Checking Triton CUDA load_binary backend..."
    TRITON_PROBE="$(mktemp "${TMPDIR:-/tmp}/complexa_triton_probe_XXXX.py")"
    cat > "$TRITON_PROBE" <<'PY'
import torch
import triton
import triton.language as tl

@triton.jit
def kernel(x, y, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    values = tl.load(x + offsets, mask=mask, other=0.0)
    tl.store(y + offsets, values + 1.0, mask=mask)

x = torch.ones((16,), device="cuda", dtype=torch.float32)
y = torch.empty_like(x)
kernel[(1,)](x, y, x.numel(), BLOCK=16)
torch.cuda.synchronize()
assert float(y[0].item()) == 2.0
print("Triton CUDA load_binary backend: ok")
PY
    python "$TRITON_PROBE"
    rm -f "$TRITON_PROBE"

    echo "[verify] Checking Proteina-Complexa RF3 evaluate smoke..."
    export RF3_CKPT_PATH="${RF3_CKPT_PATH:-$PROJECT_DIR/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt}"
    export RF3_EXEC_PATH="${RF3_EXEC_PATH:-$VENV_DIR/.venv/bin/rf3}"
    python - <<'PY'
import os
import tempfile
from pathlib import Path


def atom_line(serial, atom, res, chain, resseq, x, y, z, element):
    return (
        f"ATOM  {serial:5d} {atom:<4s} {res:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def het_line(serial, atom, res, chain, resseq, x, y, z, element):
    return (
        f"HETATM{serial:5d} {atom:<4s} {res:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2s}\n"
    )


def write_complex(path):
    serial = 1
    lines = []
    lines.append(het_line(serial, "C1", "LIG", "A", 1, 0.0, 0.0, 0.0, "C"))
    serial += 1
    for resseq in range(1, 17):
        z = (resseq - 1) * 1.5
        atoms = (
            ("N", 4.0, 0.0, z, "N"),
            ("CA", 4.5, 0.8, z + 0.3, "C"),
            ("C", 5.2, 0.1, z + 0.8, "C"),
            ("O", 5.2, -0.8, z + 1.0, "O"),
            ("CB", 4.4, 1.8, z + 0.2, "C"),
        )
        for atom, x, y, zz, element in atoms:
            lines.append(atom_line(serial, atom, "ALA", "B", resseq, x, y, zz, element))
            serial += 1
    lines.append("TER\n")
    lines.append("END\n")
    path.write_text("".join(lines))


with tempfile.TemporaryDirectory(prefix="complexa_eval_smoke_") as tmp:
    root = Path(tmp)
    sample_dir = root / "sample"
    sample_dir.mkdir()
    sample_pdb = sample_dir / "sample.pdb"
    target_pdb = root / "target_ligand.pdb"
    write_complex(sample_pdb)
    target_pdb.write_text(
        "".join(
            line
            for line in sample_pdb.read_text().splitlines(True)
            if (line.startswith("HETATM") and line[21] == "A")
            or line.startswith("END")
        )
    )

    ckpt_path = os.environ.get("RF3_CKPT_PATH")
    rf3_path = os.environ.get("RF3_EXEC_PATH")
    if not ckpt_path or not Path(ckpt_path).is_file():
        raise RuntimeError(f"RF3_CKPT_PATH is not a file: {ckpt_path}")
    if not rf3_path or not Path(rf3_path).is_file():
        raise RuntimeError(f"RF3_EXEC_PATH is not a file: {rf3_path}")

    from proteinfoundation.metrics.binder_metrics import run_binder_eval
    from proteinfoundation.rewards.rf3_reward import get_default_rf3_runner

    runner = get_default_rf3_runner(
        ckpt_path=ckpt_path,
        rf3_path=rf3_path,
        dump_dir=str(root / "rf3"),
    )
    _, _, sequence_type_stats, _ = run_binder_eval(
        pdb_file_path=str(sample_pdb),
        target_pdb_path=str(target_pdb),
        folding_model_specs={"model_name": "RF3", "runner": runner},
        tmp_path=str(sample_dir),
        target_pdb_chain=["A"],
        sequence_types=["self"],
        inverse_folding_model="ligand_mpnn",
        gen_target_chain=["A"],
        binder_chain="B",
        interface_cutoff=8.0,
        is_target_ligand=True,
        num_redesign_seqs=1,
        smiles="C",
    )
    stats = sequence_type_stats["self"]["complex_stats"][0]
    if stats.get("rf3_status") != "completed":
        raise RuntimeError(f"RF3 evaluate smoke did not complete: {stats}")
    complex_pdb_path = stats.get("complex_pdb_path")
    if not complex_pdb_path or not Path(complex_pdb_path).is_file():
        raise RuntimeError(f"RF3 evaluate smoke did not write complex PDB: {stats}")
    rmsd = sequence_type_stats["self"]["rmsd_stats"][0]
    if "binder_scRMSD_ca" not in rmsd and "binder_scRMSD" not in rmsd:
        raise RuntimeError(f"RF3 evaluate smoke did not compute RMSD: {rmsd}")
    if "ligand_scRMSD" not in rmsd:
        raise RuntimeError(f"RF3 ligand evaluate smoke did not compute ligand RMSD: {rmsd}")
print("Proteina-Complexa RF3 evaluate smoke: ok")
PY
fi

# ------------------------------------------------------------------------------
# Done!
# ------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Installation Complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/.venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c \"import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""
echo ""
echo "Note: Foldseek and MMseqs2 are not included in the public build."
echo "Install them separately if needed:"
echo "  wget https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz"
echo "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
