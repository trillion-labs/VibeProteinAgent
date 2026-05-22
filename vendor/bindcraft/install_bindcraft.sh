#!/bin/bash
################## BindCraft installation script
################## specify conda/mamba folder, and installation folder for git repositories, and whether to use mamba or $pkg_manager
# Default value for pkg_manager
pkg_manager='conda'
cuda="${VPA_BINDCRAFT_CUDA:-${VPA_CUDA_VERSION:-}}"
env_prefix=''

# Define the short and long options
OPTIONS=p:c:e:
LONGOPTIONS=pkg_manager:,cuda:,env-prefix:

# Parse the command-line options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
eval set -- "$PARSED"

# Process the command-line options
while true; do
  case "$1" in
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda)
      cuda="$2"
      shift 2
      ;;
    -e|--env-prefix)
      env_prefix="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo -e "Invalid option $1" >&2
      exit 1
      ;;
  esac
done

# Example usage of the parsed variables
echo -e "Package manager: $pkg_manager"
echo -e "CUDA: $cuda"
if [ -n "$env_prefix" ]; then
  echo -e "Environment prefix: $env_prefix"
fi

############################################################################################################
############################################################################################################
################## initialisation
SECONDS=0

# set paths needed for installation and check for package manager installation
install_dir=$(pwd)
if [ "$pkg_manager" = "micromamba" ]; then
  if ! command -v micromamba >/dev/null 2>&1; then
    if [ -n "${VPA_MICROMAMBA:-}" ] && [ -x "$VPA_MICROMAMBA" ]; then
      export PATH="$(dirname "$VPA_MICROMAMBA"):$PATH"
    fi
  fi
  MAMBA_EXE=$(command -v micromamba 2>/dev/null) || { echo -e "Error: micromamba is not installed or cannot be found."; exit 1; }
  eval "$("$MAMBA_EXE" shell hook -s bash)" || { echo -e "Error: failed to initialise micromamba shell hook."; exit 1; }
  CONDA_BASE=$("$MAMBA_EXE" info --base 2>/dev/null || true)
  echo -e "micromamba is installed at: $MAMBA_EXE"
else
  CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
  echo -e "Conda is installed at: $CONDA_BASE"
fi

### BindCraft install begin, create base environment
echo -e "Installing BindCraft environment\n"
if [ -n "$env_prefix" ]; then
  ENV_PREFIX="$env_prefix"
  if [ -x "$ENV_PREFIX/bin/python" ]; then
    echo -e "BindCraft environment already exists at ${ENV_PREFIX}; reusing it"
  else
    mkdir -p "$(dirname "$ENV_PREFIX")" || { echo -e "Error: Failed to create sidecar parent directory"; exit 1; }
    $pkg_manager create --prefix "$ENV_PREFIX" python=3.10 -y || { echo -e "Error: Failed to create BindCraft environment"; exit 1; }
  fi
else
  if conda env list | awk '{print $1}' | grep -w 'BindCraft' >/dev/null 2>&1; then
    echo -e "BindCraft conda environment already exists; reusing it"
  else
    $pkg_manager create --name BindCraft python=3.10 -y || { echo -e "Error: Failed to create BindCraft conda environment"; exit 1; }
  fi
  conda env list | grep -w 'BindCraft' >/dev/null 2>&1 || { echo -e "Error: Conda environment 'BindCraft' does not exist after creation."; exit 1; }
  ENV_PREFIX=$(conda env list | awk '$1 == "BindCraft" {print $NF; exit}')
  [ -n "$ENV_PREFIX" ] || { echo -e "Error: Could not resolve BindCraft environment prefix."; exit 1; }
fi
[ -x "$ENV_PREFIX/bin/python" ] || { echo -e "Error: BindCraft environment python is missing at ${ENV_PREFIX}/bin/python"; exit 1; }

# Load newly created BindCraft environment
echo -e "Loading BindCraft environment\n"
if [ "$pkg_manager" = "micromamba" ]; then
  micromamba activate "${ENV_PREFIX}" || { echo -e "Error: Failed to activate the BindCraft environment."; exit 1; }
else
  source ${CONDA_BASE}/bin/activate ${ENV_PREFIX} || { echo -e "Error: Failed to activate the BindCraft environment."; exit 1; }
fi
if [ -n "$env_prefix" ]; then
  [ "$CONDA_PREFIX" = "$ENV_PREFIX" ] || { echo -e "Error: The BindCraft environment is not active."; exit 1; }
else
  [ "$CONDA_DEFAULT_ENV" = "BindCraft" ] || { echo -e "Error: The BindCraft environment is not active."; exit 1; }
fi
echo -e "BindCraft environment activated at ${ENV_PREFIX}"

# install required conda packages
echo -e "Instaling conda requirements\n"
if [ -n "$cuda" ]; then
  CONDA_OVERRIDE_CUDA="$cuda" $pkg_manager install \
    pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0=*cuda*' cuda-nvcc cudnn \
    -c conda-forge -c nvidia -y \
  || { echo -e "Error: Failed to install conda packages."; exit 1; }
else
  $pkg_manager install \
    pip pandas matplotlib 'numpy<2.0.0' biopython scipy pdbfixer seaborn libgfortran5 tqdm jupyter ffmpeg fsspec py3dmol \
    chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
    'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0' \
    -c conda-forge -c nvidia -y \
  || { echo -e "Error: Failed to install conda packages."; exit 1; }
fi

# make sure all required packages were installed
required_packages=(pip pandas libgfortran5 matplotlib numpy biopython scipy pdbfixer seaborn tqdm jupyter ffmpeg fsspec py3dmol chex dm-haiku dm-tree joblib ml-collections immutabledict optax jaxlib jax cuda-nvcc cudnn)
missing_packages=()

# Check each package
for pkg in "${required_packages[@]}"; do
    $pkg_manager list "$pkg" | grep -w "$pkg" >/dev/null 2>&1 || missing_packages+=("$pkg")
done

# If any packages are missing, output error and exit
if [ ${#missing_packages[@]} -ne 0 ]; then
    echo -e "Error: The following packages are missing from the environment:"
    for pkg in "${missing_packages[@]}"; do
        echo -e " - $pkg"
    done
    exit 1
fi

# install PyTorch (CPU wheel is sufficient for BindCraft's torch import surface;
# the actual design model execution uses JAX/CUDA in this sidecar).
echo -e "Installing PyTorch\n"
python -c "import torch" >/dev/null 2>&1 || \
  pip install torch --index-url https://download.pytorch.org/whl/cpu || { echo -e "Error: Failed to install PyTorch"; exit 1; }
python -c "import torch" >/dev/null 2>&1 || { echo -e "Error: torch module not found after installation"; exit 1; }

# install VPA sidecar runtime dependencies imported before BindCraft starts.
echo -e "Installing VPA sidecar runtime dependencies\n"
python -c "import pydantic" >/dev/null 2>&1 || \
  pip install 'pydantic>=2,<3' || { echo -e "Error: Failed to install pydantic"; exit 1; }
python -c "import pydantic" >/dev/null 2>&1 || { echo -e "Error: pydantic module not found after installation"; exit 1; }
python -c "import msgpack" >/dev/null 2>&1 || \
  pip install 'msgpack>=1.0' || { echo -e "Error: Failed to install msgpack"; exit 1; }
python -c "import msgpack" >/dev/null 2>&1 || { echo -e "Error: msgpack module not found after installation"; exit 1; }

# install ColabDesign
echo -e "Installing ColabDesign\n"
pip3 install git+https://github.com/sokrypton/ColabDesign.git --no-deps || { echo -e "Error: Failed to install ColabDesign"; exit 1; }
python -c "import colabdesign" >/dev/null 2>&1 || { echo -e "Error: colabdesign module not found after installation"; exit 1; }

# install PyRosetta
echo -e "Installing PyRosetta\n"
pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release.cxx11thread.serialization
python -c "import pyrosetta" >/dev/null 2>&1 || { echo -e "Error: pyrosetta module not found after installation"; exit 1; }

# AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights \n"
params_dir="${install_dir}/params"
params_file="${params_dir}/alphafold_params_2022-12-06.tar"

# download AF2 weights
mkdir -p "${params_dir}" || { echo -e "Error: Failed to create weights directory"; exit 1; }
if [ -f "${params_dir}/params_model_5_ptm.npz" ]; then
  echo -e "AlphaFold2 model weights already extracted; skipping download"
else
  wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { echo -e "Error: Failed to download AlphaFold2 weights"; exit 1; }
  [ -s "${params_file}" ] || { echo -e "Error: Could not locate downloaded AlphaFold2 weights"; exit 1; }

  # extract AF2 weights
  tar tf "${params_file}" >/dev/null 2>&1 || { echo -e "Error: Corrupt AlphaFold2 weights download"; exit 1; }
  tar -xvf "${params_file}" -C "${params_dir}" || { echo -e "Error: Failed to extract AlphaFold2weights"; exit 1; }
  [ -f "${params_dir}/params_model_5_ptm.npz" ] || { echo -e "Error: Could not locate extracted AlphaFold2 weights"; exit 1; }
  rm "${params_file}" || { echo -e "Warning: Failed to remove AlphaFold2 weights archive"; }
fi

# chmod executables
echo -e "Changing permissions for executables\n"
chmod +x "${install_dir}/functions/dssp" || { echo -e "Error: Failed to chmod dssp"; exit 1; }
chmod +x "${install_dir}/functions/DAlphaBall.gcc" || { echo -e "Error: Failed to chmod DAlphaBall.gcc"; exit 1; }

# finish
if [ "$pkg_manager" = "micromamba" ]; then
  micromamba deactivate || true
else
  conda deactivate
fi
echo -e "BindCraft environment set up\n"

############################################################################################################
############################################################################################################
################## cleanup
echo -e "Cleaning up ${pkg_manager} temporary files to save space\n"
$pkg_manager clean -a -y
echo -e "$pkg_manager cleaned up\n"

################## finish script
t=$SECONDS 
echo -e "Successfully finished BindCraft installation!\n"
echo -e "Activate environment using command: \"$pkg_manager activate BindCraft\""
echo -e "\n"
echo -e "Installation took $(($t / 3600)) hours, $((($t / 60) % 60)) minutes and $(($t % 60)) seconds."
