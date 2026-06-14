"""Shared plumbing for the three Proteina-Complexa adapters.

The Proteina-Complexa upstream ships a single tree at
``vendor/proteina-complexa/`` (NVIDIA-Digital-Bio, NVIDIA Open Model
License) that exposes one ``complexa`` CLI with three pipeline configs:

* ``configs/search_binder_local_pipeline.yaml`` — protein binder design
* ``configs/search_ligand_binder_local_pipeline.yaml`` — small-molecule
  binder design (LoRA-adapted base + ligand conditioning)
* ``configs/search_ame_local_pipeline.yaml`` — Atomistic Motif Embedding
  (motif scaffolding into monomers, possibly with ligand context)

Each variant has its own NGC checkpoint pair (``complexa{,_ligand,_ame}``
+ matching ``_ae`` autoencoder). The CLI is hydra-based; VPA constrains
adapter runs to ``generate → evaluate`` so filtering/analyze stages cannot
turn generated artifacts into an empty candidate stream.

Why a *separate uv venv* instead of the vpa env
================================================

Upstream's ``pyproject.toml`` pins:

* ``scipy==1.12.0`` — direct conflict with our vpa-pinned ``scipy 1.17.1``
  (the Phase 12-20 hard-rule forbids modifying scipy).
* ``lightning>=2.5.0,<2.6`` — upper-bounded for a known checkpoint
  loading bug; not catastrophic but pip-installing the full upstream
  stack into vpa would resolve to this and bump our 2.5.x install.
* PyTorch/JAX/PyG CUDA wheels are profile-driven and must not leak into the
  shared vpa env.
* ``tmol`` (Rosetta force fields) — needs Python-3.12-specific
  ``llvmlite``/``numba`` workarounds per upstream's README.
* Multiple weights gated behind NGC accounts (AF2, RF3, ESM2 weights
  live outside the repo).
* Foldseek, MMseqs2, DSSP, SC, Pyrosetta binaries expected on ``PATH``.

So these adapters run Proteina-Complexa in a *separate* uv venv built
by ``vendor/proteina-complexa/env/build_uv_env.sh`` (override the venv
path via ``$VPA_PROTEINA_COMPLEXA_VENV``). The adapters only stage
inputs, materialise target/hotspot/SMILES configuration, dispatch
``complexa design <config> ++…`` through a persistent sidecar running
``<venv>/bin/python``, and parse outputs. If the venv does not exist
(or the upstream tree is unavailable) the adapters ship as PARTIAL —
fast tests still pass, the slow tests skip with a clear message.

NGC checkpoints (each ~few GiB pair) live under
``$VPA_PROTEINA_COMPLEXA_CKPTS`` (default
``vendor/proteina-complexa/ckpts/``). Operators fetch them via
``complexa download --complexa-all`` after activating the venv.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from vibeproteinagent.adapters.runtime import sanitized_python_env, smoke_import
from vibeproteinagent.core.errors import DependencyError, VPAError
from vibeproteinagent.workers.sidecar import PersistentPythonSidecar, SidecarRuntimeSpec

# --------------------------------------------------------------------------- root path resolution

# This module lives at src/vibeproteinagent/adapters/_proteina_complexa_common.py
# so the vpa repo root is four parents up.
_THIS_DIR = Path(__file__).parent
_VPA_ROOT = _THIS_DIR.parent.parent.parent

PROTEINA_COMPLEXA_TREE_DEFAULT = _VPA_ROOT / "vendor" / "proteina-complexa"
PROTEINA_COMPLEXA_ENV_NAME = "proteina-complexa-uv"

_RUNNER_SIDECAR_CALL = "vibeproteinagent.adapters._proteina_complexa_common:run_design_sidecar"


def proteina_complexa_tree() -> Path:
    """Return the resolved ``$VPA_PROTEINA_COMPLEXA_TREE`` path (env wins)."""
    return Path(
        os.environ.get(
            "VPA_PROTEINA_COMPLEXA_TREE", str(PROTEINA_COMPLEXA_TREE_DEFAULT)
        )
    ).expanduser()


def proteina_complexa_venv() -> Path:
    """Return the resolved ``$VPA_PROTEINA_COMPLEXA_VENV`` path (env wins).

    Default is ``<tree>/.venv`` — the layout produced by upstream's
    ``env/build_uv_env.sh``.
    """
    env = os.environ.get("VPA_PROTEINA_COMPLEXA_VENV")
    if env:
        return Path(env).expanduser()
    return proteina_complexa_tree() / ".venv"


def proteina_complexa_ckpts() -> Path:
    """Return the resolved ``$VPA_PROTEINA_COMPLEXA_CKPTS`` path (env wins)."""
    env = os.environ.get("VPA_PROTEINA_COMPLEXA_CKPTS")
    if env:
        return Path(env).expanduser()
    return proteina_complexa_tree() / "ckpts"


def complexa_executable() -> Path:
    """Return the path to ``<venv>/bin/complexa`` (resolved at call time).

    We deliberately resolve lazily — env vars and venv state may change
    between adapter construction and ``run()`` invocation.
    """
    return proteina_complexa_venv() / "bin" / "complexa"


def proteina_complexa_python() -> Path:
    """Return the path to ``<venv>/bin/python`` for the persistent sidecar."""
    return proteina_complexa_venv() / "bin" / "python"


# --------------------------------------------------------------------------- availability probes


def tree_present() -> bool:
    tree = proteina_complexa_tree()
    return (tree / "pyproject.toml").is_file() and (tree / "configs").is_dir()


def venv_present() -> bool:
    return complexa_executable().is_file() and proteina_complexa_python().is_file()


def ckpt_pair_present(ckpt_name: str, ae_name: str) -> bool:
    """Return True iff the two named checkpoints exist under ``$..._CKPTS``."""
    base = proteina_complexa_ckpts()
    return (base / ckpt_name).is_file() and (base / ae_name).is_file()


def pipeline_config(rel: str) -> Path:
    """Resolve a pipeline config path under the vendored tree."""
    return proteina_complexa_tree() / rel


# --------------------------------------------------------------------------- target helpers


def protein_chain_input_spec(pdb: Path, chain: str) -> str:
    """Return a Proteina-Complexa target input spec for a PDB polymer chain.

    Upstream target dictionaries use compact specs such as ``A1-112``.
    Build the same shape from ATOM records only; HETATM waters and
    ligands may share the chain ID but are not part of the target
    polymer span.
    """
    residues: set[int] = set()
    with pdb.open(errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 26 or line[21].strip() != chain:
                continue
            try:
                residues.add(int(line[22:26]))
            except ValueError:
                continue
    if not residues:
        raise ValueError(f"No polymer ATOM residues for chain {chain!r} found in {pdb}")
    return "/".join(
        f"{chain}{start}" if start == end else f"{chain}{start}-{end}"
        for start, end in residue_ranges(sorted(residues))
    )


def hydra_task_key(path: Path, *, prefix: str = "vpa") -> str:
    """Return a Hydra-dotpath-safe task key derived from a file path."""
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem).strip("_")
    if not stem:
        stem = "target"
    if not (stem[0].isalpha() or stem[0] == "_"):
        stem = f"{prefix}_{stem}"
    return stem


def residue_ranges(residues: list[int]) -> list[tuple[int, int]]:
    """Collapse residue indices into inclusive contiguous ranges."""
    if not residues:
        raise ValueError("at least one residue is required")
    ordered = sorted(set(int(r) for r in residues))
    ranges: list[tuple[int, int]] = []
    start = prev = ordered[0]
    for res in ordered[1:]:
        if res == prev + 1:
            prev = res
            continue
        ranges.append((start, prev))
        start = prev = res
    ranges.append((start, prev))
    return ranges


def motif_contig_string(*, chain: str, residues: list[int], total_length: int) -> str:
    """Build a Proteina-Complexa motif contig for indexed monomer scaffolding."""
    motif_ranges = residue_ranges(residues)
    motif_len = sum(end - start + 1 for start, end in motif_ranges)
    if motif_len >= total_length:
        raise ValueError(
            f"motif has {motif_len} residues, which must be smaller than total_length "
            f"{total_length}"
        )
    max_spacer = max(total_length - motif_len, 1)
    parts: list[str] = [f"0-{max_spacer}"]
    for start, end in motif_ranges:
        parts.append(f"{chain}{start}-{end}")
        parts.append(f"0-{max_spacer}")
    return "/".join(parts)


def motif_segment_order(residues: list[int]) -> str:
    """Return a segment-order string matching the number of motif ranges."""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    count = len(residue_ranges(residues))
    if count > len(labels):
        raise ValueError(f"too many motif segments: {count}")
    return ";".join(labels[:count])


# --------------------------------------------------------------------------- command builder


def build_design_cmd(
    *,
    config_path: Path,
    overrides: list[str],
    steps: list[str] | None = None,
    venv: Path | None = None,
) -> list[str]:
    """Build ``<venv>/bin/complexa design <config> <overrides>`` invocation.

    Overrides are hydra-style ``++key=value`` strings. The list passed
    in is appended verbatim so callers retain full control of quoting.
    """
    venv = venv or proteina_complexa_venv()
    base = [
        str(venv / "bin" / "complexa"),
        "design",
    ]
    if steps:
        # ``--steps`` is an argparse ``nargs='+'`` option upstream. Use ``--``
        # so Hydra overrides are parsed as positionals, not extra step names.
        return [*base, "--steps", *steps, "--", str(config_path), *overrides]
    return [*base, str(config_path), *overrides]


def build_isolated_env(
    *, seed: int | None, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """Sanitised environment for the dedicated Proteina-Complexa runtime.

    Drop ``PYTHONPATH`` so the uv venv is fully isolated from vpa's
    site-packages, keep ``PYTHONNOUSERSITE`` on, and route the GPU
    through whatever the parent process declares.
    """
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["DEBUG"] = "false"
    env.pop("PYTHONPATH", None)
    tree = proteina_complexa_tree()
    venv = proteina_complexa_venv()
    ckpts = proteina_complexa_ckpts()
    community = tree / "community_models"
    env.setdefault("COMPLEXA_INIT", "uv")
    if "VPA_PROTEINA_COMPLEXA_JAX_PLATFORMS" in os.environ:
        env.setdefault("JAX_PLATFORMS", os.environ["VPA_PROTEINA_COMPLEXA_JAX_PLATFORMS"])
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("LOCAL_CODE_PATH", str(tree))
    env.setdefault("LOCAL_DATA_PATH", str(tree / "data"))
    env.setdefault("LOCAL_CACHE_DIR", str(tree / ".cache"))
    env.setdefault("LOCAL_CHECKPOINT_PATH", str(ckpts))
    env.setdefault("COMMUNITY_MODELS_PATH", str(community))
    env.setdefault("CKPT_PATH", str(ckpts))
    env.setdefault("ESM_DIR", str(community / "ckpts" / "ESM2"))
    env.setdefault("AF2_DIR", str(community / "ckpts" / "AF2"))
    env.setdefault("RF3_DIR", str(community / "ckpts" / "RF3"))
    env.setdefault(
        "RF3_CKPT_PATH",
        str(community / "ckpts" / "RF3" / "rf3_foundry_01_24_latest_remapped.ckpt"),
    )
    env.setdefault("UV_VENV", str(venv))
    lib_paths = _nvidia_package_lib_paths(venv)
    if lib_paths:
        existing_ld = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*lib_paths, *([existing_ld] if existing_ld else [])]
        )
    env.setdefault(
        "FOLDSEEK_EXEC", shutil.which("foldseek") or str(venv / "bin" / "foldseek")
    )
    env.setdefault("MMSEQS_EXEC", shutil.which("mmseqs") or str(venv / "bin" / "mmseqs"))
    env.setdefault(
        "DSSP_EXEC",
        shutil.which("mkdssp") or shutil.which("dssp") or str(venv / "bin" / "dssp"),
    )
    env.setdefault("RF3_EXEC_PATH", shutil.which("rf3") or str(venv / "bin" / "rf3"))
    env.setdefault(
        "SC_EXEC", shutil.which("sc") or str(tree / "env" / "docker" / "internal" / "sc")
    )
    env.setdefault(
        "TMOL_PATH",
        str(_venv_site_packages(venv) / "tmol"),
    )
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    if extra:
        env.update(extra)
    return env


def _venv_site_packages(venv: Path) -> Path:
    candidates = sorted((venv / "lib").glob("python*/site-packages"))
    if candidates:
        return candidates[0]
    return (
        venv
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def _nvidia_package_lib_paths(venv: Path) -> list[str]:
    site_packages = _venv_site_packages(venv)
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.is_dir():
        return []
    priority = {
        "cudnn": 0,
        "cublas": 1,
        "cuda_runtime": 2,
        "cuda_nvrtc": 3,
    }
    paths = [path for path in nvidia_root.glob("*/lib") if path.is_dir()]
    paths.sort(key=lambda path: (priority.get(path.parent.name, 100), path.parent.name))
    return [str(path) for path in paths]


def build_sidecar_env(
    *, seed: int | None, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """Sanitised environment for a persistent Proteina-Complexa sidecar."""
    env = build_isolated_env(seed=seed, extra=extra)
    venv_bin = str(proteina_complexa_venv() / "bin")
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    return sanitized_python_env(
        seed=seed,
        prepend_bin=[proteina_complexa_venv() / "bin"],
        extra=env,
    )


def venv_import_skip_reason(timeout_s: int = 30) -> str | None:
    """Smoke-import the uv venv sidecar runtime before declaring availability."""
    python = proteina_complexa_python()
    reason = smoke_import(
        label="Proteina-Complexa sidecar",
        python=python,
        code=(
            "import msgpack; "
            "import vibeproteinagent.workers.sidecar_child; "
            "import proteinfoundation.cli.cli_runner; "
            "print('ok')"
        ),
        env=build_sidecar_env(seed=None),
        timeout_s=timeout_s,
    )
    if reason is None:
        return None
    return (
        f"{reason}. "
        "Check msgpack, VPA sidecar install, and proteinfoundation CLI installation."
    )


def jax_dnn_skip_reason(timeout_s: int = 45) -> str | None:
    """Verify that Proteina-Complexa's sidecar can execute a tiny GPU DNN op.

    The normal availability probe intentionally stays shallow. Installer doctor
    uses this deeper probe to catch JAX/XLA/cuDNN failures that only appear when
    AF2RewardModel-like evaluation reaches GPU DNN kernels.
    """
    python = proteina_complexa_python()
    code = r"""
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
print("ok")
"""
    env = build_sidecar_env(seed=None)
    env["JAX_TRACEBACK_FILTERING"] = "off"
    try:
        proc = subprocess.run(
            [str(python), "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except FileNotFoundError:
        reason = f"Proteina-Complexa JAX GPU DNN python not found: {python}"
    except subprocess.TimeoutExpired:
        reason = f"Proteina-Complexa JAX GPU DNN probe timed out after {timeout_s}s"
    else:
        if proc.returncode == 0:
            return None
        reason = "Proteina-Complexa JAX GPU DNN probe failed: " + _stderr_tail(
            proc.stderr or proc.stdout or f"exit code {proc.returncode}"
        )
    if not reason:
        return None
    return (
        f"{reason}. Proteina-Complexa can import, but its JAX/XLA/cuDNN GPU "
        "backend is not ready for AF2RewardModel-style evaluation."
    )


def triton_load_binary_skip_reason(timeout_s: int = 45) -> str | None:
    """Verify Triton can compile and load a tiny CUDA kernel.

    Proteina-Complexa RF3 evaluation reaches Triton's
    ``driver.active.utils.load_binary`` through cuequivariance fused ops. A
    broken Triton/native-extension stack can import cleanly but fail only when
    a compiled CUDA kernel is loaded, so doctor must exercise that exact class
    of operation.
    """

    python = proteina_complexa_python()
    code = r'''
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
print("ok")
'''
    env = build_sidecar_env(seed=None)
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix="_proteina_complexa_triton_probe.py", delete=False
        ) as handle:
            handle.write(code)
            probe_path = Path(handle.name)
        proc = subprocess.run(
            [str(python), str(probe_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
    except FileNotFoundError:
        reason = f"Proteina-Complexa Triton probe python not found: {python}"
    except subprocess.TimeoutExpired:
        reason = f"Proteina-Complexa Triton probe timed out after {timeout_s}s"
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
    if "proc" in locals() and proc.returncode == 0:
        return None
    if "reason" not in locals():
        reason = "Proteina-Complexa Triton CUDA load_binary probe failed: " + _stderr_tail(
            proc.stderr or proc.stdout or f"exit code {proc.returncode}"
        )
    return (
        f"{reason}. Proteina-Complexa can import, but its Triton/cuequivariance "
        "backend is not ready for RF3 evaluation."
    )


def evaluate_smoke_skip_reason(timeout_s: int = 900) -> str | None:
    """Run a real minimal Proteina-Complexa RF3 binder-evaluation smoke.

    This is intentionally heavier than import/JAX/Triton probes. It executes
    the vendor ``run_binder_eval()`` path with ``sequence_types=['self']`` on a
    synthetic ligand-binder complex, initializes the real RF3 runner, launches
    RF3 through its JSON ligand path, parses the generated confidence/CIF output,
    converts CIF to PDB through Biotite, and computes binder/ligand RMSD. That
    covers the evaluate stack where Proteina-Complexa previously failed after
    generation.
    """

    python = proteina_complexa_python()
    code = r'''
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
print("ok")
'''
    env = build_sidecar_env(seed=None)
    env["JAX_TRACEBACK_FILTERING"] = "off"
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix="_proteina_complexa_eval_smoke.py", delete=False
        ) as handle:
            handle.write(code)
            probe_path = Path(handle.name)
        proc = subprocess.run(
            [str(python), str(probe_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            cwd=proteina_complexa_tree(),
        )
    except FileNotFoundError:
        reason = f"Proteina-Complexa evaluate smoke python not found: {python}"
    except subprocess.TimeoutExpired:
        reason = f"Proteina-Complexa evaluate smoke timed out after {timeout_s}s"
    finally:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
    if "proc" in locals() and proc.returncode == 0:
        return None
    if "reason" not in locals():
        reason = "Proteina-Complexa RF3 evaluate smoke failed: " + _stderr_tail(
            (proc.stderr or "") + "\n" + (proc.stdout or "") or f"exit code {proc.returncode}",
            max_lines=20,
        )
    return (
        f"{reason}. Proteina-Complexa can import and may pass primitive GPU probes, "
        "but the full RF3 evaluate path is not ready."
    )


def _stderr_tail(text: str, *, max_lines: int = 8) -> str:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return "no stderr captured"
    return " | ".join(lines[-max_lines:])


# --------------------------------------------------------------------------- output discovery


def latest_inference_dir(tree: Path) -> Path | None:
    """Return the most recent ``inference/<run_name>/<target>`` directory.

    ``complexa design`` writes outputs under
    ``$CWD/inference/<run_name>/<target_name>/`` by default; we use
    mtime to identify the run we just kicked off when ``run_name`` was
    not supplied explicitly. Callers that *did* supply ``run_name``
    should construct the path directly.
    """
    inference_root = tree / "inference"
    if not inference_root.is_dir():
        return None
    candidates = [p for p in inference_root.rglob("*.pdb") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime).parent


def collect_pdbs(run_dir: Path) -> list[Path]:
    """List ``*.pdb`` files under a run output dir, sorted lexicographically."""
    if not run_dir.is_dir():
        return []
    return sorted(run_dir.rglob("*.pdb"))


def inference_run_dir(*, config_stem: str, task_name: str, run_name: str) -> Path:
    """Return the output directory shape used by ``complexa design``."""
    return proteina_complexa_tree() / "inference" / f"{config_stem}_{task_name}_{run_name}"


# --------------------------------------------------------------------------- skip helper


def availability_skip_reason() -> str | None:
    """Return a human-readable reason if the runtime is unavailable, else None.

    The two slow regression tests use this to skip cleanly without
    duplicating the probe logic.
    """
    if not tree_present():
        return (
            f"Proteina-Complexa tree not present at {proteina_complexa_tree()} "
            f"— set $VPA_PROTEINA_COMPLEXA_TREE or vendor the repo."
        )
    if not venv_present():
        return (
            f"Proteina-Complexa uv venv missing at {proteina_complexa_venv()}. "
            f"Build via `cd {proteina_complexa_tree()} && ./env/build_uv_env.sh` "
            f"(NOT into the vpa env — scipy 1.12.0 conflicts with vpa's "
            f"pinned scipy 1.17.1). See vendor/proteina-complexa/UPSTREAM.md."
        )
    return venv_import_skip_reason()


# --------------------------------------------------------------------------- persistent sidecar


def sidecar_for(
    *,
    cache: dict[tuple[int | None, tuple[tuple[str, str], ...]], PersistentPythonSidecar],
    seed: int | None,
    extra_env: dict[str, str] | None = None,
) -> PersistentPythonSidecar:
    """Return a cached Proteina-Complexa sidecar for this seed/env tuple."""
    key = (
        int(seed) if seed is not None else None,
        tuple(sorted((extra_env or {}).items())),
    )
    sidecar = cache.get(key)
    if sidecar is None:
        python = proteina_complexa_python()
        if not python.is_file():
            raise RuntimeError(
                f"Proteina-Complexa sidecar python not found at {python}. "
                f"Build the uv venv via `cd {proteina_complexa_tree()} && "
                f"./env/build_uv_env.sh`."
            )
        spec = SidecarRuntimeSpec(
            python=python,
            env=build_sidecar_env(seed=seed, extra=extra_env),
            name=f"proteina-complexa[{key[0] if key[0] is not None else 'default'}]",
            run_target=_RUNNER_SIDECAR_CALL,
            timeout_s=60.0,
        )
        sidecar = spec.spawn(PersistentPythonSidecar)
        cache[key] = sidecar
    return sidecar


def run_design(
    *,
    sidecar: PersistentPythonSidecar,
    config_path: Path,
    overrides: list[str],
    cwd: Path,
    steps: list[str] | None = None,
    timeout_s: int = 4 * 3600,
) -> dict[str, Any]:
    """Invoke ``complexa design`` through a persistent sidecar."""
    cmd = build_design_cmd(config_path=config_path, overrides=overrides, steps=steps)
    if not Path(cmd[0]).is_file():
        raise RuntimeError(
            f"complexa executable not found at {cmd[0]}. Build the uv venv "
            f"via `cd {proteina_complexa_tree()} && ./env/build_uv_env.sh`."
        )
    try:
        result = sidecar.call(
            _RUNNER_SIDECAR_CALL,
            {
                "config_path": str(config_path),
                "overrides": list(overrides),
                "steps": list(steps) if steps else None,
                "cwd": str(cwd),
            },
            timeout_s=timeout_s,
        )
        if isinstance(result, dict) and result.get("ok") is False:
            raise RuntimeError(
                f"complexa design sidecar failed. cmd: {' '.join(cmd)}. "
                f"exit_code={result.get('exit_code')}"
            )
        return result
    except VPAError as exc:
        details = _sidecar_error_details(
            exc,
            cmd=cmd,
            config_path=config_path,
            overrides=overrides,
            steps=steps,
            cwd=cwd,
        )
        if _is_dependency_sidecar_failure(details):
            raise DependencyError(
                "Proteina-Complexa sidecar failed in its runtime/evaluation "
                "backend (CUDA/JAX/XLA or Python API compatibility) while running "
                "the design pipeline. This is a dependency or runtime compatibility "
                "issue, not a candidate quality failure.",
                details=details,
            ) from exc
        raise RuntimeError(
            f"complexa design sidecar failed. cmd: {' '.join(cmd)}. {exc}"
        ) from exc


def _sidecar_error_details(
    exc: VPAError,
    *,
    cmd: list[str],
    config_path: Path,
    overrides: list[str],
    steps: list[str] | None,
    cwd: Path,
) -> dict[str, Any]:
    return {
        "cmd": cmd,
        "config_path": str(config_path),
        "overrides": list(overrides),
        "steps": list(steps) if steps else None,
        "cwd": str(cwd),
        "sidecar_message": exc.message,
        "sidecar_details": exc.details,
        "root_cause_hint": _sidecar_runtime_hint(exc),
    }


def _accelerator_backend_hint(exc: VPAError) -> str | None:
    return _sidecar_runtime_hint(exc)


def _sidecar_runtime_hint(exc: VPAError) -> str | None:
    text = _flatten_sidecar_error_text(exc)
    if "dnn_support != nullptr" in text:
        return "JAX/XLA could not initialize a GPU DNN/cuDNN backend"
    if "jaxruntimeerror" in text and ("xla" in text or "cuda" in text or "cudnn" in text):
        return "JAX/XLA GPU runtime failed"
    if "cudnn" in text and ("jax" in text or "xla" in text):
        return "cuDNN is unavailable or incompatible with the JAX/XLA runtime"
    if "py_ssize_t_clean macro must be defined" in text:
        return "Python C-extension ABI mismatch in the Proteina-Complexa evaluation runtime"
    if "a file opened in 'text' mode is required" in text:
        return "Proteina-Complexa evaluation writer API mismatch in the sidecar runtime"
    return None


def _is_accelerator_backend_failure(details: dict[str, Any]) -> bool:
    return bool(details.get("root_cause_hint"))


def _is_dependency_sidecar_failure(details: dict[str, Any]) -> bool:
    return bool(details.get("root_cause_hint"))


def _flatten_sidecar_error_text(exc: VPAError) -> str:
    parts = [exc.message, str(exc)]
    details = exc.details
    if isinstance(details, dict):
        parts.append(str(details.get("code") or ""))
        nested = details.get("details")
        if isinstance(nested, dict):
            parts.extend(str(value) for value in nested.values())
        else:
            parts.append(str(nested or ""))
    return "\n".join(parts).lower()


def run_design_sidecar(
    *, config_path: str, overrides: list[str], cwd: str, steps: list[str] | None = None
) -> dict[str, Any]:
    """Run the Proteina-Complexa CLI inside the persistent sidecar process."""
    import runpy

    entrypoint = complexa_executable()
    if not entrypoint.is_file():
        raise RuntimeError(f"complexa executable not found at {entrypoint}")
    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    argv = ["complexa", "design"]
    if steps:
        argv.extend(["--steps", *steps, "--"])
    argv.extend([config_path, *overrides])
    try:
        os.chdir(cwd)
        sys.argv = argv
        runpy.run_path(str(entrypoint), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            return {"ok": False, "exit_code": code, "cmd": argv}
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
    return {"ok": True}


# --------------------------------------------------------------------------- shutil shim


def which_complexa() -> Path | None:
    """Resolve ``complexa`` via the venv first, then PATH (for non-uv installs)."""
    direct = complexa_executable()
    if direct.is_file():
        return direct
    on_path = shutil.which("complexa")
    return Path(on_path) if on_path else None
