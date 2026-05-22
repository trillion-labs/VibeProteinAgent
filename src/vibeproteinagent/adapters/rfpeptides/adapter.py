"""RFpeptides adapter.

Pointer-style adapter wrapping the Baker lab's ``rfpeptides`` branch of
``RFdiffusion`` for head-to-tail macrocyclic peptide design.

The on-disk tree lives at ``$VPA_RFPEPTIDES_TREE`` (default
``vendor/rfpeptides/``). Weights live under ``$VPA_RFPEPTIDES_WEIGHTS``
(default ``<TREE>/weights/``). Each ``run()`` shells out to::

    python scripts/run_inference.py --config-name base ...

with a ``contigmap.contigs`` derived from peptide_length (+ target
contigs when ``target_pdb`` is provided).

Why subprocess: RFdiffusion patches ``torch.set_grad_enabled`` / hydra
globals at import time. Driving each call as a fresh subprocess keeps
the parent worker process clean for cohabiting adapters.

DGL ABI workaround: rfdiffusion eagerly imports ``dgl``; DGL 2.1.0 ships
graphbolt binaries only for torch <= 2.2.1, but the ``vpa`` env runs torch
2.7.1+cu128. We stub ``dgl.graphbolt`` before the DGL import in both the
availability probe and subprocess; rfdiffusion itself only uses
``DGLGraph`` / ``edge_softmax``. The preload is materialised inline in the
subprocess command (no extra files).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.compat import import_dgl_with_graphbolt_guard

# --------------------------------------------------------------------------- paths

_ADAPTER_DIR = Path(__file__).parent
_VPA_ROOT = _ADAPTER_DIR.parent.parent.parent.parent  # /VibeProteinAgent/

RFPEPTIDES_TREE_DEFAULT = _VPA_ROOT / "vendor" / "rfpeptides"

RFPEPTIDES_TREE = Path(
    os.environ.get("VPA_RFPEPTIDES_TREE", str(RFPEPTIDES_TREE_DEFAULT))
).expanduser()
RFPEPTIDES_WEIGHTS = Path(
    os.environ.get("VPA_RFPEPTIDES_WEIGHTS", str(RFPEPTIDES_TREE / "weights"))
).expanduser()


def _check_paths() -> None:
    if not RFPEPTIDES_TREE.is_dir():
        raise FileNotFoundError(
            f"RFpeptides tree not found at {RFPEPTIDES_TREE}. "
            "Clone via: git clone --branch rfpeptides "
            "https://github.com/RosettaCommons/RFdiffusion.git "
            f"{RFPEPTIDES_TREE} (or set $VPA_RFPEPTIDES_TREE)."
        )
    if not (RFPEPTIDES_TREE / "scripts" / "run_inference.py").is_file():
        raise FileNotFoundError(
            f"{RFPEPTIDES_TREE / 'scripts' / 'run_inference.py'} not found — "
            f"$VPA_RFPEPTIDES_TREE does not look like an RFdiffusion checkout."
        )


def _required_weight(target_pdb: str | None) -> Path:
    """Return the canonical weight file for the requested mode."""
    if target_pdb is None:
        return RFPEPTIDES_WEIGHTS / "Base_ckpt.pt"
    return RFPEPTIDES_WEIGHTS / "Complex_base_ckpt.pt"


def _import_dgl_for_rfpeptides():
    """Import DGL with the same GraphBolt stub used by the subprocess path."""
    return import_dgl_with_graphbolt_guard()


def availability_skip_reason() -> str | None:
    """Return a precise skip reason if RFpeptides cannot run, else ``None``.

    Surfaced by ``vpa daemon warm rfpeptides --json`` so operators see
    the concrete missing piece (tree, weights, or a CUDA-enabled DGL
    install for the active torch/CUDA stack). Weight selection is mode-dependent
    (monomer vs binder); the *monomer* weight is required at minimum
    because that's the lowest-friction promotion target.
    """
    run_inference = RFPEPTIDES_TREE / "scripts" / "run_inference.py"
    if not RFPEPTIDES_TREE.is_dir() or not run_inference.is_file():
        return (
            f"RFpeptides tree not present at {RFPEPTIDES_TREE}. "
            f"Clone the rfpeptides branch of RFdiffusion "
            f"(or set $VPA_RFPEPTIDES_TREE)."
        )
    if not (RFPEPTIDES_WEIGHTS / "Base_ckpt.pt").is_file():
        return (
            f"RFpeptides Base_ckpt.pt missing under {RFPEPTIDES_WEIGHTS}. "
            f"See vendor/rfpeptides/UPSTREAM.md for the canonical URL list."
        )
    try:
        dgl = _import_dgl_for_rfpeptides()
    except Exception as exc:
        return (
            f"DGL import failed under this env: {type(exc).__name__}: {exc}. "
            "RFpeptides needs a CUDA-enabled DGL build plus the GraphBolt "
            "torch-2.7 stub described in vendor/rfpeptides/UPSTREAM.md."
        )
    if not torch.cuda.is_available():
        return "RFpeptides requires CUDA for the real rfdiffusion path; no CUDA device is visible."
    try:
        g = dgl.graph(([0, 1], [1, 0])).to("cuda")
        _ = g.all_edges()
    except Exception as exc:
        return (
            f"DGL CUDA graph ops unavailable under this env: {type(exc).__name__}: {exc}. "
            "Install a CUDA-enabled DGL wheel/build compatible with this "
            "torch/CUDA/B200 environment per vendor/rfpeptides/UPSTREAM.md."
        )
    return None


# --------------------------------------------------------------------------- I/O models


class RFPeptidesInput(BaseModel):
    target_pdb: str | None = Field(default=None)
    target_chain: str = Field(default="A", min_length=1, max_length=1)
    target_residues: str | None = Field(default=None, pattern=r"^[0-9]+-[0-9]+$")
    target_hotspots: list[int] = Field(default_factory=list)
    peptide_length: int = Field(default=10, ge=5, le=50)
    num_designs: int = Field(default=10, ge=1, le=64)
    num_diffusion_steps: int = Field(default=50, ge=10, le=200)
    noise_scale: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int | None = Field(default=None)


class RFPeptidesDesign(BaseModel):
    pdb_path: str
    trb_path: str | None = None
    sequence: str | None = None
    length: int
    score: float | None = None


class RFPeptidesOutput(BaseModel):
    designs: list[RFPeptidesDesign]
    runtime_ms: int
    num_designs_returned: int


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class RFPeptidesAdapter(ToolAdapter):
    """Adapter for RFpeptides macrocyclic-peptide design.

    Each ``run()`` invocation shells out to RFdiffusion's
    ``run_inference.py`` with ``inference.cyclic=True``. Outputs are a
    set of backbone PDBs (glycine placeholders); sequence design happens
    downstream via ProteinMPNN or similar.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        self.device = device
        _check_paths()

    @torch.no_grad()
    def run(self, params: RFPeptidesInput, ctx: AdapterRunContext) -> RFPeptidesOutput:
        t0 = time.time()
        ctx.emit_progress(0.02)

        if not torch.cuda.is_available():
            raise RuntimeError("RFpeptides adapter requires CUDA")

        weight = _required_weight(params.target_pdb)
        if not weight.is_file():
            raise FileNotFoundError(
                f"Required RFpeptides weight checkpoint missing at {weight}. "
                f"Download from "
                f"https://files.ipd.uw.edu/pub/RFdiffusion/ — see "
                f"vendor/rfpeptides/UPSTREAM.md for the canonical URL list."
            )

        job_inputs = ctx.job_dir / "inputs"
        job_outputs = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)

        staged_target: Path | None = None
        if params.target_pdb is not None:
            src = Path(params.target_pdb).expanduser().resolve()
            if not src.is_file():
                raise FileNotFoundError(f"target_pdb not found: {src}")
            staged_target = job_inputs / src.name
            shutil.copyfile(src, staged_target)

        contigs = _build_contigs(params, staged_target)
        cmd = _build_cmd(
            params=params,
            output_prefix=job_outputs / "design",
            target_pdb=staged_target,
            contigs=contigs,
            weights_dir=RFPEPTIDES_WEIGHTS,
        )
        env = _build_env(params.seed)

        ctx.emit_progress(0.10)
        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                cwd=str(job_outputs),
                timeout=3600,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"RFpeptides subprocess failed with exit code {exc.returncode}"
            ) from exc
        ctx.emit_progress(0.92)

        designs = _collect_designs(
            prefix=job_outputs / "design",
            peptide_length=params.peptide_length,
        )

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return RFPeptidesOutput(
            designs=designs,
            runtime_ms=runtime_ms,
            num_designs_returned=len(designs),
        )

    def estimate_vram(self, params: RFPeptidesInput) -> int:
        del params
        return 12 * 1024

    def healthcheck(self) -> bool:
        return availability_skip_reason() is None

    def unload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- helpers


def _build_contigs(params: RFPeptidesInput, staged_target: Path | None) -> str:
    """Build the ``contigmap.contigs`` string.

    Monomer: ``[<L>-<L>]``.
    Binder:  ``[<L>-<L> <chain><start>-<end>/0]``.
    """
    length = params.peptide_length
    if staged_target is None:
        return f"[{length}-{length}]"
    chain = params.target_chain.upper()
    if params.target_residues is not None:
        return f"[{length}-{length} {chain}{params.target_residues}/0]"
    # Default to full chain (RFdiffusion accepts ``A1-XXX`` ranges only). We
    # use a generous open-ended range — RFdiffusion parses ``Achain1-9999``
    # by capping at the chain length.
    return f"[{length}-{length} {chain}1-9999/0]"


def _build_cmd(
    *,
    params: RFPeptidesInput,
    output_prefix: Path,
    target_pdb: Path | None,
    contigs: str,
    weights_dir: Path,
) -> list[str]:
    """Build the ``python -c '<preload>; runpy.run_path(...)'`` command.

    The preload step stubs ``dgl.graphbolt`` before any rfdiffusion code
    runs (see DGL ABI workaround in the module docstring).
    """
    run_inference = RFPEPTIDES_TREE / "scripts" / "run_inference.py"
    preload = (
        "import sys, types; "
        "sys.modules.setdefault('dgl.graphbolt', types.ModuleType('dgl.graphbolt')); "
        "import sys as _s; _s.argv = "
        + repr(
            [
                str(run_inference),
                "--config-name",
                "base",
                f"inference.output_prefix={output_prefix}",
                f"inference.num_designs={params.num_designs}",
                f"contigmap.contigs={contigs}",
                "inference.cyclic=True",
                "inference.cyc_chains=a",
                f"diffuser.T={params.num_diffusion_steps}",
                f"denoiser.noise_scale_ca={params.noise_scale}",
                f"denoiser.noise_scale_frame={params.noise_scale}",
                f"inference.model_directory_path={weights_dir}",
            ]
            + (
                [f"inference.input_pdb={target_pdb}"]
                if target_pdb is not None
                else []
            )
            + (
                [
                    "ppi.hotspot_res=["
                    + ",".join(
                        f"\\'{params.target_chain.upper()}{i}\\'"
                        for i in params.target_hotspots
                    )
                    + "]"
                ]
                if params.target_hotspots and target_pdb is not None
                else []
            )
        )
        + "; "
        "import runpy; runpy.run_path("
        + repr(str(run_inference))
        + ", run_name='__main__')"
    )
    return [sys.executable, "-c", preload]


def _build_env(seed: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    src = str(RFPEPTIDES_TREE)
    se3 = str(RFPEPTIDES_TREE / "env" / "SE3Transformer")
    env["PYTHONPATH"] = ":".join([src, se3])
    env["DGLBACKEND"] = "pytorch"
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    return env


def _collect_designs(*, prefix: Path, peptide_length: int) -> list[RFPeptidesDesign]:
    """Walk RFdiffusion's output layout.

    RFdiffusion writes ``<prefix>_<i>.pdb`` and ``<prefix>_<i>.trb`` files for
    each generated design.
    """
    parent = prefix.parent
    stem = prefix.name
    pdbs = sorted(parent.glob(f"{stem}_*.pdb"))
    if not pdbs:
        raise RuntimeError(
            f"RFdiffusion produced no PDB outputs under {parent} with stem '{stem}'."
        )
    designs: list[RFPeptidesDesign] = []
    for pdb in pdbs:
        trb = pdb.with_suffix(".trb")
        designs.append(
            RFPeptidesDesign(
                pdb_path=str(pdb),
                trb_path=str(trb) if trb.is_file() else None,
                sequence=None,
                length=peptide_length,
                score=None,
            )
        )
    return designs


__all__ = [
    "RFPEPTIDES_TREE",
    "RFPEPTIDES_WEIGHTS",
    "RFPeptidesAdapter",
    "RFPeptidesDesign",
    "RFPeptidesInput",
    "RFPeptidesOutput",
    "availability_skip_reason",
]
