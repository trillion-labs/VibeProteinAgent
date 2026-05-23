"""Real PXDesign adapter (ByteDance target-conditioned binder design).

Vendor-style adapter — loads PXDesign + the `protenix-pxd` fork + PXDesignBench
from `vendor/` (populated by Phase 5.1). All three vendored trees are inserted
onto ``sys.path`` *before* importing pxdesign so internal cross-package
imports (`from protenix.config import ...`, `from pxdbench.pxd_configs.eval
import eval_configs`) resolve to our vendored copies.

The Phase 5.1 hand-off (note ``9b227bfb-…``) documents:

* Three vendor trees: ``vendor/pxdesign``, ``vendor/pxdbench``, ``vendor/protenix-pxd``.
* Module-name collision with the in-place Protenix v1.0.4 used by the
  ``protenix_v2`` adapter — they must not co-load in the same worker. This is
  expressed via ``descriptor.yaml`` ``incompatible_with: [protenix_v2]`` and
  enforced by ``Worker._load_adapter`` (Phase 5.2 framework change).
* A fair-esm stub injection is required before ``import pxdesign`` to satisfy
  ``protenix-pxd/protenix/data/compute_esm.py``'s top-level
  ``from esm import FastaBatchedDataset, pretrained``.

The warm-load path stops after ``init_env / init_basics / init_model /
load_checkpoint``; ``init_dumper`` and ``init_data`` are re-run per call
because they bind the InferenceRunner to a specific input/output dir.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.runtime import EnvLocator, sanitized_python_env, smoke_import
from vibeproteinagent.core.errors import DependencyError, InvalidParamsError, VPAError
from vibeproteinagent.workers.sidecar import (
    PersistentPythonSidecar,
    SidecarRuntimeSpec,
)

# --------------------------------------------------------------------------- paths

# Three vendor roots, in the order they get prepended to sys.path. The
# protenix-pxd tree must come before pxdesign so the vendored `protenix`
# package wins over any other `protenix` already on `sys.path`. (Phase 5.1
# verified this in the smoke run.)
_REPO_ROOT = Path(__file__).resolve().parents[4]

PXDESIGN_VENDOR_DEFAULT = _REPO_ROOT / "vendor" / "pxdesign"
PXDBENCH_VENDOR_DEFAULT = _REPO_ROOT / "vendor" / "pxdbench"
PROTENIX_PXD_VENDOR_DEFAULT = _REPO_ROOT / "vendor" / "protenix-pxd"

def _default_pxdesign_ckpt_dir() -> Path:
    vpa_home = os.environ.get("VPA_HOME")
    if vpa_home:
        cache_dir = Path(vpa_home).expanduser() / "cache" / "pxdesign"
        if cache_dir.exists():
            return cache_dir
    return Path("/opt/vpa-data/pxdesign_ckpt")


PXDESIGN_CKPT_DIR_DEFAULT = _default_pxdesign_ckpt_dir()


def _default_protenix_data_root() -> str:
    explicit = os.environ.get("VPA_PROTENIX_DATA_ROOT_DIR") or os.environ.get(
        "PROTENIX_DATA_ROOT_DIR"
    )
    if explicit:
        return explicit
    ckpt_root = os.environ.get("VPA_PROTENIX_CKPT")
    if ckpt_root:
        common = Path(ckpt_root).expanduser() / "common"
        if common.is_dir():
            return str(common)
    vpa_home = os.environ.get("VPA_HOME")
    if vpa_home:
        common = Path(vpa_home).expanduser() / "cache" / "protenix" / "common"
        if common.is_dir():
            return str(common)
    return "/opt/vpa-data/protenix_ckpt/common"


PROTENIX_DATA_ROOT_DEFAULT = _default_protenix_data_root()

PXDESIGN_MODEL_NAME = "pxdesign_v0.1.0"

PXDESIGN_VENDOR = Path(
    os.environ.get("VPA_PXDESIGN_VENDOR", str(PXDESIGN_VENDOR_DEFAULT))
).expanduser()
PXDBENCH_VENDOR = Path(
    os.environ.get("VPA_PXDBENCH_VENDOR", str(PXDBENCH_VENDOR_DEFAULT))
).expanduser()
PROTENIX_PXD_VENDOR = Path(
    os.environ.get("VPA_PROTENIX_PXD_VENDOR", str(PROTENIX_PXD_VENDOR_DEFAULT))
).expanduser()
PXDESIGN_CKPT_DIR = Path(
    os.environ.get("VPA_PXDESIGN_CKPT_DIR", str(PXDESIGN_CKPT_DIR_DEFAULT))
).expanduser()

# Dedicated conda env for the `pipeline` mode (AF2-IG + ProteinMPNN + filters).
# JAX + dm-haiku + optax + ColabDesign cannot share vpa's process — JAX's
# bundled cuDNN/CUBLAS/nccl wheels would override torch's bundled CUDA libs
# (see Phase 16b note `094ffe75-...`). Pipeline mode runs inside this
# dedicated persistent sidecar; the `infer` mode keeps running in-process.
PXDESIGN_PIPELINE_ENV_DEFAULT = Path("/opt/conda/envs/pxdesign-pipeline-vpa")
PXDESIGN_PIPELINE_ENV_NAME = "pxdesign-pipeline-vpa"

_PIPELINE_RUNNER_MODULE = "vibeproteinagent.adapters.pxdesign._pipeline_runner"
_PIPELINE_SIDECAR_CALL = f"{_PIPELINE_RUNNER_MODULE}:run_sidecar"


def _resolve_checkpoint() -> Path:
    """Return the path of the PXDesign diffusion checkpoint, raising if absent."""
    ckpt = PXDESIGN_CKPT_DIR / f"{PXDESIGN_MODEL_NAME}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"PXDesign checkpoint not found at {ckpt}. "
            f"Set $VPA_PXDESIGN_CKPT_DIR to the directory holding "
            f"{PXDESIGN_MODEL_NAME}.pt (and the protenix base/mini checkpoints)."
        )
    return ckpt


def _stub_fair_esm_symbols() -> None:
    """Inject ``FastaBatchedDataset`` / ``pretrained`` into the installed ``esm``.

    The vendored ``protenix-pxd/protenix/data/compute_esm.py`` does
    ``from esm import FastaBatchedDataset, pretrained`` at module top — those
    names live in the legacy ``fair-esm`` 2.x package, which we deliberately
    don't install (it would collide on the ``esm`` namespace with ``esm 3.2.4``).
    PXDesign's diffusion path doesn't actually use ESM; stubs that raise on
    call are enough to satisfy the import.
    """
    try:
        import esm as _esm
    except ImportError:
        return

    if not hasattr(_esm, "FastaBatchedDataset"):
        class _StubFastaBatchedDataset:  # pragma: no cover - never instantiated
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise NotImplementedError(
                    "fair-esm's FastaBatchedDataset is not available in this env; "
                    "PXDesign ESM-augmented paths are not supported here."
                )

        _esm.FastaBatchedDataset = _StubFastaBatchedDataset  # type: ignore[attr-defined]

    if not hasattr(_esm, "pretrained"):
        _pretrained = types.ModuleType("esm.pretrained")

        def _raise(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
            raise NotImplementedError(
                "fair-esm's pretrained loaders are not available in this env."
            )

        _pretrained.load_model_and_alphabet_local = _raise
        _pretrained.load_model_and_alphabet = _raise
        _esm.pretrained = _pretrained  # type: ignore[attr-defined]
        sys.modules.setdefault("esm.pretrained", _pretrained)


def _purge_protenix_module_cache() -> None:
    """Drop any cached ``protenix.*`` / ``runner.*`` / ``pxdbench.*`` / ``pxdesign.*``
    module from ``sys.modules``.

    The Protenix v1.0.4 tree (consumed by ``protenix_v2``) and the vendored
    Protenix v0.5.0+pxd ship the same top-level package name. If the v2 path
    has already populated ``sys.modules['protenix']``, the next ``import
    protenix.config`` would resolve to the wrong code. Incompatibility eviction
    drops the adapter object, but the module cache is process-global; clear it
    here so the subsequent ``sys.path.insert`` + import reads our vendored copy.
    """
    prefixes = ("protenix", "pxdbench", "pxdesign", "runner")
    for mod_name in list(sys.modules.keys()):
        if any(mod_name == p or mod_name.startswith(p + ".") for p in prefixes):
            del sys.modules[mod_name]


def _ensure_pxdesign_importable() -> None:
    """Insert the three vendor roots on ``sys.path`` and export env vars.

    Order matters: ``protenix-pxd`` first so its ``protenix/`` shadows any
    other on-disk protenix that might already be on the path (notably the
    in-place Protenix v1.0.4 the ``protenix_v2`` adapter consumes). We also
    purge any cached ``protenix.*`` modules in ``sys.modules`` so a previously
    loaded peer adapter's package cache doesn't shadow our vendored copy.
    """
    for root in (PROTENIX_PXD_VENDOR, PXDBENCH_VENDOR, PXDESIGN_VENDOR):
        if not root.is_dir():
            raise FileNotFoundError(
                f"PXDesign vendor tree missing: {root}. Re-vendor via Phase 5.1."
            )
    os.environ["PROTENIX_DATA_ROOT_DIR"] = PROTENIX_DATA_ROOT_DEFAULT
    # Remove any conflicting protenix tree from sys.path (eg the v1.0.4 tree
    # the protenix_v2 adapter inserts) so vendored protenix-pxd wins.
    for path_entry in list(sys.path):
        if path_entry.endswith("/Protenix") or "/Protenix/" in path_entry + "/":
            if path_entry != str(PROTENIX_PXD_VENDOR):
                try:
                    sys.path.remove(path_entry)
                except ValueError:
                    pass
    _purge_protenix_module_cache()
    # Prepend in reverse order so final ordering is pxdesign, pxdbench,
    # protenix-pxd, then the rest of sys.path.
    for root in (PROTENIX_PXD_VENDOR, PXDBENCH_VENDOR, PXDESIGN_VENDOR):
        p = str(root)
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    _stub_fair_esm_symbols()


# --------------------------------------------------------------------------- I/O models


class PXDesignInput(BaseModel):
    target_pdb: str = Field(..., description="Path to target structure (PDB or CIF).")
    target_chain: str = Field(..., min_length=1)
    binder_length: int = Field(..., ge=30, le=200)
    num_samples: int = Field(default=5, ge=1, le=256)
    num_steps: int = Field(default=400, ge=5, le=1000)
    seed: int | None = Field(default=None)
    temperature: float = Field(default=1.0)
    hotspots: list[int] = Field(default_factory=list)
    crop: str | None = Field(default=None)
    mode: str = Field(default="infer", pattern=r"^(infer|pipeline)$")
    top_k_designs: int = Field(default=4, ge=1, le=64)
    mpnn_temperature: float = Field(default=0.1, gt=0.0, le=5.0)
    af2_init_guess: bool = Field(default=True)
    preset: str = Field(default="preview", pattern=r"^(preview|extended)$")


class PXDesignDesign(BaseModel):
    candidate_id: str | None = None
    pdb_path: str | None = None
    cif_path: str
    score: float | None = None
    sequence: str
    binder_length: int


class PXDesignOutput(BaseModel):
    designs: list[PXDesignDesign]
    runtime_ms: int
    num_samples: int
    task_name: str | None = None
    seed: int | None = None


def _design_candidate_id(task_name: str, index: int, sequence: str, cif_path: Path) -> str:
    seed = f"{task_name}:{index}:{sequence}:{cif_path.name}"
    token = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"pxdesign:{task_name}:design_{index}:{token}"


@dataclass(frozen=True)
class _PipelineInvocation:
    job_outputs: Path
    task_name: str
    seed_val: int
    argv: list[str]


def _prepare_pipeline_invocation(params: PXDesignInput, job_dir: Path) -> _PipelineInvocation:
    target = Path(params.target_pdb).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"target_pdb not found: {target}")

    job_inputs = job_dir / "inputs"
    job_outputs = job_dir / "outputs"
    job_inputs.mkdir(parents=True, exist_ok=True)
    job_outputs.mkdir(parents=True, exist_ok=True)

    staged_target = job_inputs / target.name
    if staged_target.resolve() != target:
        shutil.copyfile(target, staged_target)

    task_name = f"design_{target.stem}_{params.target_chain}_{params.binder_length}aa"
    seed_val = int(params.seed) if params.seed is not None else 101

    chain_filter: dict[str, Any] = {"chain_id": [params.target_chain]}
    if params.crop:
        chain_filter["crop"] = {params.target_chain: params.crop}
    else:
        chain_filter["crop"] = {}
    hotspot: dict[str, list[int]] = {}
    if params.hotspots:
        hotspot[params.target_chain] = list(params.hotspots)
    task = {
        "name": task_name,
        "condition": {
            "structure_file": str(staged_target),
            "filter": chain_filter,
            "msa": {},
        },
        "hotspot": hotspot,
        "generation": [
            {
                "type": "protein",
                "length": int(params.binder_length),
                "count": 1,
            }
        ],
    }
    input_json_path = job_inputs / f"{task_name}.json"
    input_json_path.write_text(json.dumps([task], indent=2))

    argv = PXDesignAdapter._build_pipeline_argv(
        params=params,
        seed_val=seed_val,
        input_json_path=input_json_path,
        job_outputs=job_outputs,
    )
    return _PipelineInvocation(
        job_outputs=job_outputs,
        task_name=task_name,
        seed_val=seed_val,
        argv=argv,
    )


# --------------------------------------------------------------------------- pipeline env

def pipeline_env() -> Path:
    """Resolve the dedicated `pxdesign-pipeline-vpa` conda env (env var wins)."""
    explicit = os.environ.get("VPA_PXDESIGN_PIPELINE_ENV")
    if explicit:
        return Path(explicit).expanduser()
    if (PXDESIGN_PIPELINE_ENV_DEFAULT / "bin" / "python").is_file():
        return PXDESIGN_PIPELINE_ENV_DEFAULT
    for root in (
        Path.home() / ".conda" / "envs",
        Path("/opt/miniconda3/envs"),
        Path("/opt/conda/envs"),
    ):
        candidate = root / PXDESIGN_PIPELINE_ENV_NAME
        if (candidate / "bin" / "python").is_file():
            return candidate
    discovered = _discover_named_conda_env(PXDESIGN_PIPELINE_ENV_NAME)
    if discovered is not None:
        return discovered
    return PXDESIGN_PIPELINE_ENV_DEFAULT


def pipeline_python() -> Path:
    return pipeline_env() / "bin" / "python"


def _find_conda_exe() -> Path | None:
    env_var = os.environ.get("CONDA_EXE")
    extra = [Path(env_var)] if env_var else []
    return EnvLocator.find_conda_exe(extra)


def _discover_named_conda_env(env_name: str) -> Path | None:
    return EnvLocator.conda_env_path(
        env_name,
        final_default=None,
        conda_exe=_find_conda_exe(),
    )


def _pipeline_subprocess_env(seed: int | None) -> dict[str, str]:
    """Sanitised env for the pipeline runner — vpa's PYTHONPATH must not leak."""
    # The PXDesign vendor trees in this checkout are source-only snapshots
    # (no setup.py / pyproject.toml), so the dedicated sidecar cannot rely on
    # `pip install -e vendor/...`. Expose the same vendor roots the in-process
    # infer path uses, while still keeping the parent env's PYTHONPATH isolated.
    # JAX must not preallocate the whole GPU — pipeline shares VRAM with
    # downstream tools.
    return sanitized_python_env(
        seed=seed,
        prepend_bin=[pipeline_env() / "bin"],
        vendor_paths=[PXDESIGN_VENDOR, PXDBENCH_VENDOR, PROTENIX_PXD_VENDOR],
        extra={
            "PROTENIX_DATA_ROOT_DIR": PROTENIX_DATA_ROOT_DEFAULT,
            "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
                "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
            ),
        },
    )


def _pipeline_import_skip_reason(python: Path, timeout_s: int = 180) -> str | None:
    """Smoke-import the pipeline sidecar runtime before marking it available."""
    reason = smoke_import(
        label="PXDesign pipeline sidecar",
        python=python,
        code=(
            "import sys; "
            "assert sys.version_info >= (3, 11), 'Python >=3.11 required'; "
            "import msgpack; "
            "import vibeproteinagent.workers.sidecar_child; "
            "from vibeproteinagent.adapters.pxdesign.adapter "
            "import _ensure_pxdesign_importable; "
            "_ensure_pxdesign_importable(); "
            "import jax; "
            "import haiku; "
            "import optax; "
            "import colabdesign; "
            "from pxdesign.utils.pipeline import check_tool_weights; "
            "check_tool_weights(); "
            "print('ok')"
        ),
        env=_pipeline_subprocess_env(seed=None),
        timeout_s=timeout_s,
    )
    if reason is None:
        return None
    return (
        f"{reason}. "
        "Check Python >=3.11, JAX, dm-haiku, optax, ColabDesign, msgpack, "
        "PXDesign/PXDBench vendor imports, VPA sidecar install, and "
        "$TOOL_WEIGHTS_ROOT AF2 tool weights."
    )


def availability_skip_reason() -> str | None:
    """Return a human-readable reason if pipeline mode is unavailable, else None."""
    env = pipeline_env()
    py = pipeline_python()
    if not py.is_file():
        return (
            f"PXDesign pipeline env not present at {env} (no python at {py}). "
            f"Provision per vendor/pxdesign/UPSTREAM.md#pipeline-mode "
            f"(needs JAX + dm-haiku + optax + ColabDesign + AF2 params). "
            f"Set $VPA_PXDESIGN_PIPELINE_ENV to override the default path."
        )
    return _pipeline_import_skip_reason(py)


# --------------------------------------------------------------------------- helpers


# Three-letter -> one-letter amino-acid code for sequence extraction from CIFs.
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _extract_binder_sequence(cif_path: Path) -> str:
    """Best-effort: pull the binder (chain B) sequence from PXDesign's CIF.

    PXDesign always writes the binder as the last polymer chain. We parse
    `_entity_poly_seq` rows for entity_id 2 (the second polymer = binder
    when target is a single chain). Diffusion-only ``infer`` outputs can
    contain PXDesign pseudo-residues such as ``xpb`` instead of designed
    amino acids; if every binder monomer is unknown, return an empty string
    rather than a misleading ``XXXXX`` sequence. The adapter still produces
    a usable structure-only design record.
    """
    try:
        text = cif_path.read_text()
    except Exception:
        return ""
    seq: list[str] = []
    in_loop = False
    columns: list[str] = []
    # Walk a simple state machine over the entity_poly_seq loop.
    for line in text.splitlines():
        if line.startswith("loop_"):
            in_loop = False
            columns = []
            continue
        if line.startswith("_entity_poly_seq."):
            in_loop = True
            columns.append(line.strip())
            continue
        if in_loop:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                in_loop = False
                columns = []
                continue
            if len(parts) < len(columns):
                in_loop = False
                continue
            try:
                entity_id = parts[0]
                mon_id = parts[2]
            except IndexError:
                continue
            if entity_id != "2":
                continue
            seq.append(_AA3_TO_1.get(mon_id.upper(), "X"))
    if seq and all(aa == "X" for aa in seq):
        return ""
    return "".join(seq)


def _summary_score(summary: dict[str, Any]) -> float | None:
    """Pick a single scalar score from a PXDesign summary_confidence JSON.

    PXDesign writes various per-sample scalars (plddt, iptm, ptm,
    ranking_score, ...). We prefer ``ranking_score`` and fall back to
    ``iptm`` then ``plddt``.
    """
    for key in ("ranking_score", "iptm", "plddt"):
        v = summary.get(key)
        if v is None:
            continue
        if isinstance(v, list):
            if not v:
                continue
            v = v[0]
        if hasattr(v, "item"):
            try:
                return float(v.item())
            except Exception:
                continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def _summary_score_from_row(row: dict[str, str]) -> float | None:
    """Pick a single scalar score from a pipeline summary CSV row.

    Column priority follows ``trim_summary_df``'s renamed AF2 metrics:
    ``af2_iptm`` > ``af2_plddt`` > ``af2_ptm`` > ``Rg``. Falls back to None
    if no parseable numeric exists.
    """
    for key in ("af2_iptm", "af2_plddt", "af2_ptm", "ranking_score", "Rg"):
        raw = row.get(key)
        if raw is None or raw == "":
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class PXDesignAdapter(ToolAdapter):
    """Adapter for ByteDance's PXDesign target-conditioned binder generator."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        # weights_dir is the worker's hardcoded placeholder; we resolve via env.
        del config, weights_dir
        self.device = device
        self._pipeline_sidecar: PersistentPythonSidecar | None = None
        self.ckpt_path = _resolve_checkpoint()
        self.runner: Any | None = None

    def _ensure_infer_runner(self) -> None:
        """Lazy-load the heavy in-process PXDesign inference runner.

        Pipeline mode runs in a dedicated sidecar and should not pay the
        CUDA/DeepSpeed/model-load cost of the plain ``infer`` path.
        """
        if self.runner is not None:
            return
        _ensure_pxdesign_importable()

        # Resolve the three pxdesign internals we need.
        infer_mod = importlib.import_module("pxdesign.utils.infer")
        inference_mod = importlib.import_module("pxdesign.runner.inference")
        inputs_mod = importlib.import_module("pxdesign.utils.inputs")
        seed_mod = importlib.import_module("protenix.utils.seed")

        self._get_configs = infer_mod.get_configs
        self._convert_to_bioassembly_dict = infer_mod.convert_to_bioassembly_dict
        self._download_inference_cache = infer_mod.download_inference_cache
        self._process_input_file = inputs_mod.process_input_file
        self._seed_everything = seed_mod.seed_everything
        self._InferenceRunner = inference_mod.InferenceRunner

        # Build a configs object good enough for env / model / checkpoint init.
        # We deliberately do NOT bind input_json_path or dump_dir — init_data is
        # deferred to run() so we can rebind per call.
        argv = [
            "--dump_dir", str(_warm_load_placeholder_dir()),
            "--load_checkpoint_dir", str(PXDESIGN_CKPT_DIR),
            "--N_sample", "1",
            "--N_step", "5",
            "--eta_type", "const",
            "--eta_min", "2.5",
            "--eta_max", "2.5",
            "--dtype", "bf16",
            "--seeds", "101",
        ]
        self._warm_configs = self._get_configs(argv)

        # Construct InferenceRunner skipping init_data — we'll re-run
        # init_dumper + init_data per call.
        runner = self._InferenceRunner.__new__(self._InferenceRunner)
        runner.configs = self._warm_configs
        runner.init_env()
        runner.init_basics()
        runner.init_model()
        runner.load_checkpoint()
        # init_dumper and init_data are deferred.
        self.runner = runner

    @torch.no_grad()
    def run(self, params: PXDesignInput, ctx: AdapterRunContext) -> PXDesignOutput:
        if params.mode == "pipeline":
            return self._run_pipeline(params, ctx)
        return self._run_infer(params, ctx)

    @torch.no_grad()
    def _run_infer(self, params: PXDesignInput, ctx: AdapterRunContext) -> PXDesignOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)
        self._ensure_infer_runner()
        assert self.runner is not None

        target = Path(params.target_pdb).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"target_pdb not found: {target}")

        job_inputs = ctx.job_dir / "inputs"
        job_outputs = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)

        task_name = f"design_{target.stem}_{params.target_chain}_{params.binder_length}aa"
        seed_val = int(params.seed) if params.seed is not None else 101

        # Build a single-task PXDesign input JSON inline (skips the YAML route).
        chain_filter: dict[str, Any] = {"chain_id": [params.target_chain]}
        if params.crop:
            chain_filter["crop"] = {params.target_chain: params.crop}
        else:
            chain_filter["crop"] = {}
        hotspot: dict[str, list[int]] = {}
        if params.hotspots:
            hotspot[params.target_chain] = list(params.hotspots)
        task = {
            "name": task_name,
            "condition": {
                "structure_file": str(target),
                "filter": chain_filter,
                "msa": {},
            },
            "hotspot": hotspot,
            "generation": [
                {
                    "type": "protein",
                    "length": int(params.binder_length),
                    "count": 1,
                }
            ],
        }
        input_json_path = job_inputs / f"{task_name}.json"
        input_json_path.write_text(json.dumps([task], indent=2))
        ctx.emit_progress(0.10)

        # Rebind the warm runner to this call's I/O + diffusion knobs.
        cfg = self.runner.configs
        cfg["input_json_path"] = str(input_json_path)
        cfg["dump_dir"] = str(job_outputs)
        cfg["seeds"] = [seed_val]
        cfg["use_msa"] = False
        # num_workers > 0 spawns DataLoader workers that try to pickle
        # PXDesign's collate-fn lambda; force single-process loading.
        cfg["num_workers"] = 0
        # diffusion knobs
        cfg.sample_diffusion.N_sample = int(params.num_samples)
        cfg.sample_diffusion.N_step = int(params.num_steps)

        # Update the runner's mirror of dump_dir / error_dir (init_basics set
        # them on the warm placeholder). DataDumper is constructor-bound so
        # we re-create it via init_dumper.
        self.runner.dump_dir = str(job_outputs)
        self.runner.error_dir = str(job_outputs / "ERR")
        Path(self.runner.error_dir).mkdir(parents=True, exist_ok=True)

        # Make sure checkpoints + CCD cache exist (no-op if everything is in
        # place; downloads heavy files otherwise — Phase 5.1 already placed
        # them under ``$PXDESIGN_CKPT_DIR``).
        self._download_inference_cache(cfg)

        # Pre-convert the target structure to a bioassembly pkl.gz (mirrors
        # PXDesign's `main()`); the dataloader expects the pre-converted file.
        bio_inputs: list[dict[str, Any]] = [task]
        for x in bio_inputs:
            self._convert_to_bioassembly_dict(x, str(job_outputs))
        bio_input_path = job_outputs / "input_tasks.json"
        bio_input_path.write_text(json.dumps(bio_inputs))
        cfg["input_json_path"] = str(bio_input_path)
        ctx.emit_progress(0.20)

        # Re-bind dataset + dumper for this call.
        self.runner.init_dumper()
        self.runner.init_data()
        ctx.emit_progress(0.30)

        # The heavy lift: diffusion sampling.
        self._seed_everything(seed=seed_val, deterministic=False)
        self.runner._inference(seed_val)
        ctx.emit_progress(0.95)

        designs = self._collect_outputs(job_outputs, task_name, seed_val,
                                        params.binder_length)
        if not designs:
            raise RuntimeError(
                f"PXDesign produced no designs under {job_outputs}. "
                f"Check {self.runner.error_dir} for per-sample errors."
            )

        ctx.emit_progress(1.0)
        runtime_ms = int((time.time() - t0) * 1000)
        return PXDesignOutput(
            designs=designs,
            runtime_ms=runtime_ms,
            num_samples=len(designs),
            task_name=task_name,
            seed=seed_val,
        )

    def _run_pipeline(self, params: PXDesignInput, ctx: AdapterRunContext) -> PXDesignOutput:
        """Persistent-sidecar path for real-sequence design.

        The dedicated env owns JAX + ColabDesign + AF2 params; vpa stays clean.
        If the env isn't present, raise
        ``DependencyError`` with an actionable provision message — same shape
        the daemon/CLI surfaces as ``E_DEPENDENCY``.
        """
        t0 = time.time()
        ctx.emit_progress(0.02)

        skip = availability_skip_reason()
        if skip is not None:
            raise DependencyError(
                skip,
                details={
                    "env": str(pipeline_env()),
                    "bootstrap": "vendor/pxdesign/UPSTREAM.md#pipeline-mode",
                },
            )

        invocation = _prepare_pipeline_invocation(params, ctx.job_dir)
        job_outputs = invocation.job_outputs
        task_name = invocation.task_name
        seed_val = invocation.seed_val
        argv = invocation.argv
        ctx.emit_progress(0.08)

        ctx.emit_progress(0.12)
        if params.preset == "preview":
            _run_preview_pipeline_until_artifacts(
                argv=argv,
                cwd=ctx.job_dir,
                job_outputs=job_outputs,
                task_name=task_name,
                seed=params.seed,
                mpnn_temperature=params.mpnn_temperature,
                timeout_s=8 * 3600,
            )
        else:
            try:
                self._pipeline_sidecar_for(params.seed).call(
                    _PIPELINE_SIDECAR_CALL,
                    {"argv": argv, "cwd": str(ctx.job_dir)},
                    timeout_s=8 * 3600,
                )
            except VPAError as exc:
                raise RuntimeError(
                    f"PXDesign pipeline runner failed. "
                    f"See {job_outputs} for partial outputs and logs."
                ) from exc

        top_k = int(params.top_k_designs)
        designs = self._collect_pipeline_outputs(
            job_outputs, task_name, params.binder_length, top_k,
        )
        if not designs and params.preset == "preview":
            designs = _collect_preview_pipeline_outputs(
                job_outputs, task_name, params.binder_length, top_k,
            )
        if not designs:
            raise RuntimeError(
                f"PXDesign pipeline produced no designs under {job_outputs}. "
                f"Check the per-run logs for filter rejections."
            )

        ctx.emit_progress(1.0)
        runtime_ms = int((time.time() - t0) * 1000)
        return PXDesignOutput(
            designs=designs,
            runtime_ms=runtime_ms,
            num_samples=len(designs),
            task_name=task_name,
            seed=seed_val,
        )

    def _pipeline_sidecar_for(self, seed: int | None) -> PersistentPythonSidecar:
        sidecar = getattr(self, "_pipeline_sidecar", None)
        if sidecar is None:
            spec = SidecarRuntimeSpec(
                python=pipeline_python(),
                env=_pipeline_subprocess_env(seed),
                name=PXDESIGN_PIPELINE_ENV_NAME,
                run_target=_PIPELINE_SIDECAR_CALL,
                timeout_s=60.0,
            )
            sidecar = spec.spawn(PersistentPythonSidecar)
            self._pipeline_sidecar = sidecar
        return sidecar

    @staticmethod
    def _build_pipeline_argv(
        *,
        params: PXDesignInput,
        seed_val: int,
        input_json_path: Path,
        job_outputs: Path,
    ) -> list[str]:
        """Construct the argv for ``pxdesign.runner.pipeline``.

        Phase 20 (Pi2 + Pi3): threads ``mpnn_temperature`` and ``preset``
        through to the upstream CLI. ``preset`` wins over ``af2_init_guess``
        when both are set; when ``preset`` is at its default ("preview")
        ``af2_init_guess=False`` falls back to the legacy "custom" preset
        for callers that pre-date the preset enum.
        """
        top_k = int(params.top_k_designs)
        preset = params.preset
        if preset == "preview" and not params.af2_init_guess:
            preset = "custom"
        cmd = [
            "--dump_dir", str(job_outputs),
            "--load_checkpoint_dir", str(PXDESIGN_CKPT_DIR),
            "--input_json_path", str(input_json_path),
            "--N_sample", str(int(params.num_samples)),
            "--N_step", str(int(params.num_steps)),
            "--eta_type", "const",
            "--eta_min", "2.5",
            "--eta_max", "2.5",
            "--dtype", "bf16",
            "--seeds", str(seed_val),
            "--preset", preset,
            "--N_max_runs", "1",
            "--min_total_return", str(top_k),
            "--max_success_return", str(top_k),
            "--use_msa", "false",
            "--eval.binder.eval_protenix", "false",
            "--eval.binder.eval_protenix_mini", "false",
        ]
        if preset == "preview":
            cmd.extend(
                [
                    "--eval.binder.eval_complex", "false",
                    "--eval.binder.eval_binder_monomer", "false",
                ]
            )
        cmd.extend(
            [
                "--eval.binder.tools.mpnn.temperature",
                str(float(params.mpnn_temperature)),
            ]
        )
        return cmd

    @staticmethod
    def _build_pipeline_cmd(
        *,
        params: PXDesignInput,
        seed_val: int,
        input_json_path: Path,
        job_outputs: Path,
    ) -> list[str]:
        """Construct the legacy cold-process command for compatibility tests."""
        return [
            str(pipeline_python()),
            "-m",
            "pxdesign.runner.pipeline",
            *PXDesignAdapter._build_pipeline_argv(
                params=params,
                seed_val=seed_val,
                input_json_path=input_json_path,
                job_outputs=job_outputs,
            ),
        ]

    @staticmethod
    def _collect_pipeline_outputs(
        dump_dir: Path, task_name: str, binder_length: int, top_k: int,
    ) -> list[PXDesignDesign]:
        """Parse the pipeline mode output layout.

        Layout:
          <dump_dir>/design_outputs/<task_name>/summary.csv
          <dump_dir>/design_outputs/<task_name>/<orig|af2>/rank_<k>.cif
        The summary CSV's `sequence` column carries the designed AAs (real
        single-letter codes, not 'X' placeholders), and `chosen_struct_path`
        points to the chosen CIF relative to the task dir.
        """
        task_dir = dump_dir / "design_outputs" / task_name
        summary_csv = task_dir / "summary.csv"
        if not summary_csv.is_file():
            return []

        designs: list[PXDesignDesign] = []
        try:
            with summary_csv.open() as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception:
            return []

        rows = rows[:top_k]
        for idx, row in enumerate(rows, start=1):
            seq = (row.get("sequence") or "").strip()
            chosen_rel = (row.get("chosen_struct_path") or "").strip()
            cif_path = task_dir / chosen_rel if chosen_rel else None
            if not cif_path or not cif_path.is_file():
                continue
            score = _summary_score_from_row(row)
            designs.append(
                PXDesignDesign(
                    candidate_id=_design_candidate_id(task_name, idx, seq, cif_path),
                    pdb_path=None,
                    cif_path=str(cif_path),
                    score=score,
                    sequence=seq,
                    binder_length=binder_length,
                )
            )
        return designs

    @staticmethod
    def _collect_outputs(
        dump_dir: Path, task_name: str, seed: int, binder_length: int,
    ) -> list[PXDesignDesign]:
        """Walk the PXDesign DataDumper layout and assemble PXDesignDesign records.

        Layout: ``<dump_dir>/<task_name>/seed_<seed>/predictions/<task_name>_sample_<i>.cif``
        plus optional summary JSONs.
        """
        seed_dir = dump_dir / task_name / f"seed_{seed}"
        success_file = seed_dir / "SUCCESS_FILE"
        predictions_dir = seed_dir / "predictions"
        if not predictions_dir.is_dir():
            return []
        results: list[PXDesignDesign] = []
        cifs = sorted(predictions_dir.glob(f"{task_name}_sample_*.cif"))
        for idx, cif in enumerate(cifs, start=1):
            try:
                rank_str = cif.stem.rsplit("_", 1)[-1]
                rank = int(rank_str)
            except ValueError:
                continue
            summary_path = predictions_dir / (
                f"{task_name}_summary_confidence_sample_{rank}.json"
            )
            score: float | None = None
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text())
                    if isinstance(summary, dict):
                        score = _summary_score(summary)
                except Exception:
                    score = None
            seq = _extract_binder_sequence(cif)
            results.append(
                PXDesignDesign(
                    candidate_id=_design_candidate_id(task_name, idx, seq, cif),
                    pdb_path=None,
                    cif_path=str(cif),
                    score=score,
                    sequence=seq,
                    binder_length=binder_length,
                )
            )
        # Best-effort success flag: don't fail collection if missing — _inference
        # writes per-sample CIFs even when SUCCESS_FILE is delayed.
        if not results and success_file.is_file():
            return []
        return results

    def estimate_vram(self, params: PXDesignInput) -> int:
        return 24 * 1024

    def healthcheck(self) -> bool:
        """Warm-readiness includes the PARTIAL pipeline availability probe.

        The worker only emits warm warnings when ``healthcheck()`` is false.
        Include ``mode=pipeline`` availability here so a warm call can surface
        the actionable sidecar-env bootstrap reason instead of silently
        reporting healthy just because ``mode=infer`` is available.
        """
        base_ok = self.ckpt_path.is_file() and PXDESIGN_VENDOR.is_dir()
        return base_ok and availability_skip_reason() is None

    def unload(self) -> None:
        sidecar = getattr(self, "_pipeline_sidecar", None)
        if sidecar is not None:
            sidecar.close()
            self._pipeline_sidecar = None
        if hasattr(self, "runner"):
            try:
                del self.runner.model
            except Exception:
                pass
            try:
                del self.runner
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _distributed_pipeline_env(params: PXDesignInput, ctx: Any) -> dict[str, str]:
    env = _pipeline_subprocess_env(params.seed)
    env.update(
        {
            "MASTER_ADDR": str(ctx.master_addr),
            "MASTER_PORT": str(ctx.master_port),
            "RANK": str(ctx.rank),
            "WORLD_SIZE": str(ctx.world_size),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": str(ctx.local_world_size),
        }
    )
    return env


def run_distributed(params: PXDesignInput, ctx: Any) -> PXDesignOutput | dict[str, Any]:
    """Distributed PXDesign pipeline entrypoint used by multi-worker dispatch."""
    if params.mode != "pipeline":
        raise InvalidParamsError(
            "pxdesign multi-GPU dispatch requires mode='pipeline'",
            details={"mode": params.mode},
        )
    if int(params.num_samples) < int(ctx.world_size):
        raise InvalidParamsError(
            "pxdesign multi-GPU pipeline requires num_samples >= world_size",
            details={"num_samples": params.num_samples, "world_size": ctx.world_size},
        )
    if int(params.num_samples) % int(ctx.world_size) != 0:
        raise InvalidParamsError(
            "pxdesign multi-GPU pipeline requires num_samples divisible by world_size",
            details={"num_samples": params.num_samples, "world_size": ctx.world_size},
        )
    skip = availability_skip_reason()
    if skip is not None:
        raise DependencyError(
            skip,
            details={
                "env": str(pipeline_env()),
                "bootstrap": "vendor/pxdesign/UPSTREAM.md#pipeline-mode",
            },
        )

    started = time.time()
    invocation = _prepare_pipeline_invocation(params, ctx.job_dir)
    sidecar = PersistentPythonSidecar(
        python=pipeline_python(),
        env=_distributed_pipeline_env(params, ctx),
        name=f"{PXDESIGN_PIPELINE_ENV_NAME}[rank{ctx.rank}]",
        startup_timeout_s=60.0,
    )
    try:
        sidecar.call(
            _PIPELINE_SIDECAR_CALL,
            {"argv": invocation.argv, "cwd": str(ctx.job_dir)},
            timeout_s=8 * 3600,
        )
    finally:
        sidecar.close()

    if int(ctx.rank) != 0:
        return {"rank_completed": True}

    top_k = int(params.top_k_designs)
    designs = PXDesignAdapter._collect_pipeline_outputs(
        invocation.job_outputs,
        invocation.task_name,
        params.binder_length,
        top_k,
    )
    if not designs and params.preset == "preview":
        designs = _collect_preview_pipeline_outputs(
            invocation.job_outputs,
            invocation.task_name,
            params.binder_length,
            top_k,
        )
    if not designs:
        raise RuntimeError(
            f"PXDesign distributed pipeline produced no designs under "
            f"{invocation.job_outputs}."
        )
    return PXDesignOutput(
        designs=designs,
        runtime_ms=int((time.time() - started) * 1000),
        num_samples=len(designs),
        task_name=invocation.task_name,
        seed=invocation.seed_val,
    )


def _warm_load_placeholder_dir() -> Path:
    """Return a stable scratch dir for init_basics during warm load.

    init_basics creates the dump_dir; we hand it a tmpdir under the adapter's
    workspace ("$VPA_HOME/pxdesign_warm/") so that it doesn't collide with
    a per-job ctx.job_dir before the first run() arrives.
    """
    home = os.environ.get("VPA_HOME") or os.path.expanduser("~/.vibeproteinagent")
    p = Path(home).expanduser() / "pxdesign_warm"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _preview_artifacts_ready(job_outputs: Path, task_name: str) -> bool:
    """Return true once preview mode has emitted final sequence-design artifacts."""
    task_dir = job_outputs / "design_outputs" / task_name
    summary_csv = task_dir / "summary.csv"
    if not summary_csv.is_file():
        return False
    try:
        with summary_csv.open() as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return False
    for row in rows:
        seq = (row.get("sequence") or "").strip()
        chosen_rel = (row.get("chosen_struct_path") or "").strip()
        if seq and "X" not in seq and chosen_rel and (task_dir / chosen_rel).is_file():
            return True
    return False


def _raw_preview_cif(job_outputs: Path, task_name: str) -> Path | None:
    for seed_dir in sorted(job_outputs.glob(f"global_run_*/{task_name}/seed_*")):
        if not seed_dir.joinpath("SUCCESS_FILE").is_file():
            continue
        predictions = seed_dir / "predictions"
        cifs = sorted(predictions.glob(f"{task_name}_sample_*.cif"))
        if cifs:
            return cifs[0]
    return None


def _run_preview_pipeline_until_artifacts(
    *,
    argv: list[str],
    cwd: Path,
    job_outputs: Path,
    task_name: str,
    seed: int | None,
    mpnn_temperature: float,
    timeout_s: int,
) -> None:
    """Run preview pipeline and return as soon as successful artifacts exist.

    PXDesign's preview path can finish the real generation work and write
    SUCCESS_FILE/CIF artifacts while optional post-run bookkeeping keeps the
    process alive. For the local-first VPA contract, the adapter should return
    the generated designs once the product artifact contract is satisfied.
    """
    log_path = job_outputs / "pipeline_preview.log"
    cmd = [str(pipeline_python()), "-m", "pxdesign.runner.pipeline", *argv]
    env = _pipeline_subprocess_env(seed=seed)
    raw_ready_since: float | None = None
    sequence_grace_s = float(
        os.environ.get("VPA_PXDESIGN_PREVIEW_SEQUENCE_GRACE_S", "30")
    )
    with log_path.open("w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                if _preview_artifacts_ready(job_outputs, task_name):
                    _terminate_process_group(proc)
                    return
                raw_cif = _raw_preview_cif(job_outputs, task_name)
                if raw_cif is not None:
                    if raw_ready_since is None:
                        raw_ready_since = time.monotonic()
                    elif time.monotonic() - raw_ready_since >= sequence_grace_s:
                        _terminate_process_group(proc)
                        _materialize_preview_sequence_outputs(
                            raw_cif=raw_cif,
                            job_outputs=job_outputs,
                            task_name=task_name,
                            seed=seed,
                            mpnn_temperature=mpnn_temperature,
                        )
                        return
                rc = proc.poll()
                if rc is not None:
                    if rc == 0:
                        if _preview_artifacts_ready(job_outputs, task_name):
                            return
                        raw_cif = _raw_preview_cif(job_outputs, task_name)
                        if raw_cif is not None:
                            _materialize_preview_sequence_outputs(
                                raw_cif=raw_cif,
                                job_outputs=job_outputs,
                                task_name=task_name,
                                seed=seed,
                                mpnn_temperature=mpnn_temperature,
                            )
                            return
                        raise RuntimeError(
                            f"PXDesign preview pipeline exited successfully, but "
                            f"no final ProteinMPNN sequence-design outputs "
                            f"are required. See {log_path}."
                        )
                    raise RuntimeError(
                        f"PXDesign preview pipeline exited with status {rc} "
                        f"before emitting design artifacts. See {log_path}."
                    )
                if time.monotonic() > deadline:
                    _terminate_process_group(proc, kill=True)
                    raise TimeoutError(
                        f"PXDesign preview pipeline timed out before emitting "
                        f"design artifacts. See {log_path}."
                    )
                time.sleep(2.0)
        except BaseException:
            if proc.poll() is None:
                _terminate_process_group(proc, kill=True)
            raise


def _materialize_preview_sequence_outputs(
    *,
    raw_cif: Path,
    job_outputs: Path,
    task_name: str,
    seed: int | None,
    mpnn_temperature: float,
) -> None:
    env = _pipeline_subprocess_env(seed=seed)
    env["JAX_PLATFORMS"] = "cpu"
    log_path = job_outputs / "pipeline_preview_fallback.log"
    cmd = [
        str(pipeline_python()),
        "-m",
        "vibeproteinagent.adapters.pxdesign._preview_fallback",
        "--raw-cif",
        str(raw_cif),
        "--task-name",
        task_name,
        "--output-root",
        str(job_outputs),
        "--temperature",
        str(float(mpnn_temperature)),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    with log_path.open("w") as log_fh:
        out = subprocess.run(
            cmd,
            check=False,
            cwd=str(job_outputs),
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
    if out.returncode != 0:
        raise RuntimeError(
            f"PXDesign preview ProteinMPNN fallback failed with status "
            f"{out.returncode}. See {log_path}."
        )


def _terminate_process_group(proc: subprocess.Popen[Any], *, kill: bool = False) -> None:
    sig = signal.SIGKILL if kill else signal.SIGTERM
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill() if kill else proc.terminate()
        except OSError:
            return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if not kill:
            _terminate_process_group(proc, kill=True)


def _collect_preview_pipeline_outputs(
    dump_dir: Path,
    task_name: str,
    binder_length: int,
    top_k: int,
) -> list[PXDesignDesign]:
    """Collect raw preview artifacts when summary.csv is not produced."""
    designs: list[PXDesignDesign] = []
    for seed_dir in sorted(dump_dir.glob(f"global_run_*/{task_name}/seed_*")):
        predictions = seed_dir / "predictions"
        if not predictions.is_dir():
            continue
        for cif in sorted(predictions.glob(f"{task_name}_sample_*.cif")):
            seq = _extract_binder_sequence(cif)
            if not seq:
                continue
            designs.append(
                PXDesignDesign(
                    candidate_id=_design_candidate_id(
                        task_name, len(designs) + 1, seq, cif
                    ),
                    pdb_path=None,
                    cif_path=str(cif),
                    score=None,
                    sequence=seq,
                    binder_length=binder_length,
                )
            )
            if len(designs) >= top_k:
                return designs
    return designs


__all__ = [
    "PROTENIX_PXD_VENDOR",
    "PXDBENCH_VENDOR",
    "PXDESIGN_CKPT_DIR",
    "PXDESIGN_MODEL_NAME",
    "PXDESIGN_PIPELINE_ENV_DEFAULT",
    "PXDESIGN_PIPELINE_ENV_NAME",
    "PXDESIGN_VENDOR",
    "PXDesignAdapter",
    "PXDesignDesign",
    "PXDesignInput",
    "PXDesignOutput",
    "availability_skip_reason",
    "pipeline_env",
    "pipeline_python",
    "run_distributed",
]
