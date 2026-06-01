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
"""
from __future__ import annotations

import csv
import hashlib
import inspect
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
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.canonical_tools import candidate_capability_flags
from vibeproteinagent.agents.planning.candidate import stable_candidate_id
from vibeproteinagent.adapters.runtime import EnvLocator, sanitized_python_env, smoke_import
from vibeproteinagent.core.errors import DependencyError, VPAError

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

# Dedicated conda env for the diffusion + AF2-IG + ProteinMPNN generation
# pipeline. JAX + dm-haiku + optax + ColabDesign cannot share vpa's process:
# JAX's bundled cuDNN/CUBLAS/nccl wheels would override torch's bundled CUDA
# libs, so the adapter launches the vendor pipeline as a subprocess in this
# env and stops it once raw diffusion artifacts are ready.
PXDESIGN_PIPELINE_ENV_DEFAULT = Path("/opt/conda/envs/pxdesign-pipeline-vpa")
PXDESIGN_PIPELINE_ENV_NAME = "pxdesign-pipeline-vpa"


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
    target_pdb: str | None = Field(None, description="Path to target structure (PDB or CIF).")
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    target_chain: str = Field(
        "A",
        min_length=1,
        description=(
            "Target chain to design against. Defaults to A when the user did "
            "not specify a chain; override for multimer targets or a specific "
            "binding region."
        ),
    )
    binder_length: int = Field(
        60,
        ge=30,
        le=200,
        description=(
            "Desired binder length. Defaults to a bounded 60 aa starting point "
            "when no explicit length is requested."
        ),
    )
    num_samples: int = Field(default=5, ge=1, le=256)
    num_steps: int = Field(default=400, ge=5, le=1000)
    seed: int | None = Field(default=None)
    temperature: float = Field(default=1.0)
    hotspots: list[int] = Field(default_factory=list)
    crop: str | None = Field(default=None)
    top_k_designs: int = Field(default=4, ge=1, le=64)
    mpnn_temperature: float = Field(default=0.1, gt=0.0, le=5.0)

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_target(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        candidates = data.get("candidates")
        if not candidates:
            return data
        canonical = [_is_canonical_candidate(item) for item in candidates]
        if any(canonical) and not all(canonical):
            raise ValueError(
                "Mixed canonical and legacy candidate items are not supported. "
                "Pass either all canonical Candidate[] or all legacy items."
            )
        if all(canonical):
            structure_path = (
                candidates[0].get("structure_path")
                or candidates[0].get("full_structure_path")
                or candidates[0].get("backbone_path")
            )
            if not structure_path:
                raise ValueError(
                    "pxdesign canonical Candidate[] input requires "
                    "structure_path, full_structure_path, or backbone_path"
                )
            out = dict(data)
            out.setdefault("target_pdb", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _require_target_pdb(self) -> PXDesignInput:
        if not self.target_pdb:
            raise ValueError("pxdesign requires target_pdb")
        return self


class PXDesignDesign(BaseModel):
    candidate_id: str | None = None
    pdb_path: str | None = None
    cif_path: str
    score: float | None = None
    sequence: str
    binder_length: int


class PXDesignOutput(BaseModel):
    designs: list[PXDesignDesign]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
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
    input_structure_state: dict[str, Any]


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
    structure_file, input_structure_state = _prepare_pxdesign_structure_file(
        staged_target,
        target_chain=params.target_chain,
    )
    input_structure_state["defaulted_inputs"] = _pxdesign_defaulted_inputs(params)

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
            "structure_file": str(structure_file),
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
        input_structure_state=input_structure_state,
    )


def _prepare_pxdesign_structure_file(
    staged_target: Path,
    *,
    target_chain: str,
) -> tuple[Path, dict[str, Any]]:
    """Return the structure path passed to PXDesign's JSON condition."""

    state: dict[str, Any] = {
        "original_structure_file": str(staged_target),
        "target_chain": target_chain,
        "target_chain_default_policy": "defaults_to_A_when_unspecified",
        "defaulted_inputs": {},
        "converted_to_cif": False,
        "pipeline_structure_file": str(staged_target),
    }
    if staged_target.suffix.lower() not in {".pdb", ".ent"}:
        _preflight_structure_chain(staged_target, target_chain=target_chain)
        state["conversion"] = "none"
        return staged_target, state

    cif_path = staged_target.with_suffix(".cif")
    _convert_pdb_to_mmcif(staged_target, cif_path, target_chain=target_chain)
    state.update(
        {
            "converted_to_cif": True,
            "pipeline_structure_file": str(cif_path),
            "conversion": "biopython_pdb_to_mmcif",
        }
    )
    return cif_path, state


def _pxdesign_defaulted_inputs(params: PXDesignInput) -> dict[str, dict[str, Any]]:
    fields_set = getattr(params, "model_fields_set", set())
    defaults: dict[str, dict[str, Any]] = {}
    if "target_chain" not in fields_set:
        defaults["target_chain"] = {
            "value": params.target_chain,
            "reason": "user did not specify chain; conventional default",
        }
    if "binder_length" not in fields_set:
        defaults["binder_length"] = {
            "value": params.binder_length,
            "reason": "user did not specify binder length; bounded compact-binder default",
        }
    return defaults


def _preflight_structure_chain(structure_path: Path, *, target_chain: str) -> None:
    suffix = structure_path.suffix.lower()
    try:
        if suffix in {".cif", ".mmcif"}:
            from Bio.PDB import MMCIFParser

            structure = MMCIFParser(QUIET=True).get_structure("target", str(structure_path))
        else:
            return
        chain_ids = sorted({chain.id for model in structure for chain in model})
        if target_chain not in chain_ids:
            raise ValueError(
                f"target_chain {target_chain!r} not present in structure chains {chain_ids!r}"
            )
    except Exception as exc:
        raise VPAError(
            "PXDesign target structure failed chain preflight before pipeline execution",
            details={
                "target_pdb": str(structure_path),
                "target_chain": target_chain,
                "exception_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc


def _convert_pdb_to_mmcif(
    pdb_path: Path,
    cif_path: Path,
    *,
    target_chain: str,
) -> None:
    try:
        from Bio.PDB import MMCIFIO, PDBParser
    except Exception as exc:  # pragma: no cover - dependency is present in vpa env
        raise VPAError(
            "PXDesign requires Bio.PDB to normalize PDB targets to mmCIF",
            details={
                "target_pdb": str(pdb_path),
                "target_chain": target_chain,
                "exception_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc

    try:
        structure = PDBParser(QUIET=True).get_structure("target", str(pdb_path))
        chain_ids = sorted({chain.id for model in structure for chain in model})
        if target_chain not in chain_ids:
            raise ValueError(
                f"target_chain {target_chain!r} not present in PDB chains {chain_ids!r}"
            )
        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(cif_path))
    except Exception as exc:
        raise VPAError(
            "PXDesign could not normalize target PDB to mmCIF before pipeline execution",
            details={
                "target_pdb": str(pdb_path),
                "target_chain": target_chain,
                "output_cif": str(cif_path),
                "exception_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc


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
    # (no setup.py / pyproject.toml), so the dedicated subprocess cannot rely on
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
    """Smoke-import the pipeline subprocess runtime before marking it available."""
    reason = smoke_import(
        label="PXDesign pipeline subprocess",
        python=python,
        code=(
            "import sys; "
            "assert sys.version_info >= (3, 11), 'Python >=3.11 required'; "
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
        "Check Python >=3.11, JAX, dm-haiku, optax, ColabDesign, "
        "PXDesign/PXDBench vendor imports, VPA source import, and "
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
        self.ckpt_path = _resolve_checkpoint()

    @torch.no_grad()
    def run(self, params: PXDesignInput, ctx: AdapterRunContext) -> PXDesignOutput:
        return self._run_pipeline(params, ctx)

    def _run_pipeline(self, params: PXDesignInput, ctx: AdapterRunContext) -> PXDesignOutput:
        """Subprocess path for real-sequence design.

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

        try:
            invocation = _prepare_pipeline_invocation(params, ctx.job_dir)
        except Exception as exc:
            job_outputs = ctx.job_dir / "outputs"
            job_outputs.mkdir(parents=True, exist_ok=True)
            stdout_path = job_outputs / "pipeline_preview.stdout.log"
            stderr_path = job_outputs / "pipeline_preview.stderr.log"
            stdout_path.touch(exist_ok=True)
            stderr_path.write_text(str(exc))
            command = [str(pipeline_python()), "-m", "pxdesign.runner.pipeline"]
            input_structure_state = {
                "original_structure_file": str(params.target_pdb),
                "target_chain": params.target_chain,
                "target_chain_default_policy": "defaults_to_A_when_unspecified",
                "converted_to_cif": False,
                "pipeline_structure_file": str(params.target_pdb),
                "defaulted_inputs": _pxdesign_defaulted_inputs(params),
            }
            debug_path = job_outputs / "pxdesign_failure_debug.json"
            details = _write_pxdesign_failure_debug(
                debug_path=debug_path,
                command=command,
                cwd=ctx.job_dir,
                returncode=None,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=_pipeline_subprocess_env(seed=params.seed),
                ckpt_path=getattr(
                    self,
                    "ckpt_path",
                    PXDESIGN_CKPT_DIR / f"{PXDESIGN_MODEL_NAME}.pt",
                ),
                job_outputs=job_outputs,
                task_name="preflight",
                preflight_result={
                    "stage": "prepare_pipeline_invocation",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "availability_skip_reason": availability_skip_reason(),
                },
                input_structure_state=input_structure_state,
            )
            raise VPAError(
                f"PXDesign input preflight failed before pipeline execution. See {debug_path}.",
                details=details,
            ) from exc
        job_outputs = invocation.job_outputs
        task_name = invocation.task_name
        seed_val = invocation.seed_val
        argv = invocation.argv
        input_structure_state = invocation.input_structure_state
        ckpt_path = getattr(
            self,
            "ckpt_path",
            PXDESIGN_CKPT_DIR / f"{PXDESIGN_MODEL_NAME}.pt",
        )
        ctx.emit_progress(0.08)

        ctx.emit_progress(0.12)
        preview_kwargs: dict[str, Any] = {
            "argv": argv,
            "cwd": ctx.job_dir,
            "job_outputs": job_outputs,
            "task_name": task_name,
            "seed": params.seed,
            "mpnn_temperature": params.mpnn_temperature,
            "timeout_s": 8 * 3600,
        }
        preview_signature = inspect.signature(_run_preview_pipeline_until_artifacts)
        if "ckpt_path" in preview_signature.parameters:
            preview_kwargs["ckpt_path"] = ckpt_path
        if "input_structure_state" in preview_signature.parameters:
            preview_kwargs["input_structure_state"] = input_structure_state
        _run_preview_pipeline_until_artifacts(**preview_kwargs)

        top_k = int(params.top_k_designs)
        designs = self._collect_pipeline_outputs(
            job_outputs, task_name, params.binder_length, top_k,
        )
        if not designs:
            designs = _collect_preview_pipeline_outputs(
                job_outputs, task_name, params.binder_length, top_k,
            )
        if not designs:
            debug_path = job_outputs / "pxdesign_failure_debug.json"
            stdout_path = job_outputs / "pipeline_preview.stdout.log"
            stderr_path = job_outputs / "pipeline_preview.stderr.log"
            command = [str(pipeline_python()), "-m", "pxdesign.runner.pipeline", *argv]
            details = _write_pxdesign_failure_debug(
                debug_path=debug_path,
                command=command,
                cwd=ctx.job_dir,
                returncode=0,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                env=_pipeline_subprocess_env(seed=params.seed),
                ckpt_path=ckpt_path,
                job_outputs=job_outputs,
                task_name=task_name,
                preflight_result={"availability_skip_reason": availability_skip_reason()},
                input_structure_state=input_structure_state,
            )
            raise VPAError(
                f"PXDesign pipeline produced no generated designs under {job_outputs}. "
                f"See {debug_path} for the debug artifact contract.",
                details=details,
            )

        ctx.emit_progress(1.0)
        runtime_ms = int((time.time() - t0) * 1000)
        return PXDesignOutput(
            designs=designs,
            candidates=_pxdesign_candidates(params, designs),
            runtime_ms=runtime_ms,
            num_samples=len(designs),
            task_name=task_name,
            seed=seed_val,
        )

    @staticmethod
    def _build_pipeline_argv(
        *,
        params: PXDesignInput,
        seed_val: int,
        input_json_path: Path,
        job_outputs: Path,
    ) -> list[str]:
        """Construct the argv for ``pxdesign.runner.pipeline``.

        VPA fixes PXDesign to the generation pipeline: diffusion, AF2 initial
        guess, and ProteinMPNN sequence generation. Optional downstream
        filtering/ranking stages are disabled by using the lightweight preview
        preset and by collecting generated sequence artifacts directly.
        """
        preset = "preview"
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
            "--use_msa", "false",
            "--eval.binder.eval_protenix", "false",
            "--eval.binder.eval_protenix_mini", "false",
        ]
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

    def estimate_vram(self, params: PXDesignInput) -> int:
        return 24 * 1024

    def healthcheck(self) -> bool:
        """Warm-readiness includes the PARTIAL pipeline availability probe.

        The worker only emits warm warnings when ``healthcheck()`` is false.
        Include pipeline availability here so a warm call can surface the
        actionable subprocess-env bootstrap reason.
        """
        base_ok = self.ckpt_path.is_file() and PXDESIGN_VENDOR.is_dir()
        return base_ok and availability_skip_reason() is None

    def unload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    ckpt_path: Path | None = None,
    timeout_s: int,
    input_structure_state: dict[str, Any] | None = None,
) -> None:
    """Run vendor diffusion only, then materialize sequence outputs via VPA.

    The adapter contract is fixed to diffusion + AF2 initial-guess +
    ProteinMPNN sequence generation. Vendor preview code performs extra
    pre-filter/ranking once raw CIFs exist; VPA terminates the process at the
    first raw diffusion CIF and runs its own ProteinMPNN materialization so
    filter/selection stages cannot produce an empty accepted stream.
    """
    stdout_path = job_outputs / "pipeline_preview.stdout.log"
    stderr_path = job_outputs / "pipeline_preview.stderr.log"
    debug_path = job_outputs / "pxdesign_failure_debug.json"
    cmd = [str(pipeline_python()), "-m", "pxdesign.runner.pipeline", *argv]
    env = _pipeline_subprocess_env(seed=seed)
    resolved_ckpt_path = ckpt_path or (PXDESIGN_CKPT_DIR / f"{PXDESIGN_MODEL_NAME}.pt")
    with stdout_path.open("w") as stdout_fh, stderr_path.open("w") as stderr_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_fh,
            stderr=stderr_fh,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                raw_cif = _raw_preview_cif(job_outputs, task_name)
                if raw_cif is not None:
                    _terminate_process_group(proc)
                    _materialize_preview_sequence_outputs(
                        raw_cif=raw_cif,
                        job_outputs=job_outputs,
                        task_name=task_name,
                        seed=seed,
                        mpnn_temperature=mpnn_temperature,
                        debug_path=debug_path,
                        ckpt_path=resolved_ckpt_path,
                        input_structure_state=input_structure_state,
                    )
                    return
                rc = proc.poll()
                if rc is not None:
                    if rc == 0:
                        raw_cif = _raw_preview_cif(job_outputs, task_name)
                        if raw_cif is not None:
                            _materialize_preview_sequence_outputs(
                                raw_cif=raw_cif,
                                job_outputs=job_outputs,
                                task_name=task_name,
                                seed=seed,
                                mpnn_temperature=mpnn_temperature,
                                debug_path=debug_path,
                                ckpt_path=resolved_ckpt_path,
                                input_structure_state=input_structure_state,
                            )
                            return
                        details = _write_pxdesign_failure_debug(
                            debug_path=debug_path,
                            command=cmd,
                            cwd=cwd,
                            returncode=rc,
                            stdout_path=stdout_path,
                            stderr_path=stderr_path,
                            env=env,
                            ckpt_path=resolved_ckpt_path,
                            job_outputs=job_outputs,
                            task_name=task_name,
                            preflight_result={
                                "availability_skip_reason": availability_skip_reason()
                            },
                            input_structure_state=input_structure_state,
                        )
                        raise VPAError(
                            f"PXDesign preview pipeline exited successfully, but "
                            f"no final ProteinMPNN sequence-design outputs "
                            f"were produced. See {debug_path}.",
                            details=details,
                        )
                    details = _write_pxdesign_failure_debug(
                        debug_path=debug_path,
                        command=cmd,
                        cwd=cwd,
                        returncode=rc,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        env=env,
                        ckpt_path=resolved_ckpt_path,
                        job_outputs=job_outputs,
                        task_name=task_name,
                        preflight_result={
                            "availability_skip_reason": availability_skip_reason()
                        },
                        input_structure_state=input_structure_state,
                    )
                    raise VPAError(
                        f"PXDesign preview pipeline exited with status {rc} "
                        f"before emitting design artifacts. See {debug_path}.",
                        details=details,
                    )
                if time.monotonic() > deadline:
                    _terminate_process_group(proc, kill=True)
                    details = _write_pxdesign_failure_debug(
                        debug_path=debug_path,
                        command=cmd,
                        cwd=cwd,
                        returncode=proc.poll(),
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        env=env,
                        ckpt_path=resolved_ckpt_path,
                        job_outputs=job_outputs,
                        task_name=task_name,
                        preflight_result={
                            "availability_skip_reason": availability_skip_reason()
                        },
                        input_structure_state=input_structure_state,
                    )
                    raise VPAError(
                        f"PXDesign preview pipeline timed out before emitting "
                        f"design artifacts. See {debug_path}.",
                        details=details,
                    )
                time.sleep(2.0)
        except BaseException:
            if proc.poll() is None:
                _terminate_process_group(proc, kill=True)
            raise


def _write_pxdesign_failure_debug(
    *,
    debug_path: Path,
    command: list[str],
    cwd: Path,
    returncode: int | None,
    stdout_path: Path,
    stderr_path: Path,
    env: dict[str, str],
    ckpt_path: Path,
    job_outputs: Path,
    task_name: str,
    preflight_result: dict[str, Any],
    input_structure_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_cif = _raw_preview_cif(job_outputs, task_name)
    chain_metadata_path = (
        job_outputs
        / "design_outputs"
        / task_name
        / "preview_fallback_chain_metadata.json"
    )
    fallback_chain_metadata = None
    fallback_chain_metadata_error = None
    if chain_metadata_path.is_file():
        try:
            fallback_chain_metadata = json.loads(chain_metadata_path.read_text())
        except Exception as exc:
            fallback_chain_metadata_error = str(exc)
    elif raw_cif is not None:
        fallback_chain_metadata_error = "metadata_not_written"
    env_keys = (
        "CONDA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "JAX_PLATFORMS",
        "PROTENIX_DATA_ROOT_DIR",
        "PYTHONPATH",
        "TOOL_WEIGHTS_ROOT",
        "VPA_PXDESIGN_PIPELINE_ENV",
        "VPA_PXDESIGN_CKPT_DIR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
    )
    payload = {
        "command": command,
        "cwd": str(cwd),
        "returncode": returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "env_summary": {key: env.get(key) for key in env_keys if key in env},
        "checkpoint_state": {
            "pxdesign_checkpoint": str(ckpt_path),
            "pxdesign_checkpoint_exists": ckpt_path.is_file(),
            "pxdesign_checkpoint_dir": str(PXDESIGN_CKPT_DIR),
            "pxdesign_checkpoint_dir_exists": PXDESIGN_CKPT_DIR.is_dir(),
            "protenix_data_root": PROTENIX_DATA_ROOT_DEFAULT,
        },
        "preflight_result": preflight_result,
        "input_structure_state": input_structure_state or {},
        "raw_artifact_state": {
            "raw_cif_exists": raw_cif is not None,
            "raw_cif_path": str(raw_cif) if raw_cif is not None else None,
            "summary_csv_exists": (
                job_outputs / "design_outputs" / task_name / "summary.csv"
            ).is_file(),
        },
        "preview_fallback_chain_metadata": fallback_chain_metadata,
        "preview_fallback_chain_metadata_path": str(chain_metadata_path),
        "preview_fallback_chain_metadata_error": fallback_chain_metadata_error,
    }
    debug_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _materialize_preview_sequence_outputs(
    *,
    raw_cif: Path,
    job_outputs: Path,
    task_name: str,
    seed: int | None,
    mpnn_temperature: float,
    debug_path: Path,
    ckpt_path: Path,
    input_structure_state: dict[str, Any] | None = None,
) -> None:
    env = _pipeline_subprocess_env(seed=seed)
    env["JAX_PLATFORMS"] = "cpu"
    stdout_path = job_outputs / "pipeline_preview_fallback.stdout.log"
    stderr_path = job_outputs / "pipeline_preview_fallback.stderr.log"
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
    try:
        with stdout_path.open("w") as stdout_fh, stderr_path.open("w") as stderr_fh:
            out = subprocess.run(
                cmd,
                check=False,
                cwd=str(job_outputs),
                env=env,
                stdout=stdout_fh,
                stderr=stderr_fh,
                text=True,
                timeout=600,
            )
    except subprocess.TimeoutExpired as exc:
        details = _write_pxdesign_failure_debug(
            debug_path=debug_path,
            command=cmd,
            cwd=job_outputs,
            returncode=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=env,
            ckpt_path=ckpt_path,
            job_outputs=job_outputs,
            task_name=task_name,
            preflight_result={
                "availability_skip_reason": availability_skip_reason(),
                "fallback_timeout_s": exc.timeout,
            },
            input_structure_state=input_structure_state,
        )
        raise VPAError(
            f"PXDesign preview ProteinMPNN fallback timed out. See {debug_path}.",
            details=details,
        ) from exc
    if out.returncode != 0:
        details = _write_pxdesign_failure_debug(
            debug_path=debug_path,
            command=cmd,
            cwd=job_outputs,
            returncode=out.returncode,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            env=env,
            ckpt_path=ckpt_path,
            job_outputs=job_outputs,
            task_name=task_name,
            preflight_result={"availability_skip_reason": availability_skip_reason()},
            input_structure_state=input_structure_state,
        )
        raise VPAError(
            f"PXDesign preview ProteinMPNN fallback failed with status "
            f"{out.returncode}. See {debug_path}.",
            details=details,
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


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _pxdesign_candidates(
    params: PXDesignInput,
    designs: list[PXDesignDesign],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, design in enumerate(designs):
        local_id = f"design_{index:04d}"
        structure_path = design.pdb_path or design.cif_path
        candidate = {
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "pxdesign",
                local_id,
                native_id=design.candidate_id,
                sequence=design.sequence,
                structure_path=structure_path,
                index=index,
            ),
            "source": {
                "tool": "pxdesign",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": design.candidate_id,
                "index": index,
            },
            "sequence": design.sequence,
            "structure_path": structure_path,
            "backbone_path": structure_path,
            "scores": {
                "pxdesign": {
                    "score": design.score,
                    "binder_length": design.binder_length,
                    "pipeline": "diffusion_af2_initial_guess_proteinmpnn",
                    "structure_detail": "binder_backbone_with_target_context",
                }
            },
            "artifacts": {"pxdesign": {"cif_path": design.cif_path, "pdb_path": design.pdb_path}},
            "capabilities": candidate_capability_flags(
                has_sequence=bool(design.sequence),
                has_backbone_structure=bool(structure_path),
                has_full_structure=False,
                has_candidate_scores=True,
            ),
        }
        if params.candidates and _is_canonical_candidate(params.candidates[0]):
            upstream_scores = params.candidates[0].get("scores")
            if isinstance(upstream_scores, dict):
                candidate["scores"] = {**upstream_scores, **candidate["scores"]}
        candidates.append(candidate)
    return candidates


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
]
