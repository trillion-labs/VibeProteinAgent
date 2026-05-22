"""BindCraft adapter.

BindCraft (Pacesa et al., ETH Zurich, upstream MIT licence) does AF2-backprop
binder hallucination, ProteinMPNN sequence design and PyRosetta
relax/scoring in one pipeline. The CLI entry point is
``bindcraft.py --settings ... --filters ... --advanced ...`` which
runs an infinite trajectory loop until ``number_of_final_designs``
binders pass the filter cascade.

Why a *separate conda env* instead of the ``vpa`` env
======================================================

BindCraft's installer pins:

* ``numpy < 2.0`` — incompatible with our ``vpa`` env which pins
  numpy 2.3.5 (and the Phase 12-20 hard-rule forbids modifying it).
* ``jaxlib==*cuda*`` — bundles a cuDNN/CUBLAS that would override
  torch's bundled CUDA libs, so it must stay in a compatibility sidecar.
* ``pyrosetta`` — installed from the PyRosetta wheel/channel.

So this adapter runs BindCraft in a *separate* conda env (default
name ``BindCraft``, override via ``$VPA_BINDCRAFT_CONDA_ENV``). The
adapter only stages I/O, materialises a per-job ``settings.json`` /
``advanced.json`` / ``filters.json``, and dispatches into a persistent
Python sidecar running inside that env. If the env does not exist (or
``CONDA_EXE`` is unresolvable), the availability probe returns a concrete
operator message and the slow test skips with that reason.

AF2 weights are kept under ``$VPA_BINDCRAFT_PARAMS`` (default
``vendor/bindcraft/params/``). Standard AF2-2022-12-06 layout, the
canonical ~5.3 GiB tarball from
``https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.runtime import EnvLocator, sanitized_python_env, smoke_import
from vibeproteinagent.core.errors import VPAError
from vibeproteinagent.workers.sidecar import PersistentPythonSidecar, SidecarRuntimeSpec

_subprocess_module = subprocess  # Preserve adapter-level monkeypatch seam in tests.

# --------------------------------------------------------------------------- paths

_ADAPTER_DIR = Path(__file__).parent
_VPA_ROOT = _ADAPTER_DIR.parent.parent.parent.parent  # /VibeProteinAgent/

BINDCRAFT_TREE_DEFAULT = _VPA_ROOT / "vendor" / "bindcraft"

BINDCRAFT_TREE = Path(
    os.environ.get("VPA_BINDCRAFT_TREE", str(BINDCRAFT_TREE_DEFAULT))
).expanduser()
BINDCRAFT_PARAMS = Path(
    os.environ.get("VPA_BINDCRAFT_PARAMS", str(BINDCRAFT_TREE / "params"))
).expanduser()
BINDCRAFT_CONDA_ENV = os.environ.get("VPA_BINDCRAFT_CONDA_ENV", "BindCraft")

_RUNNER_MODULE = "vibeproteinagent.adapters.bindcraft._runner"
_RUNNER_SIDECAR_CALL = f"{_RUNNER_MODULE}:run_sidecar"


def _check_paths() -> None:
    if not BINDCRAFT_TREE.is_dir():
        raise FileNotFoundError(
            f"BindCraft tree not found at {BINDCRAFT_TREE}. "
            f"Clone via: git clone https://github.com/martinpacesa/BindCraft.git "
            f"{BINDCRAFT_TREE} (or set $VPA_BINDCRAFT_TREE)."
        )
    if not (BINDCRAFT_TREE / "bindcraft.py").is_file():
        raise FileNotFoundError(
            f"{BINDCRAFT_TREE / 'bindcraft.py'} not found — "
            f"$VPA_BINDCRAFT_TREE does not look like a BindCraft checkout."
        )


def _af2_params_present() -> bool:
    """Return True iff the canonical AF2 params layout is present.

    BindCraft expects an ``alphafold_params_2022-12-06`` layout (loose
    ``params_model_*_ptm.npz`` files under the ``params/`` dir).
    """
    if not BINDCRAFT_PARAMS.is_dir():
        return False
    return (BINDCRAFT_PARAMS / "params_model_5_ptm.npz").is_file()


def _find_conda_exe() -> Path | None:
    """Locate ``conda`` for ``conda run -n …``. ``$CONDA_EXE`` first."""
    env_var = os.environ.get("CONDA_EXE")
    extra = [Path(env_var)] if env_var else []
    return EnvLocator.find_conda_exe(extra)


def _conda_env_exists(conda_exe: Path, env_name: str) -> bool:
    return _conda_env_path(conda_exe, env_name) is not None


def _conda_env_path(conda_exe: Path, env_name: str) -> Path | None:
    if not conda_exe.is_file():
        return None
    return EnvLocator.conda_env_path(
        env_name,
        explicit_env_var="VPA_BINDCRAFT_ENV_PATH",
        conda_exe=conda_exe,
    )


def _bindcraft_python(conda_exe: Path, env_name: str) -> Path | None:
    env_path = _conda_env_path(conda_exe, env_name)
    if env_path is None:
        return None
    py = env_path / "bin" / "python"
    return py if py.is_file() else None


def _bindcraft_import_skip_reason(python: Path, timeout_s: int = 240) -> str | None:
    """Smoke-import the dedicated sidecar runtime before declaring readiness."""
    reason = smoke_import(
        label="BindCraft sidecar",
        python=python,
        code=(
            "import msgpack; "
            "import vibeproteinagent.workers.sidecar_child; "
            "import pydantic; "
            "import vibeproteinagent.adapters.bindcraft._runner; "
            "import torch; "
            "import jax; "
            "import colabdesign; "
            "import pyrosetta; "
            "print('ok')"
        ),
        env=sanitized_python_env(
            seed=None,
            env_prefix=python.parents[1],
            extra=_build_env(
                seed=None,
                env_prefix=python.parents[1],
                include_vpa_src=True,
            ),
        ),
        timeout_s=timeout_s,
    )
    if reason is None:
        return None
    return (
        f"{reason}. Check pydantic, torch, JAX/CUDA, ColabDesign, "
        "PyRosetta, msgpack, and VPA sidecar install."
    )


# --------------------------------------------------------------------------- availability


def availability_skip_reason() -> str | None:
    """Return a precise skip reason if BindCraft cannot run, else ``None``.

    Surfaced by ``vpa daemon warm bindcraft --json`` (via the worker's
    healthcheck → availability probe path) and by the slow regression
    test so operators get a concrete bootstrap pointer instead of the
    generic "healthcheck returned False" fallback.
    """
    if not BINDCRAFT_TREE.is_dir() or not (BINDCRAFT_TREE / "bindcraft.py").is_file():
        return (
            f"BindCraft tree not present at {BINDCRAFT_TREE}. "
            f"Clone via git clone https://github.com/martinpacesa/BindCraft.git "
            f"{BINDCRAFT_TREE} (or set $VPA_BINDCRAFT_TREE)."
        )
    if not _af2_params_present():
        return (
            f"AF2 params (params_model_*_ptm.npz) not found under {BINDCRAFT_PARAMS}. "
            f"Download alphafold_params_2022-12-06.tar from "
            f"https://storage.googleapis.com/alphafold/ and untar there "
            f"(or set $VPA_BINDCRAFT_PARAMS)."
        )
    explicit_env = os.environ.get("VPA_BINDCRAFT_ENV_PATH")
    if explicit_env:
        python = Path(explicit_env).expanduser() / "bin" / "python"
        if not python.is_file():
            return (
                f"BindCraft sidecar env at $VPA_BINDCRAFT_ENV_PATH={explicit_env} "
                "does not contain bin/python. Re-run `vpa install --tools bindcraft --repair`."
            )
        return _bindcraft_import_skip_reason(python)
    conda_exe = _find_conda_exe()
    if conda_exe is None:
        return (
            "No `conda` executable on PATH or in $CONDA_EXE — BindCraft "
            "runs in a dedicated conda env and needs the conda CLI."
        )
    python = _bindcraft_python(conda_exe, BINDCRAFT_CONDA_ENV)
    if python is None:
        return (
            f"BindCraft conda env '{BINDCRAFT_CONDA_ENV}' not found. "
            f"Bootstrap via vendor/bindcraft/install_bindcraft.sh "
            f"(or override the env name with $VPA_BINDCRAFT_CONDA_ENV)."
        )
    return _bindcraft_import_skip_reason(python)


def _explicit_bindcraft_python() -> Path | None:
    explicit_env = os.environ.get("VPA_BINDCRAFT_ENV_PATH")
    if not explicit_env:
        return None
    return Path(explicit_env).expanduser() / "bin" / "python"


# --------------------------------------------------------------------------- I/O models


class BindCraftInput(BaseModel):
    target_pdb: str = Field(..., min_length=1)
    target_chain: str = Field(default="A", min_length=1)
    target_hotspots: list[int] = Field(default_factory=list)
    binder_length: int = Field(default=60, ge=30, le=150)
    num_designs: int = Field(default=10, ge=1, le=32)
    num_recycles: int = Field(default=1, ge=0, le=8)
    seed: int | None = Field(default=None)
    max_trajectories: int | None = Field(default=None, ge=1, le=10_000)
    filter_mode: Literal["default", "permissive"] = "default"


class BindCraftDesign(BaseModel):
    pdb_path: str
    sequence: str
    binder_length: int
    iptm: float | None = None
    plddt: float | None = None
    pae_interaction: float | None = None
    sap_score: float | None = None


class BindCraftOutput(BaseModel):
    designs: list[BindCraftDesign]
    runtime_ms: int
    num_designs_returned: int
    seeds_used: list[int]


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class BindCraftAdapter(ToolAdapter):
    """Adapter backed by a persistent dedicated ``BindCraft`` sidecar."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        self.device = device
        self._availability_checked = False
        self._sidecars: dict[int | None, PersistentPythonSidecar] = {}
        _check_paths()

    @torch.no_grad()
    def run(self, params: BindCraftInput, ctx: AdapterRunContext) -> BindCraftOutput:
        t0 = time.time()
        ctx.emit_progress(0.02)

        if not torch.cuda.is_available():
            raise RuntimeError("BindCraft adapter requires CUDA")

        if not _af2_params_present():
            raise FileNotFoundError(
                f"AF2 params (params_model_*_ptm.npz) not found under {BINDCRAFT_PARAMS}. "
                f"Download via "
                f"https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar "
                f"and untar into that directory — see vendor/bindcraft/UPSTREAM.md."
            )

        self._ensure_available_once()

        explicit_python = _explicit_bindcraft_python()
        conda_exe = _find_conda_exe()
        if explicit_python and explicit_python.is_file():
            python = explicit_python
        elif conda_exe is not None:
            python = _bindcraft_python(conda_exe, BINDCRAFT_CONDA_ENV)
        else:
            python = None
        if python is None:
            raise RuntimeError(
                f"BindCraft conda env '{BINDCRAFT_CONDA_ENV}' not available. "
                f"Run vendor/bindcraft/install_bindcraft.sh (uses jax+cuda+pyrosetta+colabdesign "
                f"with numpy<2 — must NOT be the vpa env) and then set "
                f"$VPA_BINDCRAFT_CONDA_ENV if the env name differs."
            )

        target_src = Path(params.target_pdb).expanduser().resolve()
        if not target_src.is_file():
            raise FileNotFoundError(f"target_pdb not found: {target_src}")

        job_inputs = ctx.job_dir / "inputs"
        job_outputs = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)

        staged_target = job_inputs / target_src.name
        shutil.copyfile(target_src, staged_target)

        settings_json = job_inputs / "target_settings.json"
        filters_json = job_inputs / "filters.json"
        advanced_json = job_inputs / "advanced.json"

        settings_json.write_text(
            json.dumps(
                _build_target_settings(params, staged_target, job_outputs), indent=2
            )
        )
        filters_json.write_text(
            json.dumps(_build_filters(BINDCRAFT_TREE, mode=params.filter_mode), indent=2)
        )
        advanced_json.write_text(
            json.dumps(_build_advanced(params, BINDCRAFT_PARAMS), indent=2)
        )

        ctx.emit_progress(0.10)
        try:
            self._sidecar_for(params.seed, python).call(
                _RUNNER_SIDECAR_CALL,
                {
                    "bindcraft_py": str(BINDCRAFT_TREE / "bindcraft.py"),
                    "tree": str(BINDCRAFT_TREE),
                    "settings_json": str(settings_json),
                    "filters_json": str(filters_json),
                    "advanced_json": str(advanced_json),
                    "seed": params.seed,
                },
                timeout_s=4 * 3600,
            )
        except VPAError as exc:
            raise RuntimeError(str(exc)) from exc
        ctx.emit_progress(0.92)

        designs = _collect_designs(
            design_path=job_outputs,
            binder_length=params.binder_length,
        )
        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return BindCraftOutput(
            designs=designs,
            runtime_ms=runtime_ms,
            num_designs_returned=len(designs),
            seeds_used=[params.seed] if params.seed is not None else [],
        )

    def estimate_vram(self, params: BindCraftInput) -> int:
        del params
        return 24 * 1024

    def healthcheck(self) -> bool:
        try:
            _check_paths()
        except FileNotFoundError:
            return False
        if not _af2_params_present():
            return False
        return availability_skip_reason() is None

    def unload(self) -> None:
        for sidecar in self._sidecars.values():
            sidecar.close()
        self._sidecars.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_available_once(self) -> None:
        if self._availability_checked:
            return
        skip = availability_skip_reason()
        if skip is not None:
            raise RuntimeError(f"BindCraft unavailable: {skip}")
        self._availability_checked = True

    def _sidecar_for(
        self, seed: int | None, python: Path
    ) -> PersistentPythonSidecar:
        key = int(seed) if seed is not None else None
        sidecar = self._sidecars.get(key)
        if sidecar is None:
            spec = SidecarRuntimeSpec(
                python=python,
                env=_build_env(
                    seed,
                    env_prefix=python.parents[1],
                    include_vpa_src=True,
                ),
                name=f"bindcraft[{key if key is not None else 'default'}]",
                warm_target=None,
                run_target=_RUNNER_SIDECAR_CALL,
                timeout_s=240.0,
            )
            sidecar = spec.spawn(PersistentPythonSidecar)
            self._sidecars[key] = sidecar
        return sidecar


# --------------------------------------------------------------------------- helpers


def _build_target_settings(
    params: BindCraftInput, staged_target: Path, design_path: Path
) -> dict:
    """Materialise the ``settings_target/*.json`` BindCraft consumes."""
    hotspots_csv = ",".join(str(i) for i in params.target_hotspots)
    return {
        "design_path": str(design_path) + os.sep,
        "binder_name": staged_target.stem,
        "starting_pdb": str(staged_target),
        "chains": params.target_chain,
        "target_hotspot_residues": hotspots_csv,
        "lengths": [params.binder_length, params.binder_length],
        "number_of_final_designs": params.num_designs,
    }


def _build_filters(tree: Path, mode: Literal["default", "permissive"] = "default") -> dict:
    """Copy BindCraft's default filter cascade verbatim.

    We re-emit it under the job dir so each run is self-contained and
    no global state from ``BINDCRAFT_TREE`` is consulted at runtime.
    """
    default = tree / "settings_filters" / "default_filters.json"
    if default.is_file():
        filters = json.loads(default.read_text())
        if mode == "permissive":
            return _without_filter_thresholds(filters)
        return filters
    return {}


def _without_filter_thresholds(filters: dict) -> dict:
    """Return a copy of a BindCraft filter cascade with all thresholds disabled."""
    out = json.loads(json.dumps(filters))
    for conditions in out.values():
        if not isinstance(conditions, dict):
            continue
        if "threshold" in conditions:
            conditions["threshold"] = None
            continue
        for aa_conditions in conditions.values():
            if isinstance(aa_conditions, dict) and "threshold" in aa_conditions:
                aa_conditions["threshold"] = None
    return out


def _build_advanced(params: BindCraftInput, params_dir: Path) -> dict:
    """Emit the ``settings_advanced/*.json`` blob BindCraft consumes."""
    advanced = {
        "omit_AAs": "C",
        "force_reject_AA": False,
        "use_multimer_design": True,
        "design_algorithm": "4stage",
        "sample_models": True,
        "rm_template_seq_design": False,
        "rm_template_seq_predict": False,
        "rm_template_sc_design": False,
        "rm_template_sc_predict": False,
        "predict_initial_guess": False,
        "predict_bigbang": False,
        "soft_iterations": 75,
        "temporary_iterations": 45,
        "hard_iterations": 5,
        "greedy_iterations": 15,
        "greedy_percentage": 1,
        "save_design_animations": False,
        "save_design_trajectory_plots": False,
        "weights_plddt": 0.1,
        "weights_pae_intra": 0.4,
        "weights_pae_inter": 0.1,
        "weights_con_intra": 1.0,
        "weights_con_inter": 1.0,
        "intra_contact_distance": 14.0,
        "inter_contact_distance": 20.0,
        "intra_contact_number": 2,
        "inter_contact_number": 2,
        "weights_helicity": -0.3,
        "random_helicity": False,
        "use_i_ptm_loss": True,
        "weights_iptm": 0.05,
        "use_rg_loss": True,
        "weights_rg": 0.3,
        "use_termini_distance_loss": False,
        "weights_termini_loss": 0.1,
        "enable_mpnn": True,
        "mpnn_fix_interface": True,
        "num_seqs": 20,
        "max_mpnn_sequences": 2,
        "sampling_temp": 0.1,
        "backbone_noise": 0.0,
        "model_path": "v_48_020",
        "mpnn_weights": "soluble",
        "save_mpnn_fasta": False,
        "num_recycles_design": params.num_recycles,
        "num_recycles_validation": max(params.num_recycles, 3),
        "optimise_beta": True,
        "optimise_beta_extra_soft": 0,
        "optimise_beta_extra_temp": 0,
        "optimise_beta_recycles_design": max(params.num_recycles, 3),
        "optimise_beta_recycles_valid": max(params.num_recycles, 3),
        "remove_unrelaxed_trajectory": True,
        "remove_unrelaxed_complex": True,
        "remove_binder_monomer": True,
        "zip_animations": False,
        "zip_plots": False,
        "save_trajectory_pickle": False,
        "af_params_dir": str(params_dir),
        "dalphaball_path": str(BINDCRAFT_TREE / "functions" / "DAlphaBall.gcc"),
        "dssp_path": str(BINDCRAFT_TREE / "functions" / "dssp"),
        "max_trajectories": params.max_trajectories or False,
        "enable_rejection_check": True,
        "acceptance_rate": 0.05,
        "start_monitoring": 1000,
    }
    return advanced


def _build_env(
    seed: int | None,
    env_prefix: Path | None = None,
    *,
    include_vpa_src: bool = False,
) -> dict[str, str]:
    extra: dict[str, str] = {}
    # JAX should release VRAM cooperatively if the same process later
    # touches torch (defence in depth — under conda run -n we already
    # have a separate process, but if someone re-routes the conda env
    # to vpa we still don't want JAX to grab the whole GPU at startup).
    extra["XLA_PYTHON_CLIENT_PREALLOCATE"] = os.environ.get(
        "XLA_PYTHON_CLIENT_PREALLOCATE", "false"
    )
    if env_prefix is not None:
        lib_dir = str(env_prefix / "lib")
        old = os.environ.get("LD_LIBRARY_PATH")
        extra["LD_LIBRARY_PATH"] = f"{lib_dir}:{old}" if old else lib_dir
    return sanitized_python_env(
        seed=seed,
        env_prefix=env_prefix,
        extra=extra,
        include_vpa_src=include_vpa_src,
    )


def _collect_designs(
    *, design_path: Path, binder_length: int
) -> list[BindCraftDesign]:
    """Walk BindCraft's output layout.

    BindCraft writes accepted binders to ``<design_path>/Accepted/`` and
    summary statistics to ``<design_path>/final_design_stats.csv``.
    """
    accepted_dir = design_path / "Accepted"
    if not accepted_dir.is_dir():
        raise RuntimeError(
            f"BindCraft produced no Accepted/ directory under {design_path}. "
            f"Likely no trajectories passed the filter cascade in the allotted budget."
        )
    pdbs = sorted(accepted_dir.glob("*.pdb"))
    if not pdbs:
        raise RuntimeError(
            f"BindCraft Accepted/ directory at {accepted_dir} contains no PDBs."
        )

    stats = _load_final_stats(design_path / "final_design_stats.csv")
    designs: list[BindCraftDesign] = []
    for pdb in pdbs:
        meta = stats.get(pdb.stem, {})
        designs.append(
            BindCraftDesign(
                pdb_path=str(pdb),
                sequence=str(meta.get("Sequence", "X" * binder_length)),
                binder_length=binder_length,
                iptm=_as_float(meta.get("Average_i_pTM")),
                plddt=_as_float(meta.get("Average_pLDDT")),
                pae_interaction=_as_float(meta.get("Average_PAE_Interaction")),
                sap_score=_as_float(meta.get("Binder_SAP_score")),
            )
        )
    return designs


def _load_final_stats(csv_path: Path) -> dict[str, dict[str, str]]:
    """Parse BindCraft's ``final_design_stats.csv`` into a dict by Design name."""
    if not csv_path.is_file():
        return {}
    import csv

    rows: dict[str, dict[str, str]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("Design") or row.get("design") or ""
            if name:
                rows[name] = row
    return rows


def _as_float(v: object) -> float | None:
    if v is None or v == "" or v == "NA":
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


__all__ = [
    "BINDCRAFT_CONDA_ENV",
    "BINDCRAFT_PARAMS",
    "BINDCRAFT_TREE",
    "BindCraftAdapter",
    "BindCraftDesign",
    "BindCraftInput",
    "BindCraftOutput",
    "availability_skip_reason",
]
