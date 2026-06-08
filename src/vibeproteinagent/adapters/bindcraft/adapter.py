"""BindCraft adapter.

BindCraft (Pacesa et al., ETH Zurich, upstream MIT licence) does AF2-backprop
binder hallucination, ProteinMPNN sequence design and PyRosetta
relax/scoring in one pipeline. The CLI entry point is
``bindcraft.py --settings ... --filters ... --advanced ...`` which
runs a trajectory loop until generated binder artifacts are available.
VPA disables filtering/rejection thresholds and returns generated trajectory
artifacts directly; evaluation tools can score and rank them downstream.

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

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.residue_mapping import (
    PositionBasis,
    parse_structure_residue_refs,
    resolve_positions,
)
from vibeproteinagent.adapters.runtime import EnvLocator, sanitized_python_env, smoke_import
from vibeproteinagent.candidates.capabilities import candidate_capability_flags
from vibeproteinagent.candidates.chain_roles import (
    chain_sequence_from_structure,
    complex_chain_metadata,
    observed_chain_for_sequence,
)
from vibeproteinagent.candidates.schema import stable_candidate_id
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


class NoGeneratedBindCraftDesigns(RuntimeError):
    """BindCraft completed but produced no generated design PDBs."""


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
    target_pdb: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    target_chain: str = Field(default="A", min_length=1)
    target_hotspots: list[int] = Field(default_factory=list)
    target_hotspot_residue_ids: list[str] = Field(default_factory=list)
    target_sequence: str | None = None
    target_fasta_path: str | None = Field(default=None, min_length=1)
    position_basis: PositionBasis = "auto"
    binder_length: int = Field(default=60, ge=30, le=150)
    num_designs: int = Field(default=10, ge=1, le=32)
    num_recycles: int = Field(default=1, ge=0, le=8)
    seed: int | None = Field(default=None)
    max_trajectories: int | None = Field(default=None, ge=1, le=10_000)
    max_attempted_trajectories: int | None = Field(default=None, ge=1, le=10_000)
    max_runtime_s: int | None = Field(default=None, ge=1)

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
        if not all(canonical):
            return data
        first = candidates[0]
        structure_path = first.get("structure_path")
        if not structure_path:
            raise ValueError("bindcraft canonical Candidate[] target requires structure_path")
        out = dict(data)
        out.setdefault("target_pdb", structure_path)
        out.setdefault("target_sequence", first.get("sequence"))
        return out

    @model_validator(mode="after")
    def _require_target_pdb(self) -> BindCraftInput:
        if not self.target_pdb:
            raise ValueError("bindcraft requires target_pdb")
        return self


class BindCraftDesign(BaseModel):
    candidate_id: str | None = None
    pdb_path: str
    sequence: str
    binder_length: int
    iptm: float | None = None
    plddt: float | None = None
    pae_interaction: float | None = None
    sap_score: float | None = None


class BindCraftOutput(BaseModel):
    status: Literal["completed"] = "completed"
    designs: list[BindCraftDesign]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int
    num_designs_returned: int
    seeds_used: list[int]
    position_mappings: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    failure_summary: dict[str, int] = Field(default_factory=dict)
    run_metadata: dict[str, Any] = Field(default_factory=dict)


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

        assert params.target_pdb is not None
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

        hotspot_resolution = _resolve_target_hotspots(params, staged_target)
        settings_json.write_text(
            json.dumps(
                _build_target_settings(
                    params,
                    staged_target,
                    job_outputs,
                    resolved_hotspots=hotspot_resolution.residue_ids,
                ),
                indent=2,
            )
        )
        filters_json.write_text(
            json.dumps(_build_generation_filters(BINDCRAFT_TREE), indent=2)
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
                    "generation_only": True,
                },
                timeout_s=params.max_runtime_s or 4 * 3600,
            )
        except VPAError as exc:
            if params.max_runtime_s is not None:
                self._sidecar_for(params.seed, python).close()
            raise RuntimeError(str(exc)) from exc
        ctx.emit_progress(0.92)

        designs, collect_warnings = _collect_designs(
            design_path=job_outputs,
            binder_length=params.binder_length,
        )
        status = "completed"
        failure_summary = _bindcraft_failure_summary(job_outputs)
        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return BindCraftOutput(
            status=status,
            designs=designs,
            candidates=[
                _canonical_candidate_from_design(design.model_dump(), index, params=params)
                for index, design in enumerate(designs)
            ],
            runtime_ms=runtime_ms,
            num_designs_returned=len(designs),
            seeds_used=[params.seed] if params.seed is not None else [],
            position_mappings={
                "target_hotspots": hotspot_resolution.to_audit()
            },
            warnings=[
                *hotspot_resolution.warnings,
                *collect_warnings,
                *_bindcraft_budget_warnings(failure_summary, designs),
            ],
            failure_summary=failure_summary,
            run_metadata={
                "status": status,
                "target_pdb": str(target_src),
                "generated_design_count": len(designs),
                "failure_summary": failure_summary,
                "filtering": "disabled",
                "max_attempted_trajectories": params.max_attempted_trajectories,
                "max_trajectories": params.max_trajectories,
            },
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
    params: BindCraftInput,
    staged_target: Path,
    design_path: Path,
    *,
    resolved_hotspots: list[str] | None = None,
) -> dict:
    """Materialise the ``settings_target/*.json`` BindCraft consumes."""
    hotspots = resolved_hotspots
    if hotspots is None:
        hotspots = _resolve_target_hotspots(params, staged_target).residue_ids
    hotspots_csv = ",".join(hotspots)
    return {
        "design_path": str(design_path) + os.sep,
        "binder_name": staged_target.stem,
        "starting_pdb": str(staged_target),
        "chains": params.target_chain,
        "target_hotspot_residues": hotspots_csv,
        "lengths": [params.binder_length, params.binder_length],
        "number_of_final_designs": params.num_designs,
    }


def _resolve_target_hotspots(params: BindCraftInput, target_pdb: Path):
    resolution = resolve_positions(
        residues=parse_structure_residue_refs(target_pdb, params.target_chain),
        positions=params.target_hotspots,
        explicit_residue_ids=params.target_hotspot_residue_ids,
        chain_id=params.target_chain,
        sequence=params.target_sequence,
        fasta_path=params.target_fasta_path,
        position_basis=params.position_basis,
        label="target_hotspots",
    )
    # BindCraft settings already carry the target chain separately; plain
    # numeric hotspot labels preserve the existing single-chain adapter contract.
    if len(params.target_chain) == 1:
        chain = params.target_chain
        resolution.residue_ids = [
            residue_id[1:] if residue_id.startswith(chain) else residue_id
            for residue_id in resolution.residue_ids
        ]
    return resolution


def _build_generation_filters(tree: Path) -> dict:
    """Build a BindCraft filters file that cannot reject generated artifacts.

    Upstream BindCraft requires a filters JSON argument. VPA's adapter
    contract is generation-only, so we keep the file shape but null thresholds
    to prevent accepted/rejected filtering from creating an empty candidate
    stream.
    """
    default = tree / "settings_filters" / "default_filters.json"
    if default.is_file():
        return _without_filter_thresholds(json.loads(default.read_text()))
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
        "remove_unrelaxed_trajectory": False,
        "remove_unrelaxed_complex": True,
        "remove_binder_monomer": True,
        "zip_animations": False,
        "zip_plots": False,
        "save_trajectory_pickle": False,
        "af_params_dir": str(params_dir),
        "dalphaball_path": str(BINDCRAFT_TREE / "functions" / "DAlphaBall.gcc"),
        "dssp_path": str(BINDCRAFT_TREE / "functions" / "dssp"),
        "max_trajectories": (
            params.max_attempted_trajectories
            or params.max_trajectories
            or False
        ),
        "enable_rejection_check": False,
        "acceptance_rate": 0.05,
        "start_monitoring": 1000,
    }
    return advanced


def _count_attempted_bindcraft_trajectories(design_path: Path) -> int:
    stems: set[str] = set()
    for rel_dir in (
        "Trajectory",
        "Trajectory/Relaxed",
        "Trajectory/Clashing",
        "Trajectory/LowConfidence",
    ):
        folder = design_path / rel_dir
        if not folder.is_dir():
            continue
        for pdb in folder.glob("*.pdb"):
            if not pdb.name.startswith("."):
                stems.add(pdb.stem)
    return len(stems)


def _bindcraft_failure_summary(design_path: Path) -> dict[str, int]:
    return {
        "attempted": _count_attempted_bindcraft_trajectories(design_path),
        "accepted": _count_pdbs(design_path / "Accepted" / "Ranked"),
        "relaxed": _count_pdbs(design_path / "Trajectory" / "Relaxed"),
        "clashing": _count_pdbs(design_path / "Trajectory" / "Clashing"),
        "low_confidence": _count_pdbs(design_path / "Trajectory" / "LowConfidence"),
        "failure_events": _count_failure_csv_events(design_path / "failure_csv.csv"),
    }


def _count_pdbs(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for item in folder.glob("*.pdb") if not item.name.startswith("."))


def _count_failure_csv_events(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return 0
    total = 0
    for row in rows:
        for value in row.values():
            try:
                total += int(float(value or 0))
            except (TypeError, ValueError):
                continue
    return total


def _bindcraft_budget_warnings(
    failure_summary: dict[str, int],
    designs: list[BindCraftDesign],
) -> list[str]:
    if designs:
        return []
    attempted = failure_summary.get("attempted", 0)
    if attempted <= 0:
        return []
    return [
        (
            "BindCraft attempted "
            f"{attempted} trajectories but produced no collectable design PDBs "
            f"(clashing={failure_summary.get('clashing', 0)}, "
            f"low_confidence={failure_summary.get('low_confidence', 0)})."
        )
    ]


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
) -> tuple[list[BindCraftDesign], list[str]]:
    """Walk BindCraft's generated-artifact layout.

    VPA does not use BindCraft's filter acceptance as the output contract.
    Prefer accepted PDBs when upstream writes them, but also collect relaxed,
    low-confidence, clashing, and raw trajectory PDBs so a bounded generation
    run returns generated candidates instead of an empty accepted stream.
    """
    pdbs = _generated_bindcraft_pdbs(design_path)
    if not pdbs:
        raise NoGeneratedBindCraftDesigns(
            f"BindCraft produced no generated PDB artifacts under {design_path}."
        )

    stats = _load_bindcraft_stats(design_path)
    designs: list[BindCraftDesign] = []
    warnings: list[str] = []
    for idx, pdb in enumerate(pdbs, start=1):
        # The CSV keys by the design name without BindCraft's AF2 model suffix
        # (the trailing ``_model{N}`` it appends when saving the accepted PDB).
        # Strip it so the lookup hits.
        lookup_key = re.sub(r"_model\d+$", "", pdb.stem)
        meta = stats.get(lookup_key) or stats.get(pdb.stem, {})

        sequence = str(meta.get("Sequence", "")).strip()
        if not sequence:
            # CSV row missing — fall back to chain B of the PDB so the report
            # surfaces the real sequence instead of placeholder X's.
            sequence = _extract_binder_sequence_from_pdb(pdb, binder_chain="B")
            if sequence:
                warnings.append(
                    f"BindCraft: no stats row for {pdb.name!s} "
                    f"(tried keys {lookup_key!r}, {pdb.stem!r}); "
                    f"recovered sequence from PDB chain B."
                )
            else:
                sequence = "X" * binder_length
                warnings.append(
                    f"BindCraft: no stats row for {pdb.name!s} and chain-B "
                    f"extraction failed; emitted placeholder sequence."
                )

        designs.append(
            BindCraftDesign(
                candidate_id=_bindcraft_candidate_id(pdb, idx, sequence),
                pdb_path=str(pdb),
                sequence=sequence,
                binder_length=binder_length,
                iptm=_first_float(meta, "Average_i_pTM"),
                plddt=_first_float(meta, "Average_pLDDT"),
                pae_interaction=_first_float(
                    meta,
                    "Average_PAE_Interaction",
                    "Average_i_pAE",
                    "Average_pAE",
                ),
                sap_score=_first_float(
                    meta,
                    "Binder_SAP_score",
                    "Average_Binder_SAP_score",
                ),
            )
        )
    return designs, warnings


def _generated_bindcraft_pdbs(design_path: Path) -> list[Path]:
    rel_dirs = (
        "Generated",
        "Accepted",
        "Accepted/Ranked",
        "MPNN/Relaxed",
        "MPNN",
        "Trajectory/Relaxed",
        "Trajectory/LowConfidence",
        "Trajectory/Clashing",
        "Trajectory",
    )
    seen: set[Path] = set()
    pdbs: list[Path] = []
    for rel_dir in rel_dirs:
        folder = design_path / rel_dir
        if not folder.is_dir():
            continue
        for pdb in sorted(folder.glob("*.pdb")):
            resolved = pdb.resolve()
            if resolved in seen or pdb.name.startswith("."):
                continue
            seen.add(resolved)
            pdbs.append(pdb)
    return pdbs


def _bindcraft_candidate_id(pdb_path: Path, index: int, sequence: str) -> str:
    stem = re.sub(r"_model\d+$", "", pdb_path.stem)
    token = hashlib.sha256(f"{stem}:{sequence}".encode()).hexdigest()[:10]
    return f"bindcraft:{stem or f'design_{index}'}:{token}"


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _canonical_candidate_from_design(
    design: dict[str, Any],
    index: int,
    *,
    params: BindCraftInput | None = None,
) -> dict[str, Any]:
    local_id = f"binder_{index:04d}"
    sequence = str(design.get("sequence") or "")
    structure_path = str(design.get("pdb_path") or "")
    bindcraft_artifacts: dict[str, Any] = {}
    if params is not None:
        target_pdb = (
            str(Path(params.target_pdb).expanduser().resolve())
            if params.target_pdb
            else None
        )
        bindcraft_artifacts.update(
            {
                "target_pdb": target_pdb,
                "target_chain": params.target_chain,
            }
        )
        binder_chain_id = observed_chain_for_sequence(structure_path, sequence)
        if binder_chain_id == params.target_chain:
            binder_chain_id = None
        chain_rows = complex_chain_metadata(
            target_sequence=chain_sequence_from_structure(target_pdb, params.target_chain),
            binder_sequence=sequence,
            structure_path=structure_path,
            target_chain_id=params.target_chain,
            binder_chain_id=binder_chain_id,
        )
        if chain_rows:
            bindcraft_artifacts["complex_chains"] = chain_rows
    return {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            "bindcraft",
            local_id,
            native_id=design.get("candidate_id"),
            sequence=sequence,
            structure_path=structure_path,
            index=index,
        ),
        "source": {
            "tool": "bindcraft",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": design.get("candidate_id"),
            "index": index,
        },
        "sequence": sequence,
        "structure_path": structure_path,
        "scores": {
            "bindcraft": {
                "iptm": design.get("iptm"),
                "plddt": design.get("plddt"),
                "pae_interaction": design.get("pae_interaction"),
                "sap_score": design.get("sap_score"),
                "binder_length": design.get("binder_length"),
            }
        },
        "artifacts": {"bindcraft": bindcraft_artifacts} if bindcraft_artifacts else {},
        "capabilities": candidate_capability_flags(
            has_sequence=bool(sequence),
            has_backbone_structure=bool(structure_path),
            has_full_structure=bool(structure_path),
            has_candidate_scores=True,
        ),
    }


# Three-letter → one-letter map for the 20 canonical amino acids. Anything
# else (modified residues, ligands, etc.) is mapped to ``X``.
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _extract_binder_sequence_from_pdb(pdb_path: Path, *, binder_chain: str) -> str:
    """Parse the binder chain's amino-acid sequence from a PDB file.

    Reads CA records on ``binder_chain`` in residue-sequence order. Used as a
    fallback when ``final_design_stats.csv`` doesn't carry a row for this
    design. Returns an empty string on any read/parse error so the caller can
    decide on a further fallback.
    """
    try:
        text = pdb_path.read_text()
    except Exception:
        return ""
    seen: set[tuple[str, int, str]] = set()
    seq: list[str] = []
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        if line[21:22] != binder_chain:
            continue
        resname = line[17:20].strip()
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        icode = line[26:27]
        key = (binder_chain, resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        seq.append(_AA3_TO_1.get(resname, "X"))
    return "".join(seq)


def _load_bindcraft_stats(design_path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for csv_name in (
        "final_design_stats.csv",
        "trajectory_stats.csv",
        "mpnn_design_stats.csv",
    ):
        rows.update(_load_stats_csv(design_path / csv_name))
    return rows


def _load_stats_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    """Parse a BindCraft stats CSV into a dict by design/stem name."""
    if not csv_path.is_file():
        return {}
    import csv

    rows: dict[str, dict[str, str]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (
                row.get("Design")
                or row.get("design")
                or row.get("Name")
                or row.get("name")
                or row.get("Trajectory")
                or row.get("trajectory")
                or ""
            )
            if name:
                stem = Path(str(name)).stem
                rows[stem] = row
                rows[re.sub(r"_model\d+$", "", stem)] = row
    return rows


def _as_float(v: object) -> float | None:
    if v is None or v == "" or v == "NA":
        return None
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _first_float(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


__all__ = [
    "BINDCRAFT_CONDA_ENV",
    "BINDCRAFT_PARAMS",
    "BINDCRAFT_TREE",
    "BindCraftAdapter",
    "BindCraftDesign",
    "BindCraftInput",
    "BindCraftOutput",
    "NoGeneratedBindCraftDesigns",
    "availability_skip_reason",
]
