"""PyRosetta adapter.

PyRosetta exposes Rosetta's full energy + sampling toolkit through Python. This
adapter wraps the four most common single-pose primitives:

* ``score`` — REF2015 total score (REU) of an input pose.
* ``interface`` — InterfaceAnalyzerMover metrics (dG_separated, dSASA,
  shape complementarity, packstat) between two specified chain groups.
* ``relax`` — FastRelax sampling (configurable cycle count); writes a
  relaxed PDB under the job dir and returns its score.
* ``pack`` — pack_rotamers (RestrictToRepacking) and re-score.

Runtime placement
=================

PyRosetta requires the RosettaCommons PyRosetta wheel/channel and a bundled
native build (~2 GiB on disk).

When PyRosetta is installed in the shared ``vpa`` env, this adapter uses that
env by default. It still runs through a persistent local sidecar so
``pyrosetta.init()`` and Rosetta global state stay isolated from the main worker
process. Operators can override the runtime with a separate env if desired:

* ``$VPA_PYROSETTA_ENV`` — explicit path (preferred).
* Shared-env default: the current Python ``sys.prefix`` when ``pyrosetta`` is
  importable.
* Legacy fallback: ``/opt/conda/envs/pyrosetta-vpa`` or whatever
  ``conda env list --json`` reports for ``pyrosetta-vpa``.

If the env does not exist (or the sidecar bootstrap imports fail inside it)
the adapter ships as PARTIAL — fast tests still pass, the slow test skips
with a clear, actionable message pointing at
``vendor/pyrosetta/UPSTREAM.md``.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.canonical_tools import candidate_capability_flags
from vibeproteinagent.agents.planning.candidate import stable_candidate_id
from vibeproteinagent.adapters.runtime import EnvLocator, sanitized_python_env, smoke_import
from vibeproteinagent.core.errors import VPAError
from vibeproteinagent.workers.sidecar import (
    PersistentPythonSidecar,
    SidecarRuntimeSpec,
)

_subprocess_module = subprocess  # Preserve adapter-level monkeypatch seam in tests.

# --------------------------------------------------------------------------- paths / env resolution

_ADAPTER_DIR = Path(__file__).parent
_VPA_ROOT = _ADAPTER_DIR.parent.parent.parent.parent  # /VibeProteinAgent/

PYROSETTA_ENV_DEFAULT = Path("/opt/conda/envs/pyrosetta-vpa")
PYROSETTA_ENV_NAME = "pyrosetta-vpa"

_RUNNER_MODULE = "vibeproteinagent.adapters.pyrosetta._runner"
_RUNNER_SIDECAR_CALL = f"{_RUNNER_MODULE}:run_sidecar"
_RUNNER_SIDECAR_WARM = f"{_RUNNER_MODULE}:warm_sidecar"


def pyrosetta_env() -> Path:
    """Resolve the PyRosetta runtime env path (env var wins)."""
    explicit = os.environ.get("VPA_PYROSETTA_ENV")
    if explicit:
        return Path(explicit).expanduser()
    if importlib.util.find_spec("pyrosetta") is not None:
        return Path(sys.prefix).resolve()
    if PYROSETTA_ENV_DEFAULT.is_dir():
        return PYROSETTA_ENV_DEFAULT
    discovered = _discover_named_env(PYROSETTA_ENV_NAME)
    if discovered is not None:
        return discovered
    return PYROSETTA_ENV_DEFAULT  # likely missing, but kept stable for messages


def pyrosetta_python() -> Path:
    """Path to ``<env>/bin/python`` (resolved lazily)."""
    return pyrosetta_env() / "bin" / "python"


def _find_conda_exe() -> Path | None:
    env_var = os.environ.get("CONDA_EXE")
    extra = [Path(env_var)] if env_var else []
    return EnvLocator.find_conda_exe(extra)


def _discover_named_env(env_name: str) -> Path | None:
    """Ask ``conda`` for an env named ``env_name``; return its path or None."""
    return EnvLocator.conda_env_path(
        env_name,
        explicit_env_var="VPA_PYROSETTA_ENV",
        final_default=None,
        conda_exe=_find_conda_exe(),
    )


def env_present() -> bool:
    return pyrosetta_python().is_file()


def _pyrosetta_import_works(python: Path, timeout_s: int = 20) -> bool:
    """Confirm the dedicated env can import PyRosetta and the sidecar runtime."""
    if not python.is_file():
        return False
    return (
        smoke_import(
            label="PyRosetta sidecar",
            python=python,
            code=(
                "import msgpack; "
                "import vibeproteinagent.workers.sidecar_child; "
                "import pyrosetta; "
                "print('ok')"
            ),
            env=_subprocess_env(seed=None),
            timeout_s=timeout_s,
        )
        is None
    )


def _subprocess_env(seed: int | None) -> dict[str, str]:
    """Sanitised env for PyRosetta sidecars and fallback runner checks."""
    return sanitized_python_env(seed=seed)


def availability_skip_reason() -> str | None:
    """Return a human-readable reason if PyRosetta is unavailable, else None."""
    env = pyrosetta_env()
    py = pyrosetta_python()
    if not py.is_file():
        return (
            f"PyRosetta env not present at {env} (no python at {py}). "
            f"Provision via vendor/pyrosetta/UPSTREAM.md "
            f"(needs the PyRosetta wheel/channel). "
            f"Set $VPA_PYROSETTA_ENV to override the default path."
        )
    if not _pyrosetta_import_works(py):
        return (
            f"PyRosetta env at {env} exists but the sidecar bootstrap import failed "
            f"(`msgpack`, `vibeproteinagent.workers.sidecar_child`, or `pyrosetta`). "
            f"Install VPA + msgpack into that env and check the PyRosetta wheel/channel "
            f"per vendor/pyrosetta/UPSTREAM.md."
        )
    return None


# --------------------------------------------------------------------------- I/O models


class PyRosettaInput(BaseModel):
    pdb_path: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = Field(default="score", pattern=r"^(score|interface|relax|pack)$")
    score_fn: str = Field(default="ref2015", min_length=1)
    chains_a: str | None = Field(default=None)
    chains_b: str | None = Field(default=None)
    max_relax_cycles: int = Field(default=5, ge=1, le=25)
    seed: int | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_pdb_path(cls, data: Any) -> Any:
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
            structure_path = candidates[0].get("structure_path")
            if not structure_path:
                raise ValueError("pyrosetta canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("pdb_path", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _interface_needs_chains(self) -> PyRosettaInput:
        if not self.pdb_path:
            raise ValueError("pyrosetta requires pdb_path")
        if self.mode == "interface":
            if not self.chains_a or not self.chains_b:
                raise ValueError(
                    "mode=interface requires both chains_a and chains_b"
                )
        return self


class PyRosettaInterfaceMetrics(BaseModel):
    dG_separated: float | None = None
    dG_separated_per_int_area: float | None = None
    dSASA: float | None = None
    sc: float | None = None
    packstat: float | None = None
    nres_int: int | None = None
    interface_hbonds: int | None = None


class PyRosettaOutput(BaseModel):
    mode: str
    total_score: float
    interface_metrics: PyRosettaInterfaceMetrics | None = None
    relaxed_pdb_path: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int
    seed_used: int | None = None


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class PyRosettaAdapter(ToolAdapter):
    """Adapter backed by a persistent dedicated ``pyrosetta-vpa`` sidecar.

    The adapter never imports ``pyrosetta`` in the vpa process — it
    only resolves the env path lazily and dispatches framed RPC calls
    to ``<env>/bin/python -m vibeproteinagent.workers.sidecar_child``.
    That sidecar imports PyRosetta once and stays alive for the lifetime
    of this warm adapter instance, preserving the local-first/low-latency
    contract while keeping PyRosetta's native stack outside the shared ``vpa`` env.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        # Cache the env path at construction. Resolution is idempotent
        # and cheap; we re-resolve in healthcheck/run to pick up env
        # var changes between admission and execution.
        self._env_path = pyrosetta_env()
        self._device = device
        self._availability_checked = False
        self._sidecars: dict[int | None, PersistentPythonSidecar] = {}

    def run(self, params: PyRosettaInput, ctx: AdapterRunContext) -> PyRosettaOutput:
        t0 = time.time()
        ctx.emit_progress(0.02)

        self._ensure_available_once()

        assert params.pdb_path is not None
        pdb_src = Path(params.pdb_path).expanduser().resolve()
        if not pdb_src.is_file():
            raise FileNotFoundError(f"pdb_path not found: {pdb_src}")

        job_inputs = ctx.job_dir / "inputs"
        job_outputs = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)

        staged_pdb = job_inputs / pdb_src.name
        if staged_pdb.resolve() != pdb_src:
            shutil.copyfile(pdb_src, staged_pdb)

        spec = {
            "pdb_path": str(staged_pdb),
            "mode": params.mode,
            "score_fn": params.score_fn,
            "chains_a": params.chains_a,
            "chains_b": params.chains_b,
            "max_relax_cycles": params.max_relax_cycles,
            "seed": params.seed,
        }
        if params.mode in ("relax", "pack"):
            out_pdb_name = f"{staged_pdb.stem}.{params.mode}.pdb"
            spec["out_pdb"] = str(job_outputs / out_pdb_name)

        spec_path = job_inputs / "_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))

        ctx.emit_progress(0.10)
        try:
            payload = self._sidecar_for(params.seed).call(
                _RUNNER_SIDECAR_CALL,
                {"spec": spec},
                timeout_s=2 * 3600,
            )
        except VPAError as exc:
            raise RuntimeError(str(exc)) from exc
        ctx.emit_progress(0.92)

        result_path = job_outputs / "_result.json"
        result_path.write_text(json.dumps(payload, indent=2))
        if "error" in payload:
            raise RuntimeError(
                f"PyRosetta runner reported error: {payload['error']}"
            )

        out = PyRosettaOutput(
            mode=payload["mode"],
            total_score=float(payload["total_score"]),
            interface_metrics=(
                PyRosettaInterfaceMetrics(**payload["interface_metrics"])
                if payload.get("interface_metrics")
                else None
            ),
            relaxed_pdb_path=payload.get("relaxed_pdb_path"),
            candidates=_pyrosetta_candidates(
                params,
                float(payload["total_score"]),
                payload.get("interface_metrics"),
                payload.get("relaxed_pdb_path") or str(staged_pdb),
            ),
            runtime_ms=int((time.time() - t0) * 1000),
            seed_used=payload.get("seed_used"),
        )
        ctx.emit_progress(1.0)
        return out

    def estimate_vram(self, params: PyRosettaInput) -> int:
        del params
        # PyRosetta is CPU-only — no VRAM at all. Return 0 so the
        # scheduler treats it as a free pass on the VRAM check.
        return 0

    def healthcheck(self) -> bool:
        try:
            self._ensure_available_once()
            self._sidecar_for(None).call(
                _RUNNER_SIDECAR_WARM,
                {"seed": None},
                timeout_s=120.0,
            )
            return True
        except Exception:
            return False

    def unload(self) -> None:
        for sidecar in self._sidecars.values():
            sidecar.close()
        self._sidecars.clear()

    def _ensure_available_once(self) -> None:
        if self._availability_checked:
            return
        skip = availability_skip_reason()
        if skip is not None:
            raise RuntimeError(f"PyRosetta unavailable: {skip}")
        self._availability_checked = True

    def _sidecar_for(self, seed: int | None) -> PersistentPythonSidecar:
        key = int(seed) if seed is not None else None
        sidecar = self._sidecars.get(key)
        if sidecar is None:
            spec = SidecarRuntimeSpec(
                python=pyrosetta_python(),
                env=_subprocess_env(seed),
                name=f"pyrosetta[{key if key is not None else 'default'}]",
                warm_target=_RUNNER_SIDECAR_WARM,
                run_target=_RUNNER_SIDECAR_CALL,
                timeout_s=60.0,
            )
            sidecar = spec.spawn(PersistentPythonSidecar)
            self._sidecars[key] = sidecar
        return sidecar


# --------------------------------------------------------------------------- helpers


def _build_cmd(*, spec_path: Path, result_path: Path) -> list[str]:
    return [
        str(pyrosetta_python()),
        "-m",
        _RUNNER_MODULE,
        "--input",
        str(spec_path),
        "--output",
        str(result_path),
    ]


def _safe_read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _pyrosetta_candidates(
    params: PyRosettaInput,
    total_score: float,
    interface_metrics: dict[str, Any] | None,
    structure_path: str,
) -> list[dict[str, Any]]:
    local_id = f"{params.mode}_0000"
    scores: dict[str, Any] = {
        "mode": params.mode,
        "total_score": total_score,
    }
    if interface_metrics:
        scores["interface_metrics"] = interface_metrics
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            "pyrosetta",
            local_id,
            structure_path=structure_path,
            index=0,
        ),
        "source": {
            "tool": "pyrosetta",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": None,
            "index": 0,
        },
        "structure_path": structure_path,
        "scores": {"pyrosetta": scores},
        "artifacts": {"pyrosetta": {"relaxed_pdb_path": structure_path}},
        "capabilities": candidate_capability_flags(
            has_backbone_structure=bool(structure_path),
            has_full_structure=bool(structure_path),
            has_candidate_scores=True,
        ),
    }
    if params.candidates and _is_canonical_candidate(params.candidates[0]):
        source = params.candidates[0]
        upstream_scores = source.get("scores")
        if isinstance(upstream_scores, dict):
            candidate["scores"] = {**upstream_scores, **candidate["scores"]}
        if source.get("sequence"):
            candidate["sequence"] = source.get("sequence")
            candidate["capabilities"]["has_sequence"] = True
        if isinstance(source.get("capabilities"), dict):
            candidate["capabilities"] = dict(source["capabilities"])
            candidate["capabilities"]["has_candidate_scores"] = True
    return [candidate]


__all__ = [
    "PYROSETTA_ENV_DEFAULT",
    "PYROSETTA_ENV_NAME",
    "PyRosettaAdapter",
    "PyRosettaInput",
    "PyRosettaInterfaceMetrics",
    "PyRosettaOutput",
    "availability_skip_reason",
    "env_present",
    "pyrosetta_env",
    "pyrosetta_python",
]
