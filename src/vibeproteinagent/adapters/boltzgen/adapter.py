"""BoltzGen adapter.

Pointer-style adapter — the BoltzGen Python source lives on disk at
``$VPA_BOLTZGEN_TREE`` (default ``$VPA_HOME/sidecars/boltzgen``)
and the HuggingFace-style model + dataset cache at ``$VPA_BOLTZGEN_CACHE``
(default ``$VPA_HOME/cache/boltzgen``).

Why subprocess: BoltzGen orchestrates a multi-step pipeline (design,
inverse_fold, folding, analyze, filter) with non-trivial state in
``omegaconf`` configs and PyTorch Lightning callbacks. Driving each
``boltzgen run`` invocation as a subprocess gives us the strongest
isolation (no global-state bleed between adapters, no leftover
Lightning ``Trainer`` reinitialisation in the same Python process).
Inside the subprocess BoltzGen runs with ``--no_subprocess`` so each
pipeline step still executes in one Python process.

Why ``--use_kernels false``: the cuequivariance CUDA wheels are not
installed in the ``vpa`` env. BoltzGen's pure-PyTorch fallback path
produces identical results, slightly slower.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.config import vpa_home

# --------------------------------------------------------------------------- paths

BOLTZGEN_TREE_DEFAULT = str(vpa_home() / "sidecars" / "boltzgen")
BOLTZGEN_CACHE_DEFAULT = str(vpa_home() / "cache" / "boltzgen")

BOLTZGEN_TREE = Path(
    os.environ.get("VPA_BOLTZGEN_TREE", BOLTZGEN_TREE_DEFAULT)
).expanduser()
BOLTZGEN_CACHE = Path(
    os.environ.get("VPA_BOLTZGEN_CACHE", BOLTZGEN_CACHE_DEFAULT)
).expanduser()


def _check_paths() -> None:
    if not BOLTZGEN_TREE.is_dir():
        raise FileNotFoundError(
            f"BoltzGen source tree not found at {BOLTZGEN_TREE}. "
            f"Set $VPA_BOLTZGEN_TREE to a BoltzGen checkout (must contain src/boltzgen/)."
        )
    main_py = BOLTZGEN_TREE / "src" / "boltzgen" / "cli" / "boltzgen.py"
    if not main_py.is_file():
        raise FileNotFoundError(
            f"{main_py} not found; $VPA_BOLTZGEN_TREE does not look like "
            "a BoltzGen source checkout."
        )
    if not BOLTZGEN_CACHE.is_dir():
        raise FileNotFoundError(
            f"BoltzGen cache directory not found at {BOLTZGEN_CACHE}. "
            "Set $VPA_BOLTZGEN_CACHE to a directory pre-populated with "
            "the boltzgen-1 HF snapshots."
        )


# --------------------------------------------------------------------------- I/O models


class BoltzGenInput(BaseModel):
    target_pdb: str | None = Field(default=None)
    target_yaml: str | None = Field(default=None)
    target_chain: str = Field(default="A", min_length=1)
    binder_length: int = Field(default=60, ge=20, le=200)
    num_designs: int = Field(default=10, ge=1, le=256)
    num_diffusion_steps: int = Field(default=500, ge=5, le=1000)
    protocol: str = Field(default="protein-anything")
    design_only: bool = Field(default=True)
    budget: int = Field(default=10, ge=1, le=256)
    seed: int | None = Field(default=None)

    @model_validator(mode="after")
    def _exactly_one_target(self) -> BoltzGenInput:
        if (self.target_pdb is None) == (self.target_yaml is None):
            raise ValueError("provide exactly one of target_pdb or target_yaml")
        return self


class BoltzGenDesign(BaseModel):
    cif_path: str
    pdb_path: str | None = None
    sequence: str | None = None
    length: int
    score: float | None = None
    affinity_pred_value: float | None = None
    affinity_probability_binary: float | None = None
    confidence: float | None = None


class BoltzGenOutput(BaseModel):
    designs: list[BoltzGenDesign]
    runtime_ms: int
    num_designs_returned: int
    seeds_used: list[int] = Field(default_factory=list)


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class BoltzGenAdapter(ToolAdapter):
    """Adapter for BoltzGen binder-design pipeline.

    Each ``run()`` shells out to ``python -m boltzgen.cli.boltzgen run``
    against the on-disk BoltzGen tree. Outputs are collected from
    ``intermediate_designs/`` (design-only mode) or
    ``final_ranked_designs/`` (full pipeline mode).
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        self.device = device
        _check_paths()

    @torch.no_grad()
    def run(self, params: BoltzGenInput, ctx: AdapterRunContext) -> BoltzGenOutput:
        t0 = time.time()
        ctx.emit_progress(0.02)

        if not torch.cuda.is_available():
            raise RuntimeError("BoltzGen adapter requires CUDA")

        job_inputs = ctx.job_dir / "inputs"
        job_outputs = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)

        design_spec = _stage_design_spec(params, job_inputs)
        ctx.emit_progress(0.08)

        # Decide which pipeline steps to run. Design-only is the fastest path
        # (single diffusion sampling), full pipeline adds inverse_fold +
        # folding + analyze + filter.
        if params.design_only:
            steps = ["design"]
        else:
            steps = None  # all steps

        cmd = _build_cmd(
            design_spec=design_spec,
            output_dir=job_outputs,
            params=params,
            steps=steps,
        )
        env = _build_env(params.seed)

        ctx.emit_progress(0.10)
        # Inherit the parent stdio so the supervisor's log fan-out can see
        # BoltzGen's progress prints; subprocess.run with no capture is fine.
        try:
            subprocess.run(
                cmd,
                check=True,
                env=env,
                cwd=str(job_outputs),  # so relative file refs resolve under our jobdir
                timeout=3600,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"BoltzGen subprocess failed with exit code {exc.returncode}"
            ) from exc
        ctx.emit_progress(0.92)

        designs = _collect_designs(
            output_dir=job_outputs,
            params=params,
        )

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        seeds = [params.seed] if params.seed is not None else []
        return BoltzGenOutput(
            designs=designs,
            runtime_ms=runtime_ms,
            num_designs_returned=len(designs),
            seeds_used=seeds,
        )

    def estimate_vram(self, params: BoltzGenInput) -> int:
        # BoltzGen design model peaks around ~10-14 GiB on B200 for short binders;
        # refolding (Boltz-2) adds another ~6-8 GiB. Conservative ceiling for now.
        del params
        return 20 * 1024

    def healthcheck(self) -> bool:
        try:
            _check_paths()
            return True
        except FileNotFoundError:
            return False

    def unload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- helpers


def _stage_design_spec(params: BoltzGenInput, dest_dir: Path) -> Path:
    """Materialise the design spec under ``dest_dir/design.yaml``.

    When ``target_yaml`` is provided, copy it verbatim. When ``target_pdb`` is
    provided, generate a minimal spec referencing the target with the user-
    specified chain and a fixed binder length.
    """
    target = dest_dir / "design.yaml"
    if params.target_yaml is not None:
        src = Path(params.target_yaml).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"target_yaml not found: {src}")
        # Sanity-load to surface YAML syntax errors as a clear early failure.
        with src.open("r") as fh:
            doc = yaml.safe_load(fh)
        if not isinstance(doc, dict) or "entities" not in doc:
            raise ValueError(
                f"target_yaml at {src} does not look like a BoltzGen spec: "
                f"missing top-level 'entities' key"
            )
        # Copy file (preserves any relative-path references that the user has
        # placed alongside the YAML — we'll launch with cwd=dest_dir so
        # ``file: path:`` entries resolve from there. Symlink the target
        # files we know about.)
        target.write_text(src.read_text())
        # Best-effort: copy any sibling files referenced via ``file: path:``
        # entries so they're visible from dest_dir.
        _stage_referenced_files(doc, src.parent, dest_dir)
        return target

    # target_pdb path — build a minimal spec inline.
    pdb_path = Path(params.target_pdb).expanduser().resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(f"target_pdb not found: {pdb_path}")
    staged_target = dest_dir / pdb_path.name
    shutil.copyfile(pdb_path, staged_target)

    binder_letter = "B" if params.target_chain.upper() == "A" else "A"
    doc = {
        "entities": [
            {
                "protein": {
                    "id": binder_letter,
                    "sequence": params.binder_length,
                }
            },
            {
                "file": {
                    "path": pdb_path.name,
                    "include": [{"chain": {"id": params.target_chain}}],
                }
            },
        ]
    }
    target.write_text(yaml.safe_dump(doc, sort_keys=False))
    return target


def _stage_referenced_files(
    doc: dict[str, Any], src_dir: Path, dest_dir: Path
) -> None:
    """Copy any ``file.path`` references in the spec into dest_dir.

    BoltzGen resolves ``file: path:`` entries relative to the YAML file's
    own directory. When we copy the YAML into the job dir, sibling files
    need to follow.
    """
    for entity in doc.get("entities", []):
        if not isinstance(entity, dict):
            continue
        file_block = entity.get("file")
        if not isinstance(file_block, dict):
            continue
        rel = file_block.get("path")
        if not isinstance(rel, str):
            continue
        src = (src_dir / rel).resolve()
        if not src.is_file():
            # Let BoltzGen produce its own (clearer) error rather than guess.
            continue
        dst = dest_dir / Path(rel).name
        if not dst.exists():
            shutil.copyfile(src, dst)
        # Rewrite the path so it's relative to dest_dir.
        file_block["path"] = Path(rel).name


def _build_cmd(
    *,
    design_spec: Path,
    output_dir: Path,
    params: BoltzGenInput,
    steps: list[str] | None,
) -> list[str]:
    """Assemble the ``boltzgen run`` command line."""
    cmd: list[str] = [
        sys.executable,
        "-m",
        "boltzgen.cli.boltzgen",
        "run",
        str(design_spec),
        "--output",
        str(output_dir),
        "--cache",
        str(BOLTZGEN_CACHE),
        "--protocol",
        params.protocol,
        "--num_designs",
        str(params.num_designs),
        "--budget",
        str(params.budget),
        "--devices",
        "1",
        "--num_workers",
        "1",
        "--use_kernels",
        "false",
        "--no_subprocess",
        "--config",
        "design",
        f"sampling_steps={params.num_diffusion_steps}",
    ]
    if steps is not None:
        cmd.extend(["--steps", *steps])
    return cmd


def _build_env(seed: int | None) -> dict[str, str]:
    """Build the environment dict for the subprocess.

    Pre-seeds the python path so BoltzGen's source tree is importable in
    the child process, and routes HF lookups to the local cache.
    """
    env = os.environ.copy()
    # vpa-env hygiene; the supervisor already sets these but the subprocess
    # might inherit a stale PYTHONPATH.
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    # Make sure the BoltzGen source tree is the first thing on sys.path so
    # `python -m boltzgen.cli.boltzgen` resolves locally even though no
    # pip install was performed.
    src = str(BOLTZGEN_TREE / "src")
    env["PYTHONPATH"] = src
    # BoltzGen reads HF_HOME / HF_HUB_CACHE for some auxiliary downloads
    # (e.g. tokenizer). Point both at our cache.
    env["HF_HUB_CACHE"] = str(BOLTZGEN_CACHE)
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    return env


def _collect_designs(
    *,
    output_dir: Path,
    params: BoltzGenInput,
) -> list[BoltzGenDesign]:
    """Walk BoltzGen's output layout and return canonical BoltzGenDesign records.

    Two cases:

    1. ``design_only=True`` — read ``output_dir/intermediate_designs/*.cif``
       (skipping ``*_native.cif`` reference structures). No sequences / scores
       yet at this stage.
    2. ``design_only=False`` — read ``output_dir/final_ranked_designs/
       final_<budget>_designs/`` and load per-design scores from
       ``all_designs_metrics.csv``.
    """
    designs: list[BoltzGenDesign] = []
    if params.design_only:
        design_dir = output_dir / "intermediate_designs"
        if design_dir.is_dir():
            search_dir = design_dir
        else:
            search_dir = output_dir
        if not search_dir.is_dir():
            raise RuntimeError(
                f"BoltzGen design step did not produce {design_dir} or {output_dir}. "
                f"Inspect the worker log for the underlying failure."
            )
        cifs = sorted(
            fp
            for fp in search_dir.glob("*.cif")
            if not fp.name.endswith("_native.cif")
        )
        if not cifs:
            raise RuntimeError(
                f"BoltzGen produced no design CIFs under {search_dir}"
            )
        for cif in cifs:
            designs.append(
                BoltzGenDesign(
                    cif_path=str(cif),
                    pdb_path=None,
                    sequence=None,
                    length=params.binder_length,
                    score=None,
                    affinity_pred_value=None,
                    affinity_probability_binary=None,
                    confidence=None,
                )
            )
        return designs

    # full pipeline path
    final_dir = output_dir / "final_ranked_designs"
    metrics_csv = final_dir / "all_designs_metrics.csv"
    final_set_dir = next(final_dir.glob("final_*_designs"), None)
    if final_set_dir is None:
        # No designs survived filtering — fall back to refold_cif.
        fallback = output_dir / "intermediate_designs_inverse_folded" / "refold_cif"
        if fallback.is_dir():
            for cif in sorted(fallback.glob("*.cif")):
                designs.append(
                    BoltzGenDesign(
                        cif_path=str(cif),
                        length=params.binder_length,
                    )
                )
            return designs
        raise RuntimeError(
            f"BoltzGen did not produce a final design set under {final_dir} "
            f"and no refold_cif fallback exists at {fallback}."
        )

    metrics_by_id: dict[str, dict[str, str]] = {}
    if metrics_csv.is_file():
        with metrics_csv.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get("id") or row.get("design_id") or row.get("name") or ""
                if key:
                    metrics_by_id[key] = row

    for cif in sorted(final_set_dir.glob("*.cif")):
        stem = cif.stem
        row = metrics_by_id.get(stem, {})
        designs.append(
            BoltzGenDesign(
                cif_path=str(cif),
                pdb_path=None,
                sequence=row.get("sequence") or None,
                length=_safe_int(row.get("design_length")) or params.binder_length,
                score=_safe_float(row.get("score")),
                affinity_pred_value=_safe_float(row.get("affinity_pred_value")),
                affinity_probability_binary=_safe_float(
                    row.get("affinity_probability_binary")
                ),
                confidence=_safe_float(row.get("iptm"))
                or _safe_float(row.get("confidence")),
            )
        )
    return designs


def _safe_int(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _safe_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BOLTZGEN_CACHE",
    "BOLTZGEN_TREE",
    "BoltzGenAdapter",
    "BoltzGenDesign",
    "BoltzGenInput",
    "BoltzGenOutput",
]
