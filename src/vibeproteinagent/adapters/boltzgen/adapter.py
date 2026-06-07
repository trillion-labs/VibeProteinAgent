"""BoltzGen adapter.

Pointer-style adapter — the BoltzGen Python source lives on disk at
``$VPA_BOLTZGEN_TREE`` (default ``$VPA_HOME/sidecars/boltzgen``)
and the HuggingFace-style model + dataset cache at ``$VPA_BOLTZGEN_CACHE``
(default ``$VPA_HOME/cache/boltzgen``).

Why subprocess: BoltzGen orchestrates a multi-step pipeline (design,
inverse_folding, folding, analysis) with non-trivial state in
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
import hashlib
import json
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
from vibeproteinagent.adapters.structure_analysis.adapter import _parse_structure
from vibeproteinagent.candidates.capabilities import candidate_capability_flags
from vibeproteinagent.candidates.schema import stable_candidate_id
from vibeproteinagent.core.config import vpa_home
from vibeproteinagent.core.errors import DependencyError, TimeoutError_

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


def availability_skip_reason() -> str | dict[str, Any] | None:
    try:
        _check_paths()
    except FileNotFoundError as exc:
        return str(exc)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import hydride; "
                "import boltzgen.task.analyze.analyze; "
                "import boltzgen.task.analyze.analyze_utils; "
                "print('ok')"
            ),
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_build_env(seed=None),
        timeout=60,
    )
    if probe.returncode == 0:
        return {
            "available": True,
            "checks": {"hf_token_present": _hf_token_present()},
        }
    stderr = (probe.stderr or probe.stdout or "").strip()
    return (
        "BoltzGen analyze-stack import probe failed. Install BoltzGen shared "
        f"runtime dependencies in the active VPA env. Details: {stderr[-1000:]}"
    )


# --------------------------------------------------------------------------- I/O models


class BoltzGenInput(BaseModel):
    target_pdb: str | None = Field(default=None)
    target_yaml: str | None = Field(default=None)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    target_chain: str = Field(default="A", min_length=1)
    binder_length: int = Field(default=60, ge=20, le=200)
    num_designs: int = Field(default=10, ge=1, le=256)
    num_diffusion_steps: int = Field(default=500, ge=5, le=1000)
    protocol: str = Field(default="protein-anything")
    seed: int | None = Field(default=None)

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
            structure_path = candidates[0].get("structure_path")
            if not structure_path:
                raise ValueError("boltzgen canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("target_pdb", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _exactly_one_target(self) -> BoltzGenInput:
        if (self.target_pdb is None) == (self.target_yaml is None):
            raise ValueError("provide exactly one of target_pdb or target_yaml")
        return self


class BoltzGenDesign(BaseModel):
    candidate_id: str | None = None
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
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    runtime_ms: int
    num_designs_returned: int
    seeds_used: list[int] = Field(default_factory=list)


# --------------------------------------------------------------------------- adapter


def _design_candidate_id(index: int, cif_path: Path, sequence: str | None = None) -> str:
    seed = f"{index}:{cif_path.name}:{sequence or ''}:{_file_digest(cif_path)}"
    token = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:10]
    return f"boltzgen:design_{index}:{token}"


@register_adapter("descriptor.yaml")
class BoltzGenAdapter(ToolAdapter):
    """Adapter for BoltzGen binder-design pipeline.

    Each ``run()`` shells out to ``python -m boltzgen.cli.boltzgen run``
    against the on-disk BoltzGen tree. Outputs are collected from
    ``intermediate_designs_inverse_folded/refold_cif/`` after folding/analysis.
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
        job_logs = ctx.job_dir / "logs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        job_outputs.mkdir(parents=True, exist_ok=True)
        job_logs.mkdir(parents=True, exist_ok=True)

        design_spec = _stage_design_spec(params, job_inputs)
        ctx.emit_progress(0.08)

        cmd = _build_cmd(
            design_spec=design_spec,
            output_dir=job_outputs,
            params=params,
        )
        env = _build_env(params.seed)

        ctx.emit_progress(0.10)
        stdout_path = job_logs / "boltzgen.stdout.log"
        stderr_path = job_logs / "boltzgen.stderr.log"
        command_path = job_logs / "boltzgen.command.json"
        command_path.write_text(
            json.dumps({"cmd": cmd, "cwd": str(job_outputs)}, indent=2, sort_keys=True)
            + "\n"
        )
        warnings: list[str] = []
        if not _hf_token_present(env):
            warnings.append(
                "HF_TOKEN is not set; BoltzGen may make unauthenticated HF Hub requests."
            )
        try:
            with stdout_path.open("w") as stdout_fh, stderr_path.open("w") as stderr_fh:
                subprocess.run(
                    cmd,
                    check=True,
                    env=env,
                    cwd=str(job_outputs),  # so relative file refs resolve under our jobdir
                    timeout=3600,
                    stdout=stdout_fh,
                    stderr=stderr_fh,
                    text=True,
                )
        except subprocess.CalledProcessError as exc:
            stage = _detect_failure_stage(stdout_path, stderr_path)
            if stage == "analysis":
                try:
                    designs = _collect_designs(output_dir=job_outputs, params=params)
                except RuntimeError:
                    designs = []
                if designs:
                    warnings.append(
                        "BoltzGen subprocess failed during analysis after producing "
                        "refolded designs; returning recovered design candidates "
                        "without analysis-derived BoltzGen scores."
                    )
                    ctx.emit_progress(0.92)
                    runtime_ms = int((time.time() - t0) * 1000)
                    ctx.emit_progress(1.0)
                    seeds = [params.seed] if params.seed is not None else []
                    return BoltzGenOutput(
                        designs=designs,
                        candidates=_boltzgen_candidates(params, designs),
                        warnings=warnings,
                        runtime_ms=runtime_ms,
                        num_designs_returned=len(designs),
                        seeds_used=seeds,
                    )
            raise DependencyError(
                f"BoltzGen subprocess failed with exit code {exc.returncode}",
                details={
                    "returncode": exc.returncode,
                    "cmd": cmd,
                    "cwd": str(job_outputs),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "command_path": str(command_path),
                    "stage": stage,
                    "stderr_tail": _tail(stderr_path),
                    "stdout_tail": _tail(stdout_path),
                },
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError_(
                "BoltzGen subprocess timed out",
                details={
                    "cmd": cmd,
                    "cwd": str(job_outputs),
                    "timeout_s": exc.timeout,
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                    "command_path": str(command_path),
                    "stage": _detect_failure_stage(stdout_path, stderr_path),
                },
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
            candidates=_boltzgen_candidates(params, designs),
            warnings=warnings,
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
        "--steps",
        "design",
        "inverse_folding",
        "folding",
        "analysis",
    ]
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
    env["HYDRA_FULL_ERROR"] = "1"
    if not env.get("HF_TOKEN"):
        for alias in ("VPA_HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            if env.get(alias):
                env["HF_TOKEN"] = env[alias]
                break
    # BoltzGen reads HF_HOME / HF_HUB_CACHE for some auxiliary downloads
    # (e.g. tokenizer). Point both at our cache.
    env["HF_HUB_CACHE"] = str(BOLTZGEN_CACHE)
    if seed is not None:
        env["PYTHONHASHSEED"] = str(seed)
    return env


def _hf_token_present(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return bool(
        env.get("HF_TOKEN")
        or env.get("VPA_HF_TOKEN")
        or env.get("HUGGINGFACE_HUB_TOKEN")
    )


def _tail(path: Path, *, limit: int = 4000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(errors="replace")
    return text[-limit:]


def _detect_failure_stage(stdout_path: Path, stderr_path: Path) -> str | None:
    text = f"{_tail(stdout_path, limit=12000)}\n{_tail(stderr_path, limit=12000)}".lower()
    stage_aliases = {
        "design": ("design",),
        "inverse_folding": ("inverse_folding", "inverse_fold"),
        "folding": ("folding",),
        "analysis": ("analysis", "analyze"),
    }
    positions = {
        stage: max(text.rfind(alias) for alias in aliases)
        for stage, aliases in stage_aliases.items()
    }
    stage, pos = max(positions.items(), key=lambda item: item[1])
    return stage if pos >= 0 else None


def _collect_designs(
    *,
    output_dir: Path,
    params: BoltzGenInput,
) -> list[BoltzGenDesign]:
    """Walk BoltzGen's output layout and return canonical BoltzGenDesign records.

    VPA runs design -> inverse_folding -> folding -> analysis and intentionally
    skips BoltzGen's filter step. The primary output is therefore the refolded
    CIF set under ``intermediate_designs_inverse_folded/refold_cif``. If a
    final-ranked directory is present from a previous/upstream-compatible run,
    it is accepted as a fallback, but filtering is never required for success.
    """
    designs: list[BoltzGenDesign] = []
    fallback = output_dir / "intermediate_designs_inverse_folded" / "refold_cif"
    if fallback.is_dir():
        metrics_by_id = _load_boltzgen_metrics(output_dir)
        for idx, cif in enumerate(sorted(fallback.glob("*.cif")), start=1):
            row = metrics_by_id.get(cif.stem, {})
            sequence = row.get("sequence") or _recover_design_sequence(cif, params)
            length = _safe_int(row.get("design_length")) or (
                len(sequence) if sequence else params.binder_length
            )
            designs.append(
                BoltzGenDesign(
                    candidate_id=_design_candidate_id(idx, cif, sequence),
                    cif_path=str(cif),
                    sequence=sequence,
                    length=length,
                    score=_safe_float(row.get("score")),
                    affinity_pred_value=_safe_float(row.get("affinity_pred_value")),
                    affinity_probability_binary=_safe_float(
                        row.get("affinity_probability_binary")
                    ),
                    confidence=_safe_float(row.get("iptm"))
                    or _safe_float(row.get("confidence")),
                )
            )
        if designs:
            return designs

    final_dir = output_dir / "final_ranked_designs"
    final_set_dir = next(final_dir.glob("final_*_designs"), None)
    if final_set_dir is None:
        raise RuntimeError(
            f"BoltzGen did not produce refolded design CIFs under {fallback}."
        )

    metrics_by_id = _load_boltzgen_metrics(output_dir)

    for idx, cif in enumerate(sorted(final_set_dir.glob("*.cif")), start=1):
        stem = cif.stem
        row = metrics_by_id.get(stem, {})
        sequence = row.get("sequence") or _recover_design_sequence(cif, params)
        length = _safe_int(row.get("design_length")) or (
            len(sequence) if sequence else params.binder_length
        )
        designs.append(
            BoltzGenDesign(
                candidate_id=_design_candidate_id(idx, cif, sequence),
                cif_path=str(cif),
                pdb_path=None,
                sequence=sequence,
                length=length,
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


def _load_boltzgen_metrics(output_dir: Path) -> dict[str, dict[str, str]]:
    metrics_by_id: dict[str, dict[str, str]] = {}
    for metrics_csv in (
        output_dir / "intermediate_designs_inverse_folded" / "all_designs_metrics.csv",
        output_dir / "final_ranked_designs" / "all_designs_metrics.csv",
    ):
        if not metrics_csv.is_file():
            continue
        with metrics_csv.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get("id") or row.get("design_id") or row.get("name") or ""
                if key:
                    metrics_by_id[Path(key).stem] = row
    return metrics_by_id


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def _recover_design_sequence(cif_path: Path, params: BoltzGenInput) -> str | None:
    try:
        residues, _ligands = _parse_structure(cif_path)
    except Exception:
        return None
    chains: dict[str, list[str]] = {}
    for residue in residues:
        if residue.aa == "X":
            continue
        chains.setdefault(residue.chain_id, []).append(residue.aa)
    if not chains:
        return None

    if params.target_pdb:
        target_chain = params.target_chain
        non_target = {
            chain_id: seq
            for chain_id, seq in chains.items()
            if chain_id != target_chain
        }
        if non_target:
            chain_id, seq = min(
                non_target.items(),
                key=lambda item: (abs(len(item[1]) - params.binder_length), item[0]),
            )
            del chain_id
            return "".join(seq)

    chain_id, seq = max(chains.items(), key=lambda item: (len(item[1]), item[0]))
    del chain_id
    return "".join(seq)


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


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _boltzgen_candidates(
    params: BoltzGenInput,
    designs: list[BoltzGenDesign],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, design in enumerate(designs):
        local_id = f"design_{index:04d}"
        structure_path = design.pdb_path or design.cif_path
        candidate: dict[str, Any] = {
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "boltzgen",
                local_id,
                native_id=design.candidate_id,
                sequence=design.sequence,
                structure_path=structure_path,
                index=index,
            ),
            "source": {
                "tool": "boltzgen",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": design.candidate_id,
                "index": index,
            },
            "structure_path": structure_path,
            "scores": {
                "boltzgen": {
                    "length": design.length,
                    "score": design.score,
                    "affinity_pred_value": design.affinity_pred_value,
                    "affinity_probability_binary": design.affinity_probability_binary,
                    "confidence": design.confidence,
                }
            },
            "artifacts": {"boltzgen": {"cif_path": design.cif_path, "pdb_path": design.pdb_path}},
            "capabilities": candidate_capability_flags(
                has_sequence=bool(design.sequence),
                has_backbone_structure=bool(structure_path),
                has_full_structure=bool(structure_path),
                has_candidate_scores=True,
            ),
        }
        if design.sequence:
            candidate["sequence"] = design.sequence
        if params.candidates and _is_canonical_candidate(params.candidates[0]):
            upstream_scores = params.candidates[0].get("scores")
            if isinstance(upstream_scores, dict):
                candidate["scores"] = {**upstream_scores, **candidate["scores"]}
        candidates.append(candidate)
    return candidates


__all__ = [
    "BOLTZGEN_CACHE",
    "BOLTZGEN_TREE",
    "BoltzGenAdapter",
    "BoltzGenDesign",
    "BoltzGenInput",
    "BoltzGenOutput",
    "availability_skip_reason",
]
