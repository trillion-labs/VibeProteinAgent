"""RFdiffusion3 adapter backed by RosettaCommons Foundry.

RFD3 is now distributed through Foundry (``rc-foundry[rfd3]``) rather
than a standalone ``RFdiffusion3`` GitHub repo. The local-first contract
for this adapter is therefore:

* keep VPA's shared env clean; Foundry may pin pandas/atomworks/wandb
  versions that should not be forced into the main ``vpa`` env
* run the ``rfd3`` CLI from a dedicated compatibility env
* fail availability truthfully when the env or checkpoint is missing
* once provisioned, submit binder-design jobs through the local CLI and
  collect generated structure files from the job output directory
"""
from __future__ import annotations

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
from vibeproteinagent.agents.planning.candidate import stable_candidate_id
from vibeproteinagent.adapters.residue_mapping import (
    PositionBasis,
    parse_structure_residue_refs,
    resolve_positions,
)

# --------------------------------------------------------------------------- paths

_ADAPTER_DIR = Path(__file__).parent
_VPA_ROOT = _ADAPTER_DIR.parent.parent.parent.parent  # /VibeProteinAgent/

RFDIFFUSION3_TREE_DEFAULT = _VPA_ROOT / "vendor" / "rfdiffusion3"
RFDIFFUSION3_ENV_NAME = "rfdiffusion3-vpa"

RFDIFFUSION3_TREE = Path(
    os.environ.get("VPA_RFDIFFUSION3_TREE", str(RFDIFFUSION3_TREE_DEFAULT))
).expanduser()
RFDIFFUSION3_WEIGHTS = Path(
    os.environ.get("VPA_RFDIFFUSION3_WEIGHTS", str(RFDIFFUSION3_TREE / "checkpoints"))
).expanduser()

_FOUNDRY_AVAILABLE_MSG = (
    "RFdiffusion3 is publicly available through RosettaCommons Foundry "
    "(`pip install \"rc-foundry[rfd3]\"`, then `foundry install rfd3`)."
)
_RFD3_CKPT_NAMES = ("rfd3_latest.ckpt", "rfdiffusion3.ckpt", "rfd3.ckpt")


def availability_skip_reason() -> str | None:
    env = rfdiffusion3_env()
    if env is None:
        return (
            f"RFdiffusion3 is released via Foundry, but no local {RFDIFFUSION3_ENV_NAME} "
            f"env with an `rfd3` executable was found. Build one under "
            f"{RFDIFFUSION3_TREE / '.venv'} or set $VPA_RFDIFFUSION3_ENV."
        )
    if rfd3_executable(env) is None:
        return (
            f"RFdiffusion3 env at {env} exists but has no `rfd3` executable. "
            f"Install `rc-foundry[rfd3]` inside that env."
        )
    if rfdiffusion3_checkpoint() is None:
        return (
            f"RFdiffusion3 checkpoint missing under {RFDIFFUSION3_WEIGHTS}. "
            f"Run `DEBUG=false {env / 'bin' / 'foundry'} install rfd3 "
            f"--checkpoint-dir {RFDIFFUSION3_WEIGHTS}` or set $VPA_RFDIFFUSION3_CKPT."
        )
    return None


# --------------------------------------------------------------------------- I/O models


class RFdiffusion3Input(BaseModel):
    target_pdb: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    target_chain: str = Field(default="A", min_length=1, max_length=1)
    target_hotspots: list[int] = Field(default_factory=list)
    target_hotspot_residue_ids: list[str] = Field(default_factory=list)
    target_sequence: str | None = None
    target_fasta_path: str | None = Field(default=None, min_length=1)
    position_basis: PositionBasis = "auto"
    binder_length: int = Field(default=60, ge=30, le=200)
    num_designs: int = Field(default=8, ge=1, le=64)
    num_diffusion_steps: int = Field(default=200, ge=10, le=200)
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
                raise ValueError("rfdiffusion3 canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("target_pdb", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _require_target_pdb(self) -> RFdiffusion3Input:
        if not self.target_pdb:
            raise ValueError("rfdiffusion3 requires target_pdb")
        return self


class RFdiffusion3Design(BaseModel):
    pdb_path: str
    sequence: str | None = None
    length: int
    score: float | None = None


class RFdiffusion3Output(BaseModel):
    designs: list[RFdiffusion3Design]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int
    num_designs_returned: int
    position_mappings: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- adapter


@register_adapter("descriptor.yaml")
class RFdiffusion3Adapter(ToolAdapter):
    """Foundry-backed adapter for RFdiffusion3 binder backbone design."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        self.device = device
        self.env = rfdiffusion3_env()

    @torch.no_grad()
    def run(
        self, params: RFdiffusion3Input, ctx: AdapterRunContext
    ) -> RFdiffusion3Output:
        t0 = time.time()
        skip = availability_skip_reason()
        if skip is not None:
            raise RuntimeError(skip)
        env = rfdiffusion3_env()
        assert env is not None
        exe = rfd3_executable(env)
        ckpt = rfdiffusion3_checkpoint()
        assert exe is not None and ckpt is not None

        assert params.target_pdb is not None
        target = Path(params.target_pdb).expanduser().resolve()
        if not target.is_file():
            raise FileNotFoundError(f"target_pdb not found: {target}")

        job_inputs = ctx.job_dir / "inputs"
        out_dir = ctx.job_dir / "outputs"
        job_inputs.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        staged_target = job_inputs / target.name
        shutil.copyfile(target, staged_target)

        spec, position_mappings, mapping_warnings = _build_input_spec_with_mappings(
            params,
            staged_target,
        )
        inputs_path = job_inputs / "rfd3_inputs.json"
        inputs_path.write_text(json.dumps({"design": spec}, indent=2))

        cmd = [
            str(exe),
            "design",
            f"out_dir={out_dir}",
            f"ckpt_path={ckpt}",
            f"inputs={inputs_path}",
            f"diffusion_batch_size={params.num_designs}",
            "n_batches=1",
            f"inference_sampler.num_timesteps={params.num_diffusion_steps}",
            "inference_sampler.step_scale=1.5",
            "inference_sampler.gamma_0=0.6",
        ]
        if params.seed is not None:
            cmd.append(f"seed={params.seed}")

        ctx.emit_progress(0.05)
        subprocess.run(
            cmd,
            check=True,
            cwd=str(ctx.job_dir),
            env=_build_env(env),
            timeout=4 * 3600,
        )
        ctx.emit_progress(0.92)
        designs = _collect_designs(out_dir, params.binder_length)
        if not designs:
            raise RuntimeError(f"RFdiffusion3 completed but produced no structures under {out_dir}")
        ctx.emit_progress(1.0)
        return RFdiffusion3Output(
            designs=designs,
            candidates=_rfdiffusion3_candidates(params, designs),
            runtime_ms=_format_runtime_ms(t0),
            num_designs_returned=len(designs),
            position_mappings=position_mappings,
            warnings=mapping_warnings,
        )

    def estimate_vram(self, params: RFdiffusion3Input) -> int:
        del params
        return 24 * 1024

    def healthcheck(self) -> bool:
        return availability_skip_reason() is None

    def unload(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- helpers


def rfdiffusion3_env() -> Path | None:
    explicit = os.environ.get("VPA_RFDIFFUSION3_ENV")
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(RFDIFFUSION3_TREE / ".venv")
    for root in (
        Path("/opt/conda/envs"),
        Path.home() / ".conda" / "envs",
        Path("/opt/miniconda3/envs"),
    ):
        candidates.append(root / RFDIFFUSION3_ENV_NAME)
    current = Path(sys.prefix)
    if rfd3_executable(current) is not None:
        candidates.insert(0, current)
    for candidate in candidates:
        if rfd3_executable(candidate) is not None:
            return candidate
    return None


def rfd3_executable(env: Path) -> Path | None:
    direct = env / "bin" / "rfd3"
    if direct.is_file():
        return direct
    if env == Path(sys.prefix):
        found = shutil.which("rfd3")
        if found:
            return Path(found)
    return None


def rfdiffusion3_checkpoint() -> Path | None:
    explicit = os.environ.get("VPA_RFDIFFUSION3_CKPT")
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for name in _RFD3_CKPT_NAMES:
        p = RFDIFFUSION3_WEIGHTS / name
        if p.is_file():
            return p
    for p in sorted(RFDIFFUSION3_WEIGHTS.glob("*.ckpt")):
        if p.is_file():
            return p
    return None


def _build_env(env: Path) -> dict[str, str]:
    proc_env = os.environ.copy()
    proc_env["PYTHONNOUSERSITE"] = "1"
    proc_env["PATH"] = f"{env / 'bin'}{os.pathsep}{proc_env.get('PATH', '')}"
    proc_env["FOUNDRY_CHECKPOINT_DIRS"] = str(RFDIFFUSION3_WEIGHTS)
    # Foundry parses DEBUG as a boolean; cluster images may set DEBUG=release.
    if proc_env.get("DEBUG") not in (None, "", "0", "1", "false", "true", "False", "True"):
        proc_env["DEBUG"] = "false"
    return proc_env


def _target_residue_span(pdb: Path, chain: str) -> tuple[int, int]:
    residues: set[int] = set()
    with pdb.open() as fh:
        for line in fh:
            # Foundry component residue IDs are derived from polymer atoms.
            # Including chain-matched waters/ligands can incorrectly select a
            # longer HETATM span, e.g. A101-A216 waters in 6MRR.
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 26 or line[21].strip() != chain:
                continue
            try:
                residues.add(int(line[22:26]))
            except ValueError:
                continue
    if not residues:
        raise ValueError(f"No residues for chain {chain!r} found in {pdb}")
    ordered = sorted(residues)
    best_start = best_end = run_start = prev = ordered[0]
    for residue in ordered[1:]:
        if residue == prev + 1:
            prev = residue
            continue
        if prev - run_start > best_end - best_start:
            best_start, best_end = run_start, prev
        run_start = prev = residue
    if prev - run_start > best_end - best_start:
        best_start, best_end = run_start, prev
    return best_start, best_end


def _build_input_spec(params: RFdiffusion3Input, staged_target: Path) -> dict[str, object]:
    return _build_input_spec_with_mappings(params, staged_target)[0]


def _build_input_spec_with_mappings(
    params: RFdiffusion3Input,
    staged_target: Path,
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    start, end = _target_residue_span(staged_target, params.target_chain)
    hotspot_resolution = resolve_positions(
        residues=parse_structure_residue_refs(staged_target, params.target_chain),
        positions=params.target_hotspots,
        explicit_residue_ids=params.target_hotspot_residue_ids,
        chain_id=params.target_chain,
        sequence=params.target_sequence,
        fasta_path=params.target_fasta_path,
        position_basis=params.position_basis,
        label="target_hotspots",
    )
    hotspots = {
        residue_id: "CA" for residue_id in hotspot_resolution.residue_ids
    }
    spec: dict[str, object] = {
        "dialect": 2,
        "input": str(staged_target),
        "contig": (
            f"{params.binder_length}-{params.binder_length},/0,"
            f"{params.target_chain}{start}-{end}"
        ),
        "is_non_loopy": True,
    }
    if hotspots:
        spec["infer_ori_strategy"] = "hotspots"
        spec["select_hotspots"] = hotspots
    mappings = {
        "target_hotspots": hotspot_resolution.to_audit()
    }
    return spec, mappings, hotspot_resolution.warnings


def _collect_designs(out_dir: Path, fallback_length: int) -> list[RFdiffusion3Design]:
    structures = sorted(
        p
        for pattern in ("*.pdb", "*.cif", "*.cif.gz")
        for p in out_dir.rglob(pattern)
        if p.is_file()
    )
    return [
        RFdiffusion3Design(
            pdb_path=str(path),
            length=_infer_length(path) or fallback_length,
            score=None,
        )
        for path in structures
    ]


def _infer_length(path: Path) -> int | None:
    if path.suffix == ".gz":
        return None
    residues: set[tuple[str, int]] = set()
    try:
        with path.open(errors="ignore") as fh:
            for line in fh:
                if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 26:
                    try:
                        residues.add((line[21], int(line[22:26])))
                    except ValueError:
                        pass
    except OSError:
        return None
    return len(residues) or None


def _format_runtime_ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _rfdiffusion3_candidates(
    params: RFdiffusion3Input,
    designs: list[RFdiffusion3Design],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, design in enumerate(designs):
        local_id = f"design_{index:04d}"
        candidate: dict[str, Any] = {
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "rfdiffusion3",
                local_id,
                sequence=design.sequence,
                structure_path=design.pdb_path,
                index=index,
            ),
            "source": {
                "tool": "rfdiffusion3",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": None,
                "index": index,
            },
            "structure_path": design.pdb_path,
            "scores": {
                "rfdiffusion3": {
                    "length": design.length,
                    "score": design.score,
                }
            },
            "artifacts": {},
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
    "RFDIFFUSION3_TREE",
    "RFDIFFUSION3_WEIGHTS",
    "_FOUNDRY_AVAILABLE_MSG",
    "RFdiffusion3Adapter",
    "RFdiffusion3Design",
    "RFdiffusion3Input",
    "RFdiffusion3Output",
    "availability_skip_reason",
    "rfdiffusion3_checkpoint",
    "rfdiffusion3_env",
]
