from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.schema import stable_candidate_id
from vibeproteinagent.core.binary import find_binary

_BINARY_CANDIDATES = ("vina",)

# Vina prints a table on stdout once docking completes:
#
#    mode |   affinity | dist from best mode
#         | (kcal/mol) | rmsd l.b.| rmsd u.b.
#    -----+------------+----------+----------
#       1       -8.123      0.000      0.000
#       2       -7.880      1.234      2.001
#
# `rmsd l.b.` and `rmsd u.b.` are zero for mode 1.
_RE_MODE = re.compile(
    r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$"
)


def _find_binary() -> Path:
    return find_binary(_BINARY_CANDIDATES, env_override="VPA_VINA_BINARY")


class VinaMode(BaseModel):
    rank: int = Field(..., ge=1)
    affinity_kcal_mol: float
    rmsd_lb: float = Field(..., ge=0.0)
    rmsd_ub: float = Field(..., ge=0.0)


class VinaInput(BaseModel):
    receptor_pdbqt: str | None = Field(None, min_length=1)
    ligand_pdbqt: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    center: list[float] = Field(..., min_length=3, max_length=3)
    box_size: list[float] = Field(..., min_length=3, max_length=3)
    exhaustiveness: int = Field(8, ge=1)
    num_modes: int = Field(9, ge=1)
    seed: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_paths(cls, data: Any) -> Any:
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
            if len(candidates) < 2:
                raise ValueError("vina canonical Candidate[] input requires receptor and ligand")
            receptor = candidates[0].get("structure_path")
            ligand = candidates[1].get("structure_path")
            if not receptor or not ligand:
                raise ValueError("vina canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("receptor_pdbqt", receptor)
            out.setdefault("ligand_pdbqt", ligand)
            return out
        return data

    @model_validator(mode="after")
    def _require_paths(self) -> VinaInput:
        if not self.receptor_pdbqt or not self.ligand_pdbqt:
            raise ValueError("vina requires receptor_pdbqt and ligand_pdbqt")
        return self

    @field_validator("box_size")
    @classmethod
    def _box_size_positive(cls, v: list[float]) -> list[float]:
        if any(d <= 0.0 for d in v):
            raise ValueError("box_size dimensions must be > 0")
        return v


class VinaOutput(BaseModel):
    docked_pdbqt: str
    best_affinity_kcal_mol: float
    modes: list[VinaMode]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int = Field(..., ge=0)


def _parse_modes(stdout: str) -> list[VinaMode]:
    modes: list[VinaMode] = []
    in_table = False
    for line in stdout.splitlines():
        if "-----+------------+----------+----------" in line:
            in_table = True
            continue
        if not in_table:
            continue
        stripped = line.strip()
        if not stripped:
            if modes:
                break
            continue
        m = _RE_MODE.match(line)
        if not m:
            if modes:
                break
            continue
        modes.append(
            VinaMode(
                rank=int(m.group(1)),
                affinity_kcal_mol=float(m.group(2)),
                rmsd_lb=float(m.group(3)),
                rmsd_ub=float(m.group(4)),
            )
        )
    return modes


@register_adapter("descriptor.yaml")
class VinaAdapter(ToolAdapter):
    """Subprocess wrapper around AutoDock Vina (Trott & Olson, 2010)."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.binary_path = _find_binary()

    def run(self, params: VinaInput, ctx: AdapterRunContext) -> VinaOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        assert params.receptor_pdbqt is not None and params.ligand_pdbqt is not None
        receptor = Path(params.receptor_pdbqt).expanduser().resolve()
        ligand = Path(params.ligand_pdbqt).expanduser().resolve()
        for label, p in (("receptor_pdbqt", receptor), ("ligand_pdbqt", ligand)):
            if not p.is_file():
                raise FileNotFoundError(f"vina: {label} not found: {p}")

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        docked = out_dir / "docked.pdbqt"

        cx, cy, cz = params.center
        sx, sy, sz = params.box_size

        cmd: list[str] = [
            str(self.binary_path),
            "--receptor", str(receptor),
            "--ligand", str(ligand),
            "--center_x", f"{cx}",
            "--center_y", f"{cy}",
            "--center_z", f"{cz}",
            "--size_x", f"{sx}",
            "--size_y", f"{sy}",
            "--size_z", f"{sz}",
            "--exhaustiveness", str(params.exhaustiveness),
            "--num_modes", str(params.num_modes),
            "--out", str(docked),
        ]
        if params.seed is not None:
            cmd += ["--seed", str(params.seed)]

        ctx.emit_progress(0.20)
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        ctx.emit_progress(0.85)

        modes = _parse_modes(proc.stdout)
        if not modes:
            raise RuntimeError(
                "vina: no modes parsed from stdout. "
                f"First 400 chars: {proc.stdout[:400]!r}"
            )
        best = min(modes, key=lambda m: m.affinity_kcal_mol)

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return VinaOutput(
            docked_pdbqt=str(docked),
            best_affinity_kcal_mol=best.affinity_kcal_mol,
            modes=modes,
            candidates=_vina_candidates(params, modes, str(docked)),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: VinaInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        try:
            _find_binary()
            return True
        except FileNotFoundError:
            return False


__all__ = ["VinaAdapter", "VinaInput", "VinaMode", "VinaOutput", "_parse_modes"]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _vina_candidates(
    params: VinaInput,
    modes: list[VinaMode],
    docked_pdbqt: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, mode in enumerate(modes):
        local_id = f"mode_{index:04d}"
        candidates.append({
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "vina",
                local_id,
                structure_path=docked_pdbqt,
                index=index,
            ),
            "source": {
                "tool": "vina",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": str(mode.rank),
                "index": index,
            },
            "structure_path": docked_pdbqt,
            "scores": {
                "vina": {
                    "rank": mode.rank,
                    "affinity_kcal_mol": mode.affinity_kcal_mol,
                    "rmsd_lb": mode.rmsd_lb,
                    "rmsd_ub": mode.rmsd_ub,
                }
            },
            "artifacts": {"vina": {"docked_pdbqt": docked_pdbqt}},
        })
    if params.candidates and all(_is_canonical_candidate(item) for item in params.candidates):
        upstream_scores = params.candidates[0].get("scores")
        if isinstance(upstream_scores, dict):
            for candidate in candidates:
                candidate["scores"] = {**upstream_scores, **candidate["scores"]}
    return candidates
