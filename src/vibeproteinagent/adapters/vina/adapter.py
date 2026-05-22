from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import torch
from pydantic import BaseModel, Field, field_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
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
    receptor_pdbqt: str = Field(..., min_length=1)
    ligand_pdbqt: str = Field(..., min_length=1)
    center: list[float] = Field(..., min_length=3, max_length=3)
    box_size: list[float] = Field(..., min_length=3, max_length=3)
    exhaustiveness: int = Field(8, ge=1)
    num_modes: int = Field(9, ge=1)
    seed: int | None = None

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
