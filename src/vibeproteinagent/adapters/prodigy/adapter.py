from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _PROJECT_ROOT / "vendor" / "prodigy"

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from prodigy_prot.modules.parsers import parse_structure as _parse_structure  # noqa: E402
from prodigy_prot.modules.prodigy import Prodigy as _Prodigy  # noqa: E402


class ProdigyInput(BaseModel):
    pdb_path: str = Field(..., min_length=1)
    chain_a: str = Field(..., min_length=1)
    chain_b: str = Field(..., min_length=1)
    temperature_k: float = Field(298.15, gt=0.0)


class ProdigyOutput(BaseModel):
    delta_g_kcal_mol: float
    kd_nm: float
    interface_contacts: int
    runtime_ms: int


@register_adapter("descriptor.yaml")
class ProdigyAdapter(ToolAdapter):
    """Wraps haddocking/prodigy's `Prodigy` class for a single chain pair.

    PRODIGY is a knowledge-based regression on contact counts and
    non-interacting-surface area; no learned weights, no GPU.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        # Cheap-warm-up: a class load already; nothing else to pre-cache.

    def run(self, params: ProdigyInput, ctx: AdapterRunContext) -> ProdigyOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        pdb_path = Path(params.pdb_path).expanduser().resolve()
        if not pdb_path.exists():
            raise FileNotFoundError(f"PRODIGY: input PDB not found: {pdb_path}")

        models, _n_chains, _n_res = _parse_structure(str(pdb_path))
        if not models:
            raise ValueError(f"PRODIGY parser returned no models for {pdb_path}")
        model = models[0]
        ctx.emit_progress(0.25)

        present = {chain.id for chain in model.get_chains()}
        missing = {params.chain_a, params.chain_b} - present
        if missing:
            raise ValueError(
                f"PRODIGY: requested chains {sorted(missing)} not in PDB "
                f"(have {sorted(present)})"
            )

        # PRODIGY's API takes Celsius; descriptor accepts Kelvin (the SI choice).
        temp_c = float(params.temperature_k) - 273.15
        runner = _Prodigy(
            model=model,
            name=pdb_path.stem,
            selection=[params.chain_a, params.chain_b],
            temp=temp_c,
        )
        ctx.emit_progress(0.45)
        runner.predict(temp=temp_c)
        ctx.emit_progress(0.95)

        kd_nm = float(runner.kd_val) * 1e9  # Kd is reported in molar.
        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return ProdigyOutput(
            delta_g_kcal_mol=float(runner.ba_val),
            kd_nm=kd_nm,
            interface_contacts=len(runner.ic_network),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: ProdigyInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        return True


__all__ = ["ProdigyAdapter", "ProdigyInput", "ProdigyOutput"]
