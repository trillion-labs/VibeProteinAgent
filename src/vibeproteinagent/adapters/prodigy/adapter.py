from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import stable_candidate_id

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _PROJECT_ROOT / "vendor" / "prodigy"

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from prodigy_prot.modules.parsers import parse_structure as _parse_structure  # noqa: E402
from prodigy_prot.modules.prodigy import Prodigy as _Prodigy  # noqa: E402


class ProdigyInput(BaseModel):
    pdb_path: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    chain_a: str = Field(..., min_length=1)
    chain_b: str = Field(..., min_length=1)
    temperature_k: float = Field(298.15, gt=0.0)

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
                raise ValueError("prodigy canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("pdb_path", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _require_pdb_path(self) -> ProdigyInput:
        if not self.pdb_path:
            raise ValueError("prodigy requires pdb_path")
        return self


class ProdigyOutput(BaseModel):
    delta_g_kcal_mol: float
    kd_nm: float
    interface_contacts: int
    candidates: list[dict[str, Any]] = Field(default_factory=list)
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

        assert params.pdb_path is not None
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
            candidates=_prodigy_candidates(
                params,
                str(pdb_path),
                {
                    "delta_g_kcal_mol": float(runner.ba_val),
                    "kd_nm": kd_nm,
                    "interface_contacts": len(runner.ic_network),
                },
            ),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: ProdigyInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        return True


__all__ = ["ProdigyAdapter", "ProdigyInput", "ProdigyOutput"]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _prodigy_candidates(
    params: ProdigyInput,
    pdb_path: str,
    scores: dict[str, Any],
) -> list[dict[str, Any]]:
    upstream = params.candidates if params.candidates else [{}]
    out: list[dict[str, Any]] = []
    for index, source_candidate in enumerate(upstream):
        structure_path = source_candidate.get("structure_path") or pdb_path
        local_id = _source_local_id(source_candidate, f"prodigy_{index:04d}")
        if _is_canonical_candidate(source_candidate):
            candidate = _copy_candidate(source_candidate)
        else:
            candidate = {
                "schema_version": "candidate_v1",
                "candidate_id": stable_candidate_id(
                    "prodigy",
                    local_id,
                    structure_path=structure_path,
                    index=index,
                ),
                "source": {
                    "tool": "prodigy",
                    "node_id": None,
                    "job_id": None,
                    "local_id": local_id,
                    "native_id": None,
                    "index": index,
                },
                "structure_path": structure_path,
                "scores": {},
                "artifacts": {},
            }
        candidate.setdefault("scores", {})["prodigy"] = scores
        out.append(candidate)
    return out


def _copy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in candidate.items()
        if key in {"schema_version", "candidate_id", "source", "sequence", "structure_path"}
    }
    out["scores"] = dict(candidate.get("scores") or {})
    out["artifacts"] = dict(candidate.get("artifacts") or {})
    return out


def _source_local_id(candidate: dict[str, Any], fallback: str) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    return str(source.get("local_id") or fallback)
