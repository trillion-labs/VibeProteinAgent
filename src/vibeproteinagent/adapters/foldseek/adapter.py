from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.capabilities import candidate_capability_flags
from vibeproteinagent.candidates.schema import stable_candidate_id
from vibeproteinagent.core.binary import find_binary

_BINARY_CANDIDATES = ("foldseek",)


def _find_binary() -> Path:
    return find_binary(_BINARY_CANDIDATES, env_override="VPA_FOLDSEEK_BINARY")


class FoldseekInput(BaseModel):
    query_pdb: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    target_db: str = Field(..., min_length=1)
    mode: str = Field("easy-search", pattern=r"^(easy-search|easy-cluster)$")
    alignment_type: int = Field(2, ge=0, le=4)
    e_value: float = Field(0.001, gt=0.0)
    max_seqs: int = Field(1000, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_query(cls, data: Any) -> Any:
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
                raise ValueError("foldseek canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("query_pdb", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _require_query_pdb(self) -> FoldseekInput:
        if not self.query_pdb:
            raise ValueError("foldseek requires query_pdb")
        return self


class FoldseekOutput(BaseModel):
    hits_tsv: str
    num_hits: int = Field(..., ge=0)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int = Field(..., ge=0)


@register_adapter("descriptor.yaml")
class FoldseekAdapter(ToolAdapter):
    """Subprocess wrapper around `foldseek easy-search` (CPU build)."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.binary_path = _find_binary()

    def run(self, params: FoldseekInput, ctx: AdapterRunContext) -> FoldseekOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        assert params.query_pdb is not None
        query = Path(params.query_pdb).expanduser().resolve()
        if not query.is_file():
            raise FileNotFoundError(f"foldseek: query PDB not found: {query}")
        target = Path(params.target_db).expanduser().resolve()
        # target may be a file (FASTA/PDB) or a Foldseek DB prefix; allow either.
        if not (target.exists() or Path(str(target) + ".dbtype").exists()):
            raise FileNotFoundError(f"foldseek: target_db not found: {target}")

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = ctx.job_dir / "_foldseek_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        hits_tsv = out_dir / "hits.tsv"

        cmd = [
            str(self.binary_path),
            params.mode,
            str(query),
            str(target),
            str(hits_tsv),
            str(tmp_dir),
            "--format-output", "query,target,fident,alnlen,evalue,bits",
            "--alignment-type", str(params.alignment_type),
            "-e", f"{params.e_value}",
            "--max-seqs", str(params.max_seqs),
        ]
        ctx.emit_progress(0.20)
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        ctx.emit_progress(0.90)

        num_hits = 0
        if hits_tsv.is_file():
            with hits_tsv.open("r") as fh:
                for line in fh:
                    if line.strip():
                        num_hits += 1

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return FoldseekOutput(
            hits_tsv=str(hits_tsv),
            num_hits=num_hits,
            candidates=_foldseek_candidates(params, hits_tsv),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: FoldseekInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        try:
            _find_binary()
            return True
        except FileNotFoundError:
            return False


__all__ = ["FoldseekAdapter", "FoldseekInput", "FoldseekOutput"]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _foldseek_candidates(params: FoldseekInput, hits_tsv: str | Path) -> list[dict[str, Any]]:
    path = Path(hits_tsv)
    rows = path.read_text().splitlines() if path.is_file() else []
    candidates: list[dict[str, Any]] = []
    for index, line in enumerate(row for row in rows if row.strip()):
        cols = line.split("\t")
        target_id = cols[1] if len(cols) > 1 else f"hit_{index:04d}"
        local_id = f"hit_{index:04d}"
        candidates.append({
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "foldseek",
                local_id,
                native_id=target_id,
                index=index,
            ),
            "source": {
                "tool": "foldseek",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": target_id,
                "index": index,
            },
            "scores": {
                "foldseek": {
                    "fident": _safe_float(cols[2]) if len(cols) > 2 else None,
                    "alnlen": _safe_float(cols[3]) if len(cols) > 3 else None,
                    "evalue": _safe_float(cols[4]) if len(cols) > 4 else None,
                    "bits": _safe_float(cols[5]) if len(cols) > 5 else None,
                    "mode": params.mode,
                }
            },
            "artifacts": {"foldseek": {"hits_tsv": str(path)}},
            "capabilities": candidate_capability_flags(
                has_candidate_scores=True,
            ),
        })
    return candidates


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
