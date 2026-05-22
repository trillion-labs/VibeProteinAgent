from __future__ import annotations

import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.binary import find_binary

_BINARY_CANDIDATES = ("foldseek",)


def _find_binary() -> Path:
    return find_binary(_BINARY_CANDIDATES, env_override="VPA_FOLDSEEK_BINARY")


class FoldseekInput(BaseModel):
    query_pdb: str = Field(..., min_length=1)
    target_db: str = Field(..., min_length=1)
    mode: str = Field("easy-search", pattern=r"^(easy-search|easy-cluster)$")
    alignment_type: int = Field(2, ge=0, le=4)
    e_value: float = Field(0.001, gt=0.0)
    max_seqs: int = Field(1000, ge=1)


class FoldseekOutput(BaseModel):
    hits_tsv: str
    num_hits: int = Field(..., ge=0)
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
