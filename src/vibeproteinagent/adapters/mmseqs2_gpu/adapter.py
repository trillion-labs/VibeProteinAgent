from __future__ import annotations

import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.binary import find_binary

_BINARY_CANDIDATES = ("mmseqs-gpu", "mmseqs")
# Default location surveyed in note 7f1c5d9b-… (Toolchain survey).
_DEFAULT_PATH = Path("/opt/vpa-data/mmseqs2-gpu/bin/mmseqs")


def _find_binary() -> Path:
    try:
        return find_binary(_BINARY_CANDIDATES, env_override="VPA_MMSEQS_BINARY")
    except FileNotFoundError:
        if _DEFAULT_PATH.is_file():
            return _DEFAULT_PATH
        raise


class MMseqs2GpuInput(BaseModel):
    query_fasta: str = Field(..., min_length=1)
    target_db: str = Field(..., min_length=1)
    min_seq_id: float = Field(0.3, ge=0.0, le=1.0)
    coverage: float = Field(0.8, ge=0.0, le=1.0)
    sensitivity: float = Field(7.5, ge=1.0, le=7.5)
    max_seqs: int = Field(300, ge=1)


class MMseqs2GpuOutput(BaseModel):
    hits_tsv: str
    num_hits: int = Field(..., ge=0)
    runtime_ms: int = Field(..., ge=0)


@register_adapter("descriptor.yaml")
class MMseqs2GpuAdapter(ToolAdapter):
    """Subprocess wrapper around `mmseqs easy-search` (GPU build).

    The MMseqs2-gpu binary uses CUDA internally; from the Python side this is
    still a plain ``subprocess.run`` call — no torch / cuda imports here.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.binary_path = _find_binary()

    def run(self, params: MMseqs2GpuInput, ctx: AdapterRunContext) -> MMseqs2GpuOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        query = Path(params.query_fasta).expanduser().resolve()
        if not query.is_file():
            raise FileNotFoundError(f"mmseqs2_gpu: query FASTA not found: {query}")
        target = Path(params.target_db).expanduser().resolve()
        # target can be either a FASTA file or an mmseqs DB prefix; we only
        # require that something at the path or its `.dbtype` index exists.
        if not (target.exists() or Path(str(target) + ".dbtype").exists()):
            raise FileNotFoundError(f"mmseqs2_gpu: target_db not found: {target}")

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = ctx.job_dir / "_mmseqs_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        hits_tsv = out_dir / "hits.tsv"

        cmd = [
            str(self.binary_path),
            "easy-search",
            str(query),
            str(target),
            str(hits_tsv),
            str(tmp_dir),
            "-s", f"{params.sensitivity}",
            "--min-seq-id", f"{params.min_seq_id}",
            "-c", f"{params.coverage}",
            "--max-seqs", f"{params.max_seqs}",
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
        return MMseqs2GpuOutput(
            hits_tsv=str(hits_tsv),
            num_hits=num_hits,
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: MMseqs2GpuInput) -> int:
        return 4 * 1024

    def healthcheck(self) -> bool:
        try:
            _find_binary()
            return True
        except FileNotFoundError:
            return False


__all__ = ["MMseqs2GpuAdapter", "MMseqs2GpuInput", "MMseqs2GpuOutput"]
