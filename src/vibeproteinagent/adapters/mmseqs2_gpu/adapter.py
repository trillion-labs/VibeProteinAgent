from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.schema import stable_candidate_id
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
    query_fasta: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    target_db: str = Field(..., min_length=1)
    min_seq_id: float = Field(0.3, ge=0.0, le=1.0)
    coverage: float = Field(0.8, ge=0.0, le=1.0)
    sensitivity: float = Field(7.5, ge=1.0, le=7.5)
    max_seqs: int = Field(300, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _reject_mixed_candidates(cls, data: Any) -> Any:
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
        return data

    @model_validator(mode="after")
    def _require_query_source(self) -> MMseqs2GpuInput:
        if not self.query_fasta and not (
            self.candidates and all(_is_canonical_candidate(item) for item in self.candidates)
        ):
            raise ValueError("mmseqs2_gpu requires query_fasta or canonical Candidate[]")
        return self


class MMseqs2GpuOutput(BaseModel):
    hits_tsv: str
    num_hits: int = Field(..., ge=0)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
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

        query = _resolve_query_fasta(params, ctx.job_dir)
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
            candidates=_mmseqs2_gpu_candidates(params, hits_tsv),
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


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _resolve_query_fasta(params: MMseqs2GpuInput, job_dir: Path) -> Path:
    if params.query_fasta:
        return Path(params.query_fasta).expanduser().resolve()
    inputs_dir = job_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    query = inputs_dir / "canonical_candidates.fa"
    with query.open("w") as fh:
        for index, candidate in enumerate(params.candidates):
            sequence = candidate.get("sequence")
            if not sequence:
                raise ValueError("mmseqs2_gpu canonical Candidate[] input requires sequence")
            local_id = _source_local_id(candidate, f"seq_{index:04d}")
            fh.write(f">{local_id}\n{sequence}\n")
    return query


def _mmseqs2_gpu_candidates(
    params: MMseqs2GpuInput,
    hits_tsv: str | Path,
) -> list[dict[str, Any]]:
    path = Path(hits_tsv)
    rows: list[str] = path.read_text().splitlines() if path.is_file() else []
    candidates: list[dict[str, Any]] = []
    for index, line in enumerate(row for row in rows if row.strip()):
        cols = line.split("\t")
        target_id = cols[1] if len(cols) > 1 else f"hit_{index:04d}"
        scores = _mmseqs_scores(cols)
        local_id = f"hit_{index:04d}"
        candidates.append({
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "mmseqs2_gpu",
                local_id,
                native_id=target_id,
                index=index,
            ),
            "source": {
                "tool": "mmseqs2_gpu",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": target_id,
                "index": index,
            },
            "scores": {"mmseqs2_gpu": scores},
            "artifacts": {"mmseqs2_gpu": {"hits_tsv": str(path)}},
        })
    return candidates


def _mmseqs_scores(cols: list[str]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    if len(cols) > 2:
        scores["pident"] = _safe_float(cols[2])
    if len(cols) > 10:
        scores["evalue"] = _safe_float(cols[10])
    if len(cols) > 11:
        scores["bits"] = _safe_float(cols[11])
    return scores


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _source_local_id(candidate: dict[str, Any], fallback: str) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    return str(source.get("local_id") or fallback)
