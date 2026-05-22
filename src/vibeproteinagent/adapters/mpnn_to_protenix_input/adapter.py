from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext

_SCORE_RE = re.compile(r"score=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


class Mpnn2ProtenixInput(BaseModel):
    designs_fasta: str = Field(..., description="Path to a ProteinMPNN FASTA.")
    top_k: int = Field(1, ge=1, le=1024)
    name_prefix: str = Field("design", min_length=1, max_length=64)


class Mpnn2ProtenixOutput(BaseModel):
    input_jsons: list[str]
    names: list[str]
    count: int


def _parse_mpnn_fasta(fasta_path: Path) -> list[tuple[str, float | None]]:
    """Return (sequence, score) for every non-input record in a ProteinMPNN FASTA.

    The proteinmpnn adapter writes the source sequence first with header
    ``>name|input, num_residues=N`` (no score), then one record per design with
    header ``>name|design=K, T=…, score=S, seq_recovery=R``. We skip ``input``
    rows and parse ``score=...`` out of the remainder.
    """
    records: list[tuple[str, float | None]] = []
    header: str | None = None
    seq_chunks: list[str] = []

    def flush() -> None:
        nonlocal header, seq_chunks
        if header is None:
            seq_chunks = []
            return
        seq = "".join(seq_chunks).strip()
        if seq and "|input" not in header:
            m = _SCORE_RE.search(header)
            score = float(m.group(1)) if m else None
            records.append((seq, score))
        header = None
        seq_chunks = []

    with fasta_path.open("r") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith(">"):
                flush()
                header = line[1:]
            else:
                seq_chunks.append(line.strip())
    flush()
    return records


@register_adapter("descriptor.yaml")
class Mpnn2ProtenixAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self, params: Mpnn2ProtenixInput, ctx: AdapterRunContext
    ) -> Mpnn2ProtenixOutput:
        ctx.emit_progress(0.05)
        fasta_path = Path(params.designs_fasta).expanduser()
        if not fasta_path.is_file():
            raise FileNotFoundError(
                f"designs_fasta does not exist: {fasta_path}"
            )

        records = _parse_mpnn_fasta(fasta_path)
        if not records:
            raise ValueError(
                f"no design records found in {fasta_path} "
                f"(expected ProteinMPNN-style FASTA with '|design=' headers)"
            )

        # Lower score == better in ProteinMPNN (it's a negative log-likelihood).
        # Records without a parsed score fall to the end so a complete pool is
        # always preferred over a header-malformed one.
        ranked = sorted(
            records,
            key=lambda r: (float("inf") if r[1] is None else r[1]),
        )
        top = ranked[: params.top_k]

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        input_jsons: list[str] = []
        names: list[str] = []
        for i, (seq, _score) in enumerate(top, start=1):
            name = f"{params.name_prefix}_{i}"
            # Protenix v1 input JSON is a list of entries (each describing one
            # molecular system). For single-chain refold we emit a 1-element list.
            payload = [
                {
                    "name": name,
                    "sequences": [
                        {"proteinChain": {"sequence": seq, "count": 1}}
                    ],
                    "modelSeeds": [101],
                }
            ]
            target = out_dir / f"{name}.json"
            target.write_text(json.dumps(payload, indent=2))
            input_jsons.append(str(target))
            names.append(name)

        ctx.emit_progress(1.0)
        return Mpnn2ProtenixOutput(
            input_jsons=input_jsons, names=names, count=len(input_jsons)
        )

    def estimate_vram(self, params: Mpnn2ProtenixInput) -> int:
        return 0


__all__ = [
    "Mpnn2ProtenixAdapter",
    "Mpnn2ProtenixInput",
    "Mpnn2ProtenixOutput",
]
