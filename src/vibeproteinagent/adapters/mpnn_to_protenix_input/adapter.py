from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext

_SCORE_RE = re.compile(r"score=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_CANDIDATE_ID_RE = re.compile(r"(?:^|[,| ]+)candidate_id=([^, ]+)")


class Mpnn2ProtenixInput(BaseModel):
    designs_fasta: str = Field(..., description="Path to a ProteinMPNN FASTA.")
    top_k: int = Field(1, ge=1, le=1024)
    name_prefix: str = Field("design", min_length=1, max_length=64)


class Mpnn2ProtenixOutput(BaseModel):
    input_jsons: list[str]
    names: list[str]
    candidates: list[dict]
    count: int


def _stable_candidate_id(index: int, sequence: str) -> str:
    token = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:10]
    return f"proteinmpnn:design_{index}:{token}"


def _parse_mpnn_fasta(fasta_path: Path) -> list[dict]:
    """Return candidate records for every non-input record in a ProteinMPNN FASTA.

    The proteinmpnn adapter writes the source sequence first with header
    ``>name|input, num_residues=N`` (no score), then one record per design with
    candidate metadata. We skip ``input`` rows and parse the ranking fields out
    of the remaining headers.
    """
    records: list[dict] = []
    header: str | None = None
    seq_chunks: list[str] = []

    def flush() -> None:
        nonlocal header, seq_chunks
        if header is None:
            seq_chunks = []
            return
        seq = "".join(seq_chunks).strip()
        if seq and "|input" not in header:
            score_match = _SCORE_RE.search(header)
            candidate_match = _CANDIDATE_ID_RE.search(header)
            index = len(records) + 1
            records.append(
                {
                    "candidate_id": (
                        candidate_match.group(1)
                        if candidate_match
                        else _stable_candidate_id(index, seq)
                    ),
                    "index": index - 1,
                    "sequence": seq,
                    "score": float(score_match.group(1)) if score_match else None,
                    "header": header,
                }
            )
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
            key=lambda r: (
                float("inf") if r.get("score") is None else float(r["score"]),
                str(r.get("candidate_id") or ""),
            ),
        )
        top = ranked[: params.top_k]

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        input_jsons: list[str] = []
        names: list[str] = []
        candidates: list[dict] = []
        for i, record in enumerate(top, start=1):
            name = f"{params.name_prefix}_{i}"
            seq = str(record["sequence"])
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
            candidates.append(
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": record["candidate_id"],
                    "sequence": seq,
                    "scores": {
                        "proteinmpnn": {
                            "score": record.get("score"),
                        }
                    },
                    "artifacts": {
                        "protenix_input_json": {
                            "path": str(target),
                            "kind": "protenix_input_json",
                        }
                    },
                    "capabilities": {
                        "has_sequence": True,
                        "has_candidate_scores": record.get("score") is not None,
                    },
                    "source": {
                        "tool": "mpnn_to_protenix_input",
                        "node_id": None,
                        "job_id": None,
                        "local_id": str(record["candidate_id"]),
                        "native_id": record.get("header"),
                        "index": i - 1,
                    },
                }
            )

        ctx.emit_progress(1.0)
        return Mpnn2ProtenixOutput(
            input_jsons=input_jsons,
            names=names,
            candidates=candidates,
            count=len(input_jsons),
        )

    def estimate_vram(self, params: Mpnn2ProtenixInput) -> int:
        return 0


__all__ = [
    "Mpnn2ProtenixAdapter",
    "Mpnn2ProtenixInput",
    "Mpnn2ProtenixOutput",
]
