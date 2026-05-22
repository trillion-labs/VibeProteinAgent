from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from vibeproteinagent.adapters._remote.fasta import parse_fasta
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext

_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class SequenceToProtenixInput(BaseModel):
    sequence: str | None = None
    fasta_path: str | None = Field(None, json_schema_extra={"vpa:digest": "sha256-if-path"})
    name: str = Field("sequence", pattern=r"^[A-Za-z0-9_.-]+$")
    count: int = Field(1, ge=1, le=1024)
    seeds: list[int] = Field(default_factory=lambda: [101], min_length=1)

    @field_validator("sequence")
    @classmethod
    def _normalize_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return "".join(value.split()).upper()

    @model_validator(mode="after")
    def _has_source(self) -> SequenceToProtenixInput:
        if not self.sequence and not self.fasta_path:
            raise ValueError("sequence_to_protenix_input requires sequence or fasta_path")
        return self


class SequenceToProtenixOutput(BaseModel):
    input_json: str
    input_jsons: list[str]
    names: list[str]
    count: int
    runtime_ms: int


def _read_sequences(params: SequenceToProtenixInput) -> list[tuple[str, str]]:
    if params.sequence:
        return [(params.name, params.sequence)]
    assert params.fasta_path is not None
    path = Path(params.fasta_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"FASTA not found: {path}")
    records = parse_fasta(path.read_text())
    if not records:
        raise ValueError(f"FASTA contains no records: {path}")
    return [
        (record.identifier or f"{params.name}_{idx}", record.sequence.upper())
        for idx, record in enumerate(records, start=1)
    ]


def _validate_sequence(sequence: str, *, name: str) -> None:
    if not sequence:
        raise ValueError(f"sequence {name!r} is empty")
    bad = sorted({char for char in sequence if char not in _AMINO_ACIDS})
    if bad:
        raise ValueError(f"sequence {name!r} contains non-canonical amino acids: {bad}")


def _payload(name: str, sequence: str, count: int, seeds: list[int]) -> list[dict]:
    return [
        {
            "name": name,
            "sequences": [{"proteinChain": {"sequence": sequence, "count": count}}],
            "modelSeeds": seeds,
        }
    ]


@register_adapter("descriptor.yaml")
class SequenceToProtenixInputAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(
        self,
        params: SequenceToProtenixInput,
        ctx: AdapterRunContext,
    ) -> SequenceToProtenixOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)
        records = _read_sequences(params)
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[str] = []
        names: list[str] = []
        for idx, (raw_name, sequence) in enumerate(records, start=1):
            name = raw_name if len(records) == 1 else f"{params.name}_{idx}"
            _validate_sequence(sequence, name=name)
            path = out_dir / f"{name}.json"
            path.write_text(
                json.dumps(_payload(name, sequence, params.count, params.seeds), indent=2)
                + "\n"
            )
            paths.append(str(path))
            names.append(name)
        ctx.emit_progress(1.0)
        return SequenceToProtenixOutput(
            input_json=paths[0],
            input_jsons=paths,
            names=names,
            count=len(paths),
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: SequenceToProtenixInput) -> int:
        return 0


__all__ = [
    "SequenceToProtenixInput",
    "SequenceToProtenixInputAdapter",
    "SequenceToProtenixOutput",
]
