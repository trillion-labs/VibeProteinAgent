from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.schema import stable_candidate_id

_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_MUTATION_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$")


class ParsedMutation(BaseModel):
    position: int
    expected_wt: str | None = None
    new_residue: str


class SequenceEditInput(BaseModel):
    operation: Literal["mutate_residues", "check_wt_consistency"]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    sequence: str | None = None
    wild_type_sequence: str | None = None
    mutant_sequence: str | None = None
    mapping: dict[str, str] = Field(default_factory=dict)
    mutation_notation: list[str] = Field(default_factory=list)
    allow_noop: bool = False
    strict_wild_type: bool = True
    output_name: str = Field("mutant", pattern=r"^[A-Za-z0-9_.-]+$")

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_sequences(cls, data: Any) -> Any:
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
        if not all(canonical):
            return data
        out = dict(data)
        operation = out.get("operation")
        if operation == "mutate_residues":
            sequence = candidates[0].get("sequence")
            if not sequence:
                raise ValueError("sequence_edit canonical Candidate[] input requires sequence")
            out.setdefault("sequence", sequence)
        elif operation == "check_wt_consistency":
            if len(candidates) >= 2:
                out.setdefault("wild_type_sequence", candidates[0].get("sequence"))
                out.setdefault("mutant_sequence", candidates[1].get("sequence"))
            elif candidates[0].get("sequence"):
                out.setdefault("mutant_sequence", candidates[0].get("sequence"))
        return out

    @field_validator("sequence", "wild_type_sequence", "mutant_sequence")
    @classmethod
    def _normalize_optional_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_sequence(value)

    @field_validator("mapping")
    @classmethod
    def _normalize_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_pos, raw_residue in value.items():
            residue = str(raw_residue).strip().upper()
            if residue not in _AMINO_ACIDS:
                raise ValueError(
                    f"mapping residue for position {raw_pos!r} is not a canonical amino acid"
                )
            pos = _parse_position(raw_pos)
            normalized[str(pos)] = residue
        return normalized

    @field_validator("mutation_notation")
    @classmethod
    def _normalize_notation(cls, value: list[str]) -> list[str]:
        return [item.strip().upper() for item in value]

    @model_validator(mode="after")
    def _validate_contract(self) -> SequenceEditInput:
        if self.operation == "mutate_residues" and not self.sequence:
            raise ValueError("mutate_residues requires sequence")
        if self.operation == "check_wt_consistency":
            if not self.wild_type_sequence or not self.mutant_sequence:
                raise ValueError(
                    "check_wt_consistency requires wild_type_sequence and mutant_sequence"
                )

        for name, seq in (
            ("sequence", self.sequence),
            ("wild_type_sequence", self.wild_type_sequence),
            ("mutant_sequence", self.mutant_sequence),
        ):
            if seq is not None:
                _validate_sequence(seq, name)

        mapping = _parse_mapping(self.mapping)
        notation = _parse_notation(self.mutation_notation)
        if mapping and notation:
            mapped = {pos: mut.new_residue for pos, mut in mapping.items()}
            noted = {pos: mut.new_residue for pos, mut in notation.items()}
            if mapped != noted:
                raise ValueError(
                    "mapping and mutation_notation must describe the same positions "
                    "and new residues"
                )

        seq_for_range = (
            self.sequence if self.operation == "mutate_residues" else self.wild_type_sequence
        )
        if seq_for_range is not None:
            length = len(seq_for_range)
            for pos in sorted(set(mapping) | set(notation)):
                if pos < 1 or pos > length:
                    raise ValueError(f"mutation position {pos} is outside sequence length {length}")

        if self.operation == "mutate_residues" and self.sequence is not None:
            for pos, mut in {**mapping, **notation}.items():
                wt = self.sequence[pos - 1]
                if mut.expected_wt is not None and self.strict_wild_type and mut.expected_wt != wt:
                    raise ValueError(
                        f"mutation notation expects {mut.expected_wt} at position {pos}, found {wt}"
                    )
                if not self.allow_noop and mut.new_residue == wt:
                    raise ValueError(f"mutation at position {pos} would not change residue {wt}")

        return self


class SequenceEditOutput(BaseModel):
    operation: Literal["mutate_residues", "check_wt_consistency"]
    wild_type_length: int
    mutant_length: int
    mutations: list[str]
    changed_positions: list[int]
    mutant_sequence: str
    mutant_fasta: str | None = None
    is_consistent: bool | None = None
    mismatch_count: int
    expected_mutations_match: bool | None = None
    warnings: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


def _clean_sequence(value: str) -> str:
    return "".join(value.split()).upper()


def _validate_sequence(sequence: str, field_name: str) -> None:
    if not sequence:
        raise ValueError(f"{field_name} must not be empty")
    bad = sorted({char for char in sequence if char not in _AMINO_ACIDS})
    if bad:
        raise ValueError(f"{field_name} contains non-canonical amino acids: {bad}")


def _parse_position(value: str | int) -> int:
    try:
        pos = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mutation position {value!r} is not an integer") from exc
    if pos < 1:
        raise ValueError(f"mutation position {pos} must be 1-indexed and positive")
    return pos


def _parse_mapping(mapping: dict[str, str]) -> dict[int, ParsedMutation]:
    parsed: dict[int, ParsedMutation] = {}
    for raw_pos, residue in mapping.items():
        pos = _parse_position(raw_pos)
        if pos in parsed:
            raise ValueError(f"duplicate mutation position {pos}")
        parsed[pos] = ParsedMutation(position=pos, new_residue=residue)
    return parsed


def _parse_notation(notations: list[str]) -> dict[int, ParsedMutation]:
    parsed: dict[int, ParsedMutation] = {}
    for notation in notations:
        match = _MUTATION_RE.match(notation)
        if not match:
            raise ValueError(f"invalid mutation notation {notation!r}; expected V73Q style")
        wt, raw_pos, new = match.groups()
        pos = int(raw_pos)
        if pos in parsed:
            raise ValueError(f"duplicate mutation position {pos}")
        parsed[pos] = ParsedMutation(position=pos, expected_wt=wt, new_residue=new)
    return parsed


def _expected_mutations(params: SequenceEditInput) -> dict[int, ParsedMutation]:
    mapping = _parse_mapping(params.mapping)
    notation = _parse_notation(params.mutation_notation)
    merged = dict(mapping)
    for pos, mut in notation.items():
        base = merged.get(pos)
        if base is None:
            merged[pos] = mut
        else:
            merged[pos] = ParsedMutation(
                position=pos,
                expected_wt=mut.expected_wt,
                new_residue=base.new_residue,
            )
    return merged


def _format_mutation(wild_type: str, position: int, new_residue: str) -> str:
    return f"{wild_type}{position}{new_residue}"


def _write_fasta(path: Path, name: str, sequence: str) -> None:
    lines = [f">{name}"]
    lines.extend(sequence[i:i + 80] for i in range(0, len(sequence), 80))
    path.write_text("\n".join(lines) + "\n")


@register_adapter("descriptor.yaml")
class SequenceEditAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.config = dict(config) if config else {}
        self.weights_dir = Path(weights_dir)
        self.device = device

    def run(self, params: SequenceEditInput, ctx: AdapterRunContext) -> SequenceEditOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)
        if params.operation == "mutate_residues":
            return self._mutate_residues(params, ctx, t0)
        return self._check_wt_consistency(params, ctx, t0)

    def _mutate_residues(
        self, params: SequenceEditInput, ctx: AdapterRunContext, t0: float
    ) -> SequenceEditOutput:
        assert params.sequence is not None
        sequence = params.sequence
        mutations_by_pos = _expected_mutations(params)
        if not mutations_by_pos:
            raise ValueError("mutate_residues requires mapping or mutation_notation")

        chars = list(sequence)
        mutations: list[str] = []
        changed_positions: list[int] = []
        warnings: list[str] = []
        for pos in sorted(mutations_by_pos):
            mut = mutations_by_pos[pos]
            wt = sequence[pos - 1]
            if mut.expected_wt is not None and mut.expected_wt != wt:
                warnings.append(f"position {pos}: notation expected {mut.expected_wt}, found {wt}")
            mutations.append(_format_mutation(wt, pos, mut.new_residue))
            if wt != mut.new_residue:
                changed_positions.append(pos)
                chars[pos - 1] = mut.new_residue

        mutant_sequence = "".join(chars)
        ctx.emit_progress(0.70)

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = out_dir / f"{params.output_name}.fasta"
        _write_fasta(fasta_path, params.output_name, mutant_sequence)

        ctx.emit_progress(1.0)
        return SequenceEditOutput(
            operation="mutate_residues",
            wild_type_length=len(sequence),
            mutant_length=len(mutant_sequence),
            mutations=mutations,
            changed_positions=changed_positions,
            mutant_sequence=mutant_sequence,
            mutant_fasta=str(fasta_path),
            mismatch_count=len(changed_positions),
            warnings=warnings,
            candidates=_sequence_edit_candidates(
                params,
                mutant_sequence,
                str(fasta_path),
                {
                    "mismatch_count": len(changed_positions),
                    "changed_positions": changed_positions,
                },
            ),
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def _check_wt_consistency(
        self, params: SequenceEditInput, ctx: AdapterRunContext, t0: float
    ) -> SequenceEditOutput:
        assert params.wild_type_sequence is not None
        assert params.mutant_sequence is not None
        wild_type = params.wild_type_sequence
        mutant = params.mutant_sequence
        if len(wild_type) != len(mutant):
            raise ValueError(
                "check_wt_consistency supports substitutions only; "
                f"wild-type length {len(wild_type)} != mutant length {len(mutant)}"
            )

        inferred: dict[int, str] = {}
        mutations: list[str] = []
        for idx, (wt, new) in enumerate(zip(wild_type, mutant, strict=True), start=1):
            if wt != new:
                inferred[idx] = new
                mutations.append(_format_mutation(wt, idx, new))

        expected = _expected_mutations(params)
        expected_map = {pos: mut.new_residue for pos, mut in expected.items()}
        expected_match = None
        warnings: list[str] = []
        if expected:
            expected_match = inferred == expected_map
            if not expected_match:
                expected_notation = [
                    _format_mutation(wild_type[pos - 1], pos, residue)
                    for pos, residue in sorted(expected_map.items())
                ]
                warnings.append(
                    "inferred mutations do not match expected mutations: "
                    f"inferred={mutations}, expected={expected_notation}"
                )

        ctx.emit_progress(1.0)
        return SequenceEditOutput(
            operation="check_wt_consistency",
            wild_type_length=len(wild_type),
            mutant_length=len(mutant),
            mutations=mutations,
            changed_positions=sorted(inferred),
            mutant_sequence=mutant,
            is_consistent=expected_match is not False,
            mismatch_count=len(inferred),
            expected_mutations_match=expected_match,
            warnings=warnings,
            candidates=_sequence_edit_candidates(
                params,
                mutant,
                None,
                {
                    "mismatch_count": len(inferred),
                    "expected_mutations_match": expected_match,
                    "is_consistent": expected_match is not False,
                },
            ),
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: SequenceEditInput) -> int:
        return 0


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _sequence_edit_candidates(
    params: SequenceEditInput,
    mutant_sequence: str,
    mutant_fasta: str | None,
    scores: dict[str, Any],
) -> list[dict[str, Any]]:
    local_id = "seq_0000"
    source_candidate = params.candidates[0] if params.candidates else {}
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            "sequence_edit",
            local_id,
            sequence=mutant_sequence,
            index=0,
        ),
        "source": {
            "tool": "sequence_edit",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": None,
            "index": 0,
        },
        "sequence": mutant_sequence,
        "scores": {"sequence_edit": scores},
        "artifacts": {"sequence_edit": {"mutant_fasta": mutant_fasta}},
    }
    if _is_canonical_candidate(source_candidate):
        upstream_scores = source_candidate.get("scores")
        if isinstance(upstream_scores, dict):
            candidate["scores"] = {**upstream_scores, **candidate["scores"]}
        if source_candidate.get("structure_path"):
            candidate["structure_path"] = source_candidate.get("structure_path")
    return [candidate]


__all__ = ["SequenceEditAdapter", "SequenceEditInput", "SequenceEditOutput"]
