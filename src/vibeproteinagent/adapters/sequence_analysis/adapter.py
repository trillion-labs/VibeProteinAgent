from __future__ import annotations

import csv
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = frozenset(_AMINO_ACIDS)
_HYDROPHOBIC = frozenset("AILMFWVY")
_POLAR = frozenset("STNQCY")
_CHARGED = frozenset("DEKRH")
_POSITIVE = frozenset("KRH")
_NEGATIVE = frozenset("DE")
_AROMATIC = frozenset("FWY")
_SULFUR = frozenset("CM")
_SMALL = frozenset("AGST")

_KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

_RESIDUE_MASS = {
    "A": 89.0935, "C": 121.159, "D": 133.1032, "E": 147.1299, "F": 165.19,
    "G": 75.0669, "H": 155.1552, "I": 131.1736, "K": 146.1882, "L": 131.1736,
    "M": 149.2124, "N": 132.1184, "P": 115.131, "Q": 146.1451, "R": 174.2017,
    "S": 105.093, "T": 119.1197, "V": 117.1469, "W": 204.2262, "Y": 181.1894,
}

_PKA_POSITIVE = {"n_term": 9.69, "K": 10.5, "R": 12.4, "H": 6.0}
_PKA_NEGATIVE = {"c_term": 2.34, "D": 3.86, "E": 4.25, "C": 8.33, "Y": 10.07}


class SequenceAnalysisInput(BaseModel):
    sequence: str | None = None
    fasta_path: str | None = Field(None, min_length=1)
    positions: list[int] = Field(default_factory=list)
    ph: float = Field(7.4, ge=0.0, le=14.0)
    max_annotations: int = Field(100, ge=0)
    include_regions: bool = True
    include_residue_table: bool = True
    output_prefix: str = Field("sequence_analysis", pattern=r"^[A-Za-z0-9_.-]+$")

    @field_validator("sequence")
    @classmethod
    def _normalize_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_sequence(value)

    @model_validator(mode="after")
    def _at_least_one_sequence_source(self) -> SequenceAnalysisInput:
        if not self.sequence and not self.fasta_path:
            raise ValueError("sequence_analysis requires sequence or fasta_path")
        if any(pos < 1 for pos in self.positions):
            raise ValueError("positions are 1-indexed and must be positive")
        return self


class SequenceLiability(BaseModel):
    type: str
    positions: list[int] = Field(default_factory=list)
    reason: str


class ResidueAnnotation(BaseModel):
    position: int
    residue: str
    biochemical_class: str
    charge_class: str
    hydrophobicity_class: str
    aromatic: bool
    sulfur_containing: bool
    glycine: bool
    proline: bool
    cysteine: bool
    caution_flags: list[str] = Field(default_factory=list)


class SequenceAnalysisOutput(BaseModel):
    sequence_length: int
    metrics: dict[str, Any]
    representative_sequences: list[dict[str, Any]] = Field(default_factory=list)
    liabilities: list[SequenceLiability]
    top_annotations: list[ResidueAnnotation]
    artifacts: dict[str, str | None]
    runtime_ms: int


def _clean_sequence(value: str) -> str:
    if value.lstrip().startswith(">"):
        records = _parse_fasta_records(value)
        return records[0][1] if records else ""
    return "".join(value.split()).upper()


def _parse_fasta_text(text: str) -> tuple[str | None, str]:
    records = _parse_fasta_records(text)
    if not records:
        return None, ""
    return records[0]


def _parse_fasta_records(text: str) -> list[tuple[str | None, str]]:
    records: list[tuple[str | None, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            if chunks:
                records.append((header, "".join(chunks).upper()))
                chunks = []
            header = stripped[1:].strip() or None
            continue
        chunks.append(stripped)
    if chunks:
        records.append((header, "".join(chunks).upper()))
    return records


def _read_sequences(params: SequenceAnalysisInput) -> list[tuple[str | None, str]]:
    if params.sequence:
        if params.sequence.lstrip().startswith(">"):
            records = _parse_fasta_records(params.sequence)
            if records:
                return records
        return [(None, params.sequence)]
    assert params.fasta_path is not None
    fasta_path = Path(params.fasta_path).expanduser()
    if not fasta_path.exists():
        raise FileNotFoundError(f"sequence_analysis FASTA not found: {fasta_path}")
    records = _parse_fasta_records(fasta_path.read_text())
    if not records:
        raise ValueError(f"sequence_analysis FASTA contains no sequence: {fasta_path}")
    return records


def _counts(sequence: str) -> dict[str, int]:
    return {aa: sequence.count(aa) for aa in _AMINO_ACIDS}


def _fractions(counts: dict[str, int], length: int) -> dict[str, float]:
    if length == 0:
        return {aa: 0.0 for aa in _AMINO_ACIDS}
    return {aa: counts[aa] / length for aa in _AMINO_ACIDS}


def _fallback_molecular_weight(sequence: str) -> float | None:
    residues = [aa for aa in sequence if aa in _AA_SET]
    if not residues:
        return None
    water = 18.01528
    return float(sum(_RESIDUE_MASS[aa] for aa in residues) - water * (len(residues) - 1))


def _fallback_gravy(sequence: str) -> float | None:
    residues = [aa for aa in sequence if aa in _AA_SET]
    if not residues:
        return None
    return float(sum(_KD[aa] for aa in residues) / len(residues))


def _charge_at_ph(sequence: str, ph: float) -> float:
    counts = _counts(sequence)
    positive = 1.0 / (1.0 + 10 ** (ph - _PKA_POSITIVE["n_term"]))
    positive += sum(
        counts[aa] / (1.0 + 10 ** (ph - pka))
        for aa, pka in _PKA_POSITIVE.items()
        if aa != "n_term"
    )
    negative = 1.0 / (1.0 + 10 ** (_PKA_NEGATIVE["c_term"] - ph))
    negative += sum(
        counts[aa] / (1.0 + 10 ** (pka - ph))
        for aa, pka in _PKA_NEGATIVE.items()
        if aa != "c_term"
    )
    return float(positive - negative)


def _fallback_pi(sequence: str) -> float | None:
    if not any(aa in _AA_SET for aa in sequence):
        return None
    low = 0.0
    high = 14.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if _charge_at_ph(sequence, mid) > 0:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def _biopython_metrics(sequence: str, ph: float) -> dict[str, float | None] | None:
    if any(aa not in _AA_SET for aa in sequence):
        return None
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except Exception:
        return None
    try:
        analysis = ProteinAnalysis(sequence)
        return {
            "molecular_weight": float(analysis.molecular_weight()),
            "pI": float(analysis.isoelectric_point()),
            "gravy": float(analysis.gravy()),
            "aromaticity": float(analysis.aromaticity()),
            "instability_index": float(analysis.instability_index()),
            "net_charge": float(analysis.charge_at_pH(ph)),
        }
    except Exception:
        return None


def _contiguous_runs(sequence: str, pattern: str, min_len: int) -> list[list[int]]:
    runs: list[list[int]] = []
    for match in re.finditer(pattern, sequence):
        if len(match.group(0)) >= min_len:
            runs.append(list(range(match.start() + 1, match.end() + 1)))
    return runs


def _low_complexity_runs(sequence: str) -> list[list[int]]:
    runs: list[list[int]] = []
    for match in re.finditer(r"(.)\1{4,}", sequence):
        if match.group(1) in _AA_SET:
            runs.append(list(range(match.start() + 1, match.end() + 1)))
    for match in re.finditer(r"([A-Z]{2})\1{3,}", sequence):
        runs.append(list(range(match.start() + 1, match.end() + 1)))
    return runs


def _residue_class(aa: str) -> str:
    if aa not in _AA_SET:
        return "noncanonical"
    if aa in _POSITIVE:
        return "basic"
    if aa in _NEGATIVE:
        return "acidic"
    if aa in _HYDROPHOBIC:
        return "hydrophobic"
    if aa in _POLAR:
        return "polar"
    if aa in _SMALL:
        return "small"
    return "special"


def _charge_class(aa: str) -> str:
    if aa in _POSITIVE:
        return "positive"
    if aa in _NEGATIVE:
        return "negative"
    if aa in _AA_SET:
        return "neutral"
    return "unknown"


def _hydrophobicity_class(aa: str) -> str:
    if aa not in _AA_SET:
        return "unknown"
    if _KD[aa] >= 1.6:
        return "hydrophobic"
    if _KD[aa] <= -1.0:
        return "hydrophilic"
    return "intermediate"


def _annotate_residue(position: int, aa: str) -> ResidueAnnotation:
    flags: list[str] = []
    if aa == "C":
        flags.append("cysteine_can_form_disulfides_or_mispaired_adducts")
    if aa == "P":
        flags.append("proline_can_disrupt_secondary_structure")
    if aa == "G":
        flags.append("glycine_increases_backbone_flexibility")
    if aa in _AROMATIC:
        flags.append("aromatic_bulk")
    if aa not in _AA_SET:
        flags.append("noncanonical_residue")
    return ResidueAnnotation(
        position=position,
        residue=aa,
        biochemical_class=_residue_class(aa),
        charge_class=_charge_class(aa),
        hydrophobicity_class=_hydrophobicity_class(aa),
        aromatic=aa in _AROMATIC,
        sulfur_containing=aa in _SULFUR,
        glycine=aa == "G",
        proline=aa == "P",
        cysteine=aa == "C",
        caution_flags=flags,
    )


def _metrics(sequence: str, ph: float) -> dict[str, Any]:
    length = len(sequence)
    counts = _counts(sequence)
    canonical_count = sum(counts.values())
    noncanonical = sorted({aa for aa in sequence if aa not in _AA_SET})
    bio = _biopython_metrics(sequence, ph)
    metrics: dict[str, Any] = {
        "length": length,
        "canonical_residue_count": canonical_count,
        "noncanonical_residue_count": length - canonical_count,
        "unusual_characters": noncanonical,
        "amino_acid_counts": counts,
        "amino_acid_fractions": _fractions(counts, length),
        "charged_fraction": sum(counts[aa] for aa in _CHARGED) / length if length else 0.0,
        "polar_fraction": sum(counts[aa] for aa in _POLAR) / length if length else 0.0,
        "hydrophobic_fraction": sum(counts[aa] for aa in _HYDROPHOBIC) / length if length else 0.0,
        "aromatic_fraction": sum(counts[aa] for aa in _AROMATIC) / length if length else 0.0,
        "cysteine_fraction": counts["C"] / length if length else 0.0,
        "glycine_fraction": counts["G"] / length if length else 0.0,
        "proline_fraction": counts["P"] / length if length else 0.0,
    }
    if bio is None:
        metrics.update({
            "molecular_weight": _fallback_molecular_weight(sequence),
            "pI": _fallback_pi(sequence),
            "gravy": _fallback_gravy(sequence),
            "aromaticity": metrics["aromatic_fraction"],
            "instability_index": None,
            "net_charge": _charge_at_ph(sequence, ph),
            "metric_backend": "fallback",
        })
    else:
        metrics.update(bio)
        metrics["metric_backend"] = "biopython"
    return metrics


def _liabilities(sequence: str, metrics: dict[str, Any]) -> list[SequenceLiability]:
    liabilities: list[SequenceLiability] = []
    unusual = metrics.get("unusual_characters") or []
    if unusual:
        positions = [idx for idx, aa in enumerate(sequence, start=1) if aa not in _AA_SET]
        liabilities.append(SequenceLiability(
            type="unusual_characters",
            positions=positions,
            reason=f"noncanonical sequence characters present: {', '.join(unusual)}",
        ))
    for positions in _contiguous_runs(sequence, r"[KR]{3,}", 3):
        liabilities.append(SequenceLiability(
            type="polybasic_patch",
            positions=positions,
            reason="three or more consecutive K/R residues",
        ))
    for positions in _contiguous_runs(sequence, r"[AILMFWVY]{8,}", 8):
        liabilities.append(SequenceLiability(
            type="hydrophobic_run",
            positions=positions,
            reason=(
                "long hydrophobic run may indicate signal peptide, transmembrane "
                "segment, or aggregation risk"
            ),
        ))
    for positions in _contiguous_runs(sequence, r"[AILMFWVY]{18,}", 18):
        liabilities.append(SequenceLiability(
            type="transmembrane_like_region",
            positions=positions,
            reason="very long hydrophobic run is transmembrane-like",
        ))
    for positions in _low_complexity_runs(sequence):
        liabilities.append(SequenceLiability(
            type="low_complexity_or_repeat",
            positions=positions,
            reason="simple residue repeat or tandem repeat detected",
        ))
    if sequence[:35] and re.search(r"[AILMFWVY]{8,}", sequence[:35]):
        liabilities.append(SequenceLiability(
            type="signal_peptide_like_n_terminus",
            positions=list(range(1, min(len(sequence), 35) + 1)),
            reason="N-terminal hydrophobic run resembles a signal-peptide hydrophobic core",
        ))
    if metrics.get("cysteine_fraction", 0.0) > 0.05:
        liabilities.append(SequenceLiability(
            type="excessive_cysteine",
            positions=[idx for idx, aa in enumerate(sequence, start=1) if aa == "C"],
            reason="cysteine fraction exceeds 5%",
        ))
    length = max(int(metrics.get("length") or 0), 1)
    net_charge = metrics.get("net_charge")
    if isinstance(net_charge, (int, float)) and abs(net_charge) / length > 0.2:
        liabilities.append(SequenceLiability(
            type="extreme_net_charge",
            positions=[],
            reason=f"net charge per residue is {net_charge / length:.3f}",
        ))
    return liabilities


def _write_annotations(path: Path, annotations: list[ResidueAnnotation]) -> None:
    fields = [
        "position",
        "residue",
        "biochemical_class",
        "charge_class",
        "hydrophobicity_class",
        "aromatic",
        "sulfur_containing",
        "glycine",
        "proline",
        "cysteine",
        "caution_flags",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for annotation in annotations:
            row = annotation.model_dump()
            row["caution_flags"] = ",".join(row["caution_flags"])
            writer.writerow(row)


def _write_per_sequence_metrics(
    path: Path,
    records: list[tuple[str | None, str]],
    ph: float,
) -> None:
    with path.open("w") as fh:
        for idx, (header, sequence) in enumerate(records, start=1):
            metrics = _metrics(sequence, ph)
            fh.write(
                json.dumps(
                    {
                        "index": idx,
                        "header": header,
                        "length": len(sequence),
                        "canonical_residue_count": metrics["canonical_residue_count"],
                        "noncanonical_residue_count": metrics["noncanonical_residue_count"],
                        "gravy": metrics.get("gravy"),
                        "pI": metrics.get("pI"),
                        "net_charge": metrics.get("net_charge"),
                        "liability_types": [item.type for item in _liabilities(sequence, metrics)],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _representative_sequences(
    records: list[tuple[str | None, str]],
    ph: float,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    representatives = []
    for idx, (header, sequence) in enumerate(records[:limit], start=1):
        metrics = _metrics(sequence, ph)
        liabilities = _liabilities(sequence, metrics)
        representatives.append(
            {
                "index": idx,
                "header": header,
                "sequence": sequence,
                "length": len(sequence),
                "gravy": metrics.get("gravy"),
                "pI": metrics.get("pI"),
                "net_charge": metrics.get("net_charge"),
                "liability_count": len(liabilities),
                "liability_types": [item.type for item in liabilities],
            }
        )
    return representatives


def _family_summary(records: list[tuple[str | None, str]]) -> dict[str, Any]:
    lengths = [len(sequence) for _header, sequence in records]
    return {
        "sequence_count": len(records),
        "total_length": sum(lengths),
        "length_min": min(lengths) if lengths else None,
        "length_median": statistics.median(lengths) if lengths else None,
        "length_mean": statistics.fmean(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "headers": [header for header, _sequence in records[:20]],
    }


@register_adapter("descriptor.yaml")
class SequenceAnalysisAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.config = dict(config) if config else {}
        self.weights_dir = Path(weights_dir)
        self.device = device

    def run(self, params: SequenceAnalysisInput, ctx: AdapterRunContext) -> SequenceAnalysisOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        records = _read_sequences(params)
        sequence = "".join(seq for _header, seq in records)
        fasta_header = records[0][0] if len(records) == 1 else None
        if not sequence:
            raise ValueError("sequence_analysis sequence must not be empty")
        if len(records) > 1 and params.positions:
            raise ValueError(
                "positions are only supported for single-sequence analysis; "
                "omit positions for multi-record FASTA family summaries"
            )
        if params.positions:
            out_of_range = [pos for pos in params.positions if pos > len(sequence)]
            if out_of_range:
                raise ValueError(
                    f"positions outside sequence length {len(sequence)}: {out_of_range}"
                )
        ctx.emit_progress(0.25)

        metrics = _metrics(sequence, params.ph)
        if fasta_header:
            metrics["fasta_header"] = fasta_header
        if len(records) > 1:
            metrics["analysis_mode"] = "multi_fasta"
            metrics["family_summary"] = _family_summary(records)
        else:
            metrics["analysis_mode"] = "single_sequence"
        liabilities = _liabilities(sequence, metrics)
        ctx.emit_progress(0.55)

        selected_positions = set(params.positions) if params.positions else None
        annotations = [
            _annotate_residue(pos, aa)
            for pos, aa in enumerate(sequence, start=1)
            if selected_positions is None or pos in selected_positions
        ]
        liability_positions = {
            pos for liability in liabilities for pos in liability.positions
        }
        top_annotations = [
            annotation for annotation in annotations if annotation.position in liability_positions
        ]
        if len(top_annotations) < params.max_annotations:
            seen = {annotation.position for annotation in top_annotations}
            for annotation in annotations:
                if annotation.position not in seen:
                    top_annotations.append(annotation)
                    seen.add(annotation.position)
                if len(top_annotations) >= params.max_annotations:
                    break
        else:
            top_annotations = top_annotations[:params.max_annotations]
        top_annotations.sort(key=lambda annotation: annotation.position)

        if not params.include_regions:
            liabilities = [
                liability.model_copy(update={"positions": []}) for liability in liabilities
            ]

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / f"{params.output_prefix}_metrics.json"
        annotations_path = out_dir / f"{params.output_prefix}_residue_annotations.tsv"
        summary_path = out_dir / f"{params.output_prefix}_summary.json"
        per_sequence_path = out_dir / f"{params.output_prefix}_per_sequence_metrics.jsonl"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        if params.include_residue_table:
            _write_annotations(annotations_path, annotations)
            residue_table: str | None = str(annotations_path)
        else:
            residue_table = None
        if len(records) > 1:
            _write_per_sequence_metrics(per_sequence_path, records, params.ph)
            per_sequence_metrics: str | None = str(per_sequence_path)
        else:
            per_sequence_metrics = None

        output = SequenceAnalysisOutput(
            sequence_length=len(sequence),
            metrics=metrics,
            representative_sequences=_representative_sequences(records, params.ph),
            liabilities=liabilities,
            top_annotations=top_annotations,
            artifacts={
                "metrics_json": str(metrics_path),
                "residue_annotations_tsv": residue_table,
                "summary_json": str(summary_path),
                "per_sequence_metrics_jsonl": per_sequence_metrics,
            },
            runtime_ms=int((time.time() - t0) * 1000),
        )
        summary_path.write_text(
            json.dumps(output.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        )

        ctx.emit_progress(1.0)
        return output

    def estimate_vram(self, params: SequenceAnalysisInput) -> int:
        return 0


__all__ = ["SequenceAnalysisAdapter", "SequenceAnalysisInput", "SequenceAnalysisOutput"]
