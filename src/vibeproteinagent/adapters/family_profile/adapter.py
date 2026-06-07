from __future__ import annotations

import copy
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.schema import validate_candidate_v1
from vibeproteinagent.candidates.capabilities import candidate_capability_flags

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET = frozenset(_AMINO_ACIDS)
_GAP_SET = frozenset({"-", "."})


class FamilyProfileInput(BaseModel):
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    fasta_path: str | None = Field(None, min_length=1)
    alignment_path: str | None = Field(None, min_length=1)
    profile: dict[str, Any] | None = None
    family_id: str | None = None
    family_label: str | None = None
    min_conservation: float = Field(0.7, ge=0.0, le=1.0)
    motif_window: int = Field(9, ge=1, le=51)
    output_prefix: str = Field("family_profile", pattern=r"^[A-Za-z0-9_.-]+$")

    @model_validator(mode="after")
    def _validate_sources(self) -> FamilyProfileInput:
        if self.candidates:
            canonical = [_is_canonical_candidate(item) for item in self.candidates]
            if any(canonical) and not all(canonical):
                raise ValueError(
                    "family_profile candidates must be all canonical Candidate[] or empty"
                )
            if not all(canonical):
                raise ValueError("family_profile only accepts canonical Candidate[]")
        if not (self.profile or self.candidates or self.fasta_path or self.alignment_path):
            raise ValueError(
                "family_profile requires profile, candidates, fasta_path, or alignment_path"
            )
        return self


class FamilyProfileOutput(BaseModel):
    mode: str
    profile: dict[str, Any]
    summary: dict[str, Any]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, str]
    runtime_ms: int


@register_adapter("descriptor.yaml")
class FamilyProfileAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: FamilyProfileInput, ctx: AdapterRunContext) -> FamilyProfileOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)
        mode = "score" if params.profile is not None else "build"
        source_records = _read_reference_records(params) if mode == "build" else []
        profile = (
            _normalize_profile(params.profile, params)
            if params.profile is not None
            else _build_profile(params, source_records)
        )
        ctx.emit_progress(0.45)
        candidates = (
            [_annotate_candidate(candidate, profile) for candidate in params.candidates]
            if mode == "score"
            else []
        )
        ctx.emit_progress(0.75)
        artifacts = _write_artifacts(ctx.job_dir, params.output_prefix, profile, source_records)
        summary = _summary(
            mode=mode,
            profile=profile,
            source_records=source_records,
            candidates=candidates,
        )
        summary_path = Path(artifacts["summary_json"])
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        ctx.emit_progress(1.0)
        return FamilyProfileOutput(
            mode=mode,
            profile=profile,
            summary=summary,
            candidates=candidates,
            artifacts=artifacts,
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: FamilyProfileInput) -> int:
        return 0


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _read_reference_records(params: FamilyProfileInput) -> list[dict[str, Any]]:
    if params.alignment_path:
        return _records_from_fasta(Path(params.alignment_path), aligned=True)
    if params.fasta_path:
        return _records_from_fasta(Path(params.fasta_path), aligned=False)
    return _records_from_candidates(params.candidates)


def _records_from_fasta(path: Path, *, aligned: bool) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"family_profile FASTA not found: {resolved}")
    records = []
    for index, (header, sequence) in enumerate(_parse_fasta_records(resolved.read_text())):
        records.append(
            {
                "id": header or f"seq_{index:04d}",
                "header": header,
                "sequence": _clean_sequence(sequence, aligned=aligned),
                "aligned": aligned,
                "source": {"kind": "alignment_path" if aligned else "fasta_path"},
            }
        )
    return records


def _records_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for index, candidate in enumerate(candidates):
        sequence = candidate.get("sequence")
        if not isinstance(sequence, str) or not sequence.strip():
            continue
        source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
        records.append(
            {
                "id": str(candidate.get("candidate_id") or source.get("local_id") or index),
                "candidate_id": candidate.get("candidate_id"),
                "sequence": _clean_sequence(sequence, aligned=False),
                "aligned": False,
                "source": source,
                "score_namespaces": sorted((candidate.get("scores") or {}).keys())
                if isinstance(candidate.get("scores"), dict)
                else [],
                "artifact_namespaces": sorted((candidate.get("artifacts") or {}).keys())
                if isinstance(candidate.get("artifacts"), dict)
                else [],
            }
        )
    return records


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
                records.append((header, "".join(chunks)))
                chunks = []
            header = stripped[1:].strip() or None
            continue
        chunks.append(stripped)
    if chunks:
        records.append((header, "".join(chunks)))
    return records


def _clean_sequence(value: str, *, aligned: bool) -> str:
    allowed = _AA_SET | (_GAP_SET if aligned else frozenset())
    return "".join(char for char in value.upper() if char in allowed)


def _build_profile(
    params: FamilyProfileInput,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    sequences = [str(record["sequence"]) for record in records if record.get("sequence")]
    aligned = bool(params.alignment_path) or any(record.get("aligned") for record in records)
    position_frequencies = _position_frequencies(sequences)
    consensus = "".join(
        item["consensus"] for item in position_frequencies if item.get("consensus") in _AA_SET
    )
    conserved_positions = [
        {
            "position": item["position"],
            "consensus": item["consensus"],
            "conservation": item["conservation"],
        }
        for item in position_frequencies
        if item.get("consensus") in _AA_SET
        and float(item.get("conservation") or 0.0) >= params.min_conservation
    ]
    motif_windows = _motif_windows(conserved_positions, params.motif_window)
    lengths = [_ungapped_length(sequence) for sequence in sequences]
    return {
        "schema_version": "family_profile_v1",
        "family_id": params.family_id,
        "family_label": params.family_label,
        "sequence_count": len(sequences),
        "aligned": aligned,
        "min_conservation": params.min_conservation,
        "length_stats": _length_stats(lengths),
        "aa_frequency": _aa_frequency(sequences),
        "consensus_sequence": consensus,
        "position_frequencies": position_frequencies,
        "conserved_positions": conserved_positions,
        "motif_windows": motif_windows,
        "evidence_summary": _evidence_summary(records),
    }


def _normalize_profile(profile: dict[str, Any] | None, params: FamilyProfileInput) -> dict[str, Any]:
    normalized = copy.deepcopy(profile or {})
    normalized.setdefault("schema_version", "family_profile_v1")
    if params.family_id is not None:
        normalized["family_id"] = params.family_id
    else:
        normalized.setdefault("family_id", None)
    if params.family_label is not None:
        normalized["family_label"] = params.family_label
    else:
        normalized.setdefault("family_label", None)
    normalized.setdefault("position_frequencies", [])
    normalized.setdefault("conserved_positions", [])
    normalized.setdefault("motif_windows", [])
    normalized.setdefault("length_stats", {})
    normalized.setdefault("aa_frequency", {})
    normalized.setdefault("evidence_summary", {})
    return normalized


def _position_frequencies(sequences: list[str]) -> list[dict[str, Any]]:
    if not sequences:
        return []
    width = max(len(sequence) for sequence in sequences)
    positions: list[dict[str, Any]] = []
    for offset in range(width):
        counts = {aa: 0 for aa in _AMINO_ACIDS}
        observed = 0
        for sequence in sequences:
            residue = sequence[offset] if offset < len(sequence) else "-"
            if residue in _AA_SET:
                counts[residue] += 1
                observed += 1
        if observed:
            consensus = max(_AMINO_ACIDS, key=lambda aa: (counts[aa], aa))
            conservation = counts[consensus] / observed
            frequencies = {aa: counts[aa] / observed for aa in _AMINO_ACIDS if counts[aa]}
        else:
            consensus = "X"
            conservation = 0.0
            frequencies = {}
        positions.append(
            {
                "position": offset + 1,
                "observed": observed,
                "consensus": consensus,
                "conservation": round(float(conservation), 6),
                "counts": {aa: count for aa, count in counts.items() if count},
                "frequencies": {aa: round(value, 6) for aa, value in frequencies.items()},
            }
        )
    return positions


def _motif_windows(
    conserved_positions: list[dict[str, Any]],
    motif_window: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    previous_position: int | None = None
    for item in conserved_positions:
        position = int(item["position"])
        if (
            current
            and (previous_position is None or position != previous_position + 1)
        ) or len(current) >= motif_window:
            windows.append(_window_record(current))
            current = []
        current.append(item)
        previous_position = position
    if current:
        windows.append(_window_record(current))
    return [window for window in windows if window["length"] > 0]


def _window_record(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start": int(items[0]["position"]),
        "end": int(items[-1]["position"]),
        "length": len(items),
        "consensus": "".join(str(item.get("consensus") or "X") for item in items),
        "mean_conservation": round(
            sum(float(item.get("conservation") or 0.0) for item in items) / len(items),
            6,
        ),
    }


def _length_stats(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
            "stdev": None,
        }
    return {
        "count": len(lengths),
        "min": min(lengths),
        "median": float(statistics.median(lengths)),
        "mean": float(statistics.mean(lengths)),
        "max": max(lengths),
        "stdev": float(statistics.pstdev(lengths)) if len(lengths) > 1 else 0.0,
    }


def _aa_frequency(sequences: list[str]) -> dict[str, float]:
    counts = {aa: 0 for aa in _AMINO_ACIDS}
    total = 0
    for sequence in sequences:
        for residue in sequence:
            if residue not in _AA_SET:
                continue
            counts[residue] += 1
            total += 1
    if total == 0:
        return {}
    return {
        aa: round(count / total, 6)
        for aa, count in counts.items()
        if count
    }


def _ungapped_length(sequence: str) -> int:
    return sum(1 for char in sequence if char in _AA_SET)


def _evidence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    score_namespaces: dict[str, int] = {}
    artifact_namespaces: dict[str, int] = {}
    source_tools: dict[str, int] = {}
    for record in records:
        source = record.get("source")
        if isinstance(source, dict) and source.get("tool"):
            tool = str(source["tool"])
            source_tools[tool] = source_tools.get(tool, 0) + 1
        for namespace in record.get("score_namespaces") or []:
            score_namespaces[str(namespace)] = score_namespaces.get(str(namespace), 0) + 1
        for namespace in record.get("artifact_namespaces") or []:
            artifact_namespaces[str(namespace)] = (
                artifact_namespaces.get(str(namespace), 0) + 1
            )
    return {
        "source_tools": source_tools,
        "score_namespaces": score_namespaces,
        "artifact_namespaces": artifact_namespaces,
    }


def _annotate_candidate(candidate: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    validate_candidate_v1(candidate)
    out = copy.deepcopy(candidate)
    scores = dict(out.get("scores") or {})
    artifacts = dict(out.get("artifacts") or {})
    family_scores = _family_scores(str(candidate.get("sequence") or ""), profile)
    scores["family_profile"] = family_scores
    artifacts["family_profile"] = {
        "family_id": profile.get("family_id"),
        "family_label": profile.get("family_label"),
        "profile_schema_version": profile.get("schema_version", "family_profile_v1"),
    }
    out["scores"] = scores
    out["artifacts"] = artifacts
    capabilities = dict(out.get("capabilities") or {})
    inferred = candidate_capability_flags(
        has_sequence=bool(out.get("sequence")),
        has_backbone_structure=bool(out.get("backbone_path") or out.get("structure_path")),
        has_full_structure=bool(out.get("full_structure_path")),
        has_candidate_scores=True,
    )
    capabilities.update({key: value for key, value in inferred.items() if value})
    out["capabilities"] = capabilities
    validate_candidate_v1(out)
    return out


def _family_scores(sequence: str, profile: dict[str, Any]) -> dict[str, Any]:
    clean = _clean_sequence(sequence, aligned=False)
    consensus_match = _consensus_match_score(clean, profile)
    motif_match = _motif_match_score(clean, profile)
    length_zscore = _length_zscore(len(clean), profile)
    length_score = 1.0 / (1.0 + abs(length_zscore)) if length_zscore is not None else 0.0
    conservation_coverage = _conservation_coverage(clean, profile)
    family_similarity = statistics.mean(
        [consensus_match, motif_match, length_score, conservation_coverage]
    )
    return {
        "ranking_score": round(float(family_similarity), 6),
        "family_similarity_score": round(float(family_similarity), 6),
        "consensus_match_score": round(float(consensus_match), 6),
        "motif_match_score": round(float(motif_match), 6),
        "length_zscore": round(float(length_zscore), 6)
        if length_zscore is not None and math.isfinite(length_zscore)
        else None,
        "conservation_coverage": round(float(conservation_coverage), 6),
    }


def _consensus_match_score(sequence: str, profile: dict[str, Any]) -> float:
    consensus = str(profile.get("consensus_sequence") or "")
    comparable = 0
    matches = 0
    for residue, expected in zip(sequence, consensus):
        if expected not in _AA_SET:
            continue
        comparable += 1
        if residue == expected:
            matches += 1
    return matches / comparable if comparable else 0.0


def _motif_match_score(sequence: str, profile: dict[str, Any]) -> float:
    windows = profile.get("motif_windows")
    if not isinstance(windows, list) or not windows:
        return _consensus_match_score(sequence, profile)
    scores: list[float] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        start = int(window.get("start") or 1) - 1
        motif = str(window.get("consensus") or "")
        if start < 0 or not motif:
            continue
        fragment = sequence[start : start + len(motif)]
        comparable = min(len(fragment), len(motif))
        if comparable <= 0:
            scores.append(0.0)
            continue
        matches = sum(1 for a, b in zip(fragment, motif) if a == b)
        scores.append(matches / comparable)
    return statistics.mean(scores) if scores else 0.0


def _length_zscore(length: int, profile: dict[str, Any]) -> float | None:
    stats = profile.get("length_stats") if isinstance(profile.get("length_stats"), dict) else {}
    mean = _safe_float(stats.get("mean"))
    stdev = _safe_float(stats.get("stdev"))
    if mean is None:
        return None
    if stdev is None or stdev == 0:
        return 0.0 if length == int(round(mean)) else float(length - mean)
    return (length - mean) / stdev


def _conservation_coverage(sequence: str, profile: dict[str, Any]) -> float:
    conserved = profile.get("conserved_positions")
    if not isinstance(conserved, list) or not conserved:
        return 0.0
    comparable = 0
    matches = 0
    for item in conserved:
        if not isinstance(item, dict):
            continue
        position = int(item.get("position") or 0)
        expected = str(item.get("consensus") or "")
        if position <= 0 or position > len(sequence) or expected not in _AA_SET:
            continue
        comparable += 1
        if sequence[position - 1] == expected:
            matches += 1
    return matches / comparable if comparable else 0.0


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _write_artifacts(
    job_dir: Path,
    output_prefix: str,
    profile: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, str]:
    out_dir = job_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_json = out_dir / f"{output_prefix}.profile.json"
    summary_json = out_dir / f"{output_prefix}.summary.json"
    sequences_fasta = out_dir / f"{output_prefix}.sequences.fasta"
    position_frequencies_json = out_dir / f"{output_prefix}.position_frequencies.json"
    motif_windows_json = out_dir / f"{output_prefix}.motif_windows.json"
    profile_json.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    position_frequencies_json.write_text(
        json.dumps(profile.get("position_frequencies") or [], indent=2, sort_keys=True)
        + "\n"
    )
    motif_windows_json.write_text(
        json.dumps(profile.get("motif_windows") or [], indent=2, sort_keys=True) + "\n"
    )
    with sequences_fasta.open("w") as fh:
        for index, record in enumerate(records):
            sequence = str(record.get("sequence") or "")
            if not sequence:
                continue
            header = str(record.get("id") or f"seq_{index:04d}")
            fh.write(f">{header}\n{sequence}\n")
    summary_json.write_text("{}\n")
    return {
        "profile_json": str(profile_json),
        "summary_json": str(summary_json),
        "sequences_fasta": str(sequences_fasta),
        "position_frequencies_json": str(position_frequencies_json),
        "motif_windows_json": str(motif_windows_json),
    }


def _summary(
    *,
    mode: str,
    profile: dict[str, Any],
    source_records: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_sequence_count = len(source_records)
    if mode == "score":
        reference_sequence_count = _safe_int(profile.get("sequence_count"), default=0)
    return {
        "mode": mode,
        "family_id": profile.get("family_id"),
        "family_label": profile.get("family_label"),
        "reference_sequence_count": reference_sequence_count,
        "annotated_candidate_count": len(candidates),
        "consensus_length": len(str(profile.get("consensus_sequence") or "")),
        "conserved_position_count": len(profile.get("conserved_positions") or []),
        "motif_window_count": len(profile.get("motif_windows") or []),
        "evidence_summary": profile.get("evidence_summary") or {},
    }


__all__ = ["FamilyProfileAdapter", "FamilyProfileInput", "FamilyProfileOutput"]
