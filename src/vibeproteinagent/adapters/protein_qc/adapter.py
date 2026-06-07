from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters._confidence import normalize_plddt
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.schema import stable_candidate_id

_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_DEFAULT_METRICS = ["plddt", "polybasic", "gravy", "instability_index", "pi"]

# Heuristic thresholds (Adaptyv Bio "protein-qc" skill).
_POLYBASIC_RUN = 3        # K/R consecutive run length that fires the flag.
_GRAVY_HIGH = 0.5         # GRAVY above this is suspiciously hydrophobic.
_PLDDT_LOW = 70.0         # pLDDT < this triggers a low-confidence flag.
_INSTABILITY_HIGH = 40.0  # Guruprasad heuristic: > 40 is "unstable".


class ProteinQCInput(BaseModel):
    sequence: str | None = Field(None)
    pdb_path: str | None = Field(None)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=lambda: list(_DEFAULT_METRICS))

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_legacy_fields(cls, data: Any) -> Any:
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
        first = candidates[0]
        out = dict(data)
        out.setdefault("sequence", first.get("sequence"))
        out.setdefault("pdb_path", first.get("structure_path"))
        return out

    @model_validator(mode="after")
    def _at_least_one(self) -> ProteinQCInput:
        if not self.sequence and not self.pdb_path:
            raise ValueError("protein_qc requires at least one of sequence / pdb_path")
        return self


class ProteinQCOutput(BaseModel):
    metrics: dict[str, float | str | None]
    flags: list[str]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


def _polybasic_hits(seq: str, run: int = _POLYBASIC_RUN) -> list[int]:
    """Return start indices of every run of `run` or more consecutive K/R."""
    return [m.start() for m in re.finditer(r"[KR]{" + str(run) + r",}", seq)]


def _extract_seq_and_plddt(pdb_path: Path) -> tuple[str, float | None]:
    """Walk a PDB/mmCIF once, returning (sequence, mean CA B-factor).

    For AF2/Chai/Boltz outputs the B-factor column carries pLDDT and the mean
    over CAs is the de-facto pLDDT for the whole chain. For experimental
    structures the value is just a B-factor — still numerically meaningful.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    parser = (
        MMCIFParser(QUIET=True)
        if pdb_path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("x", str(pdb_path))
    bfactors: list[float] = []
    seq_chars: list[str] = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                if "CA" not in res:
                    continue
                bfactors.append(float(res["CA"].get_bfactor()))
                seq_chars.append(_AA3_TO_1.get(res.get_resname(), ""))
        break
    seq = "".join(seq_chars)
    mean_b = float(np.mean(bfactors)) if bfactors else None
    return seq, mean_b


def _score_sequence(
    sequence: str | None,
    pdb_path: str | None,
    requested: set[str],
) -> tuple[dict[str, float | str | None], list[str]]:
    sequence = sequence or ""
    pdb_plddt: float | None = None
    if pdb_path:
        pdb_seq, pdb_plddt = _extract_seq_and_plddt(Path(pdb_path).expanduser())
        pdb_plddt = normalize_plddt(pdb_plddt)
        if not sequence:
            sequence = pdb_seq

    metrics: dict[str, float | str | None] = {}
    flags: list[str] = []

    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    if "plddt" in requested:
        metrics["plddt"] = pdb_plddt
        if pdb_plddt is not None and pdb_plddt < _PLDDT_LOW:
            flags.append("low_plddt")

    if "polybasic" in requested:
        hits = _polybasic_hits(sequence, run=_POLYBASIC_RUN) if sequence else []
        metrics["polybasic_hits"] = float(len(hits))
        if hits:
            flags.append("contains_polybasic_cluster")

    if sequence:
        try:
            pa = ProteinAnalysis(sequence)
            if "gravy" in requested:
                gravy = float(pa.gravy())
                metrics["gravy"] = gravy
                if gravy > _GRAVY_HIGH:
                    flags.append("high_GRAVY")
            if "instability_index" in requested:
                ii = float(pa.instability_index())
                metrics["instability_index"] = ii
                if ii > _INSTABILITY_HIGH:
                    flags.append("high_instability_index")
            if "pi" in requested:
                metrics["pi"] = float(pa.isoelectric_point())
        except Exception as exc:  # malformed sequence — surface as a flag.
            metrics.setdefault("gravy", None)
            metrics.setdefault("instability_index", None)
            metrics.setdefault("pi", None)
            flags.append(f"protparam_error:{type(exc).__name__}")
    else:
        for metric_name in ("gravy", "instability_index", "pi"):
            if metric_name in requested:
                metrics[metric_name] = None

    return metrics, flags


@register_adapter("descriptor.yaml")
class ProteinQCAdapter(ToolAdapter):
    """Curated protein-design QC metrics.

    Pattern: each metric is a self-contained branch keyed by name in
    ``params.metrics``. Adding a metric = add a key to the enum in
    ``protein_qc.input.json`` and a branch below.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)

    def run(self, params: ProteinQCInput, ctx: AdapterRunContext) -> ProteinQCOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        requested = set(params.metrics or _DEFAULT_METRICS)
        metrics, flags = _score_sequence(params.sequence, params.pdb_path, requested)
        ctx.emit_progress(0.30)

        ctx.emit_progress(1.0)
        runtime_ms = int((time.time() - t0) * 1000)
        return ProteinQCOutput(
            metrics=metrics,
            flags=flags,
            candidates=_protein_qc_candidates(params, requested),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: ProteinQCInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        return True


__all__ = ["ProteinQCAdapter", "ProteinQCInput", "ProteinQCOutput"]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _protein_qc_candidates(
    params: ProteinQCInput,
    requested: set[str],
) -> list[dict[str, Any]]:
    upstream = params.candidates if params.candidates else [{}]
    candidates: list[dict[str, Any]] = []
    for index, source_candidate in enumerate(upstream):
        sequence = source_candidate.get("sequence") or params.sequence
        structure_path = source_candidate.get("structure_path") or params.pdb_path
        local_id = _source_local_id(source_candidate, f"qc_{index:04d}")
        candidate = _base_candidate(
            "protein_qc",
            local_id,
            index,
            sequence=sequence,
            structure_path=structure_path,
            upstream=source_candidate if _is_canonical_candidate(source_candidate) else None,
        )
        metrics, flags = _score_sequence(sequence, structure_path, requested)
        candidate.setdefault("scores", {})["protein_qc"] = {
            **metrics,
            "passes_filters": not flags,
            "flags": list(flags),
        }
        candidates.append(candidate)
    return candidates


def _base_candidate(
    tool: str,
    local_id: str,
    index: int,
    *,
    sequence: str | None,
    structure_path: str | None,
    upstream: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if upstream is not None:
        candidate = {
            key: value
            for key, value in upstream.items()
            if key
            in {
                "schema_version",
                "candidate_id",
                "source",
                "sequence",
                "structure_path",
                "capabilities",
            }
        }
        candidate["scores"] = dict(upstream.get("scores") or {})
        candidate["artifacts"] = dict(upstream.get("artifacts") or {})
        return candidate
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            tool,
            local_id,
            sequence=sequence,
            structure_path=structure_path,
            index=index,
        ),
        "source": {
            "tool": tool,
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": None,
            "index": index,
        },
        "scores": {},
        "artifacts": {},
    }
    if sequence:
        candidate["sequence"] = sequence
    if structure_path:
        candidate["structure_path"] = structure_path
    return candidate


def _source_local_id(candidate: dict[str, Any], fallback: str) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    return str(source.get("local_id") or fallback)
