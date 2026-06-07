from __future__ import annotations

import json
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


def _extract_seq_and_bfactor_plddt(
    pdb_path: Path,
) -> tuple[str, float | None, float | None]:
    """Walk a PDB/mmCIF once, returning sequence and B-factor diagnostics."""
    from Bio.PDB import MMCIFParser, PDBParser

    parser = (
        MMCIFParser(QUIET=True)
        if pdb_path.suffix.lower() in {".cif", ".mmcif"}
        else PDBParser(QUIET=True)
    )
    structure = parser.get_structure("x", str(pdb_path))
    ca_bfactors: list[float] = []
    atom_bfactors: list[float] = []
    seq_chars: list[str] = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != " ":
                    continue
                for atom in res:
                    atom_bfactors.append(float(atom.get_bfactor()))
                if "CA" not in res:
                    continue
                ca_bfactors.append(float(res["CA"].get_bfactor()))
                seq_chars.append(_AA3_TO_1.get(res.get_resname(), ""))
        break
    seq = "".join(seq_chars)
    ca_mean = float(np.mean(ca_bfactors)) if ca_bfactors else None
    atom_mean = float(np.mean(atom_bfactors)) if atom_bfactors else None
    return seq, ca_mean, atom_mean


def _numeric_plddt(value: Any) -> float | None:
    try:
        return normalize_plddt(float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _plddt_from_json_artifact(path: Any) -> float | None:
    if not path:
        return None
    try:
        payload = json.loads(Path(str(path)).expanduser().read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("plddt", "mean_plddt", "global_plddt"):
        value = _numeric_plddt(payload.get(key))
        if value is not None:
            return value
    confidence = payload.get("confidence")
    if isinstance(confidence, dict):
        for key in ("plddt", "mean_plddt", "global_plddt"):
            value = _numeric_plddt(confidence.get(key))
            if value is not None:
                return value
    return None


def _official_plddt_from_candidate(
    candidate: dict[str, Any] | None,
) -> tuple[float | None, str | None]:
    if not _is_canonical_candidate(candidate):
        return None, None
    assert candidate is not None
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    score_paths = (
        ("protenix_v2", "plddt"),
        ("esm3", "mean_plddt"),
        ("esm3", "plddt"),
        ("esm2", "mean_plddt"),
        ("esm2", "plddt"),
    )
    for namespace, key in score_paths:
        value = _numeric_plddt(_nested_value(scores, (namespace, key)))
        if value is not None:
            return value, f"scores.{namespace}.{key}"

    artifacts = (
        candidate.get("artifacts")
        if isinstance(candidate.get("artifacts"), dict)
        else {}
    )
    for namespace in ("protenix_v2", "esm3", "esm2"):
        payload = artifacts.get(namespace)
        if not isinstance(payload, dict):
            continue
        for key in ("summary_json", "confidence_json", "full_confidence_json"):
            value = _plddt_from_json_artifact(payload.get(key))
            if value is not None:
                return value, f"artifacts.{namespace}.{key}.plddt"
    return None, None


def _score_sequence(
    sequence: str | None,
    pdb_path: str | None,
    requested: set[str],
    *,
    source_candidate: dict[str, Any] | None = None,
) -> tuple[dict[str, float | str | None], list[str]]:
    sequence = sequence or ""
    ca_bfactor_plddt: float | None = None
    atom_bfactor_plddt: float | None = None
    if pdb_path:
        pdb_seq, ca_bfactor_plddt, atom_bfactor_plddt = _extract_seq_and_bfactor_plddt(
            Path(pdb_path).expanduser()
        )
        ca_bfactor_plddt = normalize_plddt(ca_bfactor_plddt)
        atom_bfactor_plddt = normalize_plddt(atom_bfactor_plddt)
        if not sequence:
            sequence = pdb_seq
    official_plddt, plddt_source = _official_plddt_from_candidate(source_candidate)
    chosen_plddt = official_plddt
    if chosen_plddt is None:
        chosen_plddt = ca_bfactor_plddt if ca_bfactor_plddt is not None else atom_bfactor_plddt
        if chosen_plddt is not None:
            plddt_source = (
                "ca_bfactor_plddt"
                if ca_bfactor_plddt is not None
                else "atom_bfactor_plddt"
            )

    metrics: dict[str, float | str | None] = {}
    flags: list[str] = []

    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    if "plddt" in requested:
        metrics["plddt"] = chosen_plddt
        metrics["plddt_source"] = plddt_source
        metrics["ca_bfactor_plddt"] = ca_bfactor_plddt
        metrics["atom_bfactor_plddt"] = atom_bfactor_plddt
        if chosen_plddt is not None and chosen_plddt < _PLDDT_LOW:
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
        first_candidate = (
            params.candidates[0]
            if params.candidates and _is_canonical_candidate(params.candidates[0])
            else None
        )
        metrics, flags = _score_sequence(
            params.sequence,
            params.pdb_path,
            requested,
            source_candidate=first_candidate,
        )
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
        metrics, flags = _score_sequence(
            sequence,
            structure_path,
            requested,
            source_candidate=source_candidate
            if _is_canonical_candidate(source_candidate)
            else None,
        )
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
