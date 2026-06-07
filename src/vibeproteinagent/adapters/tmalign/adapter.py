from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.capabilities import candidate_capability_flags
from vibeproteinagent.candidates.schema import stable_candidate_id
from vibeproteinagent.core.binary import find_binary

_BINARY_CANDIDATES = ("TMalign", "tmalign")

# Stdout lines we parse:
#   Length of Chain_1: <int> residues
#   Length of Chain_2: <int> residues
#   Aligned length=  <int>, RMSD=   <f>, Seq_ID=n_identical/n_aligned= <f>
#   TM-score=  <f> (if normalized by length of Chain_1, ...)
#   TM-score=  <f> (if normalized by length of Chain_2, ...)
_RE_LEN1 = re.compile(r"Length of Chain_1:\s+(\d+)\s+residues")
_RE_LEN2 = re.compile(r"Length of Chain_2:\s+(\d+)\s+residues")
_RE_ALIGNED = re.compile(
    r"Aligned length=\s*(\d+),\s*RMSD=\s*([0-9.]+),\s*Seq_ID=n_identical/n_aligned=\s*([0-9.]+)"
)
_RE_TM1 = re.compile(r"TM-score=\s*([0-9.]+)\s*\(if normalized by length of Chain_1")
_RE_TM2 = re.compile(r"TM-score=\s*([0-9.]+)\s*\(if normalized by length of Chain_2")


def _find_binary() -> Path:
    return find_binary(_BINARY_CANDIDATES, env_override="VPA_TMALIGN_BINARY")


class TMalignInput(BaseModel):
    pdb_a: str | None = Field(None, min_length=1)
    pdb_b: str | None = Field(None, min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    normalize_by: str = Field("shorter", pattern=r"^(a|b|average|shorter|longer)$")

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_pdbs(cls, data: Any) -> Any:
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
        if all(canonical):
            if len(candidates) < 2:
                raise ValueError(
                    "tmalign canonical Candidate[] input requires at least two structure candidates"
                )
            pdb_a = candidates[0].get("structure_path")
            pdb_b = candidates[1].get("structure_path")
            if not pdb_a or not pdb_b:
                raise ValueError("tmalign canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("pdb_a", pdb_a)
            out.setdefault("pdb_b", pdb_b)
            return out
        return data

    @model_validator(mode="after")
    def _require_pdbs(self) -> TMalignInput:
        if not self.pdb_a or not self.pdb_b:
            raise ValueError("tmalign requires pdb_a and pdb_b")
        return self


class TMalignOutput(BaseModel):
    tm_score_normalized_by_a: float
    tm_score_normalized_by_b: float
    tm_score_normalized: float
    rmsd: float
    aligned_length: int
    length_a: int
    length_b: int
    seq_identity: float
    superposition_pdb: str | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


def _pick_normalized(
    *, tm_a: float, tm_b: float, length_a: int, length_b: int, normalize_by: str
) -> float:
    if normalize_by == "a":
        return tm_a
    if normalize_by == "b":
        return tm_b
    if normalize_by == "average":
        return (tm_a + tm_b) / 2.0
    if normalize_by == "shorter":
        return tm_a if length_a <= length_b else tm_b
    if normalize_by == "longer":
        return tm_a if length_a >= length_b else tm_b
    return tm_a  # unreachable — pydantic enum guards this


def _parse_tmalign_stdout(text: str) -> dict:
    out: dict = {}
    if m := _RE_LEN1.search(text):
        out["length_a"] = int(m.group(1))
    if m := _RE_LEN2.search(text):
        out["length_b"] = int(m.group(1))
    if m := _RE_ALIGNED.search(text):
        out["aligned_length"] = int(m.group(1))
        out["rmsd"] = float(m.group(2))
        out["seq_identity"] = float(m.group(3))
    if m := _RE_TM1.search(text):
        out["tm_a"] = float(m.group(1))
    if m := _RE_TM2.search(text):
        out["tm_b"] = float(m.group(1))
    return out


@register_adapter("descriptor.yaml")
class TMalignAdapter(ToolAdapter):
    """Subprocess wrapper around `TMalign` (Zhang & Skolnick)."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.binary_path = _find_binary()

    def run(self, params: TMalignInput, ctx: AdapterRunContext) -> TMalignOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        assert params.pdb_a is not None and params.pdb_b is not None
        pdb_a = Path(params.pdb_a).expanduser().resolve()
        pdb_b = Path(params.pdb_b).expanduser().resolve()
        for label, p in (("pdb_a", pdb_a), ("pdb_b", pdb_b)):
            if not p.is_file():
                raise FileNotFoundError(f"tmalign: {label} not found: {p}")

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        sup_prefix = out_dir / "superposition"

        cmd = [
            str(self.binary_path),
            str(pdb_a),
            str(pdb_b),
            "-o",
            str(sup_prefix),
        ]
        ctx.emit_progress(0.30)
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        ctx.emit_progress(0.80)
        parsed = _parse_tmalign_stdout(proc.stdout)
        required = {
            "length_a",
            "length_b",
            "aligned_length",
            "rmsd",
            "seq_identity",
            "tm_a",
            "tm_b",
        }
        missing = required - parsed.keys()
        if missing:
            raise RuntimeError(
                f"tmalign: failed to parse fields {sorted(missing)} from stdout. "
                f"First 200 chars: {proc.stdout[:200]!r}"
            )

        # TMalign writes <prefix> (no extension) and <prefix>_all_atm etc. when -o is given.
        # The canonical superposition file is the prefix itself or with _atm/_all_atm.
        # Pick whatever exists; default to the no-extension prefix.
        sup_candidates = [
            sup_prefix,
            sup_prefix.with_name(sup_prefix.name + "_all_atm"),
            sup_prefix.with_name(sup_prefix.name + "_atm"),
        ]
        sup_path: str | None = None
        for cand in sup_candidates:
            if cand.is_file():
                sup_path = str(cand)
                break

        normalized = _pick_normalized(
            tm_a=parsed["tm_a"],
            tm_b=parsed["tm_b"],
            length_a=parsed["length_a"],
            length_b=parsed["length_b"],
            normalize_by=params.normalize_by,
        )

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return TMalignOutput(
            tm_score_normalized_by_a=parsed["tm_a"],
            tm_score_normalized_by_b=parsed["tm_b"],
            tm_score_normalized=normalized,
            rmsd=parsed["rmsd"],
            aligned_length=parsed["aligned_length"],
            length_a=parsed["length_a"],
            length_b=parsed["length_b"],
            seq_identity=parsed["seq_identity"],
            superposition_pdb=sup_path,
            candidates=_tmalign_candidates(
                params,
                str(pdb_b),
                {
                    "tm_score": normalized,
                    "tm_score_normalized": normalized,
                    "tm_score_normalized_by_a": parsed["tm_a"],
                    "tm_score_normalized_by_b": parsed["tm_b"],
                    "rmsd": parsed["rmsd"],
                    "aligned_length": parsed["aligned_length"],
                    "seq_identity": parsed["seq_identity"],
                },
                sup_path,
            ),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: TMalignInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        try:
            _find_binary()
            return True
        except FileNotFoundError:
            return False


__all__ = ["TMalignAdapter", "TMalignInput", "TMalignOutput", "_parse_tmalign_stdout"]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _tmalign_candidates(
    params: TMalignInput,
    query_path: str,
    scores: dict[str, Any],
    superposition_pdb: str | None,
) -> list[dict[str, Any]]:
    if params.candidates and len(params.candidates) >= 2:
        upstream = [params.candidates[1]]
    else:
        upstream = [{}]
    out: list[dict[str, Any]] = []
    for index, source_candidate in enumerate(upstream):
        structure_path = source_candidate.get("structure_path") or query_path
        local_id = _source_local_id(source_candidate, f"tmalign_{index:04d}")
        if _is_canonical_candidate(source_candidate):
            candidate = _copy_candidate(source_candidate)
        else:
            candidate = {
                "schema_version": "candidate_v1",
                "candidate_id": stable_candidate_id(
                    "tmalign",
                    local_id,
                    structure_path=structure_path,
                    index=index,
                ),
                "source": {
                    "tool": "tmalign",
                    "node_id": None,
                    "job_id": None,
                    "local_id": local_id,
                    "native_id": None,
                    "index": index,
                },
                "structure_path": structure_path,
                "scores": {},
                "artifacts": {},
            }
        candidate.setdefault("scores", {})["tmalign"] = scores
        if superposition_pdb:
            candidate.setdefault("artifacts", {})["tmalign"] = {
                "superposition_pdb": superposition_pdb
            }
        candidate.setdefault(
            "capabilities",
            candidate_capability_flags(
                has_sequence=bool(candidate.get("sequence")),
                has_backbone_structure=bool(structure_path),
                has_full_structure=False,
                has_candidate_scores=True,
            ),
        )
        out.append(candidate)
    return out


def _copy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in candidate.items()
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
    out["scores"] = dict(candidate.get("scores") or {})
    out["artifacts"] = dict(candidate.get("artifacts") or {})
    return out


def _source_local_id(candidate: dict[str, Any], fallback: str) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    return str(source.get("local_id") or fallback)
