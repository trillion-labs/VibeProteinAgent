from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import stable_candidate_id

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _PROJECT_ROOT / "vendor" / "ipsae"
_IPSAE_SCRIPT = _VENDOR_ROOT / "ipsae_vendor" / "ipsae.py"


def _ipsae_cli_number(value: float) -> str:
    """Preserve the numeric cutoff passed to upstream ipsae.py."""
    return f"{value:g}"


def _ipsae_output_cutoff_string(value: float) -> str:
    # Upstream uses int(cutoff) for output filenames, while the calculation
    # itself uses the original float cutoff.
    return f"{int(value):02d}"


class IpSaeInput(BaseModel):
    pae_path: str = Field(..., description="AF2 PAE JSON, AF3 JSON, or Boltz NPZ.")
    pdb_path: str | None = Field(None, description="Predicted complex PDB or mmCIF.")
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    chain_a: str = Field(..., min_length=1)
    chain_b: str = Field(..., min_length=1)
    cutoff: float = Field(10.0, gt=0.0, description="PAE cutoff (Å-equivalent).")
    dist_cutoff: float = Field(10.0, gt=0.0, description="Distance cutoff (Å).")

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_pdb_path(cls, data: Any) -> Any:
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
            structure_path = candidates[0].get("structure_path")
            if not structure_path:
                raise ValueError("ipsae canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("pdb_path", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _require_pdb_path(self) -> IpSaeInput:
        if not self.pdb_path:
            raise ValueError("ipsae requires pdb_path")
        return self


class IpSaeOutput(BaseModel):
    ipsae_score: float
    ipsae_asym_ab: float
    ipsae_asym_ba: float
    ipsae_max: float
    interface_residues_a: list[int]
    interface_residues_b: list[int]
    scoreboard_path: str
    byres_path: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


def _parse_scoreboard(text: str) -> list[dict[str, str]]:
    """Parse the ipsae.py main `.txt` scoreboard into a list of rows."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("Chn1"):
            header = line.split()
            continue
        if header is None:
            continue
        parts = line.split()
        if len(parts) < len(header):
            continue
        # The trailing 'Model' column may itself be a path; collapse the tail.
        head_fields = parts[: len(header) - 1]
        tail_field = " ".join(parts[len(header) - 1 :])
        rows.append(dict(zip(header, [*head_fields, tail_field], strict=True)))
    return rows


def _parse_byres(text: str) -> list[dict[str, str]]:
    """Parse ipsae.py `_byres.txt` into a list of per-residue rows."""
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("i"):
            header = line.split()
            continue
        if header is None:
            continue
        parts = line.split()
        if len(parts) < len(header):
            continue
        rows.append(dict(zip(header, parts[: len(header)], strict=True)))
    return rows


def _find_scoreboard_row(
    rows: list[dict[str, str]],
    chain_a: str,
    chain_b: str,
    row_type: str,
) -> dict[str, str] | None:
    for row in rows:
        if (
            row.get("Chn1") == chain_a
            and row.get("Chn2") == chain_b
            and row.get("Type") == row_type
        ):
            return row
    return None


def _stage_pae_file(src: Path, out_dir: Path) -> Path:
    """Stage a PAE file for upstream ipsae.py, decompressing JSON gzip inputs."""
    if src.suffix.lower() != ".gz":
        staged = out_dir / src.name
        if staged != src:
            shutil.copy2(src, staged)
        return staged

    inner_name = src.with_suffix("").name
    staged = out_dir / inner_name
    if staged == src:
        return staged
    with gzip.open(src, "rb") as fi, staged.open("wb") as fo:
        shutil.copyfileobj(fi, fo)
    return staged


@register_adapter("descriptor.yaml")
class IpSaeAdapter(ToolAdapter):
    """Subprocess wrapper around the vendored ``ipsae.py`` script.

    The script is CLI-only (sys.argv driven, file outputs alongside the input
    PDB). We stage the PAE+PDB inputs into the job dir so the output files
    land under ``ctx.job_dir/outputs/``.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        if not _IPSAE_SCRIPT.exists():
            raise FileNotFoundError(
                f"vendored ipsae.py missing at {_IPSAE_SCRIPT!s}. "
                "See vendor/ipsae/UPSTREAM.md."
            )

    def run(self, params: IpSaeInput, ctx: AdapterRunContext) -> IpSaeOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        pae_src = Path(params.pae_path).expanduser().resolve()
        assert params.pdb_path is not None
        pdb_src = Path(params.pdb_path).expanduser().resolve()
        if not pae_src.exists():
            raise FileNotFoundError(f"ipSAE PAE file not found: {pae_src}")
        if not pdb_src.exists():
            raise FileNotFoundError(f"ipSAE PDB/CIF file not found: {pdb_src}")

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ipsae.py writes outputs alongside the PDB, so stage inputs inside the
        # job dir to keep the working tree clean and reproducible.
        staged_pae = _stage_pae_file(pae_src, out_dir)
        staged_pdb = out_dir / pdb_src.name
        if staged_pdb != pdb_src:
            shutil.copy2(pdb_src, staged_pdb)

        cmd = [
            sys.executable,
            str(_IPSAE_SCRIPT),
            str(staged_pae),
            str(staged_pdb),
            _ipsae_cli_number(params.cutoff),
            _ipsae_cli_number(params.dist_cutoff),
        ]
        ctx.emit_progress(0.30)
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            cwd=str(out_dir),
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ipsae.py exited {proc.returncode}: stderr={proc.stderr[:2000]}"
            )
        ctx.emit_progress(0.80)

        pae_string = _ipsae_output_cutoff_string(params.cutoff)
        dist_string = _ipsae_output_cutoff_string(params.dist_cutoff)
        # ipsae strips the .pdb/.cif suffix to form the output stem.
        stem_suffix = ".pdb" if staged_pdb.suffix.lower() == ".pdb" else ".cif"
        stem = str(staged_pdb).removesuffix(stem_suffix) + f"_{pae_string}_{dist_string}"
        scoreboard = Path(stem + ".txt")
        byres = Path(stem + "_byres.txt")
        if not scoreboard.exists():
            raise RuntimeError(
                f"ipsae.py did not write expected scoreboard at {scoreboard}; "
                f"stdout={proc.stdout[:1000]}"
            )

        rows = _parse_scoreboard(scoreboard.read_text())
        asym_ab_row = _find_scoreboard_row(rows, params.chain_a, params.chain_b, "asym")
        asym_ba_row = _find_scoreboard_row(rows, params.chain_b, params.chain_a, "asym")
        max_row = _find_scoreboard_row(rows, params.chain_a, params.chain_b, "max")
        if max_row is None:
            max_row = _find_scoreboard_row(rows, params.chain_b, params.chain_a, "max")
        if asym_ab_row is None or asym_ba_row is None or max_row is None:
            available = sorted({(r.get("Chn1"), r.get("Chn2")) for r in rows})
            raise ValueError(
                f"ipSAE produced no row for chain_a={params.chain_a} / "
                f"chain_b={params.chain_b}. Available pairs: {available}"
            )
        ipsae_asym_ab = float(asym_ab_row["ipSAE"])
        ipsae_asym_ba = float(asym_ba_row["ipSAE"])
        ipsae_max = float(max_row["ipSAE"])

        interface_a: list[int] = []
        interface_b: list[int] = []
        if byres.exists():
            byres_rows = _parse_byres(byres.read_text())
            seen_a: set[int] = set()
            seen_b: set[int] = set()
            for row in byres_rows:
                try:
                    val = float(row.get("ipSAE", "0"))
                except ValueError:
                    continue
                if val <= 0.0:
                    continue
                align_chain = row.get("AlignChn")
                scored_chain = row.get("ScoredChain")
                resnum = int(row.get("AlignResNum", "0"))
                if align_chain == params.chain_a and scored_chain == params.chain_b:
                    seen_a.add(resnum)
                elif align_chain == params.chain_b and scored_chain == params.chain_a:
                    seen_b.add(resnum)
            interface_a = sorted(seen_a)
            interface_b = sorted(seen_b)

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return IpSaeOutput(
            ipsae_score=ipsae_max,
            ipsae_asym_ab=ipsae_asym_ab,
            ipsae_asym_ba=ipsae_asym_ba,
            ipsae_max=ipsae_max,
            interface_residues_a=interface_a,
            interface_residues_b=interface_b,
            scoreboard_path=str(scoreboard),
            byres_path=str(byres),
            candidates=_ipsae_candidates(
                params,
                str(pdb_src),
                {
                    "interface_pae": ipsae_max,
                    "ipsae_score": ipsae_max,
                    "ipsae_asym_ab": ipsae_asym_ab,
                    "ipsae_asym_ba": ipsae_asym_ba,
                    "ipsae_max": ipsae_max,
                    "interface_residues_a": interface_a,
                    "interface_residues_b": interface_b,
                },
                str(scoreboard),
                str(byres),
            ),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: IpSaeInput) -> int:
        # CPU-only; reported as 0 but admission uses descriptor's vram_min_gib.
        return 0

    def healthcheck(self) -> bool:
        return _IPSAE_SCRIPT.exists()


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _ipsae_candidates(
    params: IpSaeInput,
    pdb_path: str,
    scores: dict[str, Any],
    scoreboard_path: str,
    byres_path: str,
) -> list[dict[str, Any]]:
    upstream = params.candidates if params.candidates else [{}]
    candidates: list[dict[str, Any]] = []
    for index, source_candidate in enumerate(upstream):
        structure_path = source_candidate.get("structure_path") or pdb_path
        local_id = _source_local_id(source_candidate, f"ipsae_{index:04d}")
        if _is_canonical_candidate(source_candidate):
            candidate = _copy_candidate(source_candidate)
        else:
            candidate = {
                "schema_version": "candidate_v1",
                "candidate_id": stable_candidate_id(
                    "ipsae",
                    local_id,
                    structure_path=structure_path,
                    index=index,
                ),
                "source": {
                    "tool": "ipsae",
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
        candidate.setdefault("scores", {})["ipsae"] = scores
        candidate.setdefault("artifacts", {})["ipsae"] = {
            "scoreboard_path": scoreboard_path,
            "byres_path": byres_path,
        }
        candidates.append(candidate)
    return candidates


def _copy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in candidate.items()
        if key in {"schema_version", "candidate_id", "source", "sequence", "structure_path"}
    }
    out["scores"] = dict(candidate.get("scores") or {})
    out["artifacts"] = dict(candidate.get("artifacts") or {})
    return out


def _source_local_id(candidate: dict[str, Any], fallback: str) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    return str(source.get("local_id") or fallback)


__all__ = ["IpSaeAdapter", "IpSaeInput", "IpSaeOutput"]
