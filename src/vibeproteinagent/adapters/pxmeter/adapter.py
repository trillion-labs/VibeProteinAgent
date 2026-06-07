from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.capabilities import candidate_capability_flags
from vibeproteinagent.candidates.schema import stable_candidate_id

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _PROJECT_ROOT / "vendor" / "pxmeter"


def _ensure_pxmeter_importable() -> None:
    """Put the vendored PXMeter on sys.path before first import.

    PXMeter is structured with absolute imports (`from pxmeter.X import …`),
    so we insert the vendor root *containing* the `pxmeter/` package — not
    the package itself — and then `import pxmeter` resolves to the vendor.
    """
    vendor_str = str(_VENDOR_ROOT)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)


class ChainPair(BaseModel):
    a: str = Field(..., min_length=1)
    b: str = Field(..., min_length=1)


class PXMeterInput(BaseModel):
    predicted_cif: str | None = Field(
        None, min_length=1, description="Model-predicted structure (mmCIF)."
    )
    reference_cif: str | None = Field(
        None,
        min_length=1,
        description="Reference / ground-truth structure (mmCIF).",
    )
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=lambda: ["lddt", "dockq"])
    chain_pairs: list[ChainPair] | None = Field(
        default=None,
        description="Restrict DockQ to these chain pairs. If null, all pairs are scored.",
    )

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_cifs(cls, data: Any) -> Any:
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
                    "pxmeter canonical Candidate[] input requires reference and predicted candidates"
                )
            reference_cif = candidates[0].get("structure_path")
            predicted_cif = candidates[1].get("structure_path")
            if not reference_cif or not predicted_cif:
                raise ValueError("pxmeter canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("reference_cif", reference_cif)
            out.setdefault("predicted_cif", predicted_cif)
            return out
        return data

    @model_validator(mode="after")
    def _require_cifs(self) -> PXMeterInput:
        if not self.predicted_cif or not self.reference_cif:
            raise ValueError("pxmeter requires predicted_cif and reference_cif")
        return self


class DockQEntry(BaseModel):
    chain_pair: str
    score: float


class PXMeterOutput(BaseModel):
    metrics: dict[str, Any]
    report_json_path: str
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int


@register_adapter("descriptor.yaml")
class PXMeterAdapter(ToolAdapter):
    """Thin wrapper around ``pxmeter.cli.run_eval_cif``.

    Mirrors the Phase 1.7 verified call: stage inputs into ``ctx.job_dir/inputs/``,
    invoke ``run_eval_cif`` to write a full PXMeter JSON under ``ctx.job_dir/outputs/``,
    then surface ``complex.lddt`` and per-interface DockQ (filtered by ``chain_pairs``
    if provided). The complete JSON path is returned so downstream consumers can
    pick any of the per-chain / per-interface / stereochemistry fields.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        _ensure_pxmeter_importable()
        # Force the import wall now so daemon warm catches missing deps
        # (rdkit METADATA shim, anarcii, gemmi, etc.) before any job runs.
        from pxmeter.cli import run_eval_cif  # noqa: F401

    def run(self, params: PXMeterInput, ctx: AdapterRunContext) -> PXMeterOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        assert params.reference_cif is not None and params.predicted_cif is not None
        ref_src = Path(params.reference_cif).expanduser().resolve()
        pred_src = Path(params.predicted_cif).expanduser().resolve()
        if not ref_src.exists():
            raise FileNotFoundError(f"PXMeter reference_cif not found: {ref_src}")
        if not pred_src.exists():
            raise FileNotFoundError(f"PXMeter predicted_cif not found: {pred_src}")

        in_dir = ctx.job_dir / "inputs"
        out_dir = ctx.job_dir / "outputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stage inputs so the on-disk layout under job_dir is self-describing.
        staged_ref = in_dir / ref_src.name
        staged_pred = in_dir / pred_src.name
        if staged_ref != ref_src:
            shutil.copy2(ref_src, staged_ref)
        if staged_pred != pred_src:
            shutil.copy2(pred_src, staged_pred)
        ctx.emit_progress(0.20)

        _ensure_pxmeter_importable()
        from pxmeter.cli import run_eval_cif

        report_path = out_dir / "pxmeter_report.json"
        run_eval_cif(
            ref_cif=staged_ref,
            model_cif=staged_pred,
            output_json=report_path,
        )
        ctx.emit_progress(0.85)

        report = json.loads(report_path.read_text())
        metrics_out = self._extract_metrics(report, params)
        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        return PXMeterOutput(
            metrics=metrics_out,
            report_json_path=str(report_path),
            candidates=_pxmeter_candidates(params, str(pred_src), metrics_out, str(report_path)),
            runtime_ms=runtime_ms,
        )

    @staticmethod
    def _extract_metrics(report: dict[str, Any], params: PXMeterInput) -> dict[str, Any]:
        requested = {m.lower() for m in params.metrics}
        wanted_pairs: set[frozenset[str]] | None = None
        if params.chain_pairs:
            wanted_pairs = {frozenset((p.a, p.b)) for p in params.chain_pairs}

        out: dict[str, Any] = {}

        if "lddt" in requested:
            complex_lddt = report.get("complex", {}).get("lddt")
            out["lddt"] = float(complex_lddt) if complex_lddt is not None else None

            per_chain = {}
            for chain, info in (report.get("chain") or {}).items():
                if isinstance(info, dict) and "lddt" in info:
                    per_chain[str(chain)] = float(info["lddt"])
            if per_chain:
                out["per_chain_lddt"] = per_chain

            per_interface_lddt = {}
            for iface, info in (report.get("interface") or {}).items():
                if isinstance(info, dict) and "lddt" in info:
                    per_interface_lddt[str(iface)] = float(info["lddt"])
            if per_interface_lddt:
                out["per_interface_lddt"] = per_interface_lddt

        if "dockq" in requested:
            dockq_rows: list[dict[str, Any]] = []
            for iface, info in (report.get("interface") or {}).items():
                if not isinstance(info, dict):
                    continue
                if "dockq" not in info:
                    continue
                a, _, b = str(iface).partition(",")
                if wanted_pairs is not None and frozenset((a, b)) not in wanted_pairs:
                    continue
                dockq_rows.append(
                    {"chain_pair": str(iface), "score": float(info["dockq"])}
                )
            out["dockq"] = dockq_rows if dockq_rows else None

        return out

    def estimate_vram(self, params: PXMeterInput) -> int:
        return 0

    def healthcheck(self) -> bool:
        return (_VENDOR_ROOT / "pxmeter" / "cli.py").is_file()


__all__ = ["ChainPair", "PXMeterAdapter", "PXMeterInput", "PXMeterOutput"]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _pxmeter_candidates(
    params: PXMeterInput,
    predicted_cif: str,
    metrics: dict[str, Any],
    report_json_path: str,
) -> list[dict[str, Any]]:
    upstream = [params.candidates[1]] if params.candidates and len(params.candidates) >= 2 else [{}]
    out: list[dict[str, Any]] = []
    for index, source_candidate in enumerate(upstream):
        structure_path = source_candidate.get("structure_path") or predicted_cif
        local_id = _source_local_id(source_candidate, f"pxmeter_{index:04d}")
        if _is_canonical_candidate(source_candidate):
            candidate = _copy_candidate(source_candidate)
        else:
            candidate = {
                "schema_version": "candidate_v1",
                "candidate_id": stable_candidate_id(
                    "pxmeter",
                    local_id,
                    structure_path=structure_path,
                    index=index,
                ),
                "source": {
                    "tool": "pxmeter",
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
        candidate.setdefault("scores", {})["pxmeter"] = metrics
        candidate.setdefault("artifacts", {})["pxmeter"] = {"report_json_path": report_json_path}
        candidate.setdefault(
            "capabilities",
            candidate_capability_flags(
                has_sequence=bool(candidate.get("sequence")),
                has_backbone_structure=bool(structure_path),
                has_full_structure=bool(structure_path),
                has_candidate_scores=True,
            ),
        )
        out.append(candidate)
    return out


def _copy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    out = {
        key: value
        for key, value in candidate.items()
        if key in {"schema_version", "candidate_id", "source", "sequence", "structure_path", "capabilities"}
    }
    out["scores"] = dict(candidate.get("scores") or {})
    out["artifacts"] = dict(candidate.get("artifacts") or {})
    return out


def _source_local_id(candidate: dict[str, Any], fallback: str) -> str:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    return str(source.get("local_id") or fallback)
