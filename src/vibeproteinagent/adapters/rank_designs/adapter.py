from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from vibeproteinagent.adapters._confidence import normalize_plddt, plddt_unit_score
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter, torch
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.workspace import JobStore, Workspace

_REFERENCE_CANDIDATE_TOOLS = frozenset({
    "interpro_sequences",
    "go_sequences",
    "family_profile",
})


class RankDesignsInput(BaseModel):
    job_ids: list[str] = Field(default_factory=list)
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    analyses: list[dict[str, Any]] = Field(default_factory=list)
    qc_results: list[dict[str, Any]] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    top_k: int = Field(10, ge=1, le=1000)
    output_prefix: str = Field("rank_designs", pattern=r"^[A-Za-z0-9_.-]+$")

    @field_validator("predictions", "analyses", "qc_results", "candidates", mode="before")
    @classmethod
    def _flatten_nested_rows(cls, value: Any, info: ValidationInfo) -> list[dict[str, Any]]:
        rows = _as_list(value)
        if info.field_name != "candidates" or not rows:
            return rows
        canonical = [_is_canonical_candidate(row) for row in rows]
        if all(canonical):
            return [_row_from_canonical_candidate(row) for row in rows]
        if any(canonical):
            raise ValueError(
                "Mixed canonical and legacy candidate items are not supported. "
                "Pass either all canonical Candidate[] or all legacy row objects."
            )
        return rows


class RankDesignsOutput(BaseModel):
    candidate_count: int
    ranked_candidates: list[dict[str, Any]]
    ranked_json: str
    ranked_tsv: str
    warnings: list[str] = Field(default_factory=list)
    runtime_ms: int


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_as_list(item))
        return rows
    return []


def _is_canonical_candidate(row: dict[str, Any]) -> bool:
    return row.get("schema_version") == "candidate_v1"


def _row_from_canonical_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    producer_tool = str(source.get("tool") or "")
    scores = candidate.get("scores") if isinstance(candidate.get("scores"), dict) else {}
    producer_scores = (
        scores.get(producer_tool)
        if producer_tool and isinstance(scores.get(producer_tool), dict)
        else {}
    )
    row: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id"),
        "sequence": candidate.get("sequence"),
        "structure_path": candidate.get("structure_path"),
        "source": source,
    }
    if isinstance(producer_scores, dict):
        row.update(producer_scores)
    if scores:
        row["scores"] = dict(scores)
    return {key: value for key, value in row.items() if value is not None}


def _candidate_rows_from_output(output: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = output.get("candidates")
    if isinstance(candidates, list) and candidates:
        rows = [item for item in candidates if isinstance(item, dict)]
        if (
            rows
            and len(rows) == len(candidates)
            and all(_is_canonical_candidate(row) for row in rows)
        ):
            return [_row_from_canonical_candidate(row) for row in rows]
    rows: list[dict[str, Any]] = []
    for key in (
        "ranked_candidates",
        "top_candidates",
        "top_designs",
        "designs",
        "predictions",
        "representative_sequences",
    ):
        rows.extend(item for item in _as_list(output.get(key)) if isinstance(item, dict))
    if not rows and output:
        rows.append(dict(output))
    return rows


def _canonical_candidates_from_output(output: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = output.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return []
    rows = [item for item in candidates if isinstance(item, dict)]
    if len(rows) != len(candidates) or not all(_is_canonical_candidate(row) for row in rows):
        return []
    return rows


def _source_value(row: dict[str, Any], key: str) -> Any:
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    return source.get(key)


def _meaningful_identity_keys(row: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for name in ("candidate_id", "name", "id", "seed", "header"):
        value = row.get(name)
        if value not in (None, ""):
            keys.append(f"{name}:{value}")
    for name in ("local_id", "native_id"):
        value = _source_value(row, name)
        if value not in (None, ""):
            keys.append(f"source.{name}:{value}")
    for name in ("structure_path", "predicted_cif", "predicted_pdb", "pdb_path", "cif_path"):
        value = row.get(name)
        if value not in (None, ""):
            keys.append(f"structure:{Path(str(value)).name}")
    sequence = row.get("sequence")
    if sequence not in (None, ""):
        keys.append(f"sequence:{sequence}")
    return keys


def _merge_mapping(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if value in (None, "", [], {}):
            continue
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_mapping(current, value)
        elif current in (None, "", [], {}):
            merged[key] = value
        elif current == value:
            continue
        elif key in {"source_job_id", "source_tool"}:
            plural = f"{key}s"
            values = merged.setdefault(plural, [])
            if isinstance(values, list):
                for item in (current, value):
                    if item not in values:
                        values.append(item)
        else:
            continue
    return merged


def _join_rows_by_identity(
    *,
    predictions: list[dict[str, Any]],
    analyses: list[dict[str, Any]],
    qc_results: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, dict[str, Any]]], list[str]]:
    groups: list[dict[str, dict[str, Any]]] = []
    key_to_group: dict[str, int] = {}
    warnings: list[str] = []

    for bucket, rows in (
        ("prediction", predictions),
        ("analysis", analyses),
        ("qc", qc_results),
        ("candidate", candidates),
    ):
        for index, row in enumerate(rows):
            meaningful_keys = _meaningful_identity_keys(row)
            position_key = f"position:{index}"
            lookup_keys = meaningful_keys or [position_key]
            group_index = next(
                (key_to_group[key] for key in lookup_keys if key in key_to_group),
                None,
            )
            if group_index is None:
                group_index = len(groups)
                groups.append({})
            group = groups[group_index]
            if bucket in group:
                group[bucket] = _merge_mapping(group[bucket], row)
            else:
                group[bucket] = dict(row)
            for key in [*meaningful_keys, position_key]:
                key_to_group.setdefault(key, group_index)

    for index, group in enumerate(groups, start=1):
        if not group.get("prediction") and not group.get("candidate"):
            buckets = ", ".join(sorted(group))
            warnings.append(
                f"rank_designs could not match {buckets} evidence group {index} "
                "to a prediction or candidate; ranked it as standalone evidence."
            )
    return groups, warnings


def _bucket_for_tool(tool_name: str) -> str:
    if tool_name in {"protenix_v2", "boltzgen", "esm3", "esm2"}:
        return "predictions"
    if tool_name in {
        "ipsae",
        "prodigy",
        "tmalign",
        "structure_analysis",
        "boltz2_affinity",
    }:
        return "analyses"
    if tool_name in {"protein_qc", "sequence_analysis"}:
        return "qc_results"
    return "candidates"


def _rows_from_jobs(
    ctx: AdapterRunContext,
    job_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    if not job_ids:
        return {"predictions": [], "analyses": [], "qc_results": [], "candidates": []}
    workspace = Workspace(ctx.workspace_root.name, ctx.workspace_root)
    store = JobStore(workspace)
    buckets: dict[str, list[dict[str, Any]]] = {
        "predictions": [],
        "analyses": [],
        "qc_results": [],
        "candidates": [],
    }
    canonical_candidates: list[dict[str, Any]] = []
    evidence_by_candidate_id: dict[str, dict[str, set[str]]] = {}
    for job_id in job_ids:
        job = store.get(job_id)
        output = job.output if isinstance(job.output, dict) else {}
        if job.tool_name in _REFERENCE_CANDIDATE_TOOLS:
            continue
        canonical_rows = _canonical_candidates_from_output(output)
        if canonical_rows:
            for candidate in canonical_rows:
                canonical_candidates.append(candidate)
                candidate_id = str(candidate.get("candidate_id") or "")
                if candidate_id:
                    evidence = evidence_by_candidate_id.setdefault(
                        candidate_id,
                        {"job_ids": set(), "tools": set()},
                    )
                    evidence["job_ids"].add(job.job_id)
                    evidence["tools"].add(job.tool_name)
            continue
        bucket = _bucket_for_tool(job.tool_name)
        for row in _candidate_rows_from_output(output):
            row.setdefault("source_job_id", job.job_id)
            row.setdefault("source_tool", job.tool_name)
            buckets[bucket].append(row)
    if canonical_candidates:
        merged = _merge_rank_canonical_candidates(canonical_candidates)
        for candidate in merged:
            candidate_id = str(candidate.get("candidate_id") or "")
            evidence = evidence_by_candidate_id.get(candidate_id, {})
            row = _row_from_canonical_candidate(candidate)
            row["evidence_job_ids"] = sorted(evidence.get("job_ids") or [])
            row["evidence_tools"] = sorted(evidence.get("tools") or [])
            buckets["candidates"].append(row)
    return buckets


def _merge_rank_canonical_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            candidate_id = f"position:{len(order)}"
        if candidate_id not in groups:
            order.append(candidate_id)
        groups.setdefault(candidate_id, []).append(candidate)

    merged: list[dict[str, Any]] = []
    for candidate_id in order:
        group = groups[candidate_id]
        current = dict(group[0])
        current["scores"] = dict(group[0].get("scores") or {})
        current["artifacts"] = dict(group[0].get("artifacts") or {})
        current["capabilities"] = dict(group[0].get("capabilities") or {})
        for candidate in group[1:]:
            for key in ("sequence", "structure_path", "backbone_path", "full_structure_path"):
                if candidate.get(key):
                    current[key] = candidate[key]
            for namespace, payload in (candidate.get("scores") or {}).items():
                current.setdefault("scores", {}).setdefault(namespace, payload)
            for namespace, payload in (candidate.get("artifacts") or {}).items():
                current.setdefault("artifacts", {}).setdefault(namespace, payload)
            for key, value in (candidate.get("capabilities") or {}).items():
                current.setdefault("capabilities", {})[key] = (
                    bool(current["capabilities"].get(key)) or bool(value)
                )
        merged.append(current)
    return merged


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None and isinstance(row.get("metrics"), dict):
        value = row["metrics"].get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _aliased_metric(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _metric(row, key)
        if value is not None:
            return value
    return None


def _path(row: dict[str, Any]) -> str | None:
    for key in ("predicted_cif", "predicted_pdb", "structure_path", "pdb_path", "cif_path"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _first_value(*rows: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value is not None:
                return value
    return None


def _candidate_id(
    idx: int,
    prediction: dict[str, Any],
    analysis: dict[str, Any],
    qc: dict[str, Any],
    candidate: dict[str, Any],
) -> str:
    value = _first_value(
        candidate,
        prediction,
        analysis,
        qc,
        keys=("candidate_id", "name", "id", "seed", "header"),
    )
    if value:
        return str(value)
    return str(idx + 1)


def _sequence(candidate: dict[str, Any], prediction: dict[str, Any]) -> str | None:
    value = _first_value(
        candidate,
        prediction,
        keys=("sequence", "protein_sequence", "designed_sequence"),
    )
    return str(value) if value else None


def _first_metric(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _score_namespace(row: dict[str, Any], namespace: str) -> dict[str, Any]:
    scores = row.get("scores")
    if not isinstance(scores, dict):
        return {}
    payload = scores.get(namespace)
    return payload if isinstance(payload, dict) else {}


def _score(
    prediction: dict[str, Any],
    analysis: dict[str, Any],
    qc: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    candidate_score = _metric(candidate, "score")
    ranking_score = _first_metric(
        _metric(prediction, "ranking_score"),
        _metric(candidate, "ranking_score"),
        candidate_score,
    )
    plddt = _first_metric(
        _aliased_metric(prediction, "plddt", "mean_plddt"),
        _aliased_metric(candidate, "plddt", "mean_plddt"),
    )
    iptm = _first_metric(_metric(prediction, "iptm"), _metric(candidate, "iptm"))
    ptm = _first_metric(_metric(prediction, "ptm"), _metric(candidate, "ptm"))
    ipsae = _first_metric(
        _aliased_metric(analysis, "ipsae_score", "ipsae", "ipsae_max", "interface_pae"),
        _aliased_metric(candidate, "ipsae_score", "ipsae", "ipsae_max", "interface_pae"),
    )
    candidate_qc = _score_namespace(candidate, "protein_qc")
    candidate_boltz2 = _score_namespace(candidate, "boltz2_affinity")
    candidate_protenix = _score_namespace(candidate, "protenix_v2")
    candidate_structure = _score_namespace(candidate, "structure_analysis")
    candidate_ipsae = _score_namespace(candidate, "ipsae")
    candidate_prodigy = _score_namespace(candidate, "prodigy")
    candidate_boltzgen = _score_namespace(candidate, "boltzgen")
    candidate_family = _score_namespace(candidate, "family_profile")
    family_profile_score = _aliased_metric(
        candidate_family,
        "ranking_score",
        "family_similarity_score",
    )
    qc_metrics = qc.get("metrics") if isinstance(qc.get("metrics"), dict) else {}
    qc_plddt = (
        _aliased_metric(qc_metrics, "plddt", "mean_plddt")
        if isinstance(qc_metrics, dict)
        else None
    )
    candidate_qc_plddt = _aliased_metric(candidate_qc, "plddt", "mean_plddt")
    liabilities = (
        analysis.get("liabilities")
        if isinstance(analysis.get("liabilities"), list)
        else []
    )
    liability_count = len(liabilities)
    structure_liability_count = _metric(candidate_structure, "liability_count")
    if structure_liability_count is not None:
        liability_count = int(structure_liability_count)
    qc_flags = qc.get("flags") if isinstance(qc.get("flags"), list) else None
    if qc_flags is None:
        qc_flags = (
            candidate_qc.get("flags") if isinstance(candidate_qc.get("flags"), list) else []
        )

    score = 0.0
    boltz2_probability = _aliased_metric(
        candidate_boltz2,
        "affinity_probability_binary",
        "binder_probability",
        "binding_probability",
    )
    boltz2_affinity = _aliased_metric(
        candidate_boltz2,
        "affinity_pred_value",
        "predicted_affinity",
        "affinity",
    )
    boltzgen_score = _aliased_metric(candidate_boltzgen, "score", "ranking_score")
    structure_ligand_contacts = None
    ligand_contacts = candidate_structure.get("ligand_contacts")
    if isinstance(ligand_contacts, dict):
        structure_ligand_contacts = _aliased_metric(
            ligand_contacts,
            "contact_residue_count",
            "contact_count",
        )
    if ranking_score is not None:
        score += ranking_score
    elif boltzgen_score is not None:
        score += boltzgen_score
    plddt_score = plddt_unit_score(
        _first_metric(
            plddt,
            qc_plddt,
            candidate_qc_plddt,
            _aliased_metric(candidate_protenix, "plddt", "mean_plddt"),
        )
    )
    if plddt_score is not None:
        score += plddt_score
    if iptm is not None:
        score += 0.5 * iptm
    if ipsae is not None:
        score += 0.5 * ipsae
    if ptm is not None:
        score += 0.25 * ptm
    if family_profile_score is not None:
        score += 0.25 * family_profile_score
    if boltz2_probability is not None:
        score += boltz2_probability
    if structure_ligand_contacts is not None:
        score += min(structure_ligand_contacts, 20.0) / 100.0
    score -= 0.05 * liability_count
    score -= 0.10 * len(qc_flags)

    reported_plddt = normalize_plddt(
        _first_metric(
            plddt,
            qc_plddt,
            candidate_qc_plddt,
            _aliased_metric(candidate_protenix, "plddt", "mean_plddt"),
        )
    )
    details = {
        "candidate_score": candidate_score,
        "ranking_score": ranking_score,
        "plddt": reported_plddt,
        "iptm": _first_metric(iptm, _aliased_metric(candidate_protenix, "iptm")),
        "ptm": _first_metric(ptm, _aliased_metric(candidate_protenix, "ptm")),
        "ipsae": _first_metric(
            ipsae,
            _aliased_metric(candidate_ipsae, "ipsae_score", "ipsae", "ipsae_max"),
        ),
        "affinity_pred_value": boltz2_affinity,
        "affinity_probability_binary": boltz2_probability,
        "ligand_contact_residue_count": structure_ligand_contacts,
        "family_profile_score": family_profile_score,
        "liability_count": liability_count,
        "qc_flag_count": len(qc_flags),
        "qc_flags": qc_flags,
        "structure_path": _path(prediction) or _path(candidate),
        "sequence": _sequence(candidate, prediction),
        "evidence_job_ids": candidate.get("evidence_job_ids") or [],
        "evidence_tools": candidate.get("evidence_tools") or [],
    }
    if candidate_prodigy:
        details["prodigy_delta_g"] = _aliased_metric(
            candidate_prodigy,
            "delta_g_kcal_mol",
            "delta_g",
            "dg",
        )
        details["prodigy_kd_nm"] = _aliased_metric(candidate_prodigy, "kd_nm", "kd")
    if isinstance(candidate.get("scores"), dict):
        details["scores"] = dict(candidate["scores"])
    return score, details


@register_adapter("descriptor.yaml")
class RankDesignsAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device

    def run(self, params: RankDesignsInput, ctx: AdapterRunContext) -> RankDesignsOutput:
        t0 = time.time()
        job_rows = _rows_from_jobs(ctx, params.job_ids)
        predictions = [*job_rows["predictions"], *_as_list(params.predictions)]
        analyses = [*job_rows["analyses"], *_as_list(params.analyses)]
        qc_results = [*job_rows["qc_results"], *_as_list(params.qc_results)]
        candidates = [*job_rows["candidates"], *_as_list(params.candidates)]
        groups, warnings = _join_rows_by_identity(
            predictions=predictions,
            analyses=analyses,
            qc_results=qc_results,
            candidates=candidates,
        )
        if not groups:
            raise ValueError(
                "rank_designs requires at least one prediction, analysis, "
                "qc result, or candidate"
            )

        ranked: list[dict[str, Any]] = []
        for idx, group in enumerate(groups):
            prediction = group.get("prediction") or {}
            analysis = group.get("analysis") or {}
            qc = group.get("qc") or {}
            candidate = group.get("candidate") or {}
            score, details = _score(prediction, analysis, qc, candidate)
            candidate_id = _candidate_id(idx, prediction, analysis, qc, candidate)
            ranked.append(
                {
                    "rank": 0,
                    "candidate_id": candidate_id,
                    "score": round(score, 6),
                    "candidate_ref": {
                        "candidate_id": candidate_id,
                        "source_job_ids": details.get("evidence_job_ids") or [],
                    },
                    **details,
                }
            )

        ranked.sort(key=lambda row: (-row["score"], str(row["candidate_id"])))
        for rank, row in enumerate(ranked, start=1):
            row["rank"] = rank
        selected = ranked[: params.top_k]

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{params.output_prefix}.json"
        tsv_path = out_dir / f"{params.output_prefix}.tsv"
        json_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        with tsv_path.open("w") as fh:
            fh.write(
                "rank\tcandidate_id\tscore\tranking_score\tplddt\tiptm\tptm\t"
                "ipsae\tliability_count\tqc_flag_count\tstructure_path\n"
            )
            for row in selected:
                fh.write(
                    f"{row['rank']}\t{row['candidate_id']}\t{row['score']}\t"
                    f"{row.get('ranking_score') or ''}\t{row.get('plddt') or ''}\t"
                    f"{row.get('iptm') or ''}\t{row.get('ptm') or ''}\t"
                    f"{row.get('ipsae') or ''}\t"
                    f"{row['liability_count']}\t{row['qc_flag_count']}\t"
                    f"{row.get('structure_path') or ''}\n"
                )
        ctx.emit_progress(1.0)
        return RankDesignsOutput(
            candidate_count=len(groups),
            ranked_candidates=selected,
            ranked_json=str(json_path),
            ranked_tsv=str(tsv_path),
            warnings=warnings,
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: RankDesignsInput) -> int:
        return 0


__all__ = ["RankDesignsAdapter", "RankDesignsInput", "RankDesignsOutput"]
