from __future__ import annotations

from copy import deepcopy
from typing import Any

_AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")


def normalize_profile_guidance(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(profile, dict) or not profile:
        return None
    normalized = deepcopy(profile)
    normalized["schema_version"] = "profile_guidance_v1"
    normalized.setdefault("family_id", profile.get("family_id"))
    normalized.setdefault("family_label", profile.get("family_label"))
    normalized.setdefault("sequence_count", _safe_int(profile.get("sequence_count"), 0))
    normalized.setdefault("consensus_sequence", str(profile.get("consensus_sequence") or ""))
    normalized.setdefault("length_stats", dict(profile.get("length_stats") or {}))
    normalized.setdefault("aa_frequency", dict(profile.get("aa_frequency") or {}))
    normalized["position_preferences"] = _position_preferences(profile)
    normalized.setdefault("conserved_positions", list(profile.get("conserved_positions") or []))
    normalized.setdefault("motif_windows", list(profile.get("motif_windows") or []))
    normalized["warnings"] = list(profile.get("warnings") or [])
    return normalized


def profile_guidance_metadata(profile: dict[str, Any] | None) -> dict[str, Any]:
    guidance = normalize_profile_guidance(profile)
    if guidance is None:
        return {}
    return {
        "schema_version": guidance.get("schema_version"),
        "family_id": guidance.get("family_id"),
        "family_label": guidance.get("family_label"),
        "sequence_count": guidance.get("sequence_count"),
        "consensus_length": len(str(guidance.get("consensus_sequence") or "")),
        "conserved_position_count": len(guidance.get("conserved_positions") or []),
        "motif_window_count": len(guidance.get("motif_windows") or []),
    }


def profile_guidance_warnings(profile: dict[str, Any] | None) -> list[str]:
    guidance = normalize_profile_guidance(profile)
    if guidance is None:
        return []
    warnings = [str(item) for item in guidance.get("warnings") or [] if item]
    if guidance.get("motif_windows"):
        warnings.append(
            "profile motif windows are sequence evidence only unless a structure-aware "
            "adapter explicitly maps them"
        )
    return warnings


def derive_profile_mask_positions(
    sequence: str,
    profile: dict[str, Any] | None,
) -> list[int]:
    guidance = normalize_profile_guidance(profile)
    if guidance is None:
        return []
    out: set[int] = set()
    sequence = str(sequence or "").upper()
    for item in guidance.get("conserved_positions") or []:
        if not isinstance(item, dict):
            continue
        pos = _safe_int(item.get("position"), 0)
        consensus = str(item.get("consensus") or "").upper()
        if pos < 1 or pos > len(sequence) or consensus not in _AA_SET:
            continue
        if sequence[pos - 1] != consensus:
            out.add(pos)
    return sorted(out)


def suggested_profile_length(profile: dict[str, Any] | None) -> int | None:
    guidance = normalize_profile_guidance(profile)
    if guidance is None:
        return None
    stats = guidance.get("length_stats") if isinstance(guidance.get("length_stats"), dict) else {}
    for key in ("median", "mean", "min"):
        value = _safe_float(stats.get(key))
        if value is not None and value > 0:
            return int(round(value))
    consensus = str(guidance.get("consensus_sequence") or "")
    return len(consensus) or None


def consensus_sequence(profile: dict[str, Any] | None) -> str | None:
    guidance = normalize_profile_guidance(profile)
    if guidance is None:
        return None
    sequence = str(guidance.get("consensus_sequence") or "").strip().upper()
    return sequence or None


def proteinmpnn_bias_applicability(
    profile: dict[str, Any] | None,
    *,
    num_residues: int,
) -> dict[str, Any]:
    guidance = normalize_profile_guidance(profile)
    if guidance is None:
        return {"applied": False, "warnings": []}
    consensus = str(guidance.get("consensus_sequence") or "")
    warnings: list[str] = []
    if not consensus:
        warnings.append("profile guidance has no consensus sequence")
        return {"applied": False, "warnings": warnings}
    if len(consensus) != int(num_residues):
        warnings.append(
            "profile guidance length does not match designed ProteinMPNN chain length"
        )
        return {"applied": False, "warnings": warnings}
    return {
        "applied": True,
        "warnings": warnings,
        "position_preference_count": len(guidance.get("position_preferences") or []),
    }


def _position_preferences(profile: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in profile.get("position_frequencies") or []:
        if not isinstance(item, dict):
            continue
        frequencies = item.get("frequencies")
        if not isinstance(frequencies, dict):
            counts = item.get("counts")
            frequencies = _frequencies_from_counts(counts if isinstance(counts, dict) else {})
        cleaned = {
            str(aa).upper(): round(float(freq), 6)
            for aa, freq in (frequencies or {}).items()
            if str(aa).upper() in _AA_SET and _safe_float(freq) is not None
        }
        if not cleaned:
            continue
        consensus = max(cleaned.items(), key=lambda pair: (pair[1], pair[0]))[0]
        out.append(
            {
                "position": _safe_int(item.get("position"), len(out) + 1),
                "consensus": str(item.get("consensus") or consensus).upper(),
                "frequencies": cleaned,
                "conservation": round(float(max(cleaned.values())), 6),
            }
        )
    if out:
        return out
    consensus = str(profile.get("consensus_sequence") or "").upper()
    return [
        {
            "position": index,
            "consensus": aa,
            "frequencies": {aa: 1.0},
            "conservation": 1.0,
        }
        for index, aa in enumerate(consensus, start=1)
        if aa in _AA_SET
    ]


def _frequencies_from_counts(counts: dict[str, Any]) -> dict[str, float]:
    cleaned = {
        str(aa).upper(): float(count)
        for aa, count in counts.items()
        if str(aa).upper() in _AA_SET and _safe_float(count) is not None
    }
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {aa: value / total for aa, value in cleaned.items()}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "consensus_sequence",
    "derive_profile_mask_positions",
    "normalize_profile_guidance",
    "profile_guidance_metadata",
    "profile_guidance_warnings",
    "proteinmpnn_bias_applicability",
    "suggested_profile_length",
]
