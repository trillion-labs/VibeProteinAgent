from __future__ import annotations

import math
from typing import Any


def normalize_plddt(value: Any) -> float | None:
    """Return pLDDT on the canonical 0-100 scale."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < 0.0:
        return None
    if numeric <= 1.0:
        return numeric * 100.0
    return numeric


def plddt_unit_score(value: Any) -> float | None:
    normalized = normalize_plddt(value)
    if normalized is None:
        return None
    return normalized / 100.0

