from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(slots=True)
class DateFilterStats:
    kept: int = 0
    temporal_filter: int = 0
    undated: int = 0


def parse_date(value: str | int | None) -> date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt, width in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            parsed = datetime.strptime(text[:width], fmt).date()
            if fmt == "%Y":
                return date(parsed.year, 1, 1)
            if fmt == "%Y-%m":
                return date(parsed.year, parsed.month, 1)
            return parsed
        except ValueError:
            continue
    return None


def normalized_date(value: str | int | None) -> str | None:
    parsed = parse_date(value)
    return parsed.isoformat() if parsed else None


def strict_min_date(*values: str | int | None) -> str | None:
    parsed = [d for d in (parse_date(v) for v in values) if d is not None]
    if not parsed:
        return None
    return min(parsed).isoformat()


def keep_record_by_date(
    record_public_date: str | int | None,
    temporal_cutoff: str | int | None,
    *,
    include_undated: bool = False,
) -> tuple[bool, str | None]:
    cutoff = parse_date(temporal_cutoff)
    public = parse_date(record_public_date)
    if cutoff is None:
        return True, None
    if public is None:
        return (True, None) if include_undated else (False, "undated")
    if public < cutoff:
        return True, None
    return False, "temporal_filter"
