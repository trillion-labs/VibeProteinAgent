from __future__ import annotations

from datetime import date


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        text = f"{text}-01-01"
    if len(text) == 7:
        text = f"{text}-01"
    return date.fromisoformat(text[:10])


def before_cutoff(
    record_public_date: str | None,
    temporal_cutoff: str | None,
    *,
    include_undated: bool = False,
) -> bool:
    cutoff = parse_date(temporal_cutoff)
    if cutoff is None:
        return True
    public = parse_date(record_public_date)
    if public is None:
        return include_undated
    return public < cutoff
