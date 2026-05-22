from __future__ import annotations

from datetime import timedelta
from typing import Any

from .dates import parse_date
from .http import HttpClient

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search_europepmc(
    client: HttpClient,
    query: str,
    *,
    max_results: int,
    page_size: int = 25,
    result_type: str = "lite",
    open_access_only: bool = False,
    published_before: str | None = None,
    published_after: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q = query
    if open_access_only:
        q = f"({q}) OPEN_ACCESS:y"
    date_clause = _first_publication_date_clause(
        published_after=published_after,
        published_before=published_before,
    )
    if date_clause:
        q = f"({q}) {date_clause}"
    cursor = "*"
    records: list[dict[str, Any]] = []
    raw_pages: list[dict[str, Any]] = []
    while len(records) < max_results:
        data = client.get_json(
            EUROPEPMC_SEARCH,
            params={
                "query": q,
                "format": "json",
                "pageSize": min(page_size, max_results - len(records)),
                "cursorMark": cursor,
                "resultType": result_type,
            },
        )
        raw_pages.append(data)
        results = data.get("resultList", {}).get("result", []) if isinstance(data, dict) else []
        if not results:
            break
        records.extend(results)
        next_cursor = data.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = str(next_cursor)
    return records[:max_results], raw_pages


def _first_publication_date_clause(
    *,
    published_after: str | None,
    published_before: str | None,
) -> str | None:
    after = parse_date(published_after)
    before = parse_date(published_before)
    if before is None and after is None:
        return None
    start = after.isoformat() if after else "1800-01-01"
    # Europe PMC range syntax is inclusive. The adapter still applies a local
    # strict `record_public_date < temporal_cutoff` filter after retrieval, so
    # subtracting one day here preserves the strict contract while improving
    # recall within the requested result budget.
    end = (before - timedelta(days=1)).isoformat() if before else "3000-12-31"
    if before and after and before <= after:
        end = start
    return f"FIRST_PDATE:[{start} TO {end}]"
