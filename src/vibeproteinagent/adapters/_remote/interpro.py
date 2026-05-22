from __future__ import annotations

from typing import Any

from .http import HttpClient

INTERPRO = "https://www.ebi.ac.uk/interpro/api"


def interpro_accessions(
    client: HttpClient,
    interpro_id: str,
    *,
    max_records: int,
    page_size: int = 200,
    taxon_id: int | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    url = f"{INTERPRO}/protein/uniprot/entry/interpro/{interpro_id}/"
    accessions: list[str] = []
    raw_pages: list[dict[str, Any]] = []
    while url and len(accessions) < max_records:
        params = None
        if "?" not in url:
            params = {"page_size": page_size, "taxon": taxon_id}
        data = client.get_json(url, params=params)
        raw_pages.append(data)
        for row in data.get("results", []) if isinstance(data, dict) else []:
            metadata = row.get("metadata") if isinstance(row, dict) else None
            acc = None
            if isinstance(metadata, dict):
                acc = metadata.get("accession")
            acc = acc or row.get("accession") if isinstance(row, dict) else None
            if acc:
                accessions.append(str(acc))
                if len(accessions) >= max_records:
                    break
        url = data.get("next") if isinstance(data, dict) else None
    return accessions, raw_pages
