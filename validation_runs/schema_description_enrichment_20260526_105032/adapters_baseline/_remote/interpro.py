from __future__ import annotations

from typing import Any

from .http import HttpClient

INTERPRO = "https://www.ebi.ac.uk/interpro/api"


def _build_accession_url(
    interpro_id: str,
    *,
    taxon_id: int | None,
    reviewed: bool | None,
) -> str:
    """Build the InterPro REST endpoint that pre-filters on the server.

    The InterPro REST API expects filters as **URL path segments**, not
    query parameters. Passing ``?taxon=9606`` to the bare endpoint is
    silently ignored by the API — historically that made the adapter
    return cross-taxon results, all of which got rejected client-side
    and the run finished with zero hits.

    Path segments used here:

    * ``protein/reviewed``     — only Swiss-Prot (reviewed=True)
    * ``protein/unreviewed``   — only TrEMBL (reviewed=False)
    * ``protein/UniProt``      — both, when ``reviewed`` is unset
    * ``taxonomy/uniprot/{taxon_id}``  — taxonomy filter, optional
    * ``entry/InterPro/{accession}/``  — the actual InterPro family

    The two modifier blocks can appear in either order; we put taxonomy
    before entry so the endpoint reads as "UniProt proteins of taxon X
    that have InterPro entry Y."
    """
    if reviewed is True:
        base = "protein/reviewed"
    elif reviewed is False:
        base = "protein/unreviewed"
    else:
        base = "protein/UniProt"
    if taxon_id is not None:
        return (
            f"{INTERPRO}/{base}/taxonomy/uniprot/{int(taxon_id)}"
            f"/entry/InterPro/{interpro_id}/"
        )
    return f"{INTERPRO}/{base}/entry/InterPro/{interpro_id}/"


def interpro_accessions(
    client: HttpClient,
    interpro_id: str,
    *,
    max_records: int,
    page_size: int = 200,
    taxon_id: int | None = None,
    reviewed: bool | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Page through InterPro's protein endpoint and collect UniProt accessions.

    Filtering by ``taxon_id`` / ``reviewed`` happens at the server (via
    URL path, not query string) — see :func:`_build_accession_url` for
    the rationale. Returns ``(accessions, raw_pages)``; the raw pages
    are kept for audit / debugging.
    """
    url = _build_accession_url(
        interpro_id, taxon_id=taxon_id, reviewed=reviewed,
    )
    accessions: list[str] = []
    raw_pages: list[dict[str, Any]] = []
    while url and len(accessions) < max_records:
        # ``page_size`` is the only query parameter the API still honours
        # on the first request; subsequent pages reuse the ``next`` URL
        # which already carries its own query string.
        params = None
        if "?" not in url:
            params = {"page_size": page_size}
        data = client.get_json(url, params=params)
        raw_pages.append(data)
        for row in data.get("results", []) if isinstance(data, dict) else []:
            metadata = row.get("metadata") if isinstance(row, dict) else None
            acc = None
            if isinstance(metadata, dict):
                acc = metadata.get("accession")
            acc = acc or (row.get("accession") if isinstance(row, dict) else None)
            if acc:
                accessions.append(str(acc))
                if len(accessions) >= max_records:
                    break
        url = data.get("next") if isinstance(data, dict) else None
    return accessions, raw_pages
