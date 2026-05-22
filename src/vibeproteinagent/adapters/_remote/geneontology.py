from __future__ import annotations

import re
from typing import Any

from .http import HttpClient

GO = "http://api.geneontology.org/api"
_UNIPROT_RE = re.compile(
    r"(?:UniProtKB:)?"
    r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])"
)


def go_associations(
    client: HttpClient,
    go_id: str,
    *,
    rows: int = 200,
    aspect: str | None = None,
    taxon_id: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aspect_path = _aspect_path(aspect)
    params: dict[str, Any] = {"rows": rows}
    if taxon_id is not None:
        params["taxon"] = f"NCBITaxon:{taxon_id}"
    data = client.get_json(f"{GO}/bioentity/{aspect_path}/{go_id}", params=params)
    assocs = data.get("associations", []) if isinstance(data, dict) else []
    return [a for a in assocs if isinstance(a, dict)], data if isinstance(data, dict) else {}


def _aspect_path(aspect: str | None) -> str:
    if not aspect:
        return "function"
    normalized = aspect.strip().lower().replace("-", "_")
    aliases = {
        "f": "function",
        "mf": "function",
        "molecular_function": "function",
        "function": "function",
        "p": "process",
        "bp": "process",
        "biological_process": "process",
        "process": "process",
        "c": "cellular_component",
        "cc": "cellular_component",
        "component": "cellular_component",
        "cellular_component": "cellular_component",
    }
    if normalized not in aliases:
        raise ValueError(
            "GO aspect must be one of function/molecular_function, "
            "process/biological_process, or component/cellular_component"
        )
    return aliases[normalized]


def extract_uniprot_accessions(
    associations: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    accessions: list[str] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in associations:
        haystack = " ".join(
            str(row.get(k, ""))
            for k in ("bioentity", "bioentity_label", "subject", "object")
        )
        match = _UNIPROT_RE.search(haystack)
        if match:
            acc = match.group(1)
            if acc not in seen:
                seen.add(acc)
                accessions.append(acc)
        else:
            unresolved.append(row)
    return accessions, unresolved
