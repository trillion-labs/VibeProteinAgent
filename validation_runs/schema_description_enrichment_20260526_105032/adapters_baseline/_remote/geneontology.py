from __future__ import annotations

import re
from typing import Any

from .http import HttpClient

GO = "http://api.geneontology.org/api"

# UniProt accession pattern (SwissProt + TrEMBL canonical form). Used to
# pull the accession out of GO API ``subject.id`` values such as
# ``"UniProtKB:Q9H0R3"``.
_UNIPROT_RE = re.compile(
    r"(?:UniProtKB:)?"
    r"([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])"
)


# ECO term IDs the GO API returns under ``association.evidence`` /
# ``evidence_types[].id``, mapped to the three-letter GO Consortium
# codes (IEA, IDA, ...) that callers actually use.
#
# Source: http://geneontology.org/docs/guide-go-evidence-codes/
_ECO_TO_LETTER: dict[str, str] = {
    # Experimental
    "ECO:0000269": "EXP",
    "ECO:0000314": "IDA",
    "ECO:0000353": "IPI",
    "ECO:0000315": "IMP",
    "ECO:0000316": "IGI",
    "ECO:0000270": "IEP",
    # High-throughput experimental
    "ECO:0006056": "HTP",
    "ECO:0007005": "HDA",
    "ECO:0007001": "HMP",
    "ECO:0007003": "HGI",
    "ECO:0007007": "HEP",
    # Phylogenetic
    "ECO:0000318": "IBA",
    "ECO:0000319": "IBD",
    "ECO:0000320": "IKR",
    "ECO:0000321": "IRD",
    # Computational / sequence similarity
    "ECO:0000250": "ISS",
    "ECO:0000266": "ISO",
    "ECO:0000247": "ISA",
    "ECO:0000255": "ISM",
    "ECO:0000317": "IGC",
    "ECO:0000245": "RCA",
    # Author / curator statements
    "ECO:0000304": "TAS",
    "ECO:0000303": "NAS",
    "ECO:0000305": "IC",
    "ECO:0000307": "ND",
    # Electronic
    "ECO:0000501": "IEA",
}


# Accepted aspect tokens on the public adapter input. They all route to
# the ``function`` endpoint server-side — see :func:`go_associations`
# for why — so this set exists purely to fail fast on typos.
_ASPECT_ALIASES: frozenset[str] = frozenset({
    "f", "mf", "function", "molecular_function",
    "p", "bp", "process", "biological_process",
    "c", "cc", "component", "cellular_component",
})


def evidence_letter_code(raw: Any) -> str:
    """Coerce a GO evidence value into its three-letter GOC code.

    ``raw`` may arrive as an ECO identifier (``"ECO:0000501"``), a
    legacy three-letter code (``"IEA"``), or a nested dict embedding
    either under ``id`` / ``type`` / ``code``. Returns ``""`` when the
    value is missing or carries an ECO id this module doesn't
    recognise — callers should treat the empty case as "unknown"
    rather than as a positive match for any specific code.
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        raw = raw.get("id") or raw.get("type") or raw.get("code") or ""
    text = str(raw).strip().upper()
    if not text:
        return ""
    if text.startswith("ECO:"):
        return _ECO_TO_LETTER.get(text, "")
    return text


def go_associations(
    client: HttpClient,
    go_id: str,
    *,
    rows: int = 200,
    aspect: str | None = None,
    taxon_id: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch gene/protein annotations for a GO term.

    Calls ``/api/bioentity/function/<go_id>/genes``. The trailing
    ``/genes`` segment is mandatory — without it the GO REST API
    returns a *list* of raw annotation records that lacks the
    ``"associations"`` key, so older callers silently saw zero hits.

    ``aspect`` is validated for shape (so obvious typos fail fast) but
    does not change the URL: the ``function`` endpoint correctly
    serves any GO term regardless of its aspect, whereas ``process``
    and ``cellular_component`` paths return 404.
    """
    _normalise_aspect(aspect)
    params: dict[str, Any] = {"rows": rows}
    if taxon_id is not None:
        params["taxon"] = f"NCBITaxon:{taxon_id}"
    data = client.get_json(f"{GO}/bioentity/function/{go_id}/genes", params=params)
    if not isinstance(data, dict):
        return [], {}
    assocs = data.get("associations", [])
    return [a for a in assocs if isinstance(a, dict)], data


def _normalise_aspect(aspect: str | None) -> None:
    """Raise ``ValueError`` for unknown aspect tokens; otherwise no-op."""
    if aspect is None:
        return
    normalized = aspect.strip().lower().replace("-", "_")
    if normalized not in _ASPECT_ALIASES:
        raise ValueError(
            "GO aspect must be one of function/molecular_function, "
            "process/biological_process, or component/cellular_component"
        )


def extract_uniprot_accessions(
    associations: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Pull UniProt accessions out of GO association rows.

    Reads ``subject.id`` (current GO API) plus legacy free-form
    ``bioentity`` / ``bioentity_label`` / ``object`` fields. Rows
    whose subject is a non-UniProt identifier (MGI:..., FB:..., ...)
    end up in the returned ``unresolved`` list so the caller can
    record them in the drop audit.
    """
    accessions: list[str] = []
    unresolved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in associations:
        haystack_parts: list[str] = []
        subject = row.get("subject")
        if isinstance(subject, dict):
            subj_id = subject.get("id")
            if subj_id:
                haystack_parts.append(str(subj_id))
        for key in ("bioentity", "bioentity_label", "object"):
            value = row.get(key)
            if value is not None:
                haystack_parts.append(str(value))
        match = _UNIPROT_RE.search(" ".join(haystack_parts))
        if match:
            acc = match.group(1)
            if acc not in seen:
                seen.add(acc)
                accessions.append(acc)
        else:
            unresolved.append(row)
    return accessions, unresolved
