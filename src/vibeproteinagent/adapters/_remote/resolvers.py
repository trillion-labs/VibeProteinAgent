from __future__ import annotations

from dataclasses import dataclass
import urllib.parse
from typing import Any

from .http import HttpClient, HttpError

QUICKGO = "https://www.ebi.ac.uk/QuickGO/services"
OLS4 = "https://www.ebi.ac.uk/ols4/api"
INTERPRO = "https://www.ebi.ac.uk/interpro/api"

_DISCRIMINATING_TOKENS = {
    "adp",
    "atp",
    "amp",
    "cdp",
    "ctp",
    "cmp",
    "gdp",
    "gtp",
    "gmp",
    "udp",
    "utp",
    "ump",
    "dna",
    "rna",
    "nad",
    "nadh",
    "nadp",
    "nadph",
    "nadhx",
    "nadphx",
}


@dataclass(frozen=True)
class IdentifierSelection:
    selected_id: str | None
    ambiguous: bool
    ready_for_execution: bool
    selection_reason: str | None
    warnings: tuple[str, ...] = ()

_GO_ASPECTS = {
    "molecular_function": {"molecular_function", "function", "mf", "f"},
    "biological_process": {"biological_process", "process", "bp", "p"},
    "cellular_component": {"cellular_component", "component", "cc", "c"},
}


def normalize_go_aspect(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_")
    for canonical, aliases in _GO_ASPECTS.items():
        if normalized in aliases:
            return canonical
    return normalized or None


def search_go_terms(
    client: HttpClient,
    query: str,
    *,
    aspect: str | None = None,
    limit: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "page": 1,
    }
    normalized_aspect = normalize_go_aspect(aspect)
    if normalized_aspect:
        params["aspect"] = normalized_aspect
    try:
        data = client.get_json(f"{QUICKGO}/ontology/go/search", params=params)
        rows = _quickgo_term_rows(data)
        source = "quickgo"
    except HttpError as exc:
        data = client.get_json(
            f"{OLS4}/search",
            params={
                "q": query,
                "ontology": "go",
                "rows": limit,
                "format": "json",
            },
        )
        rows = _ols_term_rows(data, aspect=normalized_aspect)
        source = "ols4"
        data = {"fallback_from": str(exc), "ols4": data}
    ranked = _rank_terms(rows, query=query, source=source, limit=limit)
    return ranked, data if isinstance(data, dict) else {"raw": data}


def lookup_go_term(client: HttpClient, term_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        data = client.get_json(f"{QUICKGO}/ontology/go/terms/{urllib.parse.quote(term_id, safe='')}")
        rows = _quickgo_term_rows(data)
        source = "quickgo"
    except HttpError as exc:
        iri = f"http://purl.obolibrary.org/obo/{term_id.replace(':', '_')}"
        data = client.get_json(
            f"{OLS4}/terms",
            params={"iri": iri, "ontology": "go"},
        )
        rows = _ols_term_rows(data, aspect=None)
        source = "ols4"
        data = {"fallback_from": str(exc), "ols4": data}
    row = rows[0] if rows else {"id": term_id, "valid": False}
    row = {**row, "source": source, "valid": bool(rows)}
    return row, data if isinstance(data, dict) else {"raw": data}


def search_interpro_entries(
    client: HttpClient,
    query: str,
    *,
    entry_type: str | None = None,
    limit: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {"search": query, "page_size": limit}
    data = client.get_json(f"{INTERPRO}/entry/interpro/", params=params)
    rows = _interpro_entry_rows(data)
    if entry_type:
        wanted = entry_type.strip().lower()
        rows = [
            row for row in rows
            if str(row.get("type") or "").lower() in {wanted, wanted.replace("_", " ")}
        ]
    ranked = _rank_entries(rows, query=query, source="interpro_api", limit=limit)
    return ranked, data if isinstance(data, dict) else {"raw": data}


def select_best_identifier(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    id_key: str,
    label_key: str = "name",
    obsolete_key: str = "obsolete",
    namespace: str = "term",
) -> IdentifierSelection:
    """Select a deterministic executable identifier from resolver candidates.

    Ambiguity remains metadata about close alternatives; it does not suppress a
    selected identifier when a usable candidate exists.
    """
    usable = [
        row
        for row in candidates
        if isinstance(row, dict) and row.get(id_key)
    ]
    if not usable:
        return IdentifierSelection(
            selected_id=None,
            ambiguous=False,
            ready_for_execution=False,
            selection_reason=None,
        )

    non_obsolete = [
        row for row in usable if not bool(row.get(obsolete_key))
    ]
    if non_obsolete:
        usable = non_obsolete

    ranked = sorted(
        usable,
        key=lambda row: _identifier_selection_key(
            query=query,
            row=row,
            id_key=id_key,
            label_key=label_key,
            obsolete_key=obsolete_key,
        ),
    )
    best = ranked[0]
    selected_id = str(best[id_key])
    reason = _selection_reason(query=query, row=best, label_key=label_key)
    ambiguous = _selection_is_ambiguous(
        query=query,
        ranked=ranked,
        id_key=id_key,
        label_key=label_key,
        obsolete_key=obsolete_key,
    )
    warnings: tuple[str, ...] = ()
    if ambiguous:
        warnings = (
            f"Multiple plausible {namespace} candidates were found; selected "
            "the highest-ranked candidate deterministically.",
        )
    return IdentifierSelection(
        selected_id=selected_id,
        ambiguous=ambiguous,
        ready_for_execution=True,
        selection_reason=reason,
        warnings=warnings,
    )


def lookup_interpro_entry(
    client: HttpClient,
    accession: str,
    *,
    database: str = "interpro",
) -> tuple[dict[str, Any], dict[str, Any]]:
    db_segment = _interpro_database_segment(database)
    data = client.get_json(f"{INTERPRO}/entry/{db_segment}/{accession}/")
    rows = _interpro_entry_rows(data)
    if not rows and isinstance(data, dict):
        rows = [_interpro_entry_from_mapping(data)]
    row = rows[0] if rows else {"accession": accession, "valid": False}
    row = {**row, "database": database, "valid": bool(rows)}
    return row, data if isinstance(data, dict) else {"raw": data}


def _quickgo_term_rows(data: Any) -> list[dict[str, Any]]:
    rows = data.get("results") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        term_id = row.get("id")
        if not term_id:
            continue
        out.append(
            {
                "id": str(term_id),
                "name": row.get("name"),
                "aspect": normalize_go_aspect(row.get("aspect")),
                "definition": _definition_text(row.get("definition")),
                "synonyms": _synonyms(row.get("synonyms")),
                "obsolete": bool(row.get("isObsolete") or row.get("obsolete", False)),
                "source_url": row.get("url"),
            }
        )
    return out


def _ols_term_rows(data: Any, *, aspect: str | None) -> list[dict[str, Any]]:
    docs = None
    if isinstance(data, dict):
        response = data.get("response")
        if isinstance(response, dict):
            docs = response.get("docs")
        embedded = data.get("_embedded")
        if docs is None and isinstance(embedded, dict):
            docs = embedded.get("terms")
    if not isinstance(docs, list):
        return []
    out: list[dict[str, Any]] = []
    for row in docs:
        if not isinstance(row, dict):
            continue
        term_id = _go_id_from_ols(row)
        if not term_id:
            continue
        out.append(
            {
                "id": term_id,
                "name": row.get("label") or row.get("name"),
                "aspect": aspect,
                "definition": _first(row.get("description") or row.get("definition")),
                "synonyms": [
                    str(item)
                    for item in (row.get("synonym") or row.get("synonyms") or [])
                    if item
                ],
                "obsolete": bool(row.get("is_obsolete") or row.get("obsolete", False)),
                "source_url": row.get("iri"),
            }
        )
    return out


def _go_id_from_ols(row: dict[str, Any]) -> str | None:
    for key in ("obo_id", "short_form", "id"):
        value = row.get(key)
        if isinstance(value, str):
            if value.startswith("GO:"):
                return value
            if value.startswith("GO_"):
                return value.replace("_", ":", 1)
    iri = row.get("iri")
    if isinstance(iri, str) and "GO_" in iri:
        return iri.rsplit("/", 1)[-1].replace("_", ":", 1)
    return None


def _interpro_entry_rows(data: Any) -> list[dict[str, Any]]:
    rows = data.get("results") if isinstance(data, dict) else None
    if isinstance(rows, list):
        return [_interpro_entry_from_mapping(row) for row in rows if isinstance(row, dict)]
    if isinstance(data, dict) and data.get("metadata"):
        return [_interpro_entry_from_mapping(data)]
    return []


def _interpro_entry_from_mapping(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else row
    accession = (
        metadata.get("accession")
        or row.get("accession")
        or metadata.get("id")
    )
    entry_type = metadata.get("type") or metadata.get("entry_type")
    return {
        "accession": str(accession) if accession else "",
        "name": metadata.get("name") or metadata.get("short_name"),
        "type": entry_type,
        "member_database": metadata.get("source_database") or "interpro",
        "description": _first(metadata.get("description")),
        "go_terms": _go_terms_from_interpro(metadata),
        "source_url": metadata.get("url"),
    }


def _rank_terms(
    rows: list[dict[str, Any]],
    *,
    query: str,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    return [
        {**row, "score": _text_score(query, row.get("name"), row.get("definition")), "source": source}
        for row in sorted(
            rows,
            key=lambda row: _text_score(query, row.get("name"), row.get("definition")),
            reverse=True,
        )[:limit]
    ]


def _rank_entries(
    rows: list[dict[str, Any]],
    *,
    query: str,
    source: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("accession")]
    return [
        {**row, "score": _text_score(query, row.get("name"), row.get("description")), "source": source}
        for row in sorted(
            rows,
            key=lambda row: _text_score(query, row.get("name"), row.get("description")),
            reverse=True,
        )[:limit]
    ]


def _text_score(query: str, *fields: Any) -> float:
    query_tokens = _tokens(query)
    text_tokens = _tokens(" ".join(str(field or "") for field in fields))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & text_tokens)
    return round(overlap / max(len(query_tokens), 1), 4)


def _tokens(value: str) -> set[str]:
    return {part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in value).split() if len(part) > 1}


def _normalized_phrase(value: Any) -> str:
    return " ".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in str(value or "")).split()
    )


def _score_value(row: dict[str, Any]) -> float:
    value = row.get("score")
    if value is None:
        value = row.get("confidence")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _label_tokens(row: dict[str, Any], *, label_key: str) -> set[str]:
    label = row.get(label_key) or row.get("label") or row.get("name") or ""
    return _tokens(str(label))


def _query_token_coverage(query: str, row: dict[str, Any], *, label_key: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _label_tokens(row, label_key=label_key)) / len(query_tokens)


def _exact_label_match(query: str, row: dict[str, Any], *, label_key: str) -> bool:
    query_norm = _normalized_phrase(query)
    label_norm = _normalized_phrase(row.get(label_key) or row.get("label") or row.get("name"))
    if not query_norm or not label_norm:
        return False
    if query_norm == label_norm:
        return True
    return len(label_norm.split()) > 1 and f" {label_norm} " in f" {query_norm} "


def _critical_token_score(query: str, row: dict[str, Any], *, label_key: str) -> int:
    query_critical = _tokens(query) & _DISCRIMINATING_TOKENS
    label_critical = _label_tokens(row, label_key=label_key) & _DISCRIMINATING_TOKENS
    return len(query_critical & label_critical) - len(label_critical - query_critical)


def _identifier_selection_key(
    *,
    query: str,
    row: dict[str, Any],
    id_key: str,
    label_key: str,
    obsolete_key: str,
) -> tuple[Any, ...]:
    label_norm = _normalized_phrase(row.get(label_key) or row.get("label") or row.get("name"))
    identifier = str(row.get(id_key) or "")
    return (
        -int(_exact_label_match(query, row, label_key=label_key)),
        -_critical_token_score(query, row, label_key=label_key),
        -_score_value(row),
        -_query_token_coverage(query, row, label_key=label_key),
        int(bool(row.get(obsolete_key))),
        label_norm,
        identifier,
    )


def _selection_reason(query: str, row: dict[str, Any], *, label_key: str) -> str:
    if _exact_label_match(query, row, label_key=label_key):
        return "exact_label_match"
    if _critical_token_score(query, row, label_key=label_key) > 0:
        return "exact_key_token_match"
    if _score_value(row) > 0:
        return "highest_backend_score"
    if _query_token_coverage(query, row, label_key=label_key) > 0:
        return "query_token_coverage"
    return "best_ranked_candidate_after_tiebreak"


def _selection_is_ambiguous(
    *,
    query: str,
    ranked: list[dict[str, Any]],
    id_key: str,
    label_key: str,
    obsolete_key: str,
) -> bool:
    if len(ranked) < 2:
        return False
    best = ranked[0]
    if _exact_label_match(query, best, label_key=label_key):
        return False
    if abs(_score_value(best) - _score_value(ranked[1])) < 0.05:
        return True
    best_key = _identifier_selection_key(
        query=query,
        row=best,
        id_key=id_key,
        label_key=label_key,
        obsolete_key=obsolete_key,
    )[:4]
    second_key = _identifier_selection_key(
        query=query,
        row=ranked[1],
        id_key=id_key,
        label_key=label_key,
        obsolete_key=obsolete_key,
    )[:4]
    return best_key == second_key


def _definition_text(value: Any) -> str | None:
    if isinstance(value, dict):
        return _first(value.get("text") or value.get("xrefs"))
    return _first(value)


def _first(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        return _first(value[0])
    if isinstance(value, dict):
        for key in ("text", "value", "name", "description"):
            if key in value:
                return _first(value[key])
    return None


def _synonyms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            text = _first(item.get("name") or item.get("text"))
            if text:
                out.append(text)
    return out


def _go_terms_from_interpro(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("go_terms") or metadata.get("goTerms") or []
    if isinstance(raw, dict):
        raw = raw.values()
    if not isinstance(raw, list):
        raw = list(raw) if raw else []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"id": item})
        elif isinstance(item, dict):
            term_id = item.get("identifier") or item.get("id") or item.get("accession")
            if term_id:
                out.append({"id": str(term_id), "name": item.get("name")})
    return out


def _interpro_database_segment(database: str) -> str:
    normalized = database.strip().lower().replace("_", "-")
    if normalized in {"interpro", "ipr"}:
        return "interpro"
    if normalized == "pfam":
        return "pfam"
    if normalized == "panther":
        return "panther"
    return database.strip()
