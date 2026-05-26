from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from .dates import parse_date
from .http import HttpClient

EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_ANNOTATIONS = (
    "https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds"
)

# Lucene metacharacters that, if left unescaped in a free-form query term,
# either trip Europe PMC's parser (returning hitCount=0) or are silently
# reinterpreted as operators. Tokens containing any of these are wrapped in
# double quotes so the underlying API treats them as a literal phrase.
_LUCENE_SPECIALS = set("()[]{}\"~^*?:\\/+!")


def normalize_query(text: str, *, max_terms: int = 8) -> str:
    """Make an LLM-authored natural-language query safe for Europe PMC.

    Strips commas/semicolons, quotes any token containing Lucene-special
    characters (so ``NAD(P)HX`` becomes ``"NAD(P)HX"`` rather than a broken
    grouping), drops duplicates, and truncates to ``max_terms`` tokens.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text))
    cleaned = cleaned.replace(",", " ").replace(";", " ")
    tokens: list[str] = []
    buf: list[str] = []
    in_quote = False
    for ch in cleaned:
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
            continue
        if ch.isspace() and not in_quote:
            if buf:
                tokens.append("".join(buf))
                buf = []
            continue
        buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    out: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        tok = raw.strip()
        if not tok:
            continue
        if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
            normalized = tok
        elif any(ch in tok for ch in _LUCENE_SPECIALS):
            inner = tok.replace("\\", "\\\\").replace('"', '\\"')
            normalized = f'"{inner}"'
        else:
            normalized = tok
        if normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        out.append(normalized)
        if len(out) >= max_terms:
            break
    return " ".join(out)


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


# --- text-mined annotations ---------------------------------------------------

_TM_BUCKET = {
    "Gene_Proteins": "gene_proteins",
    "Gene Proteins": "gene_proteins",
    "Chemicals": "chemicals",
    "Organisms": "organisms",
    "Accession Numbers": "accession_numbers",
    "Accession_Numbers": "accession_numbers",
    "Gene Ontology": "go_terms",
    "Gene_Ontology": "go_terms",
    "Diseases": "diseases",
}

_TM_TYPES_DEFAULT: tuple[str, ...] = (
    "Gene_Proteins",
    "Chemicals",
    "Organisms",
    "Accession Numbers",
    "Gene Ontology",
    "Diseases",
)

_GO_URI_RE = re.compile(r"GO[_:](\d{7})")
_UNIPROT_URI_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])\b"
)
_INTERPRO_URI_RE = re.compile(r"\b(IPR\d{6})\b")


def _tm_bucket_name(t: str) -> str:
    if t in _TM_BUCKET:
        return _TM_BUCKET[t]
    return t.lower().replace(" ", "_").replace("/", "_")


def fetch_text_mined_annotations(
    client: HttpClient,
    records: list[dict[str, Any]],
    *,
    types: tuple[str, ...] = _TM_TYPES_DEFAULT,
    batch_size: int = 8,
    max_records: int = 25,
    per_bucket_cap: int = 100,
) -> dict[str, list[dict[str, str]]]:
    """Pull structured entity annotations from Europe PMC's text-mining API.

    Replaces the previous regex-over-title scheme: Europe PMC has already done
    NER on these articles (when ``hasTextMinedTerms == "Y"``), and surfaces
    canonical IDs as ``tags[*].uri`` (e.g. GO/UniProt/InterPro/taxonomy URLs).
    """
    article_ids: list[str] = []
    for r in records[:max_records]:
        if str(r.get("hasTextMinedTerms", "")).upper() != "Y":
            continue
        src = r.get("source") or "MED"
        rid = r.get("id") or r.get("pmid") or r.get("pmcid")
        if not rid:
            continue
        article_ids.append(f"{src}:{rid}")
    if not article_ids:
        return {}

    aggregated: dict[str, list[dict[str, str]]] = {}
    for offset in range(0, len(article_ids), batch_size):
        chunk = article_ids[offset:offset + batch_size]
        try:
            data = client.get_json(
                EUROPEPMC_ANNOTATIONS,
                params={
                    "articleIds": chunk,
                    "type": list(types),
                    "format": "JSON",
                },
            )
        except Exception:
            # Annotation endpoint is best-effort — never let it fail the
            # overall search. The caller still gets the literature records.
            continue
        if not isinstance(data, list):
            continue
        for art in data:
            if not isinstance(art, dict):
                continue
            for ann in art.get("annotations") or []:
                if not isinstance(ann, dict):
                    continue
                kind = str(ann.get("type") or "").strip()
                exact = str(ann.get("exact") or "").strip()
                if not kind or not exact:
                    continue
                bucket = aggregated.setdefault(_tm_bucket_name(kind), [])
                entry: dict[str, str] = {"name": exact}
                tags = ann.get("tags")
                if isinstance(tags, list) and tags:
                    tag = tags[0]
                    if isinstance(tag, dict):
                        uri = tag.get("uri")
                        canonical = tag.get("name")
                        if isinstance(uri, str) and uri:
                            entry["uri"] = uri
                        if (
                            isinstance(canonical, str)
                            and canonical
                            and canonical != exact
                        ):
                            entry["canonical"] = canonical
                bucket.append(entry)

    deduped: dict[str, list[dict[str, str]]] = {}
    for bucket_name, items in aggregated.items():
        seen: set[tuple[str, str]] = set()
        kept: list[dict[str, str]] = []
        for entry in items:
            key = (entry.get("name", ""), entry.get("uri", ""))
            if key in seen:
                continue
            seen.add(key)
            kept.append(entry)
            if len(kept) >= per_bucket_cap:
                break
        deduped[bucket_name] = kept
    return deduped


def extract_ids_from_annotations(
    annotations: dict[str, list[dict[str, str]]],
) -> dict[str, list[str]]:
    """Pull canonical IDs (GO/UniProt/InterPro) out of TM annotations.

    Returns lists in the same shape downstream tools have always consumed:
    ``go_terms`` as ``["GO:NNNNNNN", ...]``, etc.
    """
    go_terms: set[str] = set()
    uniprot: set[str] = set()
    interpro: set[str] = set()

    def _scan(values: list[dict[str, str]]) -> None:
        for entry in values:
            for field in ("uri", "canonical", "name"):
                value = entry.get(field)
                if not value:
                    continue
                for m in _GO_URI_RE.finditer(value):
                    go_terms.add(f"GO:{m.group(1)}")
                for m in _UNIPROT_URI_RE.finditer(value):
                    uniprot.add(m.group(1))
                for m in _INTERPRO_URI_RE.finditer(value):
                    interpro.add(m.group(1))

    for bucket_name, entries in annotations.items():
        _scan(entries)
        # ``Gene Ontology`` bucket is the primary GO source but UniProt /
        # InterPro IDs frequently appear under ``Accession Numbers`` too —
        # the scan above already covers both.
        _ = bucket_name
    return {
        "go_terms": sorted(go_terms),
        "uniprot_accessions": sorted(uniprot),
        "interpro_ids": sorted(interpro),
    }
