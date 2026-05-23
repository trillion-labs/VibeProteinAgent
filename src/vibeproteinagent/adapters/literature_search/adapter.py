from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters._remote.europepmc import search_europepmc
from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.temporal import before_cutoff, parse_date
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext

_GO_PHRASE_MAP: tuple[tuple[str, str], ...] = (
    ("adp-dependent nad(p)h-hydrate dehydratase", "GO:0052855"),
    ("adp-dependent nadph-hydrate dehydratase", "GO:0052855"),
    ("adp-dependent nadh-hydrate dehydratase", "GO:0052855"),
    ("atp-dependent nad(p)h-hydrate dehydratase", "GO:0047453"),
    ("atp-dependent nadph-hydrate dehydratase", "GO:0047453"),
    ("atp-dependent nadh-hydrate dehydratase", "GO:0047453"),
    ("nad(p)hx epimerase", "GO:0052856"),
    ("nadphx epimerase", "GO:0052856"),
    ("nadhx epimerase", "GO:0052856"),
    ("nad(p)h-hydrate epimerase", "GO:0052856"),
)


def _record_date(record: dict[str, Any]) -> str | None:
    for key in (
        "firstPublicationDate",
        "firstIndexDate",
        "electronicPublicationDate",
        "journalInfo",
    ):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value[:10]
        if isinstance(value, dict):
            date = value.get("printPublicationDate") or value.get("dateOfPublication")
            if date:
                return str(date)[:10]
    year = record.get("pubYear")
    return f"{year}-01-01" if year else None


def _filter_dates(
    records: list[dict[str, Any]],
    *,
    before: str | None,
    after: str | None,
) -> list[dict[str, Any]]:
    after_d = parse_date(after)
    out: list[dict[str, Any]] = []
    for record in records:
        date_s = _record_date(record)
        if not before_cutoff(date_s, before, include_undated=False):
            continue
        if after_d is not None:
            public = parse_date(date_s)
            if public is None or public < after_d:
                continue
        out.append(record)
    return out


def _phrase_go_terms(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text.casefold())
    return {go_id for phrase, go_id in _GO_PHRASE_MAP if phrase in normalized}


def _evidence(records: list[dict[str, Any]], *, seed_text: str = "") -> dict[str, Any]:
    patterns = {
        "temperatures_c": re.compile(r"\b([0-9]{2,3})\s*(?:deg|degrees|°)?\s*C\b", re.I),
        "ph_values": re.compile(r"\bpH\s*([0-9]+(?:\.[0-9]+)?)\b", re.I),
        "pdb_ids": re.compile(r"\b[0-9][A-Za-z0-9]{3}\b"),
        "uniprot_accessions": re.compile(
            r"\b(?:"
            r"[OPQ][0-9][A-Z0-9]{3}[0-9]"
            r"|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]"
            r")\b"
        ),
        "interpro_ids": re.compile(r"\bIPR[0-9]{6}\b"),
        "go_terms": re.compile(r"\bGO:[0-9]{7}\b"),
    }
    found: dict[str, set[str]] = {k: set() for k in patterns}
    if seed_text:
        for key, pat in patterns.items():
            found[key].update(m.group(0) for m in pat.finditer(seed_text))
        found["go_terms"].update(_phrase_go_terms(seed_text))
    for record in records:
        text = " ".join(str(record.get(k, "")) for k in ("title", "abstractText", "keywordList"))
        for key, pat in patterns.items():
            found[key].update(m.group(0) for m in pat.finditer(text))
        found["go_terms"].update(_phrase_go_terms(text))
    return {key: sorted(values)[:100] for key, values in found.items()}


class LiteratureSearchInput(BaseModel):
    query: str = Field(min_length=1)
    functional_description: str | None = None
    must_include: list[str] = Field(default_factory=list)
    organism: str | None = None
    max_results: int = Field(25, ge=1, le=200)
    published_before: str | None = None
    published_after: str | None = None
    temporal_cutoff: str | None = None
    result_type: str = Field("lite", pattern=r"^(lite|core)$")
    open_access_only: bool = False
    include_abstracts: bool = True
    page_size: int = Field(25, ge=1, le=100)
    timeout_s: float = Field(20.0, gt=0, le=120)


class LiteratureSearchOutput(BaseModel):
    query: str
    backend: str
    result_count: int
    date_filter: dict[str, str | None]
    constraint_evidence: dict[str, Any]
    top_records: list[dict[str, Any]]
    artifacts: dict[str, str]
    runtime_ms: int


@register_adapter("descriptor.yaml")
class LiteratureSearchAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.config = config
        self.device = device

    def run(self, params: LiteratureSearchInput, ctx: AdapterRunContext) -> LiteratureSearchOutput:
        t0 = time.time()
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = out_dir / "http_cache"
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=cache_dir)
        query_parts = [params.query]
        query_parts.extend(f'"{term}"' for term in params.must_include)
        if params.organism:
            query_parts.append(f'"{params.organism}"')
        if params.functional_description:
            query_parts.append(params.functional_description)
        query = " ".join(query_parts)
        published_before = params.published_before
        if params.temporal_cutoff and (
            published_before is None
            or parse_date(params.temporal_cutoff) < parse_date(published_before)
        ):
            published_before = params.temporal_cutoff
        ctx.emit_progress(0.15)
        records, raw_pages = search_europepmc(
            client,
            query,
            max_results=params.max_results,
            page_size=params.page_size,
            result_type=params.result_type,
            open_access_only=params.open_access_only,
            published_before=published_before,
            published_after=params.published_after,
        )
        filtered = _filter_dates(records, before=published_before, after=params.published_after)
        ctx.emit_progress(0.65)
        records_path = out_dir / "literature_results.json"
        raw_path = out_dir / "literature_raw_pages.json"
        tsv_path = out_dir / "literature_results.tsv"
        evidence_path = out_dir / "constraint_evidence.json"
        records_path.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")
        raw_path.write_text(json.dumps(raw_pages, indent=2, sort_keys=True) + "\n")
        with tsv_path.open("w") as fh:
            fh.write("title\tyear\tdoi\tpmid\tfirst_publication_date\n")
            for r in filtered:
                title = str(r.get("title", "")).replace("\t", " ").replace("\n", " ")
                row = [
                    title,
                    str(r.get("pubYear", "")),
                    str(r.get("doi", "")),
                    str(r.get("pmid", "")),
                    _record_date(r) or "",
                ]
                fh.write("\t".join(row) + "\n")
        evidence = _evidence(
            filtered,
            seed_text=" ".join(
                part
                for part in (query, params.functional_description or "")
                if part
            ),
        )
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        top = []
        for r in filtered[: min(10, len(filtered))]:
            abstract = str(r.get("abstractText", "")) if params.include_abstracts else ""
            top.append(
                {
                    "title": r.get("title"),
                    "year": int(r["pubYear"]) if str(r.get("pubYear", "")).isdigit() else None,
                    "doi": r.get("doi"),
                    "pmid": r.get("pmid"),
                    "first_publication_date": _record_date(r),
                    "abstract_snippet": abstract[:240],
                }
            )
        ctx.emit_progress(1.0)
        return LiteratureSearchOutput(
            query=query,
            backend="europepmc",
            result_count=len(filtered),
            date_filter={
                "published_before": published_before,
                "published_after": params.published_after,
            },
            constraint_evidence=evidence,
            top_records=top,
            artifacts={
                "records_json": str(records_path),
                "raw_pages_json": str(raw_path),
                "records_tsv": str(tsv_path),
                "constraint_evidence_json": str(evidence_path),
            },
            runtime_ms=int((time.time() - t0) * 1000),
        )

    def estimate_vram(self, params: LiteratureSearchInput) -> int:
        return 0


__all__ = [
    "LiteratureSearchAdapter",
    "LiteratureSearchInput",
    "LiteratureSearchOutput",
]
