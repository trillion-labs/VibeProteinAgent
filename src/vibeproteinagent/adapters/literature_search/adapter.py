from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters._remote.europepmc import (
    extract_ids_from_annotations,
    fetch_text_mined_annotations,
    normalize_query,
    search_europepmc,
)
from vibeproteinagent.adapters._remote.http import HttpClient
from vibeproteinagent.adapters._remote.temporal import before_cutoff, parse_date
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.core.config import vpa_home


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


_PH_RE = re.compile(r"\bpH\s*([0-9]+(?:\.[0-9]+)?)\b", re.I)
_TEMP_RE = re.compile(r"\b([0-9]{2,3})\s*(?:deg|degrees|°)\s*C\b", re.I)


def _abstract_evidence(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Scrape narrow physicochemical hints (pH, temperature) from abstracts.

    Europe PMC's text-mining pipeline doesn't surface these as entities, so
    we keep a *tight* regex pass here — explicit ``°C`` / ``degrees C`` only,
    so generic ``"60 C"`` no longer false-positives. Runs only when abstracts
    are present (i.e. ``result_type=core``).
    """
    ph: set[str] = set()
    temps: set[str] = set()
    for r in records:
        text = str(r.get("abstractText") or "")
        if not text:
            continue
        ph.update(m.group(0) for m in _PH_RE.finditer(text))
        temps.update(m.group(0) for m in _TEMP_RE.finditer(text))
    return {
        "ph_values": sorted(ph)[:50],
        "temperatures_c": sorted(temps)[:50],
    }


def _europepmc_cache_dir() -> Path:
    """Process-wide Europe PMC HTTP cache.

    Honours ``$VPA_HOME`` (the same env the rest of the codebase uses for
    workspaces / models / sidecar trees); falls back to ``~/.vibeproteinagent``
    so unit tests and laptop installs keep working without configuration.
    No host-specific paths are baked into the source — operators point
    ``VPA_HOME`` at their shared storage location.
    """
    return vpa_home() / "cache" / "europepmc"


def _hit_count(raw_pages: list[dict[str, Any]]) -> int | None:
    if not raw_pages:
        return None
    first = raw_pages[0]
    if not isinstance(first, dict):
        return None
    value = first.get("hitCount")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _journal_fields(r: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return (journal_title, volume, issue) handling lite vs core shapes.

    Lite-mode records expose flat ``journalTitle`` / ``journalVolume`` / ``issue``;
    core-mode records nest them under ``journalInfo.{journal.title, volume, issue}``.
    """
    title = r.get("journalTitle")
    volume = r.get("journalVolume")
    issue = r.get("issue")
    info = r.get("journalInfo")
    if isinstance(info, dict):
        if not title:
            inner = info.get("journal")
            if isinstance(inner, dict):
                title = inner.get("title") or inner.get("medlineAbbreviation")
        if not volume:
            volume = info.get("volume")
        if not issue:
            issue = info.get("issue")
    return (
        str(title) if title else None,
        str(volume) if volume else None,
        str(issue) if issue else None,
    )


def _build_top_record(r: dict[str, Any], *, include_abstract_snippet: bool) -> dict[str, Any]:
    """Self-contained record shape for downstream tools and the HTML widget.

    Carries every field the report widget previously had to re-read from
    ``literature_raw_pages.json`` (authors, journal, citation chunks,
    pmcid, open-access flag) so the artifact directory is no longer a
    rendering dependency.
    """
    abstract = str(r.get("abstractText") or "")
    pub_year_raw = r.get("pubYear")
    journal, volume, issue = _journal_fields(r)
    return {
        "title": r.get("title"),
        "authors": r.get("authorString"),
        "journal": journal,
        "journal_volume": volume,
        "issue": issue,
        "page": r.get("pageInfo"),
        "year": int(pub_year_raw) if str(pub_year_raw or "").isdigit() else None,
        "doi": r.get("doi"),
        "pmid": r.get("pmid"),
        "pmcid": r.get("pmcid"),
        "source": r.get("source"),
        "first_publication_date": _record_date(r),
        "is_open_access": str(r.get("isOpenAccess") or "").upper() == "Y",
        "has_text_mined_terms": str(r.get("hasTextMinedTerms") or "").upper() == "Y",
        "cited_by_count": r.get("citedByCount"),
        "abstract_snippet": (abstract[:480] if include_abstract_snippet else ""),
    }


class LiteratureSearchInput(BaseModel):
    query: str = Field(min_length=1)
    functional_description: str | None = None
    must_include: list[str] = Field(default_factory=list)
    organism: str | None = None
    max_results: int = Field(25, ge=1, le=200)
    published_before: str | None = None
    published_after: str | None = None
    temporal_cutoff: str | None = Field(
        default=None,
        description=(
            "Deprecated alias for ``published_before``; supplying both with "
            "different values is a validation error."
        ),
    )
    result_type: str = Field("core", pattern=r"^(lite|core)$")
    open_access_only: bool = False
    include_abstracts: bool = True
    page_size: int = Field(25, ge=1, le=100)
    timeout_s: float = Field(20.0, gt=0, le=120)
    max_query_terms: int = Field(8, ge=1, le=32)
    fetch_text_mined_terms: bool = True

    @model_validator(mode="after")
    def _reconcile_dates(self) -> LiteratureSearchInput:
        if (
            self.published_before
            and self.temporal_cutoff
            and self.published_before != self.temporal_cutoff
        ):
            raise ValueError(
                "published_before and temporal_cutoff both set with different "
                "values; they are synonyms — pass only one."
            )
        if self.temporal_cutoff and not self.published_before:
            self.published_before = self.temporal_cutoff
        # ``include_abstracts`` controls whether downstream consumers receive
        # abstract snippets — that data only exists in the ``core`` resultType.
        # If a caller asks for abstracts but pinned the legacy ``lite`` mode,
        # coerce up so the flag stops being a no-op.
        if self.include_abstracts and self.result_type == "lite":
            self.result_type = "core"
        return self


class LiteratureSearchOutput(BaseModel):
    query: str
    normalized_query: str
    backend: str
    status: str  # "ok" | "no_hits" | "filtered_out"
    result_count: int
    total_hits: int | None
    date_filter: dict[str, str | None]
    constraint_evidence: dict[str, Any]
    text_mined_annotations: dict[str, Any]
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

        cache_dir = _europepmc_cache_dir()
        client = HttpClient(timeout_s=params.timeout_s, cache_dir=cache_dir)

        raw_parts = [params.query]
        raw_parts.extend(f'"{term}"' for term in params.must_include if term)
        if params.organism:
            raw_parts.append(f'"{params.organism}"')
        if params.functional_description:
            raw_parts.append(params.functional_description)
        raw_query = " ".join(raw_parts)
        query = normalize_query(raw_query, max_terms=params.max_query_terms)

        published_before = params.published_before
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
        total_hits = _hit_count(raw_pages)
        filtered = _filter_dates(records, before=published_before, after=params.published_after)
        ctx.emit_progress(0.55)

        if filtered and params.fetch_text_mined_terms:
            tm_annotations = fetch_text_mined_annotations(client, filtered)
        else:
            tm_annotations = {}
        ctx.emit_progress(0.8)

        if filtered:
            constraint_evidence = {
                **extract_ids_from_annotations(tm_annotations),
                **_abstract_evidence(filtered),
            }
            status = "ok"
        elif total_hits and total_hits > 0:
            constraint_evidence = {}
            status = "filtered_out"
        else:
            constraint_evidence = {}
            status = "no_hits"

        records_path = out_dir / "literature_results.json"
        raw_path = out_dir / "literature_raw_pages.json"
        tsv_path = out_dir / "literature_results.tsv"
        evidence_path = out_dir / "constraint_evidence.json"
        annotations_path = out_dir / "text_mined_annotations.json"
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
        evidence_path.write_text(json.dumps(constraint_evidence, indent=2, sort_keys=True) + "\n")
        annotations_path.write_text(
            json.dumps(tm_annotations, indent=2, sort_keys=True) + "\n"
        )

        top = [
            _build_top_record(r, include_abstract_snippet=params.include_abstracts)
            for r in filtered[: min(10, len(filtered))]
        ]
        ctx.emit_progress(1.0)
        return LiteratureSearchOutput(
            query=raw_query,
            normalized_query=query,
            backend="europepmc",
            status=status,
            result_count=len(filtered),
            total_hits=total_hits,
            date_filter={
                "published_before": published_before,
                "published_after": params.published_after,
            },
            constraint_evidence=constraint_evidence,
            text_mined_annotations=tm_annotations,
            top_records=top,
            artifacts={
                "records_json": str(records_path),
                "raw_pages_json": str(raw_path),
                "records_tsv": str(tsv_path),
                "constraint_evidence_json": str(evidence_path),
                "text_mined_annotations_json": str(annotations_path),
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
