from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters._remote.europepmc import normalize_query
from vibeproteinagent.adapters.literature_search import adapter as mod
from vibeproteinagent.adapters.literature_search.adapter import (
    LiteratureSearchAdapter,
    LiteratureSearchInput,
)


def _adapter(tmp_path: Path) -> LiteratureSearchAdapter:
    return LiteratureSearchAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))


def test_literature_search_mocked_europepmc_temporal_filter(tmp_path, monkeypatch):
    seen: dict[str, str | None] = {}

    def fake_search(*_args, **_kwargs):
        seen["published_before"] = _kwargs.get("published_before")
        seen["published_after"] = _kwargs.get("published_after")
        return (
            [
                {
                    "id": "1",
                    "source": "MED",
                    "title": "old cellulase",
                    "pubYear": "2020",
                    "doi": "10.1/old",
                    "pmid": "1",
                    "firstPublicationDate": "2020-05-14",
                    "abstractText": "Active at pH 5 and 60 degrees C.",
                    "hasTextMinedTerms": "Y",
                },
                {
                    "id": "2",
                    "source": "MED",
                    "title": "new cellulase",
                    "pubYear": "2024",
                    "doi": "10.1/new",
                    "pmid": "2",
                    "firstPublicationDate": "2024-01-01",
                    "abstractText": "too new",
                    "hasTextMinedTerms": "N",
                },
            ],
            [{"hitCount": 2}],
        )

    monkeypatch.setattr(mod, "search_europepmc", fake_search)
    monkeypatch.setattr(
        mod,
        "fetch_text_mined_annotations",
        lambda *_a, **_k: {
            "go_terms": [
                {
                    "name": "catalytic activity",
                    "uri": "http://purl.obolibrary.org/obo/GO_0003824",
                }
            ],
            "accession_numbers": [
                {"name": "P12345", "uri": "https://www.uniprot.org/uniprot/P12345"}
            ],
        },
    )
    out = _adapter(tmp_path).run(
        LiteratureSearchInput(query="cellulase", temporal_cutoff="2022-01-01"),
        null_context(tmp_path),
    )
    assert out.status == "ok"
    assert out.result_count == 1
    assert seen == {"published_before": "2022-01-01", "published_after": None}
    assert out.top_records[0]["title"] == "old cellulase"
    assert out.top_records[0]["is_open_access"] is False
    assert out.top_records[0]["abstract_snippet"].startswith("Active at pH 5")
    assert "GO:0003824" in out.constraint_evidence["go_terms"]
    assert "P12345" in out.constraint_evidence["uniprot_accessions"]
    assert any("pH 5" in v for v in out.constraint_evidence["ph_values"])
    assert any("60" in v for v in out.constraint_evidence["temperatures_c"])
    evidence_on_disk = json.loads(Path(out.artifacts["constraint_evidence_json"]).read_text())
    assert "GO:0003824" in evidence_on_disk["go_terms"]
    annotations_on_disk = json.loads(
        Path(out.artifacts["text_mined_annotations_json"]).read_text()
    )
    assert "go_terms" in annotations_on_disk


def test_literature_search_no_hits_emits_empty_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mod,
        "search_europepmc",
        lambda *_a, **_k: ([], [{"hitCount": 0}]),
    )

    def _unexpected(*_a, **_k):  # pragma: no cover - guard
        raise AssertionError("text-mining must not be called when there are no hits")

    monkeypatch.setattr(mod, "fetch_text_mined_annotations", _unexpected)
    out = _adapter(tmp_path).run(
        LiteratureSearchInput(
            query=(
                "Design an ADP-dependent NAD(P)H-hydrate dehydratase "
                "with NADPHX epimerase activity."
            )
        ),
        null_context(tmp_path),
    )
    assert out.status == "no_hits"
    assert out.result_count == 0
    assert out.constraint_evidence == {}
    assert out.text_mined_annotations == {}
    assert out.top_records == []


def test_literature_search_filtered_out_when_date_drops_all(tmp_path, monkeypatch):
    monkeypatch.setattr(
        mod,
        "search_europepmc",
        lambda *_a, **_k: (
            [
                {
                    "id": "1",
                    "title": "future paper",
                    "pubYear": "2030",
                    "firstPublicationDate": "2030-01-01",
                    "abstractText": "",
                    "hasTextMinedTerms": "N",
                }
            ],
            [{"hitCount": 1}],
        ),
    )
    out = _adapter(tmp_path).run(
        LiteratureSearchInput(query="cellulase", published_before="2022-01-01"),
        null_context(tmp_path),
    )
    assert out.status == "filtered_out"
    assert out.result_count == 0
    assert out.total_hits == 1
    assert out.constraint_evidence == {}


def test_normalize_query_quotes_lucene_specials_and_truncates():
    raw = (
        "ADP-dependent NAD(P)H-hydrate dehydratase, NAD(P)HX epimerase, "
        "ATP binding, AMP, NADPHX, metal ion binding, Mg2+, "
        "nicotinamide nucleotide metabolic process"
    )
    out = normalize_query(raw, max_terms=6)
    assert out.split().count("ADP-dependent") == 1
    assert '"NAD(P)H-hydrate"' in out
    assert '"NAD(P)HX"' in out
    # Comma was a separator, not part of any token.
    assert "," not in out
    # Truncated to <= 6 tokens (multi-word quoted tokens count as one).
    assert len([t for t in out.split() if not t.startswith('"') or t.endswith('"')]) <= 6


def test_normalize_query_dedupes_case_insensitive():
    out = normalize_query("cellulase Cellulase CELLULASE thermostable")
    assert out.lower().split() == ["cellulase", "thermostable"]


def test_published_before_and_temporal_cutoff_conflict_rejected():
    with pytest.raises(ValueError, match="synonyms"):
        LiteratureSearchInput(
            query="x",
            published_before="2022-01-01",
            temporal_cutoff="2024-01-01",
        )


def test_include_abstracts_coerces_lite_up_to_core():
    p = LiteratureSearchInput(query="x", result_type="lite", include_abstracts=True)
    assert p.result_type == "core"


def test_europepmc_cache_dir_honours_vpa_home(monkeypatch, tmp_path):
    monkeypatch.setenv("VPA_HOME", str(tmp_path / "vpa-root"))
    cache = mod._europepmc_cache_dir()
    assert cache == tmp_path / "vpa-root" / "cache" / "europepmc"
