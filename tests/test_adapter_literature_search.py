from __future__ import annotations

import json
from pathlib import Path

import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.literature_search import adapter as mod
from vibeproteinagent.adapters.literature_search.adapter import (
    LiteratureSearchAdapter,
    LiteratureSearchInput,
)


def test_literature_search_mocked_europepmc_temporal_filter(tmp_path, monkeypatch):
    seen: dict[str, str | None] = {}

    def fake_search(*_args, **_kwargs):
        seen["published_before"] = _kwargs.get("published_before")
        seen["published_after"] = _kwargs.get("published_after")
        return [
            {
                "title": "old cellulase",
                "pubYear": "2020",
                "doi": "10.1/old",
                "pmid": "1",
                "firstPublicationDate": "2020-05-14",
                "abstractText": "Active at pH 5 and 60 C. IPR000001 GO:0003824.",
            },
            {
                "title": "new cellulase",
                "pubYear": "2024",
                "doi": "10.1/new",
                "pmid": "2",
                "firstPublicationDate": "2024-01-01",
                "abstractText": "too new",
            },
        ], [{"page": 1}]

    monkeypatch.setattr(mod, "search_europepmc", fake_search)
    adapter = LiteratureSearchAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        LiteratureSearchInput(query="cellulase", temporal_cutoff="2022-01-01"),
        null_context(tmp_path),
    )
    assert out.result_count == 1
    assert seen == {"published_before": "2022-01-01", "published_after": None}
    assert out.top_records[0]["title"] == "old cellulase"
    assert "GO:0003824" in out.constraint_evidence["go_terms"]
    evidence = json.loads(Path(out.artifacts["constraint_evidence_json"]).read_text())
    assert "IPR000001" in evidence["interpro_ids"]


def test_literature_search_extracts_known_go_terms_from_query(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "search_europepmc", lambda *_args, **_kwargs: ([], []))
    adapter = LiteratureSearchAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        LiteratureSearchInput(
            query=(
                "Design an ADP-dependent NAD(P)H-hydrate dehydratase "
                "with NADPHX epimerase activity."
            )
        ),
        null_context(tmp_path),
    )

    assert "GO:0052855" in out.constraint_evidence["go_terms"]
    assert "GO:0052856" in out.constraint_evidence["go_terms"]
