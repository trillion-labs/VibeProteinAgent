from __future__ import annotations

from pathlib import Path

import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters._remote.uniprot import UniProtRecord
from vibeproteinagent.adapters.go_sequences import adapter as mod
from vibeproteinagent.adapters.go_sequences.adapter import GOSequencesAdapter, GOSequencesInput


def test_go_sequences_mocked_go_first_then_uniprot_filter(tmp_path, monkeypatch):
    seen: dict[str, object] = {}
    associations = [
        {"bioentity": "UniProtKB:P00001", "evidence": {"type": "EXP"}},
        {"bioentity": "MGI:Gene1", "evidence": {"type": "EXP"}},
    ]

    def fake_go_associations(*_args, **kwargs):
        seen.update(kwargs)
        return associations, {"label": "test function"}

    monkeypatch.setattr(
        mod,
        "go_associations",
        fake_go_associations,
    )
    fallback_seen: dict[str, object] = {}

    def fake_search(_client, query, **kwargs):
        fallback_seen["query"] = query
        fallback_seen.update(kwargs)
        return ["P00002"]

    monkeypatch.setattr(mod, "search_accessions", fake_search)

    def fake_fetch(_client, accession):
        date = "2020-01-01" if accession == "P00001" else "2024-01-01"
        return UniProtRecord(
            accession,
            "M" + accession[-1] * 5,
            date,
            True,
            6,
            {"organism": {"taxonId": 9606}},
        )

    monkeypatch.setattr(mod, "fetch_accession", fake_fetch)
    adapter = GOSequencesAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        GOSequencesInput(
            go_id="GO:0003824",
            aspect="function",
            max_sequences=10,
            temporal_cutoff="2022-01-01",
            taxon_id=9606,
            max_concurrency=2,
        ),
        null_context(tmp_path),
    )
    assert seen["aspect"] == "function"
    assert seen["taxon_id"] == 9606
    assert out.go_label == "test function"
    assert out.kept_sequences == 1
    assert out.sequences == [
        {
            "candidate_id": "go:GO:0003824:P00001",
            "accession": "P00001",
            "sequence": "M11111",
            "length": 6,
            "reviewed": True,
        }
    ]
    assert out.dropped["temporal_filter"] == 1
    assert out.dropped["unresolved_identifier"] == 1
    assert fallback_seen["query"].startswith("go:0003824")
    assert out.resolved_uniprot_accessions == 2
    assert Path(out.fasta_path).exists()
    assert Path(out.artifacts["uniprot_raw_records_jsonl"]).exists()
