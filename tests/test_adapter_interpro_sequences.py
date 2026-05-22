from __future__ import annotations

from pathlib import Path

import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters._remote.uniprot import UniProtRecord
from vibeproteinagent.adapters.interpro_sequences import adapter as mod
from vibeproteinagent.adapters.interpro_sequences.adapter import (
    InterProSequencesAdapter,
    InterProSequencesInput,
)


def test_interpro_sequences_mocked_temporal_filter(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def fake_interpro(*_args, **kwargs):
        seen.update(kwargs)
        return ["P00001", "P00002"], [{"raw": True}]

    monkeypatch.setattr(
        mod,
        "interpro_accessions",
        fake_interpro,
    )

    def fake_fetch(_client, accession):
        date = "2020-01-01" if accession == "P00001" else "2024-01-01"
        return UniProtRecord(
            accession,
            "MKKLL" + accession[-1],
            date,
            True,
            6,
            {"entryAudit": {"firstPublicDate": date}, "organism": {"taxonId": 9606}},
        )

    monkeypatch.setattr(mod, "fetch_accession", fake_fetch)
    adapter = InterProSequencesAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        InterProSequencesInput(
            interpro_id="IPR000001",
            max_sequences=10,
            temporal_cutoff="2022-01-01",
            taxon_id=9606,
            max_concurrency=2,
        ),
        null_context(tmp_path),
    )
    assert seen["taxon_id"] == 9606
    assert out.retrieved_accessions == 2
    assert out.kept_sequences == 1
    assert out.dropped_by_temporal_filter == 1
    assert Path(out.fasta_path).read_text().startswith(">P00001")
    assert Path(out.artifacts["uniprot_raw_records_jsonl"]).exists()
