from __future__ import annotations

import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters._remote.uniprot import UniProtRecord
from vibeproteinagent.adapters.go_sequences import adapter as mod
from vibeproteinagent.adapters.go_sequences.adapter import GOSequencesAdapter, GOSequencesInput
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _patch_remote(monkeypatch) -> None:
    associations = [
        {"bioentity": "UniProtKB:P00001", "evidence": {"type": "EXP"}},
        {"bioentity": "UniProtKB:P00002", "evidence": {"type": "EXP"}},
    ]
    monkeypatch.setattr(
        mod,
        "go_associations",
        lambda *_args, **_kwargs: (associations, {"label": "test function"}),
    )

    def fake_fetch(_client, accession):
        return UniProtRecord(
            accession,
            "M" + accession[-1] * 5,
            "2020-01-01",
            accession == "P00001",
            6,
            {"organism": {"taxonId": 9606}},
        )

    monkeypatch.setattr(mod, "fetch_accession", fake_fetch)


def _run_adapter(tmp_path, monkeypatch):
    _patch_remote(monkeypatch)
    adapter = GOSequencesAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    return adapter.run(
        GOSequencesInput(
            go_id="GO:0003824",
            max_sequences=2,
            max_concurrency=1,
        ),
        null_context(tmp_path / "job"),
    )


def test_go_sequences_emits_parallel_canonical_candidates(tmp_path, monkeypatch) -> None:
    out = _run_adapter(tmp_path, monkeypatch)

    assert len(out.candidates) == len(out.sequences) == 2
    assert out.sequences == [
        {
            "candidate_id": "go:GO:0003824:P00001",
            "accession": "P00001",
            "sequence": "M11111",
            "length": 6,
            "reviewed": True,
        },
        {
            "candidate_id": "go:GO:0003824:P00002",
            "accession": "P00002",
            "sequence": "M22222",
            "length": 6,
            "reviewed": False,
        },
    ]
    for index, candidate in enumerate(out.candidates):
        sequence = out.sequences[index]
        validate_candidate_v1(candidate)
        assert candidate["schema_version"] == "candidate_v1"
        assert candidate["source"]["tool"] == "go_sequences"
        assert candidate["source"]["local_id"] == f"seq_{index:04d}"
        assert candidate["source"]["native_id"] == sequence["accession"]
        assert candidate["source"]["index"] == index
        assert candidate["source"]["node_id"] is None
        assert candidate["source"]["job_id"] is None
        assert candidate["sequence"] == sequence["sequence"]
        assert candidate["scores"]["go_sequences"]["length"] == sequence["length"]
        assert candidate["scores"]["go_sequences"]["reviewed"] == sequence["reviewed"]


def test_go_sequences_candidate_ids_are_deterministic(tmp_path, monkeypatch) -> None:
    first = _run_adapter(tmp_path / "first", monkeypatch)
    second = _run_adapter(tmp_path / "second", monkeypatch)

    assert [row["candidate_id"] for row in first.candidates] == [
        row["candidate_id"] for row in second.candidates
    ]


def test_go_sequences_drops_empty_sequence_records(tmp_path, monkeypatch) -> None:
    _patch_remote(monkeypatch)

    def fake_fetch(_client, accession):
        if accession == "P00001":
            return UniProtRecord(
                accession,
                "",
                "2020-01-01",
                True,
                0,
                {"organism": {"taxonId": 9606}},
            )
        return UniProtRecord(
            accession,
            "MGOOD",
            "2020-01-01",
            False,
            5,
            {"organism": {"taxonId": 9606}},
        )

    monkeypatch.setattr(mod, "fetch_accession", fake_fetch)
    adapter = GOSequencesAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        GOSequencesInput(go_id="GO:0003824", max_sequences=2, max_concurrency=1),
        null_context(tmp_path / "job"),
    )

    assert out.raw_result_count == 2
    assert out.kept_sequence_count == out.kept_sequences == 1
    assert out.dropped_empty_sequence_count == 1
    assert out.dropped["empty_sequence"] == 1
    assert out.sequences[0]["sequence"] == "MGOOD"
    assert out.candidates[0]["sequence"] == "MGOOD"
