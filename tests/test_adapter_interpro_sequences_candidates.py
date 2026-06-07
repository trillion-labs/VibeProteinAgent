from __future__ import annotations

from vibeproteinagent.adapters._remote.uniprot import UniProtRecord
from vibeproteinagent.adapters.interpro_sequences.adapter import (
    InterProSequencesOutput,
    _canonical_candidate_from_record,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _record() -> UniProtRecord:
    return UniProtRecord("P12345", "ACDE", "2020-01-01", True, 4, {})


def test_interpro_sequences_emits_canonical_candidates() -> None:
    candidate = _canonical_candidate_from_record(_record(), 0)
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "interpro_sequences"
    assert candidate["source"]["local_id"] == "seq_0000"
    assert candidate["source"]["index"] == 0
    assert candidate["sequence"] == "ACDE"
    assert candidate["scores"]["interpro_sequences"]["reviewed"] is True


def test_interpro_sequences_candidate_ids_are_deterministic() -> None:
    assert _canonical_candidate_from_record(_record(), 0)["candidate_id"] == (
        _canonical_candidate_from_record(_record(), 0)["candidate_id"]
    )


def test_interpro_sequences_output_preserves_legacy_fields() -> None:
    out = InterProSequencesOutput(
        interpro_id="IPR000001",
        retrieved_accessions=1,
        kept_sequences=1,
        dropped_by_temporal_filter=0,
        fasta_path="/tmp/a.fa",
        metadata_jsonl="/tmp/a.jsonl",
        summary={"x": 1},
        artifacts={"summary_json": "/tmp/s.json"},
        runtime_ms=1,
    )
    dumped = out.model_dump()
    assert dumped["interpro_id"] == "IPR000001"
    assert dumped["candidates"] == []
