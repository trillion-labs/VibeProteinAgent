from __future__ import annotations

import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.sequence_analysis.adapter import (
    SequenceAnalysisAdapter,
    SequenceAnalysisInput,
)
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def test_sequence_analysis_emits_parallel_canonical_candidates(tmp_path) -> None:
    fasta = tmp_path / "family.fasta"
    fasta.write_text(">a\nMGPCK\n>b\nMKKLLVV\n")
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        SequenceAnalysisInput(fasta_path=str(fasta), max_annotations=3),
        null_context(tmp_path / "job"),
    )

    assert len(out.candidates) == len(out.representative_sequences) == 2
    for index, candidate in enumerate(out.candidates):
        representative = out.representative_sequences[index]
        validate_candidate_v1(candidate)
        assert candidate["schema_version"] == "candidate_v1"
        assert candidate["source"]["tool"] == "sequence_analysis"
        assert candidate["source"]["local_id"] == f"representative_{index:04d}"
        assert candidate["source"]["native_id"] is None
        assert candidate["source"]["index"] == index
        assert representative["index"] == index + 1
        assert candidate["sequence"] == representative["sequence"]
        scores = candidate["scores"]["sequence_analysis"]
        assert scores["gravy"] == representative["gravy"]
        assert scores["pI"] == representative["pI"]
        assert scores["net_charge"] == representative["net_charge"]
        assert scores["liability_count"] == representative["liability_count"]
        assert scores["liability_types"] == representative["liability_types"]


def test_sequence_analysis_candidate_ids_are_deterministic(tmp_path) -> None:
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    params = SequenceAnalysisInput(sequence="MKKLLVV", max_annotations=3)

    first = adapter.run(params, null_context(tmp_path / "first"))
    second = adapter.run(params, null_context(tmp_path / "second"))

    assert [row["candidate_id"] for row in first.candidates] == [
        row["candidate_id"] for row in second.candidates
    ]
