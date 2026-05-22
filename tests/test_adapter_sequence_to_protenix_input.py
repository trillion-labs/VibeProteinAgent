from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.sequence_to_protenix_input.adapter import (
    SequenceToProtenixInput,
    SequenceToProtenixInputAdapter,
    SequenceToProtenixOutput,
)


def test_sequence_to_protenix_input_from_sequence(tmp_path):
    adapter = SequenceToProtenixInputAdapter(
        config={},
        weights_dir=tmp_path,
        device=torch.device("cpu"),
    )
    out = adapter.run(
        SequenceToProtenixInput(sequence="MGPCK", name="repaired", seeds=[101, 202]),
        null_context(tmp_path),
    )

    assert out.count == 1
    payload = json.loads(Path(out.input_json).read_text())
    assert payload[0]["name"] == "repaired"
    assert payload[0]["modelSeeds"] == [101, 202]
    assert payload[0]["sequences"][0]["proteinChain"]["sequence"] == "MGPCK"


def test_sequence_to_protenix_input_from_multifasta(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(">a\nMGP\n>b\nMKK\n")
    adapter = SequenceToProtenixInputAdapter(
        config={},
        weights_dir=tmp_path,
        device=torch.device("cpu"),
    )
    out = adapter.run(
        SequenceToProtenixInput(fasta_path=str(fasta), name="family"),
        null_context(tmp_path),
    )

    assert out.count == 2
    assert len(out.input_jsons) == 2


def test_sequence_to_protenix_rejects_invalid_sequence(tmp_path):
    with pytest.raises(ValidationError):
        SequenceToProtenixInput()
    adapter = SequenceToProtenixInputAdapter(
        config={},
        weights_dir=tmp_path,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="non-canonical"):
        adapter.run(
            SequenceToProtenixInput(sequence="MXX"),
            null_context(tmp_path),
        )


def test_sequence_to_protenix_output_round_trip():
    out = SequenceToProtenixOutput(
        input_json="/tmp/a.json",
        input_jsons=["/tmp/a.json"],
        names=["a"],
        count=1,
        runtime_ms=1,
    )
    assert SequenceToProtenixOutput.model_validate(out.model_dump()) == out
