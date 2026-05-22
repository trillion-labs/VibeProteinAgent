from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.sequence_edit.adapter import (
    SequenceEditAdapter,
    SequenceEditInput,
    SequenceEditOutput,
)


def test_descriptor_fields():
    d = SequenceEditAdapter.descriptor
    assert d.name == "sequence_edit"
    assert d.category.value == "sequence_engineering"
    assert d.gpu_required is False


def test_mutate_residues_mapping_and_notation_writes_fasta(tmp_path):
    adapter = SequenceEditAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    sequence = "A" * 72 + "V" + "L" * 15
    out = adapter.run(
        SequenceEditInput(
            operation="mutate_residues",
            sequence=sequence,
            mapping={"73": "Q"},
            mutation_notation=["V73Q"],
            output_name="variant_v73q",
        ),
        null_context(tmp_path),
    )

    assert out.mutant_sequence[72] == "Q"
    assert out.mutations == ["V73Q"]
    assert out.changed_positions == [73]
    assert out.mismatch_count == 1
    assert out.mutant_fasta is not None
    fasta = Path(out.mutant_fasta)
    assert fasta.exists()
    assert fasta.read_text().startswith(">variant_v73q\n")


def test_mapping_and_notation_must_agree():
    with pytest.raises(ValidationError, match="same positions and new residues"):
        SequenceEditInput(
            operation="mutate_residues",
            sequence="AVVV",
            mapping={"2": "Q"},
            mutation_notation=["V3Q"],
        )


def test_rejects_invalid_amino_acid_and_out_of_range():
    with pytest.raises(ValidationError, match="canonical amino acid"):
        SequenceEditInput(
            operation="mutate_residues",
            sequence="AVVV",
            mapping={"2": "B"},
        )
    with pytest.raises(ValidationError, match="outside sequence length"):
        SequenceEditInput(
            operation="mutate_residues",
            sequence="AVVV",
            mapping={"5": "Q"},
        )


def test_strict_wild_type_rejects_wrong_notation():
    with pytest.raises(ValidationError, match="expects A at position 2"):
        SequenceEditInput(
            operation="mutate_residues",
            sequence="MVVV",
            mutation_notation=["A2Q"],
        )


def test_check_wt_consistency_infers_mutations_and_compares_expected(tmp_path):
    adapter = SequenceEditAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        SequenceEditInput(
            operation="check_wt_consistency",
            wild_type_sequence="MAVV",
            mutant_sequence="MQVA",
            mapping={"2": "Q", "4": "A"},
        ),
        null_context(tmp_path),
    )

    assert out.is_consistent is True
    assert out.expected_mutations_match is True
    assert out.mutations == ["A2Q", "V4A"]
    assert out.changed_positions == [2, 4]
    assert out.mutant_fasta is None


def test_check_wt_consistency_reports_expected_mismatch(tmp_path):
    adapter = SequenceEditAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        SequenceEditInput(
            operation="check_wt_consistency",
            wild_type_sequence="MAVV",
            mutant_sequence="MQVV",
            mapping={"3": "Q"},
        ),
        null_context(tmp_path),
    )

    assert out.is_consistent is False
    assert out.expected_mutations_match is False
    assert out.mutations == ["A2Q"]
    assert out.warnings


def test_sequence_edit_output_round_trip():
    out = SequenceEditOutput(
        operation="mutate_residues",
        wild_type_length=4,
        mutant_length=4,
        mutations=["A2Q"],
        changed_positions=[2],
        mutant_sequence="MQQQ",
        mutant_fasta="/tmp/mutant.fasta",
        mismatch_count=1,
        warnings=[],
        runtime_ms=1,
    )
    assert SequenceEditOutput.model_validate(out.model_dump()) == out


def test_sequence_edit_estimate_vram_zero(tmp_path):
    adapter = SequenceEditAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    params = SequenceEditInput(
        operation="check_wt_consistency",
        wild_type_sequence="MA",
        mutant_sequence="MQ",
    )
    assert adapter.estimate_vram(params) == 0
