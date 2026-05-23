from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.sequence_analysis.adapter import (
    SequenceAnalysisAdapter,
    SequenceAnalysisInput,
    SequenceAnalysisOutput,
)


def test_descriptor_fields():
    d = SequenceAnalysisAdapter.descriptor
    assert d.name == "sequence_analysis"
    assert d.category.value == "analysis"
    assert d.gpu_required is False


def test_input_requires_sequence_or_fasta():
    with pytest.raises(ValidationError):
        SequenceAnalysisInput()
    SequenceAnalysisInput(sequence="MKKL")
    SequenceAnalysisInput(fasta_path="/tmp/example.fasta")


def test_sequence_metrics_liabilities_and_artifacts(tmp_path):
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    sequence = "MKKKLLLLLLLLACDEFGHIKLMNPQRSTVWY"
    out = adapter.run(
        SequenceAnalysisInput(sequence=sequence, max_annotations=5),
        null_context(tmp_path),
    )

    assert out.sequence_length == len(sequence)
    assert out.metrics["length"] == len(sequence)
    assert out.representative_sequences[0]["sequence"] == sequence
    assert out.metrics["amino_acid_counts"]["K"] == 4
    assert "gravy" in out.metrics
    assert "pI" in out.metrics
    assert len(out.top_annotations) == 5
    liability_types = {liability.type for liability in out.liabilities}
    assert "polybasic_patch" in liability_types
    assert "hydrophobic_run" in liability_types

    metrics_path = Path(out.artifacts["metrics_json"])
    annotations_path = Path(out.artifacts["residue_annotations_tsv"])
    summary_path = Path(out.artifacts["summary_json"])
    assert metrics_path.exists()
    assert annotations_path.exists()
    assert summary_path.exists()
    assert json.loads(metrics_path.read_text())["length"] == len(sequence)
    assert "position\tresidue\tbiochemical_class" in annotations_path.read_text()


def test_fasta_input_and_selected_positions(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">example\nMGPCK\n")
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        SequenceAnalysisInput(fasta_path=str(fasta), positions=[2, 4], max_annotations=10),
        null_context(tmp_path),
    )

    assert out.sequence_length == 5
    assert out.metrics["fasta_header"] == "example"
    assert [annotation.position for annotation in out.top_annotations] == [2, 4]
    annotations_path = Path(out.artifacts["residue_annotations_tsv"])
    rows = annotations_path.read_text().splitlines()
    assert len(rows) == 3
    assert rows[1].startswith("2\tG\t")
    assert rows[2].startswith("4\tC\t")


def test_multi_record_fasta_reports_family_summary(tmp_path):
    fasta = tmp_path / "family.fasta"
    fasta.write_text(">a\nMGPCK\n>b\nMKKLLVV\n")
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        SequenceAnalysisInput(fasta_path=str(fasta), max_annotations=3),
        null_context(tmp_path),
    )

    assert out.sequence_length == 12
    assert out.metrics["analysis_mode"] == "multi_fasta"
    assert out.metrics["family_summary"]["sequence_count"] == 2
    assert out.representative_sequences[0]["header"] == "a"
    assert out.representative_sequences[0]["sequence"] == "MGPCK"
    per_sequence = out.artifacts["per_sequence_metrics_jsonl"]
    assert per_sequence is not None
    assert Path(per_sequence).read_text().count("\n") == 2


def test_noncanonical_characters_are_reported_not_rejected(tmp_path):
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        SequenceAnalysisInput(sequence="MXXK", max_annotations=4),
        null_context(tmp_path),
    )

    assert out.metrics["noncanonical_residue_count"] == 2
    assert out.metrics["unusual_characters"] == ["X"]
    assert any(liability.type == "unusual_characters" for liability in out.liabilities)
    assert any("noncanonical_residue" in a.caution_flags for a in out.top_annotations)


def test_out_of_range_positions_rejected_at_run(tmp_path):
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="outside sequence length"):
        adapter.run(
            SequenceAnalysisInput(sequence="MGP", positions=[4]),
            null_context(tmp_path),
        )


def test_can_skip_residue_table_artifact(tmp_path):
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        SequenceAnalysisInput(sequence="MGP", include_residue_table=False),
        null_context(tmp_path),
    )

    assert out.artifacts["residue_annotations_tsv"] is None
    assert Path(out.artifacts["metrics_json"]).exists()
    assert Path(out.artifacts["summary_json"]).exists()


def test_sequence_analysis_output_round_trip():
    out = SequenceAnalysisOutput(
        sequence_length=1,
        metrics={"length": 1},
        representative_sequences=[
            {"index": 1, "sequence": "M", "length": 1},
        ],
        liabilities=[],
        top_annotations=[],
        artifacts={
            "metrics_json": "/tmp/metrics.json",
            "residue_annotations_tsv": None,
            "summary_json": "/tmp/summary.json",
        },
        runtime_ms=1,
    )
    assert SequenceAnalysisOutput.model_validate(out.model_dump()) == out


def test_sequence_analysis_estimate_vram_zero(tmp_path):
    adapter = SequenceAnalysisAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    assert adapter.estimate_vram(SequenceAnalysisInput(sequence="MGP")) == 0
