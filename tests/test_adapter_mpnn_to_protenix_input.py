from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import ToolRegistry
from vibeproteinagent.adapters.context import null_context
from vibeproteinagent.adapters.mpnn_to_protenix_input.adapter import (
    Mpnn2ProtenixAdapter,
    Mpnn2ProtenixInput,
    _parse_mpnn_fasta,
)

_SAMPLE_FASTA = """>foo|input, num_residues=8
MKAILVAA
>foo|design=1, T=0.1, score=1.2345, seq_recovery=0.50
MKAILVAB
>foo|design=2, T=0.1, score=0.4567, seq_recovery=0.62
MKAILVCC
>foo|design=3, T=0.1, score=2.0001, seq_recovery=0.45
MKAILVDD
"""


def _make_fasta(tmp_path: Path) -> Path:
    p = tmp_path / "designs.fa"
    p.write_text(_SAMPLE_FASTA)
    return p


def test_parse_fasta_skips_input_and_extracts_scores(tmp_path: Path) -> None:
    p = _make_fasta(tmp_path)
    records = _parse_mpnn_fasta(p)
    assert len(records) == 3
    seqs = [s for s, _ in records]
    assert "MKAILVAA" not in seqs  # the |input record is skipped
    assert seqs == ["MKAILVAB", "MKAILVCC", "MKAILVDD"]
    scores = [score for _, score in records]
    assert scores == pytest.approx([1.2345, 0.4567, 2.0001])


def test_adapter_emits_top1_json(tmp_path: Path) -> None:
    fasta = _make_fasta(tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    adapter = Mpnn2ProtenixAdapter({}, tmp_path / "weights", torch.device("cpu"))
    params = Mpnn2ProtenixInput(
        designs_fasta=str(fasta), top_k=1, name_prefix="refold"
    )
    out = adapter.run(params, null_context(job_dir))

    assert out.count == 1
    assert out.names == ["refold_1"]
    assert len(out.input_jsons) == 1
    payload_path = Path(out.input_jsons[0])
    assert payload_path.is_file()
    payload = json.loads(payload_path.read_text())
    # Top score (lowest) is 0.4567 → MKAILVCC.
    assert payload == [
        {
            "name": "refold_1",
            "sequences": [
                {"proteinChain": {"sequence": "MKAILVCC", "count": 1}}
            ],
            "modelSeeds": [101],
        }
    ]


def test_adapter_emits_top_k_multiple(tmp_path: Path) -> None:
    fasta = _make_fasta(tmp_path)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    adapter = Mpnn2ProtenixAdapter({}, tmp_path / "weights", torch.device("cpu"))
    params = Mpnn2ProtenixInput(
        designs_fasta=str(fasta), top_k=3, name_prefix="d"
    )
    out = adapter.run(params, null_context(job_dir))

    assert out.count == 3
    # Ordered low→high score.
    expected_seqs = ["MKAILVCC", "MKAILVAB", "MKAILVDD"]
    payloads = [json.loads(Path(p).read_text()) for p in out.input_jsons]
    # Each payload is a 1-element list of Protenix-v1 entries.
    got_seqs = [
        p[0]["sequences"][0]["proteinChain"]["sequence"] for p in payloads
    ]
    assert got_seqs == expected_seqs
    assert out.names == ["d_1", "d_2", "d_3"]


def test_adapter_missing_fasta_raises(tmp_path: Path) -> None:
    adapter = Mpnn2ProtenixAdapter({}, tmp_path / "weights", torch.device("cpu"))
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    params = Mpnn2ProtenixInput(
        designs_fasta=str(tmp_path / "missing.fa"), top_k=1
    )
    with pytest.raises(FileNotFoundError):
        adapter.run(params, null_context(job_dir))


def test_adapter_empty_fasta_raises(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fa"
    fasta.write_text(">foo|input, num_residues=4\nMKAA\n")
    adapter = Mpnn2ProtenixAdapter({}, tmp_path / "weights", torch.device("cpu"))
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    params = Mpnn2ProtenixInput(designs_fasta=str(fasta), top_k=1)
    with pytest.raises(ValueError, match="no design records"):
        adapter.run(params, null_context(job_dir))


def test_registry_picks_up_glue_adapter() -> None:
    registry = ToolRegistry()
    registry.discover()
    descriptor, cls = registry.get("mpnn_to_protenix_input", ">=1.0.0,<2")
    assert descriptor.name == "mpnn_to_protenix_input"
    assert descriptor.gpu_required is False
    assert cls is Mpnn2ProtenixAdapter
