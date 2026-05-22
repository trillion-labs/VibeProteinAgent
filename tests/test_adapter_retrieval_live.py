from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.go_sequences.adapter import GOSequencesAdapter, GOSequencesInput
from vibeproteinagent.adapters.interpro_sequences.adapter import (
    InterProSequencesAdapter,
    InterProSequencesInput,
)
from vibeproteinagent.adapters.literature_search.adapter import (
    LiteratureSearchAdapter,
    LiteratureSearchInput,
)

pytestmark = pytest.mark.slow


def _live_enabled() -> bool:
    return os.environ.get("VPA_LIVE_RETRIEVAL_TESTS") == "1"


@pytest.mark.skipif(not _live_enabled(), reason="set VPA_LIVE_RETRIEVAL_TESTS=1")
def test_literature_search_live_europepmc_smoke(tmp_path):
    adapter = LiteratureSearchAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        LiteratureSearchInput(
            query="thermostable cellulase",
            max_results=2,
            temporal_cutoff="2022-01-01",
            timeout_s=30,
        ),
        null_context(tmp_path),
    )
    assert out.backend == "europepmc"
    assert Path(out.artifacts["records_json"]).exists()


@pytest.mark.skipif(not _live_enabled(), reason="set VPA_LIVE_RETRIEVAL_TESTS=1")
def test_interpro_sequences_live_smoke(tmp_path):
    adapter = InterProSequencesAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        InterProSequencesInput(
            interpro_id="IPR000001",
            max_sequences=1,
            page_size=5,
            timeout_s=30,
        ),
        null_context(tmp_path),
    )
    assert out.retrieved_accessions >= out.kept_sequences
    assert Path(out.fasta_path).exists()


@pytest.mark.skipif(not _live_enabled(), reason="set VPA_LIVE_RETRIEVAL_TESTS=1")
def test_go_sequences_live_smoke(tmp_path):
    adapter = GOSequencesAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        GOSequencesInput(
            go_id="GO:0003824",
            max_sequences=1,
            page_size=5,
            timeout_s=30,
        ),
        null_context(tmp_path),
    )
    assert out.association_records >= 0
    assert Path(out.fasta_path).exists()
