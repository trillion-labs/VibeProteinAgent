from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.template_select.adapter import (
    TemplateSelectAdapter,
    TemplateSelectInput,
)
from vibeproteinagent.core.binary import find_binary

_PDB = Path(__file__).resolve().parent / "fixtures" / "6MRR.pdb"


def _foldseek_available() -> bool:
    try:
        find_binary(("foldseek",), env_override="VPA_FOLDSEEK_BINARY")
    except FileNotFoundError:
        return False
    return True


def test_template_select_local_fasta_with_temporal_metadata(tmp_path):
    db = tmp_path / "db.fasta"
    db.write_text(">old_template\nMKKLLVV\n>new_template\nMKKLLVA\n>weak\nAAAAAAA\n")
    meta = tmp_path / "meta.jsonl"
    meta.write_text(
        json.dumps({
            "template_id": "old_template",
            "release_date": "2020-01-01",
            "source": "unit",
        })
        + "\n"
        + json.dumps({
            "template_id": "new_template",
            "release_date": "2024-01-01",
            "source": "unit",
        })
        + "\n"
    )
    adapter = TemplateSelectAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        TemplateSelectInput(
            query_sequence="MKKLLVV",
            template_db=str(db),
            metadata_table=str(meta),
            temporal_cutoff="2022-01-01",
            min_identity=0.5,
            max_evalue=1,
        ),
        null_context(tmp_path),
    )
    assert out.template_count == 1
    assert out.top_templates[0]["template_id"] == "old_template"
    assert Path(out.templates_json).exists()
    assert out.artifacts["template_sequences_fasta"] is not None


def test_template_select_local_fasta_handles_shifted_domain(tmp_path):
    db = tmp_path / "db.fasta"
    db.write_text(
        ">shifted\nGGGGGMKTIIALSYIFCLVFADYKDDDD\n"
        ">prefix_only\nMKTAAAAAA\n"
    )
    adapter = TemplateSelectAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        TemplateSelectInput(
            query_sequence="MKTIIALSYIFCLVFA",
            template_db=str(db),
            min_identity=0.8,
            min_coverage=0.8,
            max_evalue=1.0,
        ),
        null_context(tmp_path),
    )
    assert out.template_count == 1
    assert out.top_templates[0]["template_id"] == "shifted"
    assert out.top_templates[0]["coverage"] >= 0.8


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("VPA_LIVE_TEMPLATE_TESTS") != "1",
    reason="set VPA_LIVE_TEMPLATE_TESTS=1",
)
def test_template_select_live_mmseqs_smoke(tmp_path):
    db = tmp_path / "db.fasta"
    db.write_text(
        ">seq_a\nMKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFKKKR\n"
        ">seq_b\nMKKILLIASLLLAVLAYAQGGEYTRTRTKHTGDFEYNFGT\n"
    )
    adapter = TemplateSelectAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        TemplateSelectInput(
            query_sequence="MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFKKKR",
            template_db=str(db),
            template_db_kind="mmseqs",
            min_identity=0.1,
            min_coverage=0.5,
            max_evalue=1.0,
            max_templates=5,
        ),
        null_context(tmp_path),
    )
    assert out.engine == "mmseqs2_gpu"
    assert out.template_count >= 1
    assert Path(out.artifacts["template_sequences_fasta"]).read_text().strip()


@pytest.mark.slow
@pytest.mark.skipif(
    not _foldseek_available(),
    reason="foldseek binary unavailable",
)
def test_template_select_foldseek_self_search(tmp_path):
    adapter = TemplateSelectAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        TemplateSelectInput(
            query_structure=str(_PDB),
            template_db=str(_PDB),
            template_db_kind="foldseek",
            min_identity=0.1,
            min_coverage=0.5,
            max_evalue=1.0,
            max_templates=5,
        ),
        null_context(tmp_path),
    )
    assert out.engine == "foldseek"
    assert out.template_count >= 1
