from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.mmseqs2_gpu.adapter import (
    MMseqs2GpuInput,
    MMseqs2GpuOutput,
    _mmseqs2_gpu_candidates,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_seq", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "seq_0000", "native_id": None, "index": 0}, "sequence": "ACDE", "scores": {}, "artifacts": {}}


def test_mmseqs2_gpu_emits_canonical_candidates(tmp_path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text("q1\tt1\t77.5\t4\t0\t0\t1\t4\t1\t4\t1e-5\t42\n")
    candidates = _mmseqs2_gpu_candidates(MMseqs2GpuInput(query_fasta="/tmp/q.fa", target_db="/tmp/db"), hits)
    assert len(candidates) == 1
    validate_candidate_v1(candidates[0])
    assert candidates[0]["source"]["tool"] == "mmseqs2_gpu"
    assert candidates[0]["source"]["local_id"] == "hit_0000"
    assert candidates[0]["source"]["index"] == 0
    assert candidates[0]["scores"]["mmseqs2_gpu"]["pident"] == 77.5


def test_mmseqs2_gpu_candidate_ids_are_deterministic(tmp_path) -> None:
    hits = tmp_path / "hits.tsv"
    hits.write_text("q1\tt1\t77.5\t4\t0\t0\t1\t4\t1\t4\t1e-5\t42\n")
    params = MMseqs2GpuInput(query_fasta="/tmp/q.fa", target_db="/tmp/db")
    assert _mmseqs2_gpu_candidates(params, hits)[0]["candidate_id"] == _mmseqs2_gpu_candidates(params, hits)[0]["candidate_id"]


def test_mmseqs2_gpu_output_preserves_legacy_fields() -> None:
    out = MMseqs2GpuOutput(hits_tsv="/tmp/h.tsv", num_hits=0, runtime_ms=1)
    assert out.model_dump()["hits_tsv"] == "/tmp/h.tsv"


def test_mmseqs2_gpu_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert MMseqs2GpuInput(candidates=[_candidate()], target_db="/tmp/db").candidates[0]["candidate_id"] == "cand_seq"
    assert MMseqs2GpuInput(query_fasta="/tmp/q.fa", target_db="/tmp/db").query_fasta == "/tmp/q.fa"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        MMseqs2GpuInput(candidates=[_candidate(), {"sequence": "AAAA"}], target_db="/tmp/db")
