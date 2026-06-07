from __future__ import annotations

import pytest
from pydantic import ValidationError

pytest.importorskip("prody")

from vibeproteinagent.adapters.ligandmpnn.adapter import LigandMPNNDesign, LigandMPNNInput, LigandMPNNOutput, _ligandmpnn_candidates
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(
    structure_path: str = "/tmp/input.pdb",
    artifacts: dict | None = None,
) -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_struct", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": structure_path, "sequence": "AAAA", "scores": {}, "artifacts": artifacts or {}}


def test_ligandmpnn_emits_canonical_candidates() -> None:
    params = LigandMPNNInput(pdb_path="/tmp/input.pdb")
    design = LigandMPNNDesign(sequence="ACDE", score=0.4, seq_recovery=0.5)
    candidate = _ligandmpnn_candidates(params, [design], "/tmp/input.pdb")[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "ligandmpnn"
    assert candidate["sequence"] == "ACDE"
    assert candidate["scores"]["ligandmpnn"]["seq_recovery"] == 0.5


def test_ligandmpnn_candidate_ids_are_deterministic() -> None:
    params = LigandMPNNInput(pdb_path="/tmp/input.pdb")
    design = LigandMPNNDesign(sequence="ACDE", score=0.4)
    assert _ligandmpnn_candidates(params, [design], "/tmp/input.pdb")[0]["candidate_id"] == _ligandmpnn_candidates(params, [design], "/tmp/input.pdb")[0]["candidate_id"]


def test_ligandmpnn_output_preserves_legacy_fields() -> None:
    out = LigandMPNNOutput(designs=[LigandMPNNDesign(sequence="ACDE", score=0.4)], designs_fasta="/tmp/d.fa", num_residues=4, runtime_ms=1)
    assert out.model_dump()["designs_fasta"] == "/tmp/d.fa"


def test_ligandmpnn_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert LigandMPNNInput(candidates=[_candidate()]).pdb_path == "/tmp/input.pdb"
    assert LigandMPNNInput(
        candidates=[
            _candidate(
                "/tmp/input.cif",
                {"protenix_v2": {"predicted_pdb": "/tmp/input.pdb"}},
            )
        ]
    ).pdb_path == "/tmp/input.pdb"
    with pytest.raises(ValidationError, match="requires a PDB artifact"):
        LigandMPNNInput(candidates=[_candidate("/tmp/input.cif")])
    assert LigandMPNNInput(pdb_path="/tmp/input.pdb").pdb_path == "/tmp/input.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        LigandMPNNInput(candidates=[_candidate(), {"pdb_path": "/tmp/legacy.pdb"}])
