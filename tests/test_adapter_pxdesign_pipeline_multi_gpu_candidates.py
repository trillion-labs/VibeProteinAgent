from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.pxdesign.adapter import PXDesignDesign, PXDesignInput, PXDesignOutput
from vibeproteinagent.adapters.pxdesign_pipeline_multi_gpu.adapter import _pxdesign_pipeline_multi_gpu_candidates
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_target", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "struct_0000", "native_id": None, "index": 0}, "structure_path": "/tmp/target.pdb", "scores": {}, "artifacts": {}}


def test_pxdesign_pipeline_multi_gpu_emits_canonical_candidates() -> None:
    params = PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60, mode="pipeline")
    design = PXDesignDesign(cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4, score=0.7)
    candidate = _pxdesign_pipeline_multi_gpu_candidates(params, [design])[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "pxdesign_pipeline_multi_gpu"
    assert "pxdesign_pipeline_multi_gpu" in candidate["scores"]


def test_pxdesign_pipeline_multi_gpu_candidate_ids_are_deterministic() -> None:
    params = PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60, mode="pipeline")
    design = PXDesignDesign(cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4)
    assert _pxdesign_pipeline_multi_gpu_candidates(params, [design])[0]["candidate_id"] == _pxdesign_pipeline_multi_gpu_candidates(params, [design])[0]["candidate_id"]


def test_pxdesign_pipeline_multi_gpu_output_preserves_legacy_fields() -> None:
    out = PXDesignOutput(designs=[PXDesignDesign(cif_path="/tmp/d.cif", sequence="ACDE", binder_length=4)], runtime_ms=1, num_samples=1)
    assert out.model_dump()["designs"][0]["sequence"] == "ACDE"


def test_pxdesign_pipeline_multi_gpu_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert PXDesignInput(candidates=[_candidate()], target_chain="A", binder_length=60, mode="pipeline").target_pdb == "/tmp/target.pdb"
    assert PXDesignInput(target_pdb="/tmp/target.pdb", target_chain="A", binder_length=60, mode="pipeline").target_pdb == "/tmp/target.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        PXDesignInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}], target_chain="A", binder_length=60, mode="pipeline")
