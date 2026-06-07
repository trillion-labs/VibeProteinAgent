from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.pxmeter.adapter import PXMeterInput, PXMeterOutput, _pxmeter_candidates
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(candidate_id: str, path: str, index: int) -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": candidate_id, "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": f"struct_{index:04d}", "native_id": None, "index": index}, "structure_path": path, "scores": {}, "artifacts": {}}


def test_pxmeter_emits_canonical_candidates() -> None:
    params = PXMeterInput(predicted_cif="/tmp/pred.cif", reference_cif="/tmp/ref.cif")
    candidate = _pxmeter_candidates(params, "/tmp/pred.cif", {"lddt": 0.9}, "/tmp/report.json")[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "pxmeter"
    assert candidate["scores"]["pxmeter"]["lddt"] == 0.9


def test_pxmeter_candidate_ids_are_deterministic() -> None:
    params = PXMeterInput(predicted_cif="/tmp/pred.cif", reference_cif="/tmp/ref.cif")
    assert _pxmeter_candidates(params, "/tmp/pred.cif", {}, "/tmp/report.json")[0]["candidate_id"] == _pxmeter_candidates(params, "/tmp/pred.cif", {}, "/tmp/report.json")[0]["candidate_id"]


def test_pxmeter_output_preserves_legacy_fields() -> None:
    out = PXMeterOutput(metrics={"lddt": 0.9}, report_json_path="/tmp/report.json", runtime_ms=1)
    assert out.model_dump()["metrics"] == {"lddt": 0.9}


def test_pxmeter_consumes_canonical_legacy_and_rejects_mixed() -> None:
    params = PXMeterInput(candidates=[_candidate("ref", "/tmp/ref.cif", 0), _candidate("pred", "/tmp/pred.cif", 1)])
    assert params.reference_cif == "/tmp/ref.cif"
    assert params.predicted_cif == "/tmp/pred.cif"
    assert PXMeterInput(predicted_cif="/tmp/pred.cif", reference_cif="/tmp/ref.cif").predicted_cif == "/tmp/pred.cif"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        PXMeterInput(candidates=[_candidate("ref", "/tmp/ref.cif", 0), {"predicted_cif": "/tmp/pred.cif"}])
