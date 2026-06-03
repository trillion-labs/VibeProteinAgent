from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.protenix_v2.adapter import (
    ProtenixV2Input,
    ProtenixV2Metrics,
    ProtenixV2Output,
    ProtenixV2PredictionRecord,
    _protenix_v2_candidates,
    _write_protenix_input_json_from_candidates,
)
from vibeproteinagent.agents.planning.candidate import validate_candidate_v1


def _candidate(sequence: str = "ACDE") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_protenix_input",
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": "seq_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": sequence,
        "scores": {},
        "artifacts": {},
    }


def _pxdesign_candidate(sequence: str, target_pdb: str, index: int) -> dict:
    candidate = _candidate(sequence)
    candidate["candidate_id"] = f"cand_pxdesign_{index}"
    candidate["source"] = {
        "tool": "pxdesign",
        "node_id": None,
        "job_id": None,
        "local_id": f"design_{index:04d}",
        "native_id": None,
        "index": index,
    }
    candidate["scores"] = {"pxdesign": {"score": 0.7 - index * 0.1}}
    candidate["artifacts"] = {
        "pxdesign": {
            "target_pdb": target_pdb,
            "target_chain": "A",
        }
    }
    return candidate


def _target_pdb(tmp_path) -> str:
    path = tmp_path / "target.pdb"
    path.write_text(
        "\n".join(
            [
                "ATOM      1  N   ALA A   1      11.104  13.207   9.000  1.00 20.00           N",
                "ATOM      2  CA  ALA A   1      12.000  13.500   9.000  1.00 20.00           C",
                "ATOM      3  N   GLY A   2      13.104  14.207   9.000  1.00 20.00           N",
                "ATOM      4  CA  GLY A   2      14.000  14.500   9.000  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n"
    )
    return str(path)


def _record(index: int = 0) -> ProtenixV2PredictionRecord:
    return ProtenixV2PredictionRecord(
        candidate_id=f"protenix_v2:sample:seed_101:rank_{index}",
        seed=101,
        rank=index,
        cif_path=f"/tmp/protenix_{index}.cif",
        summary_json=f"/tmp/protenix_{index}.json",
        pae_path=f"/tmp/protenix_{index}_pae.json",
        full_confidence_json=f"/tmp/protenix_{index}_full.json",
        ranking_score=0.9 - index * 0.1,
        plddt=80.0 - index,
        iptm=0.6,
        ptm=0.7,
    )


def _pxdesign_record(index: int = 0) -> ProtenixV2PredictionRecord:
    return ProtenixV2PredictionRecord(
        candidate_id=f"protenix_v2:canonical_candidate_{index:04d}:seed_101:rank_0",
        seed=101,
        rank=0,
        cif_path=f"/tmp/protenix_{index}.cif",
        summary_json=f"/tmp/protenix_{index}.json",
        pae_path=f"/tmp/protenix_{index}_pae.json",
        full_confidence_json=f"/tmp/protenix_{index}_full.json",
        ranking_score=0.9 - index * 0.1,
        plddt=80.0 - index,
        iptm=0.6,
        ptm=0.7,
    )


def test_protenix_v2_emits_canonical_candidates() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL")])
    records = [_record(0), _record(1)]
    candidates = _protenix_v2_candidates(params, records)

    assert len(candidates) == len(records)
    validate_candidate_v1(candidates[0])
    assert candidates[0]["sequence"] == "MKKL"
    assert candidates[0]["structure_path"] == "/tmp/protenix_0.cif"
    assert candidates[0]["scores"]["protenix_v2"]["ranking_score"] == 0.9


def test_protenix_v2_candidate_ids_are_deterministic() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL")])
    records = [_record(0)]
    assert _protenix_v2_candidates(params, records)[0]["candidate_id"] == _protenix_v2_candidates(params, records)[0]["candidate_id"]


def test_protenix_v2_candidate_source_fields() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL")])
    source = _protenix_v2_candidates(params, [_record(0)])[0]["source"]
    assert source["tool"] == "protenix_v2"
    assert source["local_id"] == "protenix_0000"
    assert source["index"] == 0


def test_protenix_v2_output_preserves_legacy_fields() -> None:
    out = ProtenixV2Output(
        predicted_cif="/tmp/protenix_0.cif",
        predicted_pdb=None,
        pae_path="/tmp/protenix_0_pae.json",
        metrics=ProtenixV2Metrics(
            plddt=80.0,
            iptm=0.6,
            ptm=0.7,
            ranking_score=0.9,
            summary_json="/tmp/protenix_0.json",
        ),
        all_predictions=[_record(0)],
        runtime_ms=1,
        seeds_used=[101],
    )
    dumped = out.model_dump()
    assert dumped["predicted_cif"] == "/tmp/protenix_0.cif"
    assert dumped["pae_path"] == "/tmp/protenix_0_pae.json"
    assert dumped["metrics"]["ranking_score"] == 0.9
    assert dumped["all_predictions"][0]["cif_path"] == "/tmp/protenix_0.cif"


def test_protenix_v2_consumes_canonical_candidates(tmp_path) -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL")], seeds=[101])
    assert params.input_json is None

    path = _write_protenix_input_json_from_candidates(
        params.candidates,
        params.seeds,
        tmp_path / "canonical_candidates.json",
    )
    payload = json.loads(path.read_text())
    assert payload[0]["sequences"][0]["proteinChain"]["sequence"] == "MKKL"
    assert payload[0]["modelSeeds"] == [101]


def test_protenix_v2_writes_pxdesign_target_binder_complex_entries(tmp_path) -> None:
    target = _target_pdb(tmp_path)
    params = ProtenixV2Input(
        candidates=[
            _pxdesign_candidate("MKKL", target, 0),
            _pxdesign_candidate("ACDE", target, 1),
        ],
        seeds=[101],
    )

    path = _write_protenix_input_json_from_candidates(
        params.candidates,
        params.seeds,
        tmp_path / "canonical_candidates.json",
    )

    payload = json.loads(path.read_text())
    assert [item["name"] for item in payload] == [
        "canonical_candidate_0000",
        "canonical_candidate_0001",
    ]
    assert payload[0]["sequences"][0]["proteinChain"]["sequence"] == "AG"
    assert payload[0]["sequences"][1]["proteinChain"]["sequence"] == "MKKL"
    assert payload[1]["sequences"][0]["proteinChain"]["sequence"] == "AG"
    assert payload[1]["sequences"][1]["proteinChain"]["sequence"] == "ACDE"


def test_protenix_v2_rejects_mixed_pxdesign_target_context_candidates(tmp_path) -> None:
    target = _target_pdb(tmp_path)
    params = ProtenixV2Input(
        candidates=[
            _pxdesign_candidate("MKKL", target, 0),
            _candidate("ACDE"),
        ],
        seeds=[101],
    )

    with pytest.raises(ValueError, match="cannot mix PXDesign target-context"):
        _write_protenix_input_json_from_candidates(
            params.candidates,
            params.seeds,
            tmp_path / "canonical_candidates.json",
        )


def test_protenix_v2_candidates_preserve_pxdesign_sequence_mapping(tmp_path) -> None:
    target = _target_pdb(tmp_path)
    params = ProtenixV2Input(
        candidates=[
            _pxdesign_candidate("MKKL", target, 0),
            _pxdesign_candidate("ACDE", target, 1),
        ],
        seeds=[101],
    )

    candidates = _protenix_v2_candidates(
        params,
        [_pxdesign_record(0), _pxdesign_record(1)],
    )

    assert candidates[0]["sequence"] == "MKKL"
    assert candidates[1]["sequence"] == "ACDE"
    assert candidates[0]["scores"]["pxdesign"]["score"] == 0.7
    assert candidates[1]["scores"]["pxdesign"]["score"] == pytest.approx(0.6)
    assert candidates[0]["artifacts"]["pxdesign"]["target_pdb"] == target
    assert (
        candidates[0]["artifacts"]["protenix_v2"]["complex_input_mode"]
        == "pxdesign_target_sequence_binder_sequence"
    )


def test_protenix_v2_consumes_legacy_input_json() -> None:
    assert ProtenixV2Input(input_json="/tmp/input.json").input_json == "/tmp/input.json"


def test_protenix_v2_rejects_mixed_candidates() -> None:
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ProtenixV2Input(candidates=[_candidate("MKKL"), {"sequence": "ACDE"}])
