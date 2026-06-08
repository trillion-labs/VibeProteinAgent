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
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate(sequence: str = "ACDE", candidate_id: str = "cand_protenix_input") -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": "seq_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": sequence,
        "scores": {"esm2": {"score": 0.42}},
        "artifacts": {"esm2": {"fasta": "/tmp/esm2.fa"}},
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


def _bindcraft_candidate(sequence: str, target_pdb: str, index: int = 0) -> dict:
    candidate = _candidate(sequence, candidate_id=f"cand_bindcraft_{index}")
    candidate["source"] = {
        "tool": "bindcraft",
        "node_id": None,
        "job_id": None,
        "local_id": f"design_{index:04d}",
        "native_id": None,
        "index": index,
    }
    candidate["scores"] = {"bindcraft": {"iptm": 0.78, "plddt": 0.85}}
    candidate["artifacts"] = {
        "bindcraft": {
            "target_pdb": target_pdb,
            "target_chain": "A",
            "complex_chains": [
                {
                    "role": "target",
                    "chain_id": "A",
                    "input_order": 0,
                    "sequence_length": 2,
                },
                {
                    "role": "binder",
                    "chain_id": "B",
                    "input_order": 1,
                    "sequence_length": len(sequence),
                },
            ],
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


def _target_input_json(tmp_path) -> str:
    path = tmp_path / "target_with_msa.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "agarase_target",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": "AG",
                                "count": 1,
                                "pairedMsaPath": "/tmp/pairing.a3m",
                                "unpairedMsaPath": "/tmp/non_pairing.a3m",
                            }
                        }
                    ],
                    "modelSeeds": [7],
                }
            ]
        )
    )
    return str(path)


def _record(index: int = 0) -> ProtenixV2PredictionRecord:
    return ProtenixV2PredictionRecord(
        candidate_id=f"protenix_v2:sample:seed_101:rank_{index}",
        seed=101,
        rank=index,
        cif_path=f"/tmp/protenix_{index}.cif",
        pdb_path=f"/tmp/protenix_{index}.pdb",
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
        pdb_path=f"/tmp/protenix_{index}.pdb",
        summary_json=f"/tmp/protenix_{index}.json",
        pae_path=f"/tmp/protenix_{index}_pae.json",
        full_confidence_json=f"/tmp/protenix_{index}_full.json",
        ranking_score=0.9 - index * 0.1,
        plddt=80.0 - index,
        iptm=0.6,
        ptm=0.7,
    )


def _canonical_candidates_record(
    rank: int,
    *,
    ranking_score: float,
    seed: int = 101,
) -> ProtenixV2PredictionRecord:
    return ProtenixV2PredictionRecord(
        candidate_id=f"protenix_v2:canonical_candidates:seed_{seed}:rank_{rank}",
        seed=seed,
        rank=rank,
        cif_path=f"/tmp/canonical_candidates_rank_{rank}.cif",
        pdb_path=f"/tmp/canonical_candidates_rank_{rank}.pdb",
        summary_json=f"/tmp/canonical_candidates_rank_{rank}.json",
        pae_path=f"/tmp/canonical_candidates_rank_{rank}_pae.json",
        full_confidence_json=f"/tmp/canonical_candidates_rank_{rank}_full.json",
        ranking_score=ranking_score,
        plddt=90.0 - rank,
        iptm=0.6,
        ptm=0.7,
    )


def _canonical_candidate_record(
    index: int,
    rank: int,
    *,
    ranking_score: float,
    seed: int = 101,
) -> ProtenixV2PredictionRecord:
    return ProtenixV2PredictionRecord(
        candidate_id=f"protenix_v2:canonical_candidate_{index:04d}:seed_{seed}:rank_{rank}",
        seed=seed,
        rank=rank,
        cif_path=f"/tmp/canonical_candidate_{index}_rank_{rank}.cif",
        pdb_path=f"/tmp/canonical_candidate_{index}_rank_{rank}.pdb",
        summary_json=f"/tmp/canonical_candidate_{index}_rank_{rank}.json",
        pae_path=f"/tmp/canonical_candidate_{index}_rank_{rank}_pae.json",
        full_confidence_json=f"/tmp/canonical_candidate_{index}_rank_{rank}_full.json",
        ranking_score=ranking_score,
        plddt=90.0 - rank,
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
    assert candidates[0]["artifacts"]["protenix_v2"]["predicted_cif"] == "/tmp/protenix_0.cif"
    assert candidates[0]["artifacts"]["protenix_v2"]["predicted_pdb"] == "/tmp/protenix_0.pdb"
    assert candidates[0]["scores"]["protenix_v2"]["ranking_score"] == 0.9


def test_protenix_v2_candidate_ids_are_deterministic() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL")])
    records = [_record(0)]
    first = _protenix_v2_candidates(params, records)
    second = _protenix_v2_candidates(params, records)
    assert first[0]["candidate_id"] == second[0]["candidate_id"]


def test_protenix_v2_candidate_source_fields() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL")])
    source = _protenix_v2_candidates(params, [_record(0)])[0]["source"]
    assert source["tool"] == "protenix_v2"
    assert source["local_id"] == "protenix_0000"
    assert source["index"] == 0


def test_protenix_v2_preserves_upstream_identity_for_canonical_sample() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL", candidate_id="v40")])

    candidate = _protenix_v2_candidates(params, [_pxdesign_record(0)])[0]

    assert candidate["candidate_id"] == "v40"
    assert candidate["sequence"] == "MKKL"
    assert candidate["structure_path"] == "/tmp/protenix_0.cif"
    assert candidate["scores"]["esm2"]["score"] == 0.42
    assert candidate["scores"]["protenix_v2"]["plddt"] == 80.0
    assert candidate["artifacts"]["esm2"]["fasta"] == "/tmp/esm2.fa"
    assert candidate["artifacts"]["protenix_v2"]["predicted_cif"] == "/tmp/protenix_0.cif"


def test_protenix_v2_preserves_upstream_identity_for_single_canonical_candidates_sample() -> None:
    params = ProtenixV2Input(
        candidates=[_candidate("MKKL", candidate_id="cand_a691984f1120")]
    )
    record = ProtenixV2PredictionRecord(
        candidate_id="protenix_v2:canonical_candidates:seed_101:rank_0",
        seed=101,
        rank=0,
        cif_path="/tmp/canonical_candidates_sample_0.cif",
        pdb_path="/tmp/canonical_candidates_sample_0.pdb",
        summary_json="/tmp/canonical_candidates_summary_confidence_sample_0.json",
        pae_path="/tmp/canonical_candidates_confidences_sample_0.json",
        full_confidence_json="/tmp/canonical_candidates_full_data_sample_0.json",
        ranking_score=0.9,
        plddt=88.5,
        iptm=0.0,
        ptm=0.86,
    )

    candidate = _protenix_v2_candidates(params, [record])[0]

    assert candidate["candidate_id"] == "cand_a691984f1120"
    assert candidate["sequence"] == "MKKL"
    assert candidate["structure_path"] == "/tmp/canonical_candidates_sample_0.cif"
    assert candidate["scores"]["esm2"]["score"] == 0.42
    assert candidate["scores"]["protenix_v2"]["plddt"] == 88.5
    assert (
        candidate["artifacts"]["protenix_v2"]["native_id"]
        == "protenix_v2:canonical_candidates:seed_101:rank_0"
    )
    assert candidate["artifacts"]["protenix_v2"]["local_id"] == "protenix_0000"


def test_protenix_v2_emits_one_best_candidate_per_upstream_logical_candidate() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL", candidate_id="cand_same")])
    records = [
        _canonical_candidates_record(1, ranking_score=0.5),
        _canonical_candidates_record(0, ranking_score=0.9),
    ]

    candidates = _protenix_v2_candidates(params, records)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["candidate_id"] == "cand_same"
    assert candidate["structure_path"] == "/tmp/canonical_candidates_rank_0.cif"
    assert candidate["scores"]["protenix_v2"]["ranking_score"] == 0.9
    assert (
        candidate["artifacts"]["protenix_v2"]["native_id"]
        == "protenix_v2:canonical_candidates:seed_101:rank_0"
    )
    assert candidate["artifacts"]["protenix_v2"]["local_id"] == "protenix_0001"


def test_protenix_v2_keeps_distinct_upstream_candidates_when_each_has_multiple_records() -> None:
    params = ProtenixV2Input(
        candidates=[
            _candidate("MKKL", candidate_id="cand_a"),
            _candidate("ACDE", candidate_id="cand_b"),
        ]
    )
    records = [
        _canonical_candidate_record(0, 1, ranking_score=0.2),
        _canonical_candidate_record(0, 0, ranking_score=0.8),
        _canonical_candidate_record(1, 1, ranking_score=0.3),
        _canonical_candidate_record(1, 0, ranking_score=0.7),
    ]

    candidates = _protenix_v2_candidates(params, records)

    assert len(candidates) == 2
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    assert by_id["cand_a"]["sequence"] == "MKKL"
    assert by_id["cand_a"]["structure_path"] == "/tmp/canonical_candidate_0_rank_0.cif"
    assert by_id["cand_a"]["scores"]["protenix_v2"]["ranking_score"] == pytest.approx(0.8)
    assert by_id["cand_b"]["sequence"] == "ACDE"
    assert by_id["cand_b"]["structure_path"] == "/tmp/canonical_candidate_1_rank_0.cif"
    assert by_id["cand_b"]["scores"]["protenix_v2"]["ranking_score"] == pytest.approx(0.7)


def test_protenix_v2_legacy_unmapped_records_still_emit_record_level_candidates() -> None:
    params = ProtenixV2Input(candidates=[_candidate("MKKL", candidate_id="cand_same")])
    records = [_record(0), _record(1)]

    candidates = _protenix_v2_candidates(params, records)

    assert len(candidates) == 2
    assert {candidate["structure_path"] for candidate in candidates} == {
        "/tmp/protenix_0.cif",
        "/tmp/protenix_1.cif",
    }
    assert len({candidate["candidate_id"] for candidate in candidates}) == 2


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
    assert dumped["all_predictions"][0]["pdb_path"] == "/tmp/protenix_0.pdb"


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


def test_protenix_v2_writes_target_input_json_binder_complex_entries(tmp_path) -> None:
    target_json = _target_input_json(tmp_path)
    params = ProtenixV2Input(
        target_input_json=target_json,
        candidates=[
            _candidate("MKKL", candidate_id="cand_a"),
            _candidate("ACDE", candidate_id="cand_b"),
        ],
        seeds=[101],
    )

    path = _write_protenix_input_json_from_candidates(
        params.candidates,
        params.seeds,
        tmp_path / "canonical_candidates.json",
        target_input_json=params.target_input_json,
    )

    payload = json.loads(path.read_text())
    assert [item["name"] for item in payload] == [
        "canonical_candidate_0000",
        "canonical_candidate_0001",
    ]
    target_chain = payload[0]["sequences"][0]["proteinChain"]
    assert target_chain["sequence"] == "AG"
    assert target_chain["pairedMsaPath"] == "/tmp/pairing.a3m"
    assert target_chain["unpairedMsaPath"] == "/tmp/non_pairing.a3m"
    assert payload[0]["sequences"][1]["proteinChain"]["sequence"] == "MKKL"
    assert payload[1]["sequences"][1]["proteinChain"]["sequence"] == "ACDE"
    assert payload[0]["modelSeeds"] == [101]


def test_protenix_v2_writes_bindcraft_target_binder_complex_entries(tmp_path) -> None:
    target = _target_pdb(tmp_path)
    params = ProtenixV2Input(
        candidates=[_bindcraft_candidate("MKKL", target)],
        seeds=[101],
    )

    path = _write_protenix_input_json_from_candidates(
        params.candidates,
        params.seeds,
        tmp_path / "canonical_candidates.json",
    )

    payload = json.loads(path.read_text())
    assert payload[0]["sequences"][0]["proteinChain"]["sequence"] == "AG"
    assert payload[0]["sequences"][1]["proteinChain"]["sequence"] == "MKKL"


def test_protenix_v2_rejects_mixed_pxdesign_target_context_candidates(tmp_path) -> None:
    target = _target_pdb(tmp_path)
    params = ProtenixV2Input(
        candidates=[
            _pxdesign_candidate("MKKL", target, 0),
            _candidate("ACDE"),
        ],
        seeds=[101],
    )

    with pytest.raises(ValueError, match="cannot mix target-context"):
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
    assert candidates[0]["artifacts"]["protenix_v2"]["complex_chains"][0] == {
        "role": "target",
        "input_order": 0,
        "sequence_length": 2,
        "observed_chain_id": None,
    }
    assert candidates[0]["artifacts"]["protenix_v2"]["complex_chains"][1] == {
        "role": "binder",
        "input_order": 1,
        "sequence_length": 4,
        "observed_chain_id": None,
    }


def test_protenix_v2_candidates_preserve_bindcraft_identity_with_target_input_json(
    tmp_path,
) -> None:
    target = _target_pdb(tmp_path)
    target_json = _target_input_json(tmp_path)
    params = ProtenixV2Input(
        target_input_json=target_json,
        candidates=[_bindcraft_candidate("MKKL", target, 0)],
        seeds=[101],
    )

    candidate = _protenix_v2_candidates(params, [_pxdesign_record(0)])[0]

    assert candidate["candidate_id"] == "cand_bindcraft_0"
    assert candidate["sequence"] == "MKKL"
    assert candidate["scores"]["bindcraft"]["iptm"] == pytest.approx(0.78)
    assert candidate["scores"]["protenix_v2"]["iptm"] == pytest.approx(0.6)
    assert candidate["artifacts"]["bindcraft"]["target_pdb"] == target
    assert (
        candidate["artifacts"]["protenix_v2"]["complex_input_mode"]
        == "target_input_json_target_sequence_binder_sequence"
    )
    assert candidate["artifacts"]["protenix_v2"]["target_input_json"] == target_json
    assert candidate["artifacts"]["protenix_v2"]["complex_chains"] == [
        {
            "role": "target",
            "input_order": 0,
            "sequence_length": 2,
            "observed_chain_id": None,
        },
        {
            "role": "binder",
            "input_order": 1,
            "sequence_length": 4,
            "observed_chain_id": None,
        },
    ]


def test_protenix_v2_candidates_preserve_bindcraft_identity_from_target_context(
    tmp_path,
) -> None:
    target = _target_pdb(tmp_path)
    params = ProtenixV2Input(
        candidates=[_bindcraft_candidate("MKKL", target, 0)],
        seeds=[101],
    )

    candidate = _protenix_v2_candidates(params, [_pxdesign_record(0)])[0]

    assert candidate["candidate_id"] == "cand_bindcraft_0"
    assert candidate["sequence"] == "MKKL"
    assert (
        candidate["artifacts"]["protenix_v2"]["complex_input_mode"]
        == "bindcraft_target_sequence_binder_sequence"
    )
    assert candidate["artifacts"]["protenix_v2"]["complex_chains"][1]["sequence_length"] == 4


def test_protenix_v2_consumes_legacy_input_json() -> None:
    assert ProtenixV2Input(input_json="/tmp/input.json").input_json == "/tmp/input.json"


def test_protenix_v2_rejects_target_input_json_with_raw_input_json() -> None:
    with pytest.raises(ValidationError, match="do not combine it with input_json"):
        ProtenixV2Input(
            input_json="/tmp/input.json",
            target_input_json="/tmp/target.json",
        )


def test_protenix_v2_rejects_target_input_json_without_candidates() -> None:
    with pytest.raises(ValidationError, match="requires canonical Candidate"):
        ProtenixV2Input(target_input_json="/tmp/target.json")


def test_protenix_v2_raw_complex_input_uses_binder_sequence(tmp_path) -> None:
    input_json = tmp_path / "complex.json"
    input_json.write_text(
        json.dumps(
            [
                {
                    "name": "cplx_design",
                    "sequences": [
                        {"proteinChain": {"sequence": "A" * 809, "count": 1}},
                        {"proteinChain": {"sequence": "ACDEF" * 14, "count": 1}},
                    ],
                    "modelSeeds": [101],
                }
            ]
        )
    )
    params = ProtenixV2Input(input_json=str(input_json))
    record = ProtenixV2PredictionRecord(
        candidate_id="protenix_v2:cplx_design:seed_101:rank_0",
        seed=101,
        rank=0,
        cif_path=str(tmp_path / "complex.cif"),
        pdb_path=None,
        summary_json=str(tmp_path / "summary.json"),
        ranking_score=0.9,
        plddt=80.0,
        iptm=0.65,
        ptm=0.7,
    )

    candidate = _protenix_v2_candidates(params, [record])[0]

    assert len(candidate["sequence"]) == 70
    assert candidate["sequence"] == "ACDEF" * 14
    chains = candidate["artifacts"]["protenix_v2"]["complex_chains"]
    assert chains == [
        {
            "role": "target",
            "input_order": 0,
            "sequence_length": 809,
            "observed_chain_id": None,
        },
        {
            "role": "binder",
            "input_order": 1,
            "sequence_length": 70,
            "observed_chain_id": None,
        },
    ]
    assert (
        candidate["artifacts"]["protenix_v2"]["complex_input_mode"]
        == "input_json_target_sequence_binder_sequence"
    )


def test_protenix_v2_rejects_mixed_candidates() -> None:
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        ProtenixV2Input(candidates=[_candidate("MKKL"), {"sequence": "ACDE"}])
