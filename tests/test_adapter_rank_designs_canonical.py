from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.base import torch
from vibeproteinagent.adapters.rank_designs.adapter import RankDesignsAdapter, RankDesignsInput
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _canonical_candidate(
    candidate_id: str,
    *,
    local_id: str,
    sequence: str,
    score: float,
) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": "go_sequences",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": None,
            "index": int(local_id.rsplit("_", 1)[1]),
        },
        "sequence": sequence,
        "scores": {"go_sequences": {"score": score, "plddt": 90.0}},
        "artifacts": {},
    }


def _complete_job(
    workspace: Workspace,
    tool: str,
    output: dict,
    *,
    params: dict | None = None,
) -> str:
    store = JobStore(workspace)
    job = store.submit(tool, "1.0.0", params or {}, trajectory_id=None)
    store.transition(job.job_id, JobState.QUEUED)
    store.transition(job.job_id, JobState.RUNNING)
    store.transition(job.job_id, JobState.COMPLETED, output=output)
    return job.job_id


def _protenix_candidate(
    candidate_id: str,
    *,
    sequence: str,
    structure_path: str,
    ranking_score: float,
    native_id: str,
) -> dict:
    candidate = _canonical_candidate(
        candidate_id,
        local_id="protenix_0000",
        sequence=sequence,
        score=0.0,
    )
    candidate["source"]["tool"] = "protenix_v2"
    candidate["source"]["native_id"] = native_id
    candidate["structure_path"] = structure_path
    candidate["scores"] = {
        "protenix_v2": {
            "ranking_score": ranking_score,
            "plddt": 85.0,
            "iptm": 0.0,
            "ptm": 0.8,
        }
    }
    candidate["artifacts"] = {
        "protenix_v2": {
            "predicted_cif": structure_path,
            "native_id": native_id,
            "local_id": "protenix_0000",
        }
    }
    return candidate


def test_rank_designs_accepts_all_canonical_candidates(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            candidates=[
                _canonical_candidate("cand_weak", local_id="seq_0000", sequence="AAAA", score=0.1),
                _canonical_candidate(
                    "cand_strong",
                    local_id="seq_0001",
                    sequence="CCCC",
                    score=0.9,
                ),
            ],
            top_k=1,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.ranked_candidates[0]["candidate_id"] == "cand_strong"
    assert out.ranked_candidates[0]["sequence"] == "CCCC"
    assert out.ranked_candidates[0]["candidate_score"] == 0.9


def test_rank_designs_preserves_canonical_score_namespaces(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = _canonical_candidate(
        "cand_qc",
        local_id="seq_0000",
        sequence="AAAA",
        score=0.5,
    )
    candidate["source"]["tool"] = "protenix_v2"
    candidate["scores"] = {
        "protenix_v2": {"ranking_score": 0.8, "plddt": 91.0},
        "protein_qc": {"flags": ["high_instability_index"], "plddt": 88.0},
        "sequence_analysis": {"hydrophobic_fraction": 0.2},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["ranking_score"] == 0.8
    assert row["qc_flags"] == ["high_instability_index"]
    assert row["qc_flag_count"] == 1
    assert row["scores"]["protein_qc"]["flags"] == ["high_instability_index"]
    assert row["scores"]["sequence_analysis"]["hydrophobic_fraction"] == 0.2


def test_rank_designs_preserves_canonical_artifact_metadata(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = _canonical_candidate(
        "cand_complex",
        local_id="seq_0000",
        sequence="ACDEF",
        score=0.5,
    )
    candidate["source"]["tool"] = "protenix_v2"
    candidate["structure_path"] = "/tmp/complex.cif"
    candidate["artifacts"] = {
        "protenix_v2": {
            "complex_chains": [
                {"role": "target", "input_order": 0, "sequence_length": 809},
                {"role": "binder", "input_order": 1, "sequence_length": 5},
            ]
        }
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["sequence"] == "ACDEF"
    assert row["structure_path"] == "/tmp/complex.cif"
    assert row["artifacts"]["protenix_v2"]["complex_chains"][1]["role"] == "binder"


def test_rank_designs_uses_family_profile_score_and_preserves_namespace(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    weak = _canonical_candidate(
        "cand_weak_profile",
        local_id="seq_0000",
        sequence="AAAA",
        score=0.5,
    )
    strong = _canonical_candidate(
        "cand_strong_profile",
        local_id="seq_0001",
        sequence="CCCC",
        score=0.5,
    )
    weak["scores"]["family_profile"] = {"ranking_score": 0.1}
    strong["scores"]["family_profile"] = {
        "ranking_score": 0.9,
        "family_similarity_score": 0.8,
        "motif_match_score": 0.7,
    }

    out = adapter.run(
        RankDesignsInput(candidates=[weak, strong], top_k=2),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["candidate_id"] == "cand_strong_profile"
    assert row["family_profile_score"] == pytest.approx(0.9)
    assert row["scores"]["family_profile"]["motif_match_score"] == 0.7


def test_rank_designs_normalizes_esm3_plddt_and_reads_ptm_namespace(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_esm3",
        "source": {
            "tool": "esm3",
            "node_id": None,
            "job_id": None,
            "local_id": "esm3_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": "ACDE",
        "scores": {
            "esm3": {"mean_plddt": 0.981, "ptm": 0.864},
            "protein_qc": {"flags": [], "plddt": 98.1},
        },
        "artifacts": {},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["plddt"] == pytest.approx(98.1)
    assert row["ptm"] == pytest.approx(0.864)
    assert row["score"] == pytest.approx(0.981 + 0.25 * 0.864)
    assert row["score"] > 0


def test_rank_designs_applies_qc_penalty_without_double_dividing_plddt(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_esm3_flags",
        "source": {
            "tool": "esm3",
            "node_id": None,
            "job_id": None,
            "local_id": "esm3_0000",
            "native_id": None,
            "index": 0,
        },
        "sequence": "ACDE",
        "scores": {
            "esm3": {"mean_plddt": 0.981, "ptm": 0.864},
            "protein_qc": {
                "flags": [
                    "low_plddt",
                    "contains_polybasic_cluster",
                    "high_instability_index",
                ],
                "plddt": 98.1,
            },
        },
        "artifacts": {},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["qc_flag_count"] == 3
    assert row["score"] == pytest.approx(0.981 + 0.25 * 0.864 - 0.30)


def test_rank_designs_keeps_protenix_plddt_percent_scale_semantics(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    candidate = _canonical_candidate(
        "cand_protenix",
        local_id="seq_0000",
        sequence="AAAA",
        score=0.5,
    )
    candidate["source"]["tool"] = "protenix_v2"
    candidate["scores"] = {
        "protenix_v2": {
            "ranking_score": 0.8,
            "plddt": 91.0,
            "iptm": 0.6,
            "ptm": 0.7,
        },
        "protein_qc": {"flags": ["high_instability_index"], "plddt": 88.0},
    }

    out = adapter.run(
        RankDesignsInput(candidates=[candidate], top_k=1),
        null_context(tmp_path / "job"),
    )

    row = out.ranked_candidates[0]
    assert row["plddt"] == pytest.approx(91.0)
    assert row["score"] == pytest.approx(0.8 + 0.91 + 0.5 * 0.6 + 0.25 * 0.7 - 0.10)


def test_rank_designs_legacy_candidate_path_still_works(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            candidates=[
                {"candidate_id": "weak", "sequence": "AAAA", "score": 0.1},
                {"candidate_id": "strong", "sequence": "CCCC", "score": 0.9},
            ],
            top_k=1,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.ranked_candidates[0]["candidate_id"] == "strong"
    assert out.ranked_candidates[0]["sequence"] == "CCCC"


def test_rank_designs_rejects_mixed_canonical_and_legacy_candidates() -> None:
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        RankDesignsInput(
            candidates=[
                _canonical_candidate("cand_a", local_id="seq_0000", sequence="AAAA", score=0.1),
                {"candidate_id": "legacy", "sequence": "CCCC", "score": 0.9},
            ]
        )


def test_rank_designs_canonical_detection_only_applies_to_candidates_slot(tmp_path) -> None:
    # PR #3 intentionally routes canonical Candidate only in the candidates slot.
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            predictions=[
                _canonical_candidate(
                    "cand_prediction",
                    local_id="seq_0000",
                    sequence="AAAA",
                    score=0.9,
                )
            ],
            top_k=1,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.ranked_candidates[0]["candidate_id"] == "cand_prediction"
    assert out.ranked_candidates[0]["sequence"] == "AAAA"


def test_rank_designs_keeps_distinct_candidate_ids_with_same_sequence_separate(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        RankDesignsInput(
            candidates=[
                _canonical_candidate("cand_a", local_id="seq_0000", sequence="AAAA", score=0.2),
                _canonical_candidate("cand_b", local_id="seq_0001", sequence="AAAA", score=0.9),
            ],
            top_k=10,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.candidate_count == 2
    assert [row["candidate_id"] for row in out.ranked_candidates] == ["cand_b", "cand_a"]
    assert [row["candidate_score"] for row in out.ranked_candidates] == [0.9, 0.2]


def test_rank_designs_does_not_mix_same_candidate_id_alternate_structures(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    weak = _protenix_candidate(
        "cand_same",
        sequence="AAAA",
        structure_path="/tmp/rank1.cif",
        ranking_score=0.5,
        native_id="protenix_v2:canonical_candidates:seed_101:rank_1",
    )
    strong = _protenix_candidate(
        "cand_same",
        sequence="AAAA",
        structure_path="/tmp/rank0.cif",
        ranking_score=0.9,
        native_id="protenix_v2:canonical_candidates:seed_101:rank_0",
    )

    out = adapter.run(
        RankDesignsInput(candidates=[weak, strong], top_k=10),
        null_context(tmp_path / "job"),
    )

    assert out.candidate_count == 1
    assert len(out.ranked_candidates) == 1
    row = out.ranked_candidates[0]
    assert row["candidate_id"] == "cand_same"
    assert row["structure_path"] == "/tmp/rank0.cif"
    assert row["scores"]["protenix_v2"]["ranking_score"] == pytest.approx(0.9)
    assert (
        row["artifacts"]["protenix_v2"]["native_id"]
        == "protenix_v2:canonical_candidates:seed_101:rank_0"
    )
    assert any("alternate structure variant" in warning for warning in out.warnings)


def test_rank_designs_attaches_sequence_level_qc_to_best_alternate_structure(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    weak = _protenix_candidate(
        "cand_same",
        sequence="AAAA",
        structure_path="/tmp/rank1.cif",
        ranking_score=0.5,
        native_id="protenix_v2:canonical_candidates:seed_101:rank_1",
    )
    strong = _protenix_candidate(
        "cand_same",
        sequence="AAAA",
        structure_path="/tmp/rank0.cif",
        ranking_score=0.9,
        native_id="protenix_v2:canonical_candidates:seed_101:rank_0",
    )

    out = adapter.run(
        RankDesignsInput(
            candidates=[weak, strong],
            qc_results=[
                {
                    "candidate_id": "cand_same",
                    "sequence": "AAAA",
                    "flags": ["high_instability_index"],
                }
            ],
            top_k=10,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.candidate_count == 1
    row = out.ranked_candidates[0]
    assert row["structure_path"] == "/tmp/rank0.cif"
    assert row["qc_flags"] == ["high_instability_index"]
    assert row["qc_flag_count"] == 1
    assert row["score"] == pytest.approx(0.9 + 0.85 + 0.25 * 0.8 - 0.10)


def test_rank_designs_sequence_only_evidence_fans_out_without_collapsing_candidates(
    tmp_path,
) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))

    out = adapter.run(
        RankDesignsInput(
            candidates=[
                _canonical_candidate("cand_a", local_id="seq_0000", sequence="AAAA", score=0.2),
                _canonical_candidate("cand_b", local_id="seq_0001", sequence="AAAA", score=0.9),
            ],
            qc_results=[{"sequence": "AAAA", "flags": ["shared_sequence_flag"]}],
            top_k=10,
        ),
        null_context(tmp_path / "job"),
    )

    assert out.candidate_count == 2
    by_id = {row["candidate_id"]: row for row in out.ranked_candidates}
    assert set(by_id) == {"cand_a", "cand_b"}
    assert by_id["cand_a"]["qc_flags"] == ["shared_sequence_flag"]
    assert by_id["cand_b"]["qc_flags"] == ["shared_sequence_flag"]


def test_rank_designs_keeps_same_candidate_id_different_sequences_separate(tmp_path) -> None:
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    first = _protenix_candidate(
        "cand_same",
        sequence="AAAA",
        structure_path="/tmp/aaaa.cif",
        ranking_score=0.8,
        native_id="protenix_v2:canonical_candidate_0000:seed_101:rank_0",
    )
    second = _protenix_candidate(
        "cand_same",
        sequence="CCCC",
        structure_path="/tmp/cccc.cif",
        ranking_score=0.7,
        native_id="protenix_v2:canonical_candidate_0001:seed_101:rank_0",
    )

    out = adapter.run(
        RankDesignsInput(candidates=[first, second], top_k=10),
        null_context(tmp_path / "job"),
    )

    assert out.candidate_count == 2
    assert {row["sequence"] for row in out.ranked_candidates} == {"AAAA", "CCCC"}
    assert any("material conflicts" in warning for warning in out.warnings)


def test_rank_designs_does_not_collapse_distinct_protenix_jobs_with_same_output_basename(
    tmp_workspace: Workspace,
    tmp_path,
) -> None:
    job_ids = []
    for index, sequence in enumerate(("AAAA", "CCCC", "DDDD", "EEEE")):
        candidate = _canonical_candidate(
            f"cand_{index}",
            local_id="protenix_0000",
            sequence=sequence,
            score=0.0,
        )
        candidate["source"]["tool"] = "protenix_v2"
        candidate["source"]["native_id"] = "protenix_v2:canonical_candidates:seed_101:rank_0"
        candidate["structure_path"] = (
            f"/tmp/protenix_job_{index}/predictions/canonical_candidates_sample_0.cif"
        )
        candidate["scores"] = {
            "protenix_v2": {
                "ranking_score": 0.1 + index,
                "plddt": 80.0 + index,
                "iptm": 0.0,
                "ptm": 0.8,
            }
        }
        candidate["artifacts"] = {
            "protenix_v2": {
                "predicted_cif": candidate["structure_path"],
                "native_id": candidate["source"]["native_id"],
                "local_id": "protenix_0000",
            }
        }
        job_ids.append(
            _complete_job(
                tmp_workspace,
                "protenix_v2",
                {
                    "candidates": [candidate],
                    "predicted_cif": candidate["structure_path"],
                    "metrics": candidate["scores"]["protenix_v2"],
                },
                params={"sequence": sequence},
            )
        )

    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(job_ids=job_ids, top_k=4),
        null_context(tmp_workspace.root / "jobs" / "job_rank"),
    )

    assert out.candidate_count == 4
    assert {row["candidate_id"] for row in out.ranked_candidates} == {
        "cand_0",
        "cand_1",
        "cand_2",
        "cand_3",
    }
    assert {row["sequence"] for row in out.ranked_candidates} == {
        "AAAA",
        "CCCC",
        "DDDD",
        "EEEE",
    }
