from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.rank_designs.adapter import (
    RankDesignsAdapter,
    RankDesignsInput,
    RankDesignsOutput,
)
from vibeproteinagent.workspace import JobState, JobStore, Workspace


def _completed_job(workspace: Workspace, tool: str, output: dict) -> str:
    store = JobStore(workspace)
    rec = store.submit(tool, "1.0.0", {"test": tool})
    store.transition(rec.job_id, JobState.QUEUED)
    store.transition(rec.job_id, JobState.RUNNING)
    store.transition(rec.job_id, JobState.COMPLETED, output=output)
    return rec.job_id


def test_rank_designs_combines_prediction_analysis_and_qc(tmp_path):
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            predictions=[
                {"name": "weak", "metrics": {"ranking_score": 0.3, "plddt": 70}},
                {"name": "strong", "metrics": {"ranking_score": 0.8, "plddt": 90}},
            ],
            analyses=[
                {"liabilities": [{"type": "surface", "positions": [1], "reason": "x"}]},
                {"liabilities": []},
            ],
            qc_results=[{"flags": ["low_plddt"]}, {"flags": []}],
            top_k=2,
        ),
        null_context(tmp_path),
    )

    assert out.candidate_count == 2
    assert out.ranked_candidates[0]["candidate_id"] == "strong"
    assert Path(out.ranked_json).exists()
    assert Path(out.ranked_tsv).exists()


def test_rank_designs_preserves_candidate_sequence_and_uses_candidate_score(tmp_path):
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            candidates=[
                {"candidate_id": "weak", "sequence": "AAAA", "score": 0.1},
                {"candidate_id": "strong", "sequence": "CCCC", "score": 0.9},
            ],
            top_k=1,
        ),
        null_context(tmp_path),
    )

    assert out.ranked_candidates[0]["candidate_id"] == "strong"
    assert out.ranked_candidates[0]["sequence"] == "CCCC"
    assert out.ranked_candidates[0]["candidate_score"] == 0.9


def test_rank_designs_flattens_nested_candidate_lists(tmp_path):
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            candidates=[
                [{"candidate_id": "a", "sequence": "AAAA", "score": 0.2}],
                [{"candidate_id": "b", "sequence": "BBBB", "score": 0.7}],
            ],
            top_k=2,
        ),
        null_context(tmp_path),
    )

    assert out.candidate_count == 2
    assert [row["candidate_id"] for row in out.ranked_candidates] == ["b", "a"]


def test_rank_designs_accepts_single_prediction_object(tmp_path):
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = adapter.run(
        RankDesignsInput(
            predictions={"name": "single", "ranking_score": 0.5, "plddt": 80},
            top_k=1,
        ),
        null_context(tmp_path),
    )

    assert out.candidate_count == 1
    assert out.ranked_candidates[0]["candidate_id"] == "single"
    assert out.ranked_candidates[0]["ranking_score"] == 0.5


def test_rank_designs_requires_input(tmp_path):
    adapter = RankDesignsAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="requires at least one"):
        adapter.run(RankDesignsInput(), null_context(tmp_path))


def test_rank_designs_reads_workspace_job_ids(tmp_workspace: Workspace):
    predict_job = _completed_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_a",
                    "source": {"tool": "protenix_v2", "local_id": "a", "index": 0},
                    "sequence": "AAAA",
                    "structure_path": "/tmp/a.cif",
                    "scores": {
                        "protenix_v2": {
                            "ranking_score": 0.7,
                            "plddt": 82,
                            "iptm": 0.6,
                            "ptm": 0.5,
                        }
                    },
                }
            ]
        },
    )
    ipsae_job = _completed_job(
        tmp_workspace,
        "ipsae",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_a",
                    "source": {"tool": "ipsae", "local_id": "a", "index": 0},
                    "scores": {"ipsae": {"ipsae_score": 0.8}},
                }
            ]
        },
    )

    adapter = RankDesignsAdapter(
        config={}, weights_dir=tmp_workspace.root / "weights", device=torch.device("cpu")
    )
    out = adapter.run(
        RankDesignsInput(job_ids=[predict_job, ipsae_job], top_k=1),
        null_context(tmp_workspace.job_dir("job_rank")),
    )

    row = out.ranked_candidates[0]
    assert out.candidate_count == 1
    assert row["candidate_id"] == "cand_a"
    assert row["ipsae"] == 0.8
    assert row["structure_path"] == "/tmp/a.cif"


def test_rank_designs_job_id_evidence_joins_by_candidate_identity(
    tmp_workspace: Workspace,
) -> None:
    predict_job = _completed_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_a",
                    "source": {"tool": "protenix_v2", "local_id": "a", "index": 0},
                    "sequence": "AAAA",
                    "structure_path": "/tmp/a.cif",
                    "scores": {"protenix_v2": {"ranking_score": 0.7, "plddt": 82}},
                },
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_b",
                    "source": {"tool": "protenix_v2", "local_id": "b", "index": 1},
                    "sequence": "BBBB",
                    "structure_path": "/tmp/b.cif",
                    "scores": {"protenix_v2": {"ranking_score": 0.7, "plddt": 82}},
                },
            ]
        },
    )
    ipsae_job = _completed_job(
        tmp_workspace,
        "ipsae",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_b",
                    "source": {"tool": "ipsae", "local_id": "b", "index": 1},
                    "scores": {"ipsae": {"ipsae_score": 0.1}},
                },
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_a",
                    "source": {"tool": "ipsae", "local_id": "a", "index": 0},
                    "scores": {"ipsae": {"ipsae_score": 0.9}},
                },
            ]
        },
    )

    adapter = RankDesignsAdapter(
        config={}, weights_dir=tmp_workspace.root / "weights", device=torch.device("cpu")
    )
    out = adapter.run(
        RankDesignsInput(job_ids=[predict_job, ipsae_job], top_k=2),
        null_context(tmp_workspace.job_dir("job_rank_identity")),
    )

    by_id = {row["candidate_id"]: row for row in out.ranked_candidates}
    assert out.candidate_count == 2
    assert by_id["cand_a"]["ipsae"] == 0.9
    assert by_id["cand_b"]["ipsae"] == 0.1
    assert out.warnings == []


def test_rank_designs_warns_on_unmatched_job_id_evidence(
    tmp_workspace: Workspace,
) -> None:
    predict_job = _completed_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": "cand_a",
                    "source": {"tool": "protenix_v2", "local_id": "a", "index": 0},
                    "structure_path": "/tmp/a.cif",
                    "scores": {"protenix_v2": {"ranking_score": 0.7}},
                }
            ]
        },
    )
    ipsae_job = _completed_job(
        tmp_workspace,
        "ipsae",
        {
            "ipsae_score": 0.8,
            "metrics": {"ipsae_score": 0.8},
        },
    )

    adapter = RankDesignsAdapter(
        config={}, weights_dir=tmp_workspace.root / "weights", device=torch.device("cpu")
    )
    out = adapter.run(
        RankDesignsInput(job_ids=[predict_job, ipsae_job], top_k=5),
        null_context(tmp_workspace.job_dir("job_rank_unmatched")),
    )

    assert out.candidate_count == 2
    assert out.warnings
    assert "standalone evidence" in out.warnings[0]


def test_rank_designs_merges_enriched_canonical_workflow_by_candidate_id(
    tmp_workspace: Workspace,
) -> None:
    candidate_id = "cand_small_molecule"
    sequence = "ACDEFGHIK"
    boltzgen_job = _completed_job(
        tmp_workspace,
        "boltzgen",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "boltzgen", "local_id": "design_0000", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/boltzgen.cif",
                    "scores": {"boltzgen": {"score": 0.4, "length": len(sequence)}},
                    "artifacts": {"boltzgen": {"cif_path": "/tmp/boltzgen.cif"}},
                }
            ]
        },
    )
    boltz2_job = _completed_job(
        tmp_workspace,
        "boltz2_affinity",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "boltzgen", "local_id": "design_0000", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/boltz2_pred.cif",
                    "scores": {
                        "boltzgen": {"score": 0.4, "length": len(sequence)},
                        "boltz2_affinity": {
                            "affinity_pred_value": -1.2,
                            "affinity_probability_binary": 0.9,
                        },
                    },
                    "artifacts": {
                        "boltzgen": {"cif_path": "/tmp/boltzgen.cif"},
                        "boltz2_affinity": {"predicted_cif": "/tmp/boltz2_pred.cif"},
                    },
                }
            ]
        },
    )
    structure_job = _completed_job(
        tmp_workspace,
        "structure_analysis",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "boltzgen", "local_id": "design_0000", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/boltz2_pred.cif",
                    "scores": {
                        "structure_analysis": {
                            "interface": {"ligand_contact_count": 3},
                            "ligand_contacts": {
                                "contact_count": 3,
                                "contact_residue_count": 3,
                            },
                            "ligand_pocket": {
                                "pocket_residue_count": 3,
                                "ligand_heavy_atom_count": 4,
                                "ligand_buried_atom_count": 2,
                                "ligand_buried_fraction": 0.5,
                                "distance_cutoff_angstrom": 5.0,
                            },
                            "liability_count": 0,
                        }
                    },
                    "artifacts": {"structure_analysis": {"summary_json": "/tmp/summary.json"}},
                }
            ]
        },
    )
    qc_job = _completed_job(
        tmp_workspace,
        "protein_qc",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "boltzgen", "local_id": "design_0000", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/boltz2_pred.cif",
                    "scores": {"protein_qc": {"plddt": 88.0, "flags": []}},
                }
            ]
        },
    )

    adapter = RankDesignsAdapter(
        config={}, weights_dir=tmp_workspace.root / "weights", device=torch.device("cpu")
    )
    out = adapter.run(
        RankDesignsInput(
            job_ids=[boltzgen_job, boltz2_job, structure_job, qc_job],
            top_k=1,
        ),
        null_context(tmp_workspace.job_dir("job_rank_small_molecule")),
    )

    assert out.candidate_count == 1
    assert any("alternate structure variant" in warning for warning in out.warnings)
    row = out.ranked_candidates[0]
    assert row["candidate_id"] == candidate_id
    assert row["structure_path"] == "/tmp/boltz2_pred.cif"
    assert row["sequence"] == sequence
    assert row["candidate_ref"]["candidate_id"] == candidate_id
    assert set(row["evidence_job_ids"]) == {boltzgen_job, boltz2_job, structure_job, qc_job}
    assert set(row["evidence_tools"]) == {
        "boltzgen",
        "boltz2_affinity",
        "structure_analysis",
        "protein_qc",
    }
    assert "boltz2_affinity" in row["scores"]
    assert row["scores"]["structure_analysis"]["ligand_pocket"]["ligand_buried_fraction"] == 0.5
    assert row["scores"]["protein_qc"]["plddt"] == 88.0


def test_rank_designs_merges_miniprotein_binder_validation_by_candidate_id(
    tmp_workspace: Workspace,
) -> None:
    candidate_id = "cand_b7c39f5fe83a"
    sequence = "TRKVYINNRGVVLPPREQRLMDMHMGEQDPFLEQNPQIKRVSRYLSNKMMEKHMRVVYVP"
    bindcraft_job = _completed_job(
        tmp_workspace,
        "bindcraft",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "bindcraft", "local_id": "s585903", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/designed_pose.pdb",
                    "scores": {"bindcraft": {"iptm": 0.78, "plddt": 0.85}},
                    "artifacts": {"bindcraft": {"target_pdb": "/tmp/target.pdb"}},
                }
            ]
        },
    )
    protenix_job = _completed_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "bindcraft", "local_id": "s585903", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/full_target_complex.cif",
                    "scores": {
                        "bindcraft": {"iptm": 0.78, "plddt": 0.85},
                        "protenix_v2": {"iptm": 0.31, "plddt": 83.7, "ptm": 0.62},
                    },
                    "artifacts": {
                        "bindcraft": {"target_pdb": "/tmp/target.pdb"},
                        "protenix_v2": {
                            "predicted_cif": "/tmp/full_target_complex.cif",
                            "pae_path": "/tmp/full_target_complex_pae.json",
                        },
                    },
                }
            ]
        },
    )
    ipsae_job = _completed_job(
        tmp_workspace,
        "ipsae",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "bindcraft", "local_id": "s585903", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/full_target_complex.cif",
                    "scores": {"ipsae": {"ipsae_score": 0.0, "interface_pae": 0.0}},
                }
            ]
        },
    )
    prodigy_job = _completed_job(
        tmp_workspace,
        "prodigy",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "bindcraft", "local_id": "s585903", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/designed_pose.pdb",
                    "scores": {
                        "prodigy": {
                            "delta_g_kcal_mol": -10.2,
                            "kd_nm": 36.0,
                            "interface_contacts": 86,
                        }
                    },
                }
            ]
        },
    )
    qc_job = _completed_job(
        tmp_workspace,
        "protein_qc",
        {
            "candidates": [
                {
                    "schema_version": "candidate_v1",
                    "candidate_id": candidate_id,
                    "source": {"tool": "bindcraft", "local_id": "s585903", "index": 0},
                    "sequence": sequence,
                    "structure_path": "/tmp/full_target_complex.cif",
                    "scores": {
                        "protein_qc": {
                            "plddt": 79.9,
                            "gravy": -0.83,
                            "passes_filters": True,
                        }
                    },
                }
            ]
        },
    )

    adapter = RankDesignsAdapter(
        config={}, weights_dir=tmp_workspace.root / "weights", device=torch.device("cpu")
    )
    out = adapter.run(
        RankDesignsInput(
            job_ids=[bindcraft_job, protenix_job, ipsae_job, prodigy_job, qc_job],
            top_k=1,
        ),
        null_context(tmp_workspace.job_dir("job_rank_miniprotein")),
    )

    assert out.candidate_count == 1
    assert any("alternate structure variant" in warning for warning in out.warnings)
    row = out.ranked_candidates[0]
    assert row["candidate_id"] == candidate_id
    assert row["sequence"] == sequence
    assert row["structure_path"] == "/tmp/full_target_complex.cif"
    assert row["candidate_ref"]["candidate_id"] == candidate_id
    assert set(row["evidence_job_ids"]) == {
        bindcraft_job,
        protenix_job,
        ipsae_job,
        prodigy_job,
        qc_job,
    }
    assert set(row["scores"]) >= {
        "bindcraft",
        "protenix_v2",
        "ipsae",
        "prodigy",
        "protein_qc",
    }
    assert row["scores"]["ipsae"]["ipsae_score"] == 0.0
    assert row["scores"]["prodigy"]["kd_nm"] == pytest.approx(36.0)
    assert row["scores"]["protein_qc"]["gravy"] == pytest.approx(-0.83)


def _functional_candidate(
    candidate_id: str,
    *,
    tool: str,
    sequence: str,
    structure_path: str | None = None,
    scores: dict | None = None,
) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {"tool": tool, "local_id": candidate_id, "index": 0},
        "sequence": sequence,
        **({"structure_path": structure_path} if structure_path else {}),
        "scores": scores or {},
        "artifacts": {},
    }


def test_rank_designs_ignores_reference_hits_and_merges_functional_validation(
    tmp_workspace: Workspace,
) -> None:
    interpro_job = _completed_job(
        tmp_workspace,
        "interpro_sequences",
        {
            "candidates": [
                _functional_candidate(
                    f"ipr_hit_{idx}",
                    tool="interpro_sequences",
                    sequence="REFERENCE",
                    scores={"interpro_sequences": {"score": 0.99}},
                )
                for idx in range(3)
            ]
        },
    )
    esm2_job = _completed_job(
        tmp_workspace,
        "esm2",
        {
            "candidates": [
                _functional_candidate(
                    "v25",
                    tool="esm2",
                    sequence="ACDEFG",
                    scores={"esm2": {"score": 0.6}},
                ),
                _functional_candidate(
                    "v40",
                    tool="esm2",
                    sequence="ACDEFA",
                    scores={"esm2": {"score": 0.8}},
                ),
            ]
        },
    )
    protenix_job = _completed_job(
        tmp_workspace,
        "protenix_v2",
        {
            "candidates": [
                _functional_candidate(
                    "v25",
                    tool="esm2",
                    sequence="ACDEFG",
                    structure_path="/tmp/v25.cif",
                    scores={
                        "esm2": {"score": 0.6},
                        "protenix_v2": {"plddt": 75.7, "ptm": 0.661},
                    },
                ),
                _functional_candidate(
                    "v40",
                    tool="esm2",
                    sequence="ACDEFA",
                    structure_path="/tmp/v40.cif",
                    scores={
                        "esm2": {"score": 0.8},
                        "protenix_v2": {"plddt": 79.512, "ptm": 0.745},
                    },
                ),
            ]
        },
    )
    qc_job = _completed_job(
        tmp_workspace,
        "protein_qc",
        {
            "candidates": [
                _functional_candidate(
                    "v40",
                    tool="esm2",
                    sequence="ACDEFA",
                    structure_path="/tmp/v40.cif",
                    scores={
                        "protein_qc": {
                            "plddt": 79.512,
                            "ca_bfactor_plddt": 83.363,
                            "flags": [],
                        }
                    },
                )
            ]
        },
    )
    tmalign_job = _completed_job(
        tmp_workspace,
        "tmalign",
        {
            "candidates": [
                _functional_candidate(
                    "v40",
                    tool="esm2",
                    sequence="ACDEFA",
                    structure_path="/tmp/v40.cif",
                    scores={"tmalign": {"tm_score_normalized": 0.88058, "rmsd": 1.2}},
                )
            ]
        },
    )

    adapter = RankDesignsAdapter(
        config={}, weights_dir=tmp_workspace.root / "weights", device=torch.device("cpu")
    )
    out = adapter.run(
        RankDesignsInput(
            job_ids=[interpro_job, esm2_job, protenix_job, qc_job, tmalign_job],
            top_k=5,
        ),
        null_context(tmp_workspace.job_dir("job_rank_functional")),
    )

    by_id = {row["candidate_id"]: row for row in out.ranked_candidates}
    assert out.candidate_count == 2
    assert set(by_id) == {"v25", "v40"}
    assert out.warnings == []
    v40 = by_id["v40"]
    assert v40["structure_path"] == "/tmp/v40.cif"
    assert v40["scores"]["esm2"]["score"] == 0.8
    assert v40["scores"]["protenix_v2"]["plddt"] == pytest.approx(79.512)
    assert v40["scores"]["protein_qc"]["plddt"] == pytest.approx(79.512)
    assert v40["scores"]["protein_qc"]["ca_bfactor_plddt"] == pytest.approx(83.363)
    assert v40["scores"]["tmalign"]["tm_score_normalized"] == pytest.approx(0.88058)
    assert set(v40["evidence_tools"]) == {"esm2", "protenix_v2", "protein_qc", "tmalign"}


def test_rank_designs_output_round_trip():
    out = RankDesignsOutput(
        candidate_count=1,
        ranked_candidates=[{"rank": 1, "candidate_id": "a", "score": 1.0}],
        ranked_json="/tmp/r.json",
        ranked_tsv="/tmp/r.tsv",
        runtime_ms=1,
    )
    assert RankDesignsOutput.model_validate(out.model_dump()) == out
