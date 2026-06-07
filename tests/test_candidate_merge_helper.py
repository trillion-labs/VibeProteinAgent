from __future__ import annotations

from vibeproteinagent.candidates.merge import merge_candidate_records


def _candidate(
    candidate_id: str,
    *,
    sequence: str = "ACDE",
    tool: str = "go_sequences",
    index: int = 0,
) -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": candidate_id,
        "source": {
            "tool": tool,
            "node_id": f"node_{index}",
            "job_id": f"job_{index}",
            "local_id": f"seq_{index:04d}",
            "native_id": f"ACC{index}",
            "index": index,
        },
        "sequence": sequence,
        "scores": {tool: {"score": index}},
        "artifacts": {},
    }


def test_same_candidate_id_merges_score_artifact_and_capability_namespaces() -> None:
    source = _candidate("cand_x", tool="go_sequences", index=0)
    enriched = _candidate("cand_x", tool="sequence_analysis", index=1)
    enriched["scores"] = {"sequence_analysis": {"gravy": -0.1}}
    enriched["artifacts"] = {"analysis_json": {"path": "/tmp/analysis.json"}}
    enriched["capabilities"] = {
        "has_sequence": True,
        "has_candidate_scores": True,
    }

    result = merge_candidate_records([[source], [enriched]])

    assert len(result.candidates) == 1
    merged = result.candidates[0]
    assert merged["candidate_id"] == "cand_x"
    assert merged["source"]["tool"] == "sequence_analysis"
    assert set(merged["scores"]) == {"go_sequences", "sequence_analysis"}
    assert "analysis_json" in merged["artifacts"]
    assert merged["capabilities"]["has_sequence"] is True
    assert result.audit["duplicates_removed"] == 1
    assert result.audit["duplicate_groups"][0]["versions"][0]["source"]["native_id"] == "ACC0"


def test_conflicting_namespace_keeps_representative_value_and_audits_conflict() -> None:
    sparse = _candidate("cand_x", tool="go_sequences", index=0)
    rich = _candidate("cand_x", tool="go_sequences", index=1)
    rich["scores"] = {
        "go_sequences": {"score": 99},
        "protein_qc": {"flags": []},
    }

    result = merge_candidate_records([sparse, rich])

    assert result.candidates[0]["scores"]["go_sequences"]["score"] == 99
    assert result.audit["conflicts"][0]["field"] == "scores.go_sequences"


def test_richer_candidate_wins_over_older_and_latest_breaks_ties() -> None:
    older_rich = _candidate("cand_x", tool="go_sequences", index=0)
    older_rich["scores"]["protein_qc"] = {"flags": []}
    latest_sparse = _candidate("cand_x", tool="go_sequences", index=1)

    rich_result = merge_candidate_records([older_rich, latest_sparse])
    assert rich_result.candidates[0]["source"]["job_id"] == "job_0"

    tied_a = _candidate("cand_y", tool="go_sequences", index=0)
    tied_b = _candidate("cand_y", tool="go_sequences", index=1)
    tied_result = merge_candidate_records([tied_a, tied_b])
    assert tied_result.candidates[0]["source"]["job_id"] == "job_1"


def test_sequence_duplicate_preserves_sources_only_in_audit() -> None:
    go = _candidate("cand_go", sequence="MSEQ", tool="go_sequences", index=0)
    interpro = _candidate("cand_interpro", sequence="MSEQ", tool="interpro_sequences", index=1)

    result = merge_candidate_records([go, interpro])

    assert len(result.candidates) == 1
    assert "versions" not in result.candidates[0]
    assert result.audit["duplicate_groups"][0]["identity_type"] == "sequence"
    assert {
        version["source"]["tool"]
        for version in result.audit["duplicate_groups"][0]["versions"]
    } == {"go_sequences", "interpro_sequences"}


def test_sequence_duplicate_does_not_mix_structure_candidates() -> None:
    sequence_only = _candidate("cand_seq", sequence="MSEQ", index=0)
    structure = _candidate("cand_struct", sequence="MSEQ", index=1)
    structure["structure_path"] = "/tmp/model.pdb"

    result = merge_candidate_records([sequence_only, structure])

    assert [candidate["candidate_id"] for candidate in result.candidates] == [
        "cand_seq",
        "cand_struct",
    ]


def test_structure_identity_merges_across_path_field_names() -> None:
    structure_path = _candidate("cand_struct_path", sequence="", index=0)
    structure_path.pop("sequence", None)
    structure_path["structure_path"] = "/tmp/model.pdb"
    full_structure_path = _candidate("cand_full_path", sequence="", index=1)
    full_structure_path.pop("sequence", None)
    full_structure_path["full_structure_path"] = "/tmp/model.pdb"
    full_structure_path["scores"] = {"structure_analysis": {"helix_pct": 0.4}}

    result = merge_candidate_records([structure_path, full_structure_path])

    assert len(result.candidates) == 1
    assert result.audit["duplicate_groups"][0]["identity_type"] == "structure"
    assert result.audit["duplicate_groups"][0]["identity_key"] == "structure:/tmp/model.pdb"
    assert result.candidates[0]["candidate_id"] == "cand_full_path"


def test_structure_identity_does_not_merge_conflicting_sequences() -> None:
    first = _candidate("cand_first", sequence="AAAA", index=0)
    first["structure_path"] = "/tmp/model.pdb"
    second = _candidate("cand_second", sequence="GGGG", index=1)
    second["structure_path"] = "/tmp/model.pdb"

    result = merge_candidate_records([first, second])

    assert len(result.candidates) == 2
    assert [candidate["candidate_id"] for candidate in result.candidates] == [
        "cand_first",
        "cand_second",
    ]
    assert result.audit["duplicates_removed"] == 0


def test_private_merge_positions_are_internal_only_by_default() -> None:
    source = _candidate("cand_x", index=0)
    enriched = _candidate("cand_x", index=1)
    enriched["scores"]["sequence_analysis"] = {"gravy": -0.1}

    stripped = merge_candidate_records(
        [source, enriched],
        include_private_positions=True,
    )
    internal = merge_candidate_records(
        [source, enriched],
        include_private_positions=True,
        strip_private_keys=False,
    )

    assert "__candidate_merge_positions" not in stripped.candidates[0]
    assert "__candidate_merge_representative_position" not in stripped.candidates[0]
    assert internal.candidates[0]["__candidate_merge_positions"] == [0, 1]
    assert internal.candidates[0]["__candidate_merge_representative_position"] == 1
