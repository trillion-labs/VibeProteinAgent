from __future__ import annotations

import pytest

from vibeproteinagent.agents.planning.candidate import stable_candidate_id


def test_stable_candidate_id_is_deterministic() -> None:
    first = stable_candidate_id("go_sequences", "seq_0000", sequence="ACDE")
    second = stable_candidate_id("go_sequences", "seq_0000", sequence="ACDE")
    assert first == second
    assert first.startswith("cand_")


def test_different_producer_tool_changes_id() -> None:
    a = stable_candidate_id("go_sequences", "seq_0000", sequence="ACDE")
    b = stable_candidate_id("sequence_analysis", "seq_0000", sequence="ACDE")
    assert a != b


def test_different_local_id_changes_id() -> None:
    a = stable_candidate_id("go_sequences", "seq_0000", sequence="ACDE")
    b = stable_candidate_id("go_sequences", "seq_0001", sequence="ACDE")
    assert a != b


def test_different_sequence_changes_id() -> None:
    a = stable_candidate_id("go_sequences", "seq_0000", sequence="ACDE")
    b = stable_candidate_id("go_sequences", "seq_0000", sequence="WXYZ")
    assert a != b


def test_de_novo_candidate_without_native_id_or_structure_is_stable() -> None:
    first = stable_candidate_id(
        "rfdiffusion",
        "design_0004",
        producer_node="generate",
        sequence="ACDEFGHIK",
    )
    second = stable_candidate_id(
        "rfdiffusion",
        "design_0004",
        producer_node="generate",
        sequence="ACDEFGHIK",
    )
    assert first == second


@pytest.mark.parametrize("producer_tool,local_id", [("", "seq_0000"), ("go_sequences", "")])
def test_missing_required_identity_fields_raise(
    producer_tool: str,
    local_id: str,
) -> None:
    with pytest.raises(ValueError):
        stable_candidate_id(producer_tool, local_id)


def test_structure_path_uses_basename_not_absolute_developer_path() -> None:
    a = stable_candidate_id("boltzgen", "design_0000", structure_path="/home/foo/x.pdb")
    b = stable_candidate_id("boltzgen", "design_0000", structure_path="/home/bar/x.pdb")
    assert a == b
