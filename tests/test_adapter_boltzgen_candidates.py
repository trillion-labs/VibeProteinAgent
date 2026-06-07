from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.boltzgen.adapter import (
    BoltzGenDesign,
    BoltzGenInput,
    BoltzGenOutput,
    _boltzgen_candidates,
    _collect_designs,
)
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {
        "schema_version": "candidate_v1",
        "candidate_id": "cand_target",
        "source": {
            "tool": "upstream",
            "node_id": None,
            "job_id": None,
            "local_id": "struct_0000",
            "native_id": None,
            "index": 0,
        },
        "structure_path": "/tmp/target.pdb",
        "scores": {},
        "artifacts": {},
    }


def test_boltzgen_emits_canonical_candidates() -> None:
    params = BoltzGenInput(target_pdb="/tmp/target.pdb")
    design = BoltzGenDesign(
        candidate_id="legacy",
        cif_path="/tmp/d.cif",
        sequence="ACDE",
        length=4,
        score=0.4,
    )
    candidate = _boltzgen_candidates(params, [design])[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "boltzgen"
    assert candidate["source"]["native_id"] == "legacy"
    assert candidate["scores"]["boltzgen"]["score"] == 0.4


def test_boltzgen_candidate_ids_are_deterministic() -> None:
    params = BoltzGenInput(target_pdb="/tmp/target.pdb")
    design = BoltzGenDesign(cif_path="/tmp/d.cif", length=4)
    first = _boltzgen_candidates(params, [design])[0]
    second = _boltzgen_candidates(params, [design])[0]
    assert first["candidate_id"] == second["candidate_id"]


def test_boltzgen_output_preserves_legacy_fields() -> None:
    out = BoltzGenOutput(
        designs=[BoltzGenDesign(cif_path="/tmp/d.cif", length=4)],
        runtime_ms=1,
        num_designs_returned=1,
    )
    assert out.model_dump()["num_designs_returned"] == 1


def test_boltzgen_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert BoltzGenInput(candidates=[_candidate()]).target_pdb == "/tmp/target.pdb"
    assert BoltzGenInput(target_pdb="/tmp/target.pdb").target_pdb == "/tmp/target.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        BoltzGenInput(candidates=[_candidate(), {"target_pdb": "/tmp/legacy.pdb"}])


def _minimal_design_cif(sequence: str) -> str:
    aa3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU"}
    rows = [
        "data_design",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    for index, aa in enumerate(sequence, start=1):
        rows.append(
            f"ATOM CA {aa3[aa]} A {index} {float(index):.1f} 0.0 0.0"
        )
    rows.append("#")
    return "\n".join(rows) + "\n"


def test_boltzgen_recovers_sequence_and_length_from_refolded_cif(tmp_path) -> None:
    refold = tmp_path / "intermediate_designs_inverse_folded" / "refold_cif"
    refold.mkdir(parents=True)
    (refold / "design_00.cif").write_text(_minimal_design_cif("ACDE"))

    designs = _collect_designs(
        output_dir=tmp_path,
        params=BoltzGenInput(target_yaml="/tmp/spec.yaml"),
    )

    assert designs[0].sequence == "ACDE"
    assert designs[0].length == 4


def test_boltzgen_recovered_ids_include_content_digest(tmp_path) -> None:
    first = tmp_path / "a" / "intermediate_designs_inverse_folded" / "refold_cif"
    second = tmp_path / "b" / "intermediate_designs_inverse_folded" / "refold_cif"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "design_00.cif").write_text(_minimal_design_cif("AAAA"))
    (second / "design_00.cif").write_text(_minimal_design_cif("CCCC"))
    params = BoltzGenInput(target_yaml="/tmp/spec.yaml")

    first_candidate = _boltzgen_candidates(
        params,
        _collect_designs(output_dir=tmp_path / "a", params=params),
    )[0]
    second_candidate = _boltzgen_candidates(
        params,
        _collect_designs(output_dir=tmp_path / "b", params=params),
    )[0]

    assert first_candidate["candidate_id"] != second_candidate["candidate_id"]
