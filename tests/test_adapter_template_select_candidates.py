from __future__ import annotations

import pytest
from pydantic import ValidationError

from vibeproteinagent.adapters.template_select.adapter import TemplateSelectInput, TemplateSelectOutput, _template_select_candidates
from vibeproteinagent.candidates.schema import validate_candidate_v1


def _candidate() -> dict:
    return {"schema_version": "candidate_v1", "candidate_id": "cand_seq", "source": {"tool": "upstream", "node_id": None, "job_id": None, "local_id": "seq_0000", "native_id": None, "index": 0}, "sequence": "ACDE", "structure_path": "/tmp/a.pdb", "scores": {}, "artifacts": {}}


def test_template_select_emits_canonical_candidates() -> None:
    row = {"template_id": "1abc_A", "identity": 0.8, "coverage": 0.9, "evalue": 1e-6}
    candidate = _template_select_candidates([row], "/tmp/templates.tsv")[0]
    validate_candidate_v1(candidate)
    assert candidate["source"]["tool"] == "template_select"
    assert candidate["source"]["native_id"] == "1abc_A"
    assert candidate["scores"]["template_select"]["identity"] == 0.8


def test_template_select_candidate_ids_are_deterministic() -> None:
    row = {"template_id": "1abc_A", "identity": 0.8}
    assert _template_select_candidates([row], "/tmp/templates.tsv")[0]["candidate_id"] == _template_select_candidates([row], "/tmp/templates.tsv")[0]["candidate_id"]


def test_template_select_output_preserves_legacy_fields() -> None:
    out = TemplateSelectOutput(query_kind="sequence", engine="local", template_count=0, templates_json="/tmp/t.json", templates_tsv="/tmp/t.tsv", top_templates=[], artifacts={}, runtime_ms=1)
    assert out.model_dump()["top_templates"] == []


def test_template_select_consumes_canonical_legacy_and_rejects_mixed() -> None:
    assert TemplateSelectInput(candidates=[_candidate()], template_db="/tmp/db").query_sequence == "ACDE"
    assert TemplateSelectInput(query_sequence="ACDE", template_db="/tmp/db").query_sequence == "ACDE"
    assert TemplateSelectInput(candidates=[_candidate()], template_db="/tmp/db", template_db_kind="foldseek").query_structure == "/tmp/a.pdb"
    with pytest.raises(ValidationError, match="Mixed canonical and legacy"):
        TemplateSelectInput(candidates=[_candidate(), {"query_sequence": "AAAA"}], template_db="/tmp/db")
