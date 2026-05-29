from __future__ import annotations

from vibeproteinagent.adapters._remote.sequence_relevance import (
    intent_quota_top_k,
    sequence_relevance_score,
    top_k_by_sequence_relevance,
)
from vibeproteinagent.adapters.go_sequences.adapter import (
    GOSequencesInput,
    _canonical_candidate_from_sequence as _go_candidate,
    _retrieval_empty_facets as _go_empty_facets,
)
from vibeproteinagent.adapters.interpro_sequences.adapter import (
    InterProSequencesInput,
    _canonical_candidate_from_record as _interpro_candidate,
    _retrieval_empty_facets as _interpro_empty_facets,
)
from vibeproteinagent.adapters._remote.uniprot import UniProtRecord


def test_sequence_relevance_prefers_reviewed_keyword_match() -> None:
    high, high_breakdown = sequence_relevance_score(
        metadata={"proteinDescription": {"recommendedName": {"fullName": {"value": "alpha kinase"}}}},
        reviewed=True,
        length=250,
        intent_query="alpha kinase activity",
    )
    low, low_breakdown = sequence_relevance_score(
        metadata={"proteinDescription": {"recommendedName": {"fullName": {"value": "uncharacterized protein"}}}},
        reviewed=False,
        length=250,
        intent_query="alpha kinase activity",
    )

    assert high > low
    assert high_breakdown["annotation_keyword_score"] > low_breakdown["annotation_keyword_score"]
    assert {
        "term_match_score",
        "annotation_keyword_score",
        "reviewed_swissprot_bonus",
        "evidence_quality_score",
        "diversity_bonus",
        "length_or_fragment_penalty",
        "weighted_components",
    } <= set(high_breakdown)
    assert high_breakdown["reviewed_swissprot_bonus"] == 1.0
    assert high_breakdown["weighted_components"]["reviewed_swissprot_bonus"] == 0.15


def test_sequence_relevance_penalty_uses_plan_weight() -> None:
    base, _ = sequence_relevance_score(
        metadata={"proteinDescription": {"recommendedName": {"fullName": {"value": "alpha kinase"}}}},
        reviewed=False,
        length=250,
        intent_query="alpha kinase",
        term_match_score=1.0,
    )
    penalized, breakdown = sequence_relevance_score(
        metadata={"proteinDescription": {"flag": "Fragment", "recommendedName": {"fullName": {"value": "alpha kinase fragment"}}}},
        reviewed=False,
        length=20,
        intent_query="alpha kinase",
        term_match_score=1.0,
    )

    assert breakdown["length_or_fragment_penalty"] == 1.0
    assert breakdown["weighted_components"]["length_or_fragment_penalty"] == -0.05
    assert base - penalized <= 0.05 + 1e-6


def test_top_k_by_sequence_relevance_bounds_downstream_candidates_stably() -> None:
    rows = [
        {"accession": "low", "sequence_relevance_score": 0.1},
        {"accession": "high", "sequence_relevance_score": 0.9},
        {"accession": "middle", "sequence_relevance_score": 0.5},
    ]

    selected = top_k_by_sequence_relevance(rows, intent_query="query", top_k=2)

    assert [row["accession"] for row in selected] == ["high", "middle"]


def test_intent_quota_top_k_preserves_non_empty_facets() -> None:
    rows = [
        {"accession": "a1", "intent_id": "a", "sequence_relevance_score": 0.9},
        {"accession": "a2", "intent_id": "a", "sequence_relevance_score": 0.8},
        {"accession": "b1", "intent_id": "b", "sequence_relevance_score": 0.7},
    ]

    selected = intent_quota_top_k(rows, max_candidates=2)

    assert {row["intent_id"] for row in selected} == {"a", "b"}


def test_go_candidate_uses_canonical_sequence_relevance_metadata() -> None:
    candidate = _go_candidate(
        {
            "accession": "P12345",
            "sequence": "MKKLL",
            "length": 5,
            "reviewed": True,
            "sequence_relevance_score": 0.87,
            "term_relevance_score": 0.72,
            "score_breakdown": {"annotation_keyword_score": 0.5},
            "intent": {
                "intent_id": "facet_a",
                "term_id": "GO:0000001",
                "label": "alpha activity",
                "selected_reason": "selected for query facet 'alpha activity'",
            },
        },
        0,
    )

    scores = candidate["scores"]["go_sequences"]
    canonical = candidate["scores"]["semantic_relevance"]
    assert scores["sequence_relevance_score"] == 0.87
    assert scores["term_relevance_score"] == 0.72
    assert canonical["sequence_relevance_score"] == 0.87
    assert canonical["term_relevance_score"] == 0.72
    assert scores["semantic_relevance"]["score_breakdown"] == {
        "annotation_keyword_score": 0.5
    }
    assert scores["semantic_relevance"]["selected_reason"].startswith("selected")


def test_empty_facets_include_selection_evidence() -> None:
    go_facets = _go_empty_facets(
        GOSequencesInput(
            go_id="GO:0000001",
            intent_id="facet_a",
            intent_label="alpha activity",
            term_relevance_score=0.9,
            term_score_breakdown={"label": 0.5},
        ),
        [],
    )
    interpro_facets = _interpro_empty_facets(
        InterProSequencesInput(
            interpro_id="IPR000001",
            intent_id="facet_b",
            intent_label="beta domain",
            term_relevance_score=0.8,
            term_score_breakdown={"label": 0.4},
        ),
        [],
    )

    for facet in [*go_facets, *interpro_facets]:
        assert facet["candidate_count"] == 0
        assert facet["retrieval_status"] == "empty"
        assert facet["selected_reason"]
        assert facet["resolver_evidence"]["term_score_breakdown"]


def test_interpro_candidate_mirrors_canonical_semantic_relevance() -> None:
    record = UniProtRecord(
        accession="Q12345",
        sequence="MKKLL",
        metadata={},
        reviewed=False,
        length=5,
        first_public_date=None,
    )
    candidate = _interpro_candidate(
        record,
        0,
        semantic_relevance=0.66,
        semantic_relevance_breakdown={"annotation_keyword_score": 0.4},
        intent={
            "intent_id": "facet_b",
            "term_id": "IPR000001",
            "term_relevance_score": 0.7,
            "selected_reason": "selected term IPR000001",
        },
    )

    canonical = candidate["scores"]["semantic_relevance"]
    assert canonical["sequence_relevance_score"] == 0.66
    assert canonical["term_relevance_score"] == 0.7
    assert canonical["score_breakdown"]["annotation_keyword_score"] == 0.4
