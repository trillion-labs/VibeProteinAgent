"""Unit tests for the candidate-filterable per-tool widget section.

These lock the *display-layer* contract added so the appendix "Per-tool
widgets" section can be filtered to a single candidate's result (mirroring the
candidate review console), plus the chain-extraction fix that lets a binder
complex be detected when its target chain is marked ``HETATM``.

All functions under test are pure helpers in :mod:`reporting.render`; none of
them touch the engine.
"""

from __future__ import annotations

from pathlib import Path

from vibeproteinagent.reporting.models import ReportCandidate, WorkflowNodeSummary
from vibeproteinagent.reporting.render import (
    _extract_chains_from_structure,
    _node_candidate_map,
    _node_details,
    _PerToolCand,
    _score_bars,
    _search_sequence_widget,
    _slice_node_output_for_candidate,
    _structure_analysis_widget,
    _structure_has_plddt,
    _tool_display_name,
    _viewer_block,
)


def _pt(cid, status_label, status_cls, is_final, rank=None, metric=""):
    """Build a per-tool selector entry (mirrors a board card)."""
    return _PerToolCand(
        candidate_id=cid,
        short_id=cid,
        status_label=status_label,
        status_cls=status_cls,
        is_final=is_final,
        rank=rank,
        metric=metric,
    )


def _qc_node(candidates):
    return WorkflowNodeSummary(
        node_id="qc_candidates",
        tool="protein_qc",
        version="1",
        state="completed",
        job_id="job_qc",
        output={
            # Top-level aggregate the old widget used to read exclusively.
            "metrics": {"plddt": 50.0, "gravy": 0.0},
            "flags": [],
            "candidates": candidates,
        },
    )


def _cand(cid, *, plddt, gravy, flags):
    return {
        "candidate_id": cid,
        "sequence": "ACDEFGHIK",
        "structure_path": "",
        "scores": {
            "protein_qc": {
                "plddt": plddt,
                "gravy": gravy,
                "instability_index": 30.0,
                "pi": 7.0,
                "polybasic_hits": 0.0,
                "flags": flags,
                "passes_filters": not flags,
            }
        },
    }


# --------------------------------------------------------------------------- #
# Per-candidate slicing
# --------------------------------------------------------------------------- #


def test_slice_promotes_per_candidate_qc_into_metrics_and_flags():
    cand = _cand("cand_a", plddt=88.0, gravy=-0.4, flags=["low_plddt"])
    node = _qc_node([cand])
    sliced = _slice_node_output_for_candidate(node.output, cand)
    # The candidate's own QC must replace the node-level aggregate so the
    # widget renders this design's numbers, not the batch blob.
    assert sliced["metrics"]["plddt"] == 88.0
    assert sliced["metrics"]["gravy"] == -0.4
    # passes_filters / flags are stripped from metrics; flags promoted separately.
    assert "flags" not in sliced["metrics"]
    assert "passes_filters" not in sliced["metrics"]
    assert sliced["flags"] == ["low_plddt"]


def test_slice_filters_candidate_list_to_the_selected_design():
    a = _cand("cand_a", plddt=88.0, gravy=-0.4, flags=[])
    b = _cand("cand_b", plddt=20.0, gravy=0.9, flags=["low_plddt"])
    node = _qc_node([a, b])
    sliced = _slice_node_output_for_candidate(node.output, b)
    ids = [c["candidate_id"] for c in sliced["candidates"]]
    assert ids == ["cand_b"]


def test_slice_reframes_multifasta_sequence_analysis_to_one_design():
    # The node stores a multi-FASTA *pool aggregate*: top-level sequence_length
    # is the SUM across designs and metrics holds the pooled composition. Slicing
    # to one candidate must show THAT design's residue count + scored
    # descriptors, never the batch sum.
    node = WorkflowNodeSummary(
        node_id="seqana",
        tool="sequence_analysis",
        version="1",
        state="completed",
        job_id="job_sa",
        output={
            "sequence_length": 12859,
            "metrics": {
                "analysis_mode": "multi_fasta",
                "length": 12859,
                "pI": 6.21,
                "gravy": 0.06,
                "amino_acid_counts": {"A": 1476, "L": 1505},
            },
            "representative_sequences": [{"id": "x"}, {"id": "y"}],
            "liabilities": [{"t": "a"}, {"t": "b"}, {"t": "c"}],
            "candidates": [
                {
                    "candidate_id": "cand_a",
                    "sequence": "ACDEFGHIK",  # 9 residues
                    "scores": {
                        "sequence_analysis": {
                            "pI": 5.57,
                            "gravy": -0.1,
                            "net_charge": -3.0,
                            "liability_count": 1,
                            "liability_types": ["deamidation"],
                        }
                    },
                }
            ],
        },
    )
    cand = node.output["candidates"][0]
    sliced = _slice_node_output_for_candidate(node.output, cand)
    # Length is this design's own residue count, not the 12,859 pool sum.
    assert sliced["sequence_length"] == 9
    assert sliced["metrics"]["length"] == 9
    # Re-framed as a single protein so the widget drops multi-FASTA aggregation.
    assert sliced["metrics"]["analysis_mode"] == "single"
    # This design's own scored descriptors win over the pooled aggregate.
    assert sliced["metrics"]["pI"] == 5.57
    assert sliced["metrics"]["gravy"] == -0.1
    # Pool-level composition + representative pool are dropped, liabilities are
    # this design's own.
    assert "amino_acid_counts" not in sliced["metrics"]
    assert "representative_sequences" not in sliced
    assert sliced["liabilities"] == ["deamidation"]


def test_node_candidate_map_keys_by_candidate_id():
    a = _cand("cand_a", plddt=88.0, gravy=-0.4, flags=[])
    b = _cand("cand_b", plddt=20.0, gravy=0.9, flags=[])
    node = _qc_node([a, b])
    cmap = _node_candidate_map(node.output)
    assert set(cmap) == {"cand_a", "cand_b"}


def test_slice_swaps_in_per_candidate_structure_analysis_artifacts():
    # The structure_analysis widget loads ``artifacts.summary_json`` (which
    # carries that design's burial / interface / liabilities / secondary
    # structure) and the per-residue TSVs. Slicing to one candidate must point
    # those at the candidate's OWN bundle, not the node's shared aggregate, and
    # must drop the node-level liabilities/SS/ligand aggregates so they can't
    # leak into the per-candidate panel.
    node_arts = {
        "summary_json": "/jobs/shared/summary.json",
        "residue_annotations_tsv": "/jobs/shared/res.tsv",
    }
    cand_arts = {
        "summary_json": "/jobs/cand_a/summary.json",
        "residue_annotations_tsv": "/jobs/cand_a/res.tsv",
    }
    cand = {
        "candidate_id": "cand_a",
        "sequence": "ACDEFGHIK",
        "structure_path": "/jobs/cand_a/model.pdb",
        "artifacts": {"structure_analysis": cand_arts, "pxdesign": {"x": "y"}},
        "scores": {
            "structure_analysis": {
                "burial": {"buried": 7, "exposed": 37, "intermediate": 16},
                "interface": {"ligand_contact_count": 4},
                "site_contacts": {"contact_count": 2, "mapped_position_count": 3},
                "secondary_structure": {"helix": 49, "sheet": 0, "loop": 11},
                "ligand_contacts": {"contact_count": 4},
                "ligand_pocket": {"ligand_buried_fraction": 0.5},
                "liability_count": 2,
            }
        },
    }
    node = WorkflowNodeSummary(
        node_id="analyze_candidates",
        tool="structure_analysis",
        version="1",
        state="completed",
        job_id="job_sa",
        output={
            "artifacts": node_arts,
            "burial": {"buried": 0, "exposed": 0, "intermediate": 0},
            "interface": {"ligand_contact_count": 0},
            "liabilities": [{"type": "node_level_leak"}],
            "secondary_structure": {"helix": 1, "sheet": 1, "loop": 1},
            "ligand_contacts": {"contact_count": 0},
            "ligand_pocket": {"ligand_buried_fraction": 0.0},
            "candidates": [cand],
        },
    )
    sliced = _slice_node_output_for_candidate(node.output, cand)
    # The candidate's own analysis bundle replaces the shared one.
    assert sliced["artifacts"] == cand_arts
    assert sliced["artifacts"]["summary_json"] == "/jobs/cand_a/summary.json"
    # Per-candidate burial/interface are promoted (fallback when summary_json
    # can't be loaded).
    assert sliced["burial"]["buried"] == 7
    assert sliced["interface"]["ligand_contact_count"] == 4
    assert sliced["site_contacts"]["contact_count"] == 2
    assert sliced["ligand_pocket"]["ligand_buried_fraction"] == 0.5
    # The node-level aggregates must NOT leak into this candidate's panel.
    assert "liabilities" not in sliced
    assert "secondary_structure" not in sliced
    assert "ligand_contacts" not in sliced


def test_slice_keeps_node_artifacts_when_candidate_has_none():
    # A candidate without its own structure_analysis bundle keeps the node-level
    # artifacts (safe fallback), and burial/interface still come from its scores.
    node_arts = {"summary_json": "/jobs/shared/summary.json"}
    cand = {
        "candidate_id": "cand_a",
        "scores": {
            "structure_analysis": {
                "burial": {"buried": 5, "exposed": 5, "intermediate": 5},
                "interface": {"ligand_contact_count": 1},
            }
        },
    }
    node = WorkflowNodeSummary(
        node_id="analyze",
        tool="structure_analysis",
        version="1",
        state="completed",
        job_id="job_sa",
        output={"artifacts": node_arts, "candidates": [cand]},
    )
    sliced = _slice_node_output_for_candidate(node.output, cand)
    assert sliced["artifacts"] == node_arts
    assert sliced["burial"]["buried"] == 5


def test_structure_analysis_widget_uses_corrected_ligand_contact_residue_count():
    html = _structure_analysis_widget(
        {
            "artifacts": {},
            "interface": {"ligand_contact_count": 1},
            "site_contacts": {"contact_count": 2, "mapped_position_count": 3},
            "ligand_pocket": {"ligand_buried_fraction": 0.5},
        }
    )

    assert "Ligand contacts" in html
    assert "Site contacts" in html
    assert "Ligand buried" in html
    assert ">1</text>" in html
    assert ">2</text>" in html
    assert ">50</text>" in html


# --------------------------------------------------------------------------- #
# Section assembly: selector + per-candidate variants
# --------------------------------------------------------------------------- #


def test_node_details_emits_sample_selector_and_per_candidate_variants():
    a = _cand("cand_a", plddt=88.0, gravy=-0.4, flags=[])
    b = _cand("cand_b", plddt=20.0, gravy=0.9, flags=["low_plddt"])
    node = _qc_node([a, b])
    cand_order = [
        _pt("cand_a", "Final", "final", True, rank=1),
        _pt("cand_b", "Ranked", "ranked", False, rank=2),
    ]
    html = _node_details([node], cand_order)
    # A selector with one card per sample, final pick first + active.
    assert 'class="node-cand-toolbar"' in html
    assert html.count("node-cand-btn") >= 2
    assert 'data-pt-cand="cand_a"' in html
    assert 'data-pt-cand="cand_b"' in html
    # An evaluator (protein_qc) lives in the per-design band scoped by the selector.
    assert "nodeband-perdesign" in html
    # One variant per candidate inside the block, the final pick active first.
    assert 'data-cand="cand_a"' in html
    assert 'data-cand="cand_b"' in html
    assert "node-cand-variant is-active" in html
    # Per-candidate numbers actually differ between variants.
    assert "88" in html and "20" in html


def test_node_details_single_candidate_renders_no_selector():
    a = _cand("cand_a", plddt=88.0, gravy=-0.4, flags=[])
    node = _qc_node([a])
    html = _node_details([node], [_pt("cand_a", "Final", "final", True, rank=1)])
    # A one-sample run needs no switch — and no banding (flat stream).
    assert 'class="node-cand-toolbar"' not in html
    assert "nodeband-perdesign" not in html
    # The single evaluator panel is still sliced to that design (its own value).
    assert 'data-cand="cand_a"' in html


def test_node_details_marks_samples_a_node_did_not_process():
    # Node only processed cand_a; the selector also lists cand_b.
    a = _cand("cand_a", plddt=88.0, gravy=-0.4, flags=[])
    node = _qc_node([a])
    cand_order = [
        _pt("cand_a", "Final", "final", True, rank=1),
        _pt("cand_b", "Ranked", "ranked", False, rank=2),
    ]
    html = _node_details([node], cand_order)
    assert 'data-cand="cand_b"' in html
    assert "node-cand-na-note" in html


def test_node_details_no_candidate_scope_uses_shared_variant():
    # A retrieval-style node with no candidate list stays visible for all.
    node = WorkflowNodeSummary(
        node_id="search_sequence",
        tool="search_sequence",
        version="1",
        state="completed",
        job_id="job_s",
        output={"hits": [{"id": "P12345"}]},
    )
    cand_order = [
        _pt("cand_a", "Final", "final", True, rank=1),
        _pt("cand_b", "Ranked", "ranked", False, rank=2),
    ]
    html = _node_details([node], cand_order)
    assert 'data-cand="__shared__"' in html
    # A generator/retrieval node is a cohort result → all-designs band, with a
    # "shows all designs" badge, not the per-design selector slicing.
    assert "nodeband-cohort" in html
    assert "node-block-cohort" in html


# --------------------------------------------------------------------------- #
# Chain extraction keeps a HETATM-marked target chain (binder complex)
# --------------------------------------------------------------------------- #


_PDB_BINDER_COMPLEX = """\
ATOM      1  CA  MET B   1      0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA B   2      1.000   0.000   0.000  1.00  0.00           C
HETATM    3  CA  GLY A   1      5.000   0.000   0.000  1.00  0.00           C
HETATM    4  CA  LEU A   2      6.000   0.000   0.000  1.00  0.00           C
HETATM    5  CA  HOH W   1      9.000   0.000   0.000  1.00  0.00           O
HETATM    6 ZN    ZN A 100     10.000   0.000   0.000  1.00  0.00          ZN
END
"""


def test_extract_chains_keeps_hetatm_marked_standard_aa_chain(tmp_path: Path):
    # Target chain A is flagged HETATM (pxdesign / Protenix non-polymer); it
    # must still be returned so the binder complex is detected.
    p = tmp_path / "complex.pdb"
    p.write_text(_PDB_BINDER_COMPLEX)
    chains = dict(_extract_chains_from_structure(p))
    assert chains.get("B") == "MA"          # binder (ATOM)
    assert chains.get("A") == "GL"          # target (HETATM standard AA kept)
    # Water and the zinc ion are genuine heteroatoms — excluded.
    assert "W" not in chains


# --------------------------------------------------------------------------- #
# Q4: pLDDT availability detection (B-factor column)
# --------------------------------------------------------------------------- #


_PDB_WITH_PLDDT = """\
ATOM      1  N   MET A   1      0.000   0.000   0.000  1.00 87.21           N
ATOM      2  CA  MET A   1      1.000   0.000   0.000  1.00 91.40           C
ATOM      3  CA  ALA A   2      2.000   0.000   0.000  1.00 88.05           C
END
"""

# Coordinate-only generator output: B-factor column flat zero.
_PDB_NO_PLDDT = """\
ATOM      1  N   MET A   1      0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  MET A   1      1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   2      2.000   0.000   0.000  1.00  0.00           C
END
"""


def test_structure_has_plddt_true_for_real_bfactors():
    assert _structure_has_plddt(_PDB_WITH_PLDDT, "pdb") is True


def test_structure_has_plddt_false_for_flat_zero_column():
    # PXDesign / RFdiffusion coordinate-only output: no usable pLDDT.
    assert _structure_has_plddt(_PDB_NO_PLDDT, "pdb") is False


def test_structure_has_plddt_falls_back_true_on_unparseable():
    # An empty / non-structure blob must not disable the control.
    assert _structure_has_plddt("", "pdb") is True


def test_viewer_note_surfaces_qc_mean_when_no_per_residue_plddt(tmp_path: Path):
    # Coordinate-only structure (flat-zero B-factor) but the candidate still
    # carries a chain-level QC mean pLDDT. The viewer must explain that the
    # mean exists while per-residue colouring is impossible, so "mean pLDDT
    # shown but pLDDT view disabled" doesn't read as a bug.
    p = tmp_path / "design.pdb"
    p.write_text(_PDB_NO_PLDDT)
    cand = ReportCandidate(
        candidate_id="cand_x",
        sequence="MA",
        structure_path=str(p),
        scores={"plddt": 11.81485294117647, "score": -0.18},
    )
    html = _viewer_block(cand)
    assert 'value="plddt" disabled' in html
    assert "chain-level QC mean" in html
    assert "&asymp;12" in html  # 11.81 -> unit 0.118 -> 12


def test_viewer_note_falls_back_to_generator_text_without_qc_mean(tmp_path: Path):
    # No scalar pLDDT anywhere: keep the original coordinate-only explanation
    # (and no fabricated mean).
    p = tmp_path / "design.pdb"
    p.write_text(_PDB_NO_PLDDT)
    cand = ReportCandidate(
        candidate_id="cand_y",
        sequence="MA",
        structure_path=str(p),
        scores={"score": -0.18},
    )
    html = _viewer_block(cand)
    assert "coordinate-only generator" in html
    assert "chain-level QC mean" not in html


def test_score_bars_render_from_nested_review_bundle_candidate():
    # Review bundles carry archive-nested scores (``{tool: {metric: val}}``)
    # because review.py needs the nesting to group evidence. The Quality
    # analysis tab feeds that same candidate to ``_score_bars``, which keys on
    # flat ``tool:metric`` strings — so the bars must not silently go blank on
    # the nested shape (regression: the strip vanished after the dedup/flatten
    # split). ``_flatten_scores`` is idempotent, so nested and flat candidates
    # describing the same metrics must produce identical bar HTML.
    nested = ReportCandidate(
        candidate_id="design_1",
        score=-0.18,
        scores={
            "protein_qc": {"gravy": -0.4, "instability_index": 32.1, "plddt": 0.88},
            "boltz2_affinity": {"iptm": 0.83},
        },
    )
    flat = ReportCandidate(
        candidate_id="design_1",
        score=-0.18,
        scores={
            "protein_qc:gravy": -0.4,
            "protein_qc:instability_index": 32.1,
            "protein_qc:plddt": 0.88,
            "boltz2_affinity:iptm": 0.83,
        },
    )
    nested_html = _score_bars(nested)
    assert nested_html  # the strip is populated, not blank
    assert "gravy" in nested_html.lower()
    assert nested_html == _score_bars(flat)


def test_structure_has_plddt_reads_cif_b_iso_column():
    cif = (
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.occupancy\n"
        "_atom_site.B_iso_or_equiv\n"
        "ATOM 1 CA MET A 1 0.0 0.0 0.0 1.0 0.00\n"
        "ATOM 2 CA ALA A 2 1.0 0.0 0.0 1.0 0.00\n"
    )
    assert _structure_has_plddt(cif, "cif") is False


# --------------------------------------------------------------------------- #
# Q2-1: friendly per-tool block titles
# --------------------------------------------------------------------------- #


def test_tool_display_name_maps_known_tools():
    assert _tool_display_name("protein_qc") == "Protein QC"
    assert _tool_display_name("search_sequence") == "Sequence retrieval"
    assert _tool_display_name("rank_designs") == "Design ranking"


def test_tool_display_name_prettifies_unknown_tool():
    # Unknown tool ids still read cleanly instead of as snake_case.
    assert _tool_display_name("some_new_tool") == "Some New Tool"
    assert _tool_display_name(None) == "Tool"


def test_node_details_leads_with_friendly_tool_name_not_node_id():
    # The codex planner names nodes opaquely ("nadphx_sequence_qc"); the block
    # must lead with the tool's friendly name, keeping the node id only as muted
    # meta so the appendix is readable.
    node = WorkflowNodeSummary(
        node_id="nadphx_sequence_qc",
        tool="protein_qc",
        version="1",
        state="completed",
        job_id="job_qc",
        output={"metrics": {"plddt": 50.0}, "flags": []},
    )
    html = _node_details([node], [])
    assert 'class="node-block-title">Protein QC<' in html
    # The opaque planner id is demoted to muted meta, not the headline.
    assert "node-block-meta" in html
    assert "nadphx_sequence_qc" in html


# --------------------------------------------------------------------------- #
# Q1 + Q3: sequence retrieval widget — relevance bars + per-source yield
# --------------------------------------------------------------------------- #


def _search_output():
    return {
        "query": "NAD(P)HX repair proteins",
        "retrieval_status": "partial",
        "retrieval_sources": ["go"],
        "resolved_terms": {
            "go_terms": [
                {
                    "term_id": "GO:0052855",
                    "label": "ADP-dependent NAD(P)H-hydrate dehydratase activity",
                    "term_relevance_score": 0.473,
                    "selected_reason": "exact_label_match",
                }
            ],
            "interpro_terms": [],
        },
        "source_summaries": {
            "go": {"candidate_count": 34, "status": "completed", "term_count": 1},
            "interpro": {"candidate_count": 0, "status": "no_terms", "term_count": 0},
        },
        "empty_facets": [
            {
                "stage": "interpro_sequences",
                "selected_reason": "resolver did not select any INTERPRO term",
                "resolver_evidence": {"errors": ["interpro: returned invalid JSON"]},
            }
        ],
        "summary": {"downstream_candidate_count": 34},
    }


def test_search_widget_renders_relevance_bar_for_resolved_term():
    html = _search_sequence_widget(_search_output())
    # Q1: a relevance-score bar for the resolved GO term, with the score shown.
    assert "seqret-relbar-fill" in html
    assert "GO:0052855" in html
    assert "0.47" in html


def test_search_widget_renders_per_source_yield_bars():
    html = _search_sequence_widget(_search_output())
    # Q3: a per-source yield chart, GO contributing 34 and InterPro 0.
    assert "seqret-yield" in html
    assert "34" in html
    # The empty InterPro stream is marked, not hidden.
    assert "is-empty" in html


def test_search_widget_drops_raw_facet_garbage():
    html = _search_sequence_widget(_search_output())
    # Q3: the old widget dumped the raw intent label + stage as a bullet
    # ("... interpro_sequences"); that garbage must be gone.
    assert "interpro_sequences" not in html
    # Replaced by a concise unresolved-source note that surfaces the reason.
    assert "seqret-unresolved" in html
    assert "INTERPRO" in html
