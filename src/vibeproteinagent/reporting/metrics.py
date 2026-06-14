"""Shared metric + tool metadata used by the report renderer and plot builder.

Lives in its own module so we have a single source of truth for "what does
this metric / tool name mean and how do we display it." Before this
module existed the same data lived in private dicts on both ``render.py``
and ``plots.py``, and adding a new tool required updating both — a
recipe for drift between the candidate-card tooltip and the
score-distribution plot caption that surfaced the same number.

Three concerns live here:

1. **Metric registry**: per-metric label, explainer, and gauge range,
   keyed by canonical lowercase metric name (``plddt``, ``iptm``,
   ``seq_recovery``, ...).
2. **Tool registry**: per-tool pretty name, output kind, and the
   tool-specific explanation of the ``score`` scalar (which collides
   across tools — inverse-folding NLL vs. ranking confidence vs.
   composite reward).
3. **Recovery-swap policy**: which inverse-folding tools should plot
   ``seq_recovery`` instead of the raw NLL ``score`` in the
   distribution chart, plus the matching captions.

Helpers:

* :func:`metric_explainer` — best human-readable explanation for a
  ``(metric, tool)`` pair, preferring the per-tool override.
* :func:`pretty_metric_name` — canonical lowercase key to a display
  label, peeling unit suffixes and respecting science-token casing.
* :func:`pretty_tool_name` — lowercase adapter id to a display name.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- metric labels
#
# Report surfaces often compare metrics by human labels rather than raw adapter
# keys. Those labels drift: objective_rank may emit ``cx ipTM min``, a predictor
# may emit ``ipTM``, and a rationale may refer to ``complex ipTM``. Keep the
# canonical matching rules here so summary cards, matrices, charts, and future
# report widgets do not each grow one-off alias tables.


def metric_token(label: object) -> str:
    """Normalize a metric label/key into a stable lowercase token."""

    text = str(label or "").strip()
    text = text.replace("Δ", "delta").replace("δ", "delta")
    text = text.replace("Å", "a").replace("å", "a")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


_METRIC_CANONICAL_KEYS: dict[str, str] = {
    # Structure confidence.
    "plddt": "plddt",
    "mean_plddt": "plddt",
    "ca_bfactor_plddt": "plddt",
    "monomer_plddt": "plddt",
    "mono_plddt": "plddt",
    "complex_plddt": "plddt",
    "cx_plddt": "plddt",
    "ptm": "ptm",
    "mean_ptm": "ptm",
    "pae": "pae",
    "mean_pae": "pae",
    "rank_score": "ranking_score",
    "ranking_score": "ranking_score",
    "confidence": "confidence",
    "confidence_score": "confidence",
    "model_confidence": "confidence",
    "gpde": "gpde",
    "lddt": "lddt",
    "interface_lddt": "interface_lddt",
    "per_interface_lddt": "interface_lddt",
    # Interface and binding.
    "iptm": "iptm",
    "cx_iptm": "iptm",
    "cx_iptm_min": "iptm",
    "cx_iptm_mean": "iptm",
    "cx_iptm_max": "iptm",
    "complex_iptm": "iptm",
    "complex_iptm_min": "iptm",
    "complex_iptm_mean": "iptm",
    "complex_iptm_max": "iptm",
    "ipsae": "ipsae",
    "ipsae_score": "ipsae",
    "ipsae_max": "ipsae",
    "ipae": "ipae",
    "ipae_interaction": "interaction_pae",
    "pae_interaction": "interaction_pae",
    "interaction_pae": "interaction_pae",
    "iplddt": "iplddt",
    "dockq": "dockq",
    "binding_affinity": "delta_g",
    "delta_g": "delta_g",
    "dg": "delta_g",
    "affinity": "affinity",
    "predicted_affinity": "affinity",
    "affinity_pred_value": "affinity",
    "binder_probability": "binder_probability",
    "affinity_probability": "binder_probability",
    "affinity_probability_binary": "binder_probability",
    "ligand_buried_fraction": "ligand_buried_fraction",
    "ligand_buried_atom_count": "ligand_buried_atom_count",
    "contact_residue_count": "ligand_contact_residues",
    "ligand_contact_residues": "ligand_contact_residues",
    "contact_atom_pair_count": "ligand_atom_contacts",
    "ligand_atom_contacts": "ligand_atom_contacts",
    "contact_ligand_atom_count": "contacted_ligand_atoms",
    "contacted_ligand_atoms": "contacted_ligand_atoms",
    "ligand_heavy_atom_count": "ligand_heavy_atoms",
    "pocket_residue_count": "pocket_residues",
    "ligand_contacts": "ligand_contacts",
    "interface_contacts": "interface_contacts",
    "n_contact_pairs": "interface_contacts",
    "target_residues": "target_residues",
    "n_target_res": "target_residues",
    "liability_count": "liability_count",
    # Sequence/developability.
    "length": "length",
    "sequence_length": "length",
    "gravy": "gravy",
    "pi": "pi",
    "p_i": "pi",
    "isoelectric_point": "pi",
    "net_charge": "net_charge",
    "charge": "net_charge",
    "instability": "instability",
    "instability_index": "instability",
    "aromatic_fraction": "aromatic_fraction",
    # Stability/scoring.
    "rosetta_total_score": "rosetta_total_score",
    "total_score": "rosetta_total_score",
    "mean_total_score": "rosetta_total_score",
    "ddg": "ddg",
    "ddg_reu": "ddg",
    "mean_ddg": "mean_ddg",
    "delta_delta_g": "ddg",
    "seq_recovery": "seq_recovery",
    "sequence_recovery": "seq_recovery",
    "mpnn_score": "mpnn_score",
    # Rank/novelty.
    "objective_score": "objective_score",
    "objective_rank": "objective_rank",
    "rank": "objective_rank",
    "window_identity": "window_identity",
    "win_id": "window_identity",
    "msa_mean_identity": "msa_mean_identity",
    "mean_identity": "msa_mean_identity",
    # Misc.
    "qc": "qc",
    "generation_score": "generation_score",
    "refold_rmsd": "refold_rmsd",
    "binder_refold_rmsd": "refold_rmsd",
    "ligand_rmsd": "ligand_rmsd",
    "motif_rmsd": "motif_rmsd",
    "designability": "designability",
}


_METRIC_DISPLAY_LABELS: dict[str, str] = {
    "plddt": "pLDDT",
    "ptm": "pTM",
    "pae": "PAE",
    "ranking_score": "Rank score",
    "confidence": "Confidence",
    "gpde": "gPDE",
    "lddt": "lDDT",
    "interface_lddt": "Interface lDDT",
    "iptm": "ipTM",
    "ipsae": "ipSAE",
    "ipae": "ipAE",
    "interaction_pae": "Interaction PAE",
    "iplddt": "ipLDDT",
    "dockq": "DockQ",
    "delta_g": "Delta G",
    "affinity": "Affinity",
    "binder_probability": "Binder probability",
    "ligand_buried_fraction": "Ligand buried fraction",
    "ligand_buried_atom_count": "Ligand buried atoms",
    "ligand_contact_residues": "Ligand contact residues",
    "ligand_atom_contacts": "Ligand atom contacts",
    "contacted_ligand_atoms": "Contacted ligand atoms",
    "ligand_heavy_atoms": "Ligand heavy atoms",
    "pocket_residues": "Pocket residues",
    "ligand_contacts": "Ligand contacts",
    "interface_contacts": "Interface contacts",
    "target_residues": "Target residues",
    "liability_count": "Liability count",
    "length": "Length",
    "gravy": "GRAVY",
    "pi": "pI",
    "net_charge": "Net charge",
    "instability": "Instability",
    "aromatic_fraction": "Aromatic fraction",
    "rosetta_total_score": "Rosetta total score",
    "ddg": "ddG",
    "mean_ddg": "Mean ddG",
    "seq_recovery": "Seq recovery",
    "mpnn_score": "MPNN score",
    "objective_score": "Objective score",
    "objective_rank": "Objective rank",
    "window_identity": "Window identity",
    "msa_mean_identity": "MSA mean identity",
    "qc": "QC",
    "generation_score": "Generation score",
    "refold_rmsd": "Refold RMSD",
    "ligand_rmsd": "Ligand RMSD",
    "motif_rmsd": "Motif RMSD",
    "designability": "Designability",
}


_VARIANT_PRIORITY: dict[str, dict[str, int]] = {
    # Binder summaries should prefer robust multi-seed aggregates over a single
    # predictor row when the query asks for generic ipTM.
    "iptm": {
        "cx_iptm_min": 0,
        "complex_iptm_min": 0,
        "cx_iptm": 5,
        "complex_iptm": 5,
        "iptm": 10,
        "cx_iptm_mean": 20,
        "complex_iptm_mean": 20,
        "cx_iptm_max": 30,
        "complex_iptm_max": 30,
    },
    # In binder reports, raw pLDDT support rows usually describe the selected
    # complex structure, while monomer pLDDT is a foldability support signal.
    "plddt": {
        "complex_plddt": 0,
        "cx_plddt": 0,
        "plddt": 10,
        "mean_plddt": 10,
        "monomer_plddt": 30,
        "mono_plddt": 30,
        "ca_bfactor_plddt": 40,
    },
    "ipsae": {"ipsae": 0, "ipsae_score": 5, "ipsae_max": 10},
    "affinity": {"predicted_affinity": 0, "affinity_pred_value": 0, "affinity": 10},
    "binder_probability": {
        "binder_probability": 0,
        "affinity_probability_binary": 5,
        "affinity_probability": 10,
    },
    "pae": {"mean_pae": 0, "pae": 10},
    "interaction_pae": {"interaction_pae": 0, "pae_interaction": 0, "ipae_interaction": 10},
    "instability": {"instability": 0, "instability_index": 0},
    "pi": {"pi": 0, "p_i": 0, "isoelectric_point": 5},
    "interface_contacts": {"interface_contacts": 0, "n_contact_pairs": 5},
    "target_residues": {"target_residues": 0, "n_target_res": 5},
    "ligand_contact_residues": {"ligand_contact_residues": 0, "contact_residue_count": 5},
    "ligand_atom_contacts": {"ligand_atom_contacts": 0, "contact_atom_pair_count": 5},
    "contacted_ligand_atoms": {"contacted_ligand_atoms": 0, "contact_ligand_atom_count": 5},
    "pocket_residues": {"pocket_residues": 0, "pocket_residue_count": 5},
}


def canonical_metric_key(label: object) -> str:
    """Return the canonical comparison key for a metric label/key."""

    token = metric_token(label)
    if token in _METRIC_CANONICAL_KEYS:
        return _METRIC_CANONICAL_KEYS[token]
    if token.endswith("_score"):
        base = token[: -len("_score")]
        if base in _METRIC_CANONICAL_KEYS:
            return _METRIC_CANONICAL_KEYS[base]
    return token


def display_metric_label(label: object) -> str:
    """Human display label for a metric, after canonical alias resolution."""

    key = canonical_metric_key(label)
    if key in _METRIC_DISPLAY_LABELS:
        return _METRIC_DISPLAY_LABELS[key]
    text = str(label or "").strip()
    if text == "ΔG":
        return "Delta G"
    return text


def metric_variant_rank(label: object, target_label: object | None = None) -> tuple[int, str]:
    """Sort key for choosing a recorded variant for a requested metric label."""

    token = metric_token(label)
    key = canonical_metric_key(target_label if target_label is not None else label)
    priorities = _VARIANT_PRIORITY.get(key, {})
    return priorities.get(token, 100), token


def metric_matches(label: object, target_label: object) -> bool:
    return canonical_metric_key(label) == canonical_metric_key(target_label)


def metric_variant_title(label: object, *, workflow_type: str | None = None) -> str:
    """Specific title for a recorded metric variant, preserving useful context."""

    token = metric_token(label)
    key = canonical_metric_key(label)
    if workflow_type == "binder":
        if token == "monomer_plddt" or token == "mono_plddt":
            return "Monomer pLDDT"
        if token in {"complex_plddt", "cx_plddt", "plddt", "mean_plddt"}:
            return "Complex pLDDT"
        if token in {"cx_iptm_min", "complex_iptm_min"}:
            return "Complex ipTM min"
        if token in {"cx_iptm_mean", "complex_iptm_mean"}:
            return "Complex ipTM mean"
        if token in {"cx_iptm", "complex_iptm"}:
            return "Complex ipTM"
    if token in {"cx_iptm_min", "complex_iptm_min"}:
        return "ipTM min"
    if token in {"cx_iptm_mean", "complex_iptm_mean"}:
        return "ipTM mean"
    if key == "plddt" and token in {"monomer_plddt", "mono_plddt"}:
        return "Monomer pLDDT"
    return display_metric_label(label)

# --------------------------------------------------------------------------- metrics
#
# Single source of truth for metric semantics. Adapters and the
# candidate-card / qc-card / score-bar renderers all read from this map so
# adding a new metric is a one-line registry update - no need to teach
# the renderer about each tool individually.
#
# Each entry: ``key -> (display_name, explainer, gauge_lo, gauge_hi)``
METRIC_REGISTRY: dict[str, tuple[str, str, float, float]] = {
    "plddt": (
        "Mean pLDDT",
        "Per-residue confidence averaged across the chain (0-100). "
        ">=70 = confident, >=90 = very high.",
        0.0, 100.0,
    ),
    "mean_plddt": (
        "Mean pLDDT",
        "Per-residue confidence averaged across the chain (0-100). "
        ">=70 = confident, >=90 = very high.",
        0.0, 100.0,
    ),
    "ptm": (
        "pTM",
        "Predicted TM-score (0-1). >0.5 implies a likely-correct overall fold.",
        0.0, 1.0,
    ),
    "iptm": (
        "ipTM",
        "Interface pTM (0-1). >0.6 generally trustworthy for complex / binder interfaces.",
        0.0, 1.0,
    ),
    "pae": (
        "PAE",
        "Predicted aligned error in angstroms. Lower is better; "
        "<5 indicates well-resolved geometry.",
        0.0, 30.0,
    ),
    "mean_pae": (
        "PAE",
        "Predicted aligned error in angstroms. Lower is better; "
        "<5 indicates well-resolved geometry.",
        0.0, 30.0,
    ),
    "dockq": (
        "DockQ",
        "Interface accuracy score (0-1). >0.23 acceptable, >0.49 medium, >0.80 high.",
        0.0, 1.0,
    ),
    "ipsae": (
        "ipSAE",
        "Interface predicted symmetry-aware error. Lower is better.",
        0.0, 30.0,
    ),
    "ranking_confidence": (
        "Ranking confidence",
        "Composite head used to rank candidate structures. Scale matches the predictor.",
        0.0, 1.0,
    ),
    "ranking_score": (
        "Ranking score",
        "Pipeline-specific aggregate score used to order candidates.",
        0.0, 1.0,
    ),
    "score": (
        "Score",
        "Generic composite score reported by the tool.",
        0.0, 1.0,
    ),
    "delta_g": (
        "ΔG (kcal/mol)",
        "Predicted binding free energy. More negative = stronger binding.",
        -16.0, 0.0,
    ),
    "dg": (
        "ΔG (kcal/mol)",
        "Predicted binding free energy. More negative = stronger binding.",
        -16.0, 0.0,
    ),
    "affinity": (
        "Affinity",
        "Predicted binding strength on the tool's native scale.",
        0.0, 12.0,
    ),
    "affinity_pred_value": (
        "Affinity (pred)",
        "Boltz-2 predicted affinity value (-log10 IC50). Higher = tighter.",
        0.0, 12.0,
    ),
    "affinity_probability": (
        "Binder probability",
        "Calibrated probability that the candidate binds (0-1).",
        0.0, 1.0,
    ),
    "affinity_probability_binary": (
        "Binder probability",
        "Boltz-2 binary binder probability (0-1).",
        0.0, 1.0,
    ),
    "confidence_score": (
        "Model confidence",
        "Boltz-2 composite confidence head (0-1).",
        0.0, 1.0,
    ),
    "iplddt": (
        "ipLDDT",
        "Interface pLDDT — per-residue confidence at the binding interface.",
        0.0, 100.0,
    ),
    "recovery": (
        "Recovery",
        "Fraction of native residues recovered by the design (0-1).",
        0.0, 1.0,
    ),
    "seq_recovery": (
        "Seq recovery",
        "Fraction of native residues recovered by the designed sequence (0-1).",
        0.0, 1.0,
    ),
    "instability_index": (
        "Instability index",
        "Guruprasad heuristic. <40 = predicted stable in vitro.",
        0.0, 80.0,
    ),
    "gravy": (
        "GRAVY",
        "Grand average of hydropathy (Kyte-Doolittle). Negative = hydrophilic / likely soluble.",
        -2.0, 2.0,
    ),
    "pi": (
        "Isoelectric point",
        "pH at net-zero charge. Typical soluble proteins span 4-9.",
        0.0, 14.0,
    ),
    "polybasic_hits": (
        "Polybasic motifs",
        "Count of K/R-rich runs that drive non-specific membrane / RNA binding.",
        0.0, 10.0,
    ),
    "aromatic_fraction": (
        "Aromatic fraction",
        "Fraction of F/Y/W residues. Relevant for UV absorbance / aggregation propensity.",
        0.0, 0.25,
    ),
    "total_score": (
        "Rosetta total score",
        "Rosetta score-function total energy in REU. Lower / more negative is better "
        "within a comparable scoring protocol.",
        -1000.0, 100.0,
    ),
    "ddg": (
        "ddG (REU)",
        "Rosetta energy difference versus the reference design. Negative values are "
        "stabilizing under the recorded protocol.",
        -100.0, 20.0,
    ),
    "ddg_reu": (
        "ddG (REU)",
        "Rosetta energy difference versus the reference design. Negative values are "
        "stabilizing under the recorded protocol.",
        -100.0, 20.0,
    ),
    "mean_ddg": (
        "Mean ddG (REU)",
        "Mean Rosetta ddG across repeated seeds. Negative values are stabilizing under "
        "the recorded protocol.",
        -100.0, 20.0,
    ),
    "delta_delta_g": (
        "ddG (REU)",
        "Rosetta energy difference versus the reference design. Negative values are "
        "stabilizing under the recorded protocol.",
        -100.0, 20.0,
    ),
    "rmsd": (
        "RMSD (A)",
        "Root-mean-square deviation over aligned atoms. Lower = closer "
        "structural match; <2 A is a tight fit.",
        0.0, 10.0,
    ),
}


# Metrics that conventionally arrive on a 0-1 scale but may also appear as
# 0-100 (pLDDT is the canonical example). Used by the gauge-rendering code
# to autoscale the donut so it fills correctly either way.
DUAL_SCALE_METRICS: frozenset[str] = frozenset({
    "plddt", "mean_plddt", "iptm", "ptm", "recovery", "seq_recovery",
})


# Per-(tool, metric) explainer overrides. The bare ``METRIC_REGISTRY``
# entry covers the canonical meaning; this layer carries the *tool-specific*
# definition when the same metric name means different things across tools.
# Keys are ``(tool_name_lowercase, canonical_metric)``; values replace the
# registry explainer when both the tool and metric match.
#
# Scope: every tool whose data routes into ``candidate.score`` gets an
# entry for ``"score"`` here so the bar tooltip is unambiguous. Tools that
# don't populate candidate.score (bindcraft, boltz2_affinity, rfdiffusion,
    # embedding tools, ...) are deliberately absent - they'd never trigger.
# Promotion path of ``ranking_score`` -> ``score`` happens in
# ``collect._candidates_from_output``, which is why ``protenix_v2`` has an
# entry despite emitting ``ranking_score``.
TOOL_METRIC_EXPLAINERS: dict[tuple[str, str], str] = {
    # --- Inverse-folding / sequence-design family (NLL, lower = better) ---
    # The bar width visualised against a 0-1 axis is a layout aid, NOT a
    # confidence - these are negative-log-likelihoods of the sampled
    # sequence under the model.
    ("proteinmpnn", "score"): (
        "ProteinMPNN per-residue negative log-likelihood of the sampled "
        "sequence under the inverse-folding model - lower = better "
        "(higher native-likely probability). Typical good designs land "
        "around 0.7-1.2."
    ),
    ("ligandmpnn", "score"): (
        "LigandMPNN per-residue negative log-likelihood of the sampled "
        "sequence under the ligand-conditioned inverse-folding model - "
        "lower = better (higher native-likely probability given the "
        "ligand context). Typical good designs land around 0.7-1.2."
    ),

    # --- Structure-prediction / ranking-score family (0-1, higher = better) ---
    ("protenix_v2", "score"): (
        "Protenix-v2 ranking_score (0-1) - composite confidence head used "
        "to order predicted complexes. Higher = more confident. Promoted "
        "into candidate.score from the ranking_score field."
    ),
    ("pxdesign", "score"): (
        "PXDesign composite score (0-1) - in infer mode this is the "
        "Protenix ranking_score (fallback iptm/plddt); in pipeline mode "
        "the AF2 iptm of the design (fallback plddt/ptm/ranking_score). "
        "Higher = better in both modes."
    ),

    # --- Binder-design family ---
    ("boltzgen", "score"): (
        "BoltzGen final-set Boltz ranking score (0-1) - composite "
        "confidence + structural quality after the design + filter "
        "cascade. Higher = better."
    ),
    ("proteina_complexa", "score"): (
        "Proteina-Complexa search-cascade reward - composite of "
        "fold quality + interface metrics (or raw composite ``score`` "
        "when reward isn't populated). Higher = better."
    ),

    # --- Objective-aware reranker ---
    ("objective_rank", "objective_score"): (
        "objective_rank objective_score evaluates the agent-supplied "
        "score_formula after applying explicit objective gates. Direction "
        "is recorded in the ranking output."
    ),
    ("objective_rank", "score"): (
        "objective_rank score mirrors objective_score for generic candidate "
        "resolution. Direction is recorded in the ranking output."
    ),
}


def metric_explainer(canonical: str, tool: str = "") -> str:
    """Return the best human-readable explanation for ``(tool, metric)``.

    Prefers the per-tool override in :data:`TOOL_METRIC_EXPLAINERS`, then
    falls back to the generic explainer from :data:`METRIC_REGISTRY`.
    Returns an empty string when neither is registered so callers can omit
    the explainer rather than show a vacuous tooltip.
    """
    tool_norm = (tool or "").strip().lower()
    if tool_norm:
        override = TOOL_METRIC_EXPLAINERS.get((tool_norm, canonical))
        if override:
            return override
    spec = METRIC_REGISTRY.get(canonical)
    if spec is not None:
        return spec[1]
    return ""


# --------------------------------------------------------------------------- pretty metric name
#
# Science tokens that should keep a fixed casing when a snake_case metric
# key is expanded into a human label (so ``tm_score`` -> "TM score", not
# "Tm score"). Anything not listed is sentence-cased normally.
_METRIC_ABBR: dict[str, str] = {
    "tm": "TM", "rmsd": "RMSD", "plddt": "pLDDT", "iplddt": "ipLDDT",
    "ptm": "pTM", "iptm": "ipTM", "pae": "PAE", "lddt": "lDDT",
    "ipsae": "ipSAE", "dockq": "DockQ", "kd": "Kd", "nm": "nM",
    "pi": "pI", "ic50": "IC50", "qc": "QC", "ss": "SS", "id": "ID",
    "uniprot": "UniProt", "dna": "DNA", "rna": "RNA", "cif": "CIF",
    "pdb": "PDB", "msa": "MSA", "a": "A", "b": "B", "ab": "AB", "ba": "BA",
}

# Trailing unit suffixes pulled out of a metric key and rendered as a
# parenthesised unit ("best_affinity_kcal_mol" -> "Best affinity (kcal/mol)").
# Longest suffix first so "_kcal_mol" wins over a bare "_mol".
_METRIC_UNIT_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_kcal_mol", "kcal/mol"),
    ("_nm", "nM"),
    ("_um", "µM"),
)


def pretty_metric_name(key: str) -> str:
    """Convert a canonical metric key into a display-friendly label.

    Hard-coded short labels win first; then unit suffixes are peeled and
    parenthesised; then snake_case is turned into "Sentence case" with
    known science abbreviations preserved (``tm_score`` → ``"TM score"``).
    """
    known = {
        "plddt": "pLDDT",
        "mean_plddt": "pLDDT",
        "iptm": "ipTM",
        "ptm": "pTM",
        "pae": "PAE",
        "mean_pae": "PAE",
        "ipsae": "ipSAE",
        "dockq": "DockQ",
        "delta_g": "ΔG",
        "seq_recovery": "Seq recov.",
        "recovery": "Recovery",
        "score": "Score",
        "ranking_score": "Rank score",
    }
    if key in known:
        return known[key]
    # Peel a trailing unit suffix and render it parenthesised on the base name.
    for suffix, unit in _METRIC_UNIT_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix):
            return f"{pretty_metric_name(key[: -len(suffix)])} ({unit})"
    # snake_case / kebab-case -> "Sentence case", upper-casing known science
    # abbreviations so auto-generated metric labels read cleanly.
    tokens = [t for t in key.replace("-", "_").split("_") if t]
    if not tokens:
        return key
    words: list[str] = []
    for idx, tok in enumerate(tokens):
        low = tok.lower()
        if low in _METRIC_ABBR:
            words.append(_METRIC_ABBR[low])
        elif idx == 0:
            words.append(tok[:1].upper() + tok[1:].lower())
        else:
            words.append(tok.lower())
    return " ".join(words)


# --------------------------------------------------------------------------- tools
#
# Pretty display name per adapter / tool id. Adapter directory names use
# all-lowercase compound style (``ligandmpnn``, ``boltz2_affinity``) which
# reads badly in plot titles. This map handles the cases the report has
# seen; unknown tools fall through to a Title-cased best effort in
# :func:`pretty_tool_name`.
TOOL_PRETTY_NAMES: dict[str, str] = {
    "proteinmpnn": "ProteinMPNN",
    "ligandmpnn": "LigandMPNN",
    "soluble_mpnn": "SolubleMPNN",
    "bindcraft": "BindCraft",
    "boltzgen": "BoltzGen",
    "boltz2": "Boltz-2",
    "boltz2_affinity": "Boltz-2 affinity",
    "protenix": "Protenix",
    "protenix_v2": "Protenix v2",
    "proteina_complexa": "Proteina-Complexa",
    "rfdiffusion": "RFdiffusion",
    "alphafold": "AlphaFold",
    "alphafold2": "AlphaFold2",
    "alphafold3": "AlphaFold3",
    "esmfold": "ESMFold",
    "esm2": "ESM2",
    "esm2_embedding": "ESM2 embedding",
    "esm3": "ESM3",
    "esm3_tiny": "ESM3 (tiny)",
    "pxdesign": "PXDesign",
    "template_select": "Template selector",
}


# What kind of artifact / output category the tool produces. Drives the
# plot subtitle so a reader sees "sequence design" / "structure prediction"
# next to the tool name and knows what universe the score lives in.
# Conservative — only includes tools whose category is unambiguous.
TOOL_OUTPUT_KIND: dict[str, str] = {
    "proteinmpnn": "sequence design",
    "ligandmpnn": "sequence design (ligand-aware)",
    "soluble_mpnn": "sequence design (solubility-biased)",
    "bindcraft": "binder design (target + binder complex)",
    "boltzgen": "binder design (target + binder complex)",
    "proteina_complexa": "binder design (target + binder complex)",
    "rfdiffusion": "backbone scaffolding",
    "boltz2": "structure prediction",
    "boltz2_affinity": "structure + affinity prediction",
    "protenix": "structure prediction",
    "protenix_v2": "structure prediction",
    "alphafold": "structure prediction",
    "alphafold2": "structure prediction",
    "alphafold3": "structure prediction",
    "esmfold": "structure prediction",
    "esm2_embedding": "sequence embedding",
    "esm3_tiny": "sequence language model",
    "pxdesign": "binder design",
    "template_select": "template retrieval",
}


# Per-tool short definition of the ``candidate.score`` scalar plotted in
# the score-distribution chart. Each entry is the one-line "what is this
# number?" caption that lands in the plot subtitle. Direction (lower vs
# higher = better) is part of the caption because it's not visually
# obvious from the bar chart.
#
# Mirrors :data:`TOOL_METRIC_EXPLAINERS` in long form. Only tools whose
# value reaches ``candidate.score`` belong here. For tools that swap to
# recovery via :data:`TOOLS_PREFER_RECOVERY`, the entry here is the
# *fallback* caption used only when recovery is unavailable; the recovery
# caption lives in :data:`RECOVERY_DEFINITIONS`.
SCORE_DEFINITIONS: dict[str, str] = {
    # Inverse-folding family (NLL fallback when seq_recovery missing).
    "proteinmpnn": "per-residue NLL under the inverse-folding model; lower = better",
    "ligandmpnn": (
        "per-residue NLL under the ligand-conditioned inverse-folding "
        "model; lower = better"
    ),

    # Structure-prediction / ranking-score family.
    "protenix_v2": "Protenix ranking_score (0-1); higher = better",
    "pxdesign": (
        "Protenix ranking_score (infer mode) or AF2 iptm of the design "
        "(pipeline mode); 0-1; higher = better"
    ),

    # Binder-design family.
    "boltzgen": (
        "Boltz ranking_score after design + filter cascade (0-1); higher = better"
    ),
    "proteina_complexa": "search-cascade reward; higher = better",

    # Objective-aware reranker.
    "objective_rank": (
        "agent-supplied gates and score_formula; direction recorded in output"
    ),
}


# Inverse-folding tools where the canonical ``candidate.score`` is a
    # per-residue NLL - a value that is (a) on an unfamiliar scale, (b)
# direction-flipped (lower = better), and (c) bar-charted misleadingly
# (longer bar visually reads as "better" but is actually worse). For these
# tools the score-distribution plot swaps the y-axis to ``seq_recovery``
    # - the fraction of native residues kept by the design - which is the
# metric a sequence-design user actually cares about, on a 0-1 scale
# where higher = better and longer bar = better.
TOOLS_PREFER_RECOVERY: frozenset[str] = frozenset({
    "proteinmpnn", "ligandmpnn",
})


# Recovery captions used when the swap (above) actually fired - i.e. when
# we found a real ``seq_recovery`` value to plot. When recovery is
# missing we fall through to the NLL path and use SCORE_DEFINITIONS
# instead, so the caption never lies about which axis the bars show.
RECOVERY_DEFINITIONS: dict[str, str] = {
    "proteinmpnn": (
        "fraction of native residues recovered by the designed sequence "
        "(0-1); higher = better"
    ),
    "ligandmpnn": (
        "fraction of native residues recovered by the ligand-aware "
        "designed sequence (0-1); higher = better"
    ),
}


# Title suffix used when the recovery swap fires. Plain "score
# distribution" otherwise.
RECOVERY_PLOT_TITLE_SUFFIX = "sequence recovery distribution"


def pretty_tool_name(tool_lc: str) -> str:
    """Map a lowercase adapter / tool id to a display-friendly name.

    Known tools come from :data:`TOOL_PRETTY_NAMES`; everything else
    falls back to a Title-cased best effort so an unknown tool still
    reads cleanly in a plot title.
    """
    if not tool_lc:
        return ""
    if tool_lc in TOOL_PRETTY_NAMES:
        return TOOL_PRETTY_NAMES[tool_lc]
    # Best-effort fallback: underscore to space, then title-case.
    return tool_lc.replace("_", " ").title()


__all__ = [
    "DUAL_SCALE_METRICS",
    "METRIC_REGISTRY",
    "RECOVERY_DEFINITIONS",
    "RECOVERY_PLOT_TITLE_SUFFIX",
    "SCORE_DEFINITIONS",
    "TOOLS_PREFER_RECOVERY",
    "TOOL_METRIC_EXPLAINERS",
    "TOOL_OUTPUT_KIND",
    "TOOL_PRETTY_NAMES",
    "metric_explainer",
    "pretty_metric_name",
    "pretty_tool_name",
]
