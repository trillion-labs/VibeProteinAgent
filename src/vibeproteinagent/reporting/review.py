"""Candidate review console: derive a candidate-centric presentation model.

``build_review_model`` is a pure function of an already-collected
:class:`~vibeproteinagent.reporting.models.ReportModel`. It re-frames the report
as a *candidate review surface* — a ranked board of candidates, each backed by
an evidence dossier (status, pipeline path, score families, structure, MSA).

Design rules locked with the team:

* The workflow DAG is a **shared** pipeline every candidate traverses, so node
  topology is workflow-level truth. Per-candidate attribution onto a node is
  often partial in the recorded data, so each overlay atom carries a
  :class:`Provenance` tag and may be ``NOT_RECORDED`` — the UI never fabricates.
* Tool is **not** primary navigation. Per-candidate evidence comes from
  ``candidate.scores`` (already keyed by tool), regrouped into researcher-facing
  metric *families*, not tool dumps.
* ``grouping_policy="raw"`` keeps one bundle per recorded candidate (MVP). A
  future ``"design_family"`` can roll shards of one design together without
  changing this contract.

This module is part of the report rendering layer only; it never touches the
execution engine.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from .collect import _as_float, _as_int, _string_list
from .identity import (
    candidate_manifest_identity as _candidate_manifest_identity,
)
from .identity import (
    candidate_material_key as _candidate_material_key,
)
from .identity import (
    candidate_ranked_candidate_id as _candidate_ranked_candidate_id,
)
from .identity import (
    candidate_sequence_identity as _candidate_sequence_identity,
)
from .identity import (
    candidate_structure_identity as _candidate_structure_identity,
)
from .identity import (
    row_material_key as _row_material_key,
)
from .identity import (
    row_ranked_candidate_id as _row_ranked_candidate_id,
)
from .identity import (
    strong_candidate_manifest_identity as _strong_candidate_manifest_identity,
)
from .identity import (
    submitted_material_key as _submitted_material_key,
)
from .identity import (
    submitted_ranked_candidate_id as _submitted_ranked_candidate_id,
)
from .models import (
    BoardGroup,
    CandidateEvidenceBundle,
    CandidateStatus,
    EvidencePath,
    EvidenceStep,
    MSABlock,
    Provenance,
    ReportCandidate,
    ReportModel,
    ReportReviewModel,
    ScoreGroup,
    ToolRole,
    WorkflowNodeSummary,
    WorkflowType,
)

# --------------------------------------------------------------------------- #
# Tool classification (report-layer heuristic; unknown -> OTHER, never mis-roled)
# --------------------------------------------------------------------------- #

_GENERATOR_TOOLS = frozenset(
    {
        "rfdiffusion",
        "rfdiffusion3",
        "rfpeptides",
        "pxdesign",
        "bindcraft",
        "proteinmpnn",
        "ligandmpnn",
        "boltzgen",
        "sequence_edit",
        "proteina_complexa",
        # retrieval generators (produce candidate sequences)
        "search_sequence",
        "go_sequences",
        "interpro_sequences",
    }
)

_EVALUATOR_TOOLS = frozenset(
    {
        "esm3",
        "esm2",
        "boltz2_affinity",
        "protenix_v2",
        "structure_analysis",
        "sequence_analysis",
        "protein_qc",
        "pxmeter",
        "ipsae",
        "prodigy",
        "tmalign",
        "vina",
    }
)

_RANKER_TOOLS = frozenset({"objective_rank"})
_SUBMISSION_TOOLS = frozenset({"submit_candidate"})
_PLUMBING_TOOLS = frozenset(
    {
        "merge_candidates",
        "mpnn_to_protenix_input",
        "sequence_to_protenix_input",
        "noop",
        "mmseqs2_gpu",
        "foldseek",
        "template_select",
    }
)

# Structure-producing generators — used only for workflow-type detection.
_STRUCTURE_GENERATOR_TOOLS = frozenset(
    {
        "rfdiffusion",
        "rfdiffusion3",
        "rfpeptides",
        "pxdesign",
        "bindcraft",
        "boltzgen",
        "proteina_complexa",
    }
)
_BINDER_GENERATOR_TOOLS = frozenset({"pxdesign", "bindcraft", "boltzgen"})
_BACKBONE_STAGE_GENERATOR_TOOLS = frozenset({"rfdiffusion", "rfdiffusion3", "rfpeptides"})
_BINDER_STAGE_GENERATOR_TOOLS = frozenset({"pxdesign", "bindcraft", "boltzgen"})
_SEQUENCE_STAGE_GENERATOR_TOOLS = frozenset(
    {
        "esm2",
        "proteinmpnn",
        "ligandmpnn",
        "search_sequence",
        "go_sequences",
        "interpro_sequences",
    }
)
_MODED_BINDER_TOOLS = {"proteina_complexa": frozenset({"protein_binder", "ligand_binder"})}
_INTERFACE_EVALUATOR_TOOLS = frozenset({"boltz2_affinity", "ipsae", "prodigy", "pxmeter"})
_RETRIEVAL_TOOLS = frozenset({"search_sequence", "go_sequences", "interpro_sequences"})
_STRUCTURE_PATH_KEYS: tuple[str, ...] = (
    "structure_path",
    "pdb_path",
    "cif_path",
    "predicted_pdb",
    "predicted_cif",
)

# Evaluator tools whose presence in candidate.scores counts as real evaluation
# evidence (drives RANKED_ALTERNATIVE vs INCOMPLETE_EVIDENCE).
_EVIDENCE_EVALUATORS = _EVALUATOR_TOOLS


def classify_tool_role(tool: str | None) -> str:
    """Map a tool name to its pipeline role (string constant)."""
    name = (tool or "").strip().lower()
    if not name:
        return ToolRole.OTHER
    if name in _RANKER_TOOLS:
        return ToolRole.RANKER
    if name in _SUBMISSION_TOOLS:
        return ToolRole.SUBMISSION
    if name in _PLUMBING_TOOLS:
        return ToolRole.PLUMBING
    if name in _EVALUATOR_TOOLS:
        return ToolRole.EVALUATOR
    if name in _GENERATOR_TOOLS:
        return ToolRole.GENERATOR
    return ToolRole.OTHER


def classify_generation_stage_key(node: WorkflowNodeSummary) -> str | None:
    """Return the product-stage key for generator-like tools, if known."""
    tool = (node.tool or "").strip().lower()
    if tool == "esm3":
        return _esm3_stage_key(node)
    if tool in _BACKBONE_STAGE_GENERATOR_TOOLS:
        return "backbone_generation"
    if tool in _BINDER_STAGE_GENERATOR_TOOLS:
        return "binder_generation"
    if tool == "proteina_complexa":
        allowed_modes = _MODED_BINDER_TOOLS.get(tool, frozenset())
        if _node_mode(node) in allowed_modes:
            return "binder_generation"
        return "backbone_generation"
    if tool in _SEQUENCE_STAGE_GENERATOR_TOOLS:
        return "generation"
    return None


def classify_workflow_stage_key(
    node: WorkflowNodeSummary,
    *,
    include_intent: bool = False,
) -> str:
    """Map a recorded tool call to the report workflow stage it contributed to."""
    tool = (node.tool or "").strip().lower()
    if include_intent and tool in {"agent_intent", "user_request"}:
        return "intent"
    explicit = {
        "sequence_edit": "engineering",
        "protein_qc": "qc",
        "sequence_analysis": "qc",
        "protenix_v2": "structure",
        "structure_analysis": "structure",
        "tmalign": "structure_comparison",
        "pyrosetta": "energy",
    }
    if tool in explicit:
        return explicit[tool]
    generator_stage = classify_generation_stage_key(node)
    if generator_stage:
        return generator_stage
    role = classify_tool_role(tool)
    if role == ToolRole.RANKER:
        return "ranking"
    if role == ToolRole.SUBMISSION:
        return "recommendation"
    if role == ToolRole.EVALUATOR:
        if tool in {"boltz2_affinity", "pxmeter", "ipsae", "prodigy", "vina"}:
            return "interface"
        return "other"
    if role == ToolRole.GENERATOR:
        return "generation"
    return "other"


def _esm3_stage_key(node: WorkflowNodeSummary) -> str:
    if _esm3_has_structure_output(node):
        return "backbone_generation"
    tracks = _esm3_tracks(node)
    if tracks:
        return "backbone_generation" if "structure" in tracks else "generation"
    return "backbone_generation"


def _esm3_tracks(node: WorkflowNodeSummary) -> set[str]:
    for container in (node.params, node.output):
        if not isinstance(container, dict):
            continue
        tracks = container.get("tracks")
        if isinstance(tracks, str):
            return {tracks.strip().lower()}
        if isinstance(tracks, (list, tuple, set)):
            return {str(track).strip().lower() for track in tracks if str(track).strip()}
    return set()


def _esm3_has_structure_output(node: WorkflowNodeSummary) -> bool:
    output = node.output if isinstance(node.output, dict) else {}
    if _nonempty_text(output.get("predicted_pdb")):
        return True
    confidence = output.get("confidence_scores")
    if isinstance(confidence, dict) and any(
        confidence.get(key) is not None for key in ("mean_plddt", "plddt", "ptm")
    ):
        return True
    candidates = output.get("candidates")
    if isinstance(candidates, list):
        return any(_candidate_has_structure_output(candidate) for candidate in candidates)
    return False


def _candidate_has_structure_output(candidate: Any) -> bool:
    if not isinstance(candidate, dict):
        return False
    if any(
        _nonempty_text(candidate.get(key))
        for key in ("structure_path", "predicted_pdb", "predicted_cif", "pdb_path", "cif_path")
    ):
        return True
    capabilities = candidate.get("capabilities")
    if isinstance(capabilities, dict) and any(
        bool(capabilities.get(key)) for key in ("has_backbone_structure", "has_full_structure")
    ):
        return True
    scores = candidate.get("scores")
    esm3_scores = scores.get("esm3") if isinstance(scores, dict) else None
    return isinstance(esm3_scores, dict) and bool(esm3_scores.get("structure_predicted"))


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def detect_workflow_type(model: ReportModel) -> str:
    """Infer the coarse workflow shape from the tool set + structure presence."""
    tools = {(node.tool or "").strip().lower() for node in model.graph.nodes}
    has_structures = any((c.structure_path or "").strip() for c in model.candidates) or any(
        c.structure_path for c in model.final_candidates
    )
    if _has_binder_workflow_signal(model, tools):
        return WorkflowType.BINDER
    if (
        (tools & _RETRIEVAL_TOOLS)
        and not (tools & _STRUCTURE_GENERATOR_TOOLS)
        and not has_structures
    ):
        return WorkflowType.RETRIEVAL
    if has_structures or (tools & _STRUCTURE_GENERATOR_TOOLS):
        return WorkflowType.DENOVO
    return WorkflowType.GENERIC


def _has_binder_workflow_signal(model: ReportModel, tools: set[str]) -> bool:
    if tools & _BINDER_GENERATOR_TOOLS:
        return True
    if tools & _INTERFACE_EVALUATOR_TOOLS:
        return True
    for node in model.graph.nodes:
        tool = (node.tool or "").strip().lower()
        allowed_modes = _MODED_BINDER_TOOLS.get(tool)
        if allowed_modes and _node_mode(node) in allowed_modes:
            return True
    return False


def _node_mode(node: WorkflowNodeSummary) -> str:
    for container in (node.output, node.params):
        if isinstance(container, dict):
            mode = container.get("mode")
            if mode not in (None, ""):
                return str(mode).strip().lower()
    return ""


# --------------------------------------------------------------------------- #
# Small display helpers
# --------------------------------------------------------------------------- #


def _plddt_100(value: Any) -> float | None:
    """Normalise a pLDDT-family value to the [0, 100] scale for display.

    The recorded data mixes [0, 1] and [0, 100] conventions; a value <= 1.0 is
    treated as a fraction and scaled up.
    """
    num = _as_float(value)
    if num is None:
        return None
    if 0.0 <= num <= 1.0:
        return round(num * 100.0, 1)
    return round(num, 1)


def _plddt_disp(value: Any) -> str | None:
    """Normalised pLDDT as a one-decimal display string (keeps the 0..100 scale
    visually distinct from a 0..1 fraction)."""
    num = _plddt_100(value)
    return None if num is None else f"{num:.1f}"


def _fmt(value: Any, digits: int = 2) -> str | None:
    num = _as_float(value)
    if num is None:
        return None
    if abs(num - round(num)) < 1e-9:
        return str(round(num))
    return f"{num:.{digits}f}"


def _fraction_pct_disp(value: Any) -> str | None:
    num = _as_float(value)
    if num is None:
        return None
    return _fmt(num * 100.0 if 0.0 <= num <= 1.0 else num)


def _max_numeric_from_rows(value: Any, key: str = "score") -> str | None:
    values: list[float] = []
    if isinstance(value, list):
        for row in value:
            if isinstance(row, dict):
                num = _as_float(row.get(key))
            else:
                num = _as_float(row)
            if num is not None:
                values.append(num)
    elif isinstance(value, dict):
        for item in value.values():
            num = _as_float(item)
            if num is not None:
                values.append(num)
    else:
        num = _as_float(value)
        if num is not None:
            values.append(num)
    if not values:
        return None
    return _fmt(max(values))


def _dockq_disp(value: Any) -> str | None:
    return _max_numeric_from_rows(value, key="score")


def _first_recorded(*values: Any) -> Any | None:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _degap(seq: str) -> str:
    return "".join(ch for ch in seq if ch.isalpha()).upper()


def _topo_order(nodes: list[WorkflowNodeSummary]) -> list[WorkflowNodeSummary]:
    """Kahn topological sort over node.upstream; stable, cycle-safe."""
    by_id = {n.node_id: n for n in nodes}
    indeg: dict[str, int] = {n.node_id: 0 for n in nodes}
    for n in nodes:
        for up in n.upstream:
            if up in by_id:
                indeg[n.node_id] += 1
    order: list[WorkflowNodeSummary] = []
    ready = [n for n in nodes if indeg[n.node_id] == 0]
    seen: set[str] = set()
    while ready:
        node = ready.pop(0)
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        order.append(node)
        for other in nodes:
            if node.node_id in other.upstream and other.node_id not in seen:
                indeg[other.node_id] -= 1
                if indeg[other.node_id] <= 0:
                    ready.append(other)
    # Append any nodes left out by cycles, preserving input order.
    for n in nodes:
        if n.node_id not in seen:
            order.append(n)
    return order


# --------------------------------------------------------------------------- #
# Score families (metric-family view, not a tool dump)
# --------------------------------------------------------------------------- #

# (tool, metric_key, label, unit, transform)
_FAMILY_SPECS: list[tuple[str, list[tuple[str, str, str, str | None, Any]]]] = [
    (
        "Structure confidence",
        [
            ("val", "mono_plddt", "Monomer pLDDT", None, _plddt_disp),
            ("esm3", "mean_plddt", "pLDDT", None, _plddt_disp),
            ("esm3", "ptm", "pTM", None, _fmt),
            ("boltz2_affinity", "plddt", "pLDDT", None, _plddt_disp),
            ("boltz2_affinity", "ptm", "pTM", None, _fmt),
            ("boltz2_affinity", "pae", "PAE", "Å", _fmt),
            ("boltz2_affinity", "mean_pae", "PAE", "Å", _fmt),
            ("boltz2_affinity", "confidence_score", "Confidence", None, _fmt),
            ("bindcraft", "plddt", "pLDDT", None, _plddt_disp),
            ("boltzgen", "confidence", "Confidence", None, _fmt),
            ("boltzgen", "score", "Generation score", None, _fmt),
            ("pxdesign", "score", "Generation score", None, _fmt),
            ("proteina_complexa", "plddt", "pLDDT", None, _plddt_disp),
            ("proteina_complexa", "scaffold_plddt", "Scaffold pLDDT", None, _plddt_disp),
            ("proteina_complexa", "binder_refold_rmsd", "Refold RMSD", "Å", _fmt),
            ("proteina_complexa", "ligand_rmsd", "Ligand RMSD", "Å", _fmt),
            ("proteina_complexa", "motif_rmsd", "Motif RMSD", "Å", _fmt),
            ("proteina_complexa", "designability", "Designability", None, _fmt),
            ("protenix_v2", "plddt", "pLDDT", None, _plddt_disp),
            ("protenix_v2", "ptm", "pTM", None, _fmt),
            ("protenix_v2", "pae", "PAE", "Å", _fmt),
            ("protenix_v2", "mean_pae", "PAE", "Å", _fmt),
            ("protenix_v2", "ranking_score", "Rank score", None, _fmt),
            ("protenix_v2", "gpde", "gPDE", None, _fmt),
            ("pxmeter", "lddt", "lDDT", None, _fmt),
            ("pxmeter", "per_interface_lddt", "Interface lDDT", None, _max_numeric_from_rows),
            ("protein_qc", "plddt", "pLDDT", None, _plddt_disp),
            ("protein_qc", "ca_bfactor_plddt", "CA B-factor pLDDT", None, _plddt_disp),
        ],
    ),
    (
        "Interface & binding",
        [
            ("val", "ipsae", "ipSAE", None, _fmt),
            ("val", "cx_iptm_min", "cx ipTM min", None, _fmt),
            ("val", "cx_iptm_mean", "cx ipTM mean", None, _fmt),
            ("val", "n_contact_pairs", "Interface contacts", None, _fmt),
            ("val", "n_target_res", "Target residues", None, _fmt),
            ("boltz2_affinity", "iptm", "ipTM", None, _fmt),
            ("boltz2_affinity", "predicted_affinity", "Affinity", None, _fmt),
            ("boltz2_affinity", "affinity", "Affinity", None, _fmt),
            ("boltz2_affinity", "affinity_pred_value", "Affinity", None, _fmt),
            (
                "boltz2_affinity",
                "affinity_probability_binary",
                "Binder probability",
                None,
                _fmt,
            ),
            ("boltz2_affinity", "pae_interaction", "Interaction PAE", "Å", _fmt),
            ("boltz2_affinity", "iplddt", "ipLDDT", None, _fmt),
            ("bindcraft", "iptm", "ipTM", None, _fmt),
            ("bindcraft", "pae_interaction", "Interaction PAE", "Å", _fmt),
            ("boltzgen", "affinity_pred_value", "Affinity", None, _fmt),
            ("boltzgen", "affinity_probability_binary", "Binder probability", None, _fmt),
            ("proteina_complexa", "iptm", "ipTM", None, _fmt),
            ("proteina_complexa", "ligand_contacts", "Ligand contacts", None, _fmt),
            ("protenix_v2", "iptm", "ipTM", None, _fmt),
            ("protenix_v2", "ipae", "ipAE", "Å", _fmt),
            ("protenix_v2", "iplddt", "ipLDDT", None, _fmt),
            ("ipsae", "ipsae_score", "ipSAE", None, _fmt),
            ("ipsae", "ipsae_max", "ipSAE", None, _fmt),
            ("ipsae", "ipsae", "ipSAE", None, _fmt),
            ("pxmeter", "dockq", "DockQ", None, _dockq_disp),
            ("prodigy", "binding_affinity", "ΔG", "kcal/mol", _fmt),
            ("structure_analysis", "interface_contacts", "Interface contacts", None, _fmt),
            (
                "structure_analysis",
                "ligand_pocket.ligand_buried_fraction",
                "Ligand buried fraction",
                "%",
                _fraction_pct_disp,
            ),
            (
                "structure_analysis",
                "ligand_contacts.contact_residue_count",
                "Ligand contact residues",
                None,
                _fmt,
            ),
            (
                "structure_analysis",
                "ligand_contacts.contact_atom_pair_count",
                "Ligand atom contacts",
                None,
                _fmt,
            ),
            (
                "structure_analysis",
                "ligand_contacts.contact_ligand_atom_count",
                "Contacted ligand atoms",
                None,
                _fmt,
            ),
            (
                "structure_analysis",
                "ligand_pocket.pocket_residue_count",
                "Pocket residues",
                None,
                _fmt,
            ),
        ],
    ),
    (
        "Sequence biophysics",
        [
            ("val", "length", "Length", "aa", _fmt),
            ("sequence_analysis", "gravy", "GRAVY", None, _fmt),
            ("sequence_analysis", "pI", "pI", None, _fmt),
            ("sequence_analysis", "net_charge", "Net charge", None, _fmt),
            ("sequence_analysis", "instability_index", "Instability", None, _fmt),
            ("sequence_analysis", "length", "Length", "aa", _fmt),
            ("sequence_analysis", "aromatic_fraction", "Aromatic fraction", None, _fmt),
            ("protein_qc", "instability_index", "Instability", None, _fmt),
            ("protein_qc", "pi", "pI", None, _fmt),
            ("protein_qc", "gravy", "GRAVY", None, _fmt),
        ],
    ),
    (
        "Stability scoring",
        [
            ("pyrosetta", "total_score", "Rosetta total score", "REU", _fmt),
            ("pyrosetta", "mean_total_score", "Mean Rosetta total score", "REU", _fmt),
            ("pyrosetta", "ddg", "ddG", "REU", _fmt),
            ("pyrosetta", "ddg_reu", "ddG", "REU", _fmt),
            ("pyrosetta", "mean_ddg", "Mean ddG", "REU", _fmt),
            ("pyrosetta", "delta_delta_g", "ddG", "REU", _fmt),
            ("pyrosetta", "delta_g", "Delta G", "REU", _fmt),
            ("pyrosetta", "dg", "Delta G", "REU", _fmt),
            ("submit_candidate", "ddg_reu", "ddG", "REU", _fmt),
            ("submit_candidate", "mean_ddg", "Mean ddG", "REU", _fmt),
        ],
    ),
    (
        "Objective ranking",
        [
            ("objective_rank", "objective_score", "Objective score", None, _fmt),
            ("objective_rank", "rank", "Objective rank", None, _fmt),
        ],
    ),
    (
        "Novelty & similarity",
        [
            ("val", "win_id", "Window identity", None, _fmt),
            ("proteinmpnn", "seq_recovery", "Seq recovery", "%", lambda v: _fmt(_pct(v))),
            ("proteinmpnn", "score", "MPNN score", None, _fmt),
        ],
    ),
]


def _pct(value: Any) -> float | None:
    num = _as_float(value)
    if num is None:
        return None
    return round(num * 100.0, 1) if 0.0 <= num <= 1.0 else round(num, 1)


def _score_namespace(candidate: ReportCandidate, tool: str) -> dict[str, Any]:
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    out: dict[str, Any] = {}
    block = scores.get(tool)
    if isinstance(block, dict):
        out.update(block)
    prefix = f"{tool}:"
    for raw_key, value in scores.items():
        if not isinstance(raw_key, str) or not raw_key.startswith(prefix):
            continue
        metric = raw_key[len(prefix) :].strip()
        if metric:
            out.setdefault(metric, value)
    if tool == "sequence_analysis":
        # Some adapters publish per-candidate sequence-analysis fields as bare
        # keys rather than inside a ``sequence_analysis`` namespace. Keep this
        # fallback candidate-local; collector no longer attaches global
        # sequence-analysis summaries to objective-rank rows.
        for key in (
            "gravy",
            "pI",
            "pi",
            "net_charge",
            "instability_index",
            "aromatic_fraction",
            "length",
        ):
            if key not in scores:
                continue
            out.setdefault("pI" if key == "pi" else key, scores[key])
    source = candidate.source if isinstance(candidate.source, dict) else {}
    source_tool = str(source.get("tool") or "").strip().lower()
    if source_tool == tool and isinstance(candidate.scores, dict):
        for raw_key, value in scores.items():
            if (
                isinstance(raw_key, str)
                and raw_key
                and ":" not in raw_key
                and not isinstance(value, dict)
            ):
                out.setdefault(raw_key, value)
    return out


# Short, human-readable source-tool labels for metric provenance. Every score
# in a family is read from a specific tool's namespace; surfacing that tool
# (e.g. "pLDDT — ESM-3", "ipTM — Boltz-2") makes a confidence number traceable
# to the model that produced it rather than reading as a free-floating value.
_METRIC_SOURCE_NAME: dict[str, str] = {
    "esm3": "ESM-3",
    "esm2": "ESM-2",
    "boltz2_affinity": "Boltz-2",
    "bindcraft": "BindCraft",
    "boltzgen": "BoltzGen",
    "proteina_complexa": "Proteina-Complexa",
    "protenix_v2": "Protenix",
    "protein_qc": "Protein QC",
    "pxdesign": "PXDesign",
    "pxmeter": "PXMeter",
    "sequence_analysis": "Seq. analysis",
    "structure_analysis": "Struct. analysis",
    "proteinmpnn": "ProteinMPNN",
    "ipsae": "ipSAE",
    "prodigy": "PRODIGY",
    "pyrosetta": "PyRosetta",
    "objective_rank": "Objective rank",
    "submit_candidate": "Submit candidate",
    "val": "Objective metrics",
}


def _source_name(tool: str) -> str:
    return _METRIC_SOURCE_NAME.get(tool, tool.replace("_", " ").title() if tool else "")


def _score_value(block: dict[str, Any], key: str) -> Any:
    """Read a score by literal key or dotted nested path.

    Candidate score blocks are not uniform: predictors usually emit flat
    metrics, while analyzers such as structure_analysis publish structured
    sub-blocks. Keeping dotted lookup here lets the family registry remain a
    declarative list instead of growing tool-specific extraction branches.
    """

    if key in block:
        return block[key]
    current: Any = block
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _build_score_groups(candidate: ReportCandidate, msa_blocks: list[MSABlock]) -> list[ScoreGroup]:
    groups: list[ScoreGroup] = []
    suppress_protenix_interface_metrics = _is_protenix_monomer_candidate(candidate)
    for family, specs in _FAMILY_SPECS:
        # (label, value, unit, source_tool) — source carries the producing tool
        # so the renderer can tag each confidence metric with its origin.
        metrics: list[tuple[str, str, str | None, str]] = []
        seen_metrics: set[tuple[str, str]] = set()
        seen_metric_values: set[tuple[str, str, str | None]] = set()
        for tool, key, label, unit, transform in specs:
            if (
                suppress_protenix_interface_metrics
                and family == "Interface & binding"
                and tool == "protenix_v2"
            ):
                continue
            block = _score_namespace(candidate, tool)
            raw_value = _score_value(block, key)
            if raw_value is None:
                continue
            shown = transform(raw_value) if transform else _fmt(raw_value)
            if shown is None:
                continue
            source = _source_name(tool)
            identity = (label, source)
            value_identity = (label, shown, unit)
            duplicate_value = (
                label == "Length" and unit == "aa" and value_identity in seen_metric_values
            )
            if identity in seen_metrics or duplicate_value:
                continue
            seen_metrics.add(identity)
            seen_metric_values.add(value_identity)
            metrics.append((label, shown, unit, source))
        if family == "Novelty & similarity" and msa_blocks:
            ident = max((mb.mean_identity for mb in msa_blocks), default=0.0)
            if ident:
                metrics.append(("MSA mean identity", _fmt(ident * 100.0) or "0", "%", "MSA"))
        if metrics:
            groups.append(ScoreGroup(family=family, metrics=metrics, provenance=Provenance.DERIVED))

    # Flags & liabilities family — always shown when present.
    flag_metrics: list[tuple[str, str, str | None, str]] = []
    qc = _score_namespace(candidate, "protein_qc")
    if "passes_filters" in qc:
        flag_metrics.append(
            ("QC filters", "pass" if qc.get("passes_filters") else "fail", None, "Protein QC")
        )
    flags = _string_list(candidate.qc_flags) or _string_list(qc.get("flags"))
    if flags:
        flag_metrics.append(("Flags", ", ".join(flags), None, "Protein QC"))
    liab = _liability_labels(candidate)
    if liab:
        flag_metrics.append(("Liabilities", ", ".join(liab), None, ""))
    if flag_metrics:
        groups.append(
            ScoreGroup(
                family="Flags & liabilities",
                metrics=flag_metrics,
                provenance=Provenance.DERIVED,
            )
        )
    return groups


def _is_protenix_monomer_candidate(candidate: ReportCandidate) -> bool:
    """Return true for Protenix-scored candidates with one protein chain.

    Protenix emits ``iptm: 0.0`` for monomer predictions. That value should not
    promote an enzyme/foldability report into a binder/interface review.
    """

    if not _score_namespace(candidate, "protenix_v2"):
        return False
    path = (candidate.structure_path or "").strip()
    if not path:
        return False
    return _protein_chain_count(Path(path)) == 1


def _protein_chain_count(path: Path) -> int | None:
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    if path.suffix.lower() in {".cif", ".mmcif"} or "_atom_site." in text:
        return _cif_protein_chain_count(text)
    return _pdb_protein_chain_count(text)


def _pdb_protein_chain_count(text: str) -> int | None:
    chains: set[str] = set()
    for line in text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        residue = line[17:20].strip().upper()
        if residue not in _STANDARD_AA3:
            continue
        chains.add(line[21:22].strip() or "_")
    return len(chains) if chains else None


def _cif_protein_chain_count(text: str) -> int | None:
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        cols: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_atom_site."):
            cols.append(lines[i].strip().split(".", 1)[1])
            i += 1
        if not cols:
            continue
        if "label_comp_id" not in cols:
            continue
        comp_idx = cols.index("label_comp_id")
        chain_idx = next(
            (cols.index(name) for name in ("label_asym_id", "auth_asym_id") if name in cols),
            -1,
        )
        if chain_idx < 0:
            continue
        chains: set[str] = set()
        while i < len(lines):
            stripped = lines[i].strip()
            if (
                not stripped
                or stripped.startswith("#")
                or stripped == "loop_"
                or stripped.startswith("_")
            ):
                break
            parts = stripped.split()
            if len(parts) > max(comp_idx, chain_idx):
                residue = parts[comp_idx].strip("'\"").upper()
                if residue in _STANDARD_AA3:
                    chains.add(parts[chain_idx].strip("'\"") or "_")
            i += 1
        if chains:
            return len(chains)
    return None


_STANDARD_AA3: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "SEC",
        "PYL",
    }
)


def _liability_labels(candidate: ReportCandidate) -> list[str]:
    out: list[str] = []
    for item in candidate.liabilities or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            label = item.get("type") or item.get("name") or item.get("label")
            if label:
                out.append(str(label))
    seq = _score_namespace(candidate, "sequence_analysis")
    out.extend(str(x) for x in _string_list(seq.get("liability_types")))
    # de-dup preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _has_evaluation_evidence(candidate: ReportCandidate) -> bool:
    if not isinstance(candidate.scores, dict):
        return bool((candidate.structure_path or "").strip())
    for tool in candidate.scores:
        if (tool or "").strip().lower() in _EVIDENCE_EVALUATORS:
            return True
    return bool((candidate.structure_path or "").strip())


# --------------------------------------------------------------------------- #
# Status classification + selection basis
# --------------------------------------------------------------------------- #


def _qc_fails(candidate: ReportCandidate) -> bool:
    qc = _score_namespace(candidate, "protein_qc")
    if "passes_filters" in qc:
        return not bool(qc.get("passes_filters"))
    return False


def _scoring_rank_map(model: ReportModel) -> dict[str, int]:
    """Map objective-ranked identities to authoritative scoring ranks."""
    rank_nodes = [
        n for n in model.graph.nodes if (n.tool or "").strip().lower() == "objective_rank"
    ]
    if not rank_nodes:
        return {}
    out: dict[str, int] = {}
    for node in rank_nodes:  # later nodes win (final ranking)
        output = node.output if isinstance(node.output, dict) else {}
        rows = output.get("ranked_candidates")
        if not isinstance(rows, list):
            continue
        candidate_counts: dict[str, int] = {}
        for row in rows:
            if isinstance(row, dict):
                cid = row.get("candidate_id")
                if isinstance(cid, str) and cid:
                    candidate_counts[cid] = candidate_counts.get(cid, 0) + 1
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            cid = row.get("candidate_id")
            rank = _as_int(row.get("rank"))
            resolved_rank = rank if rank is not None else idx + 1
            ranked_id = _row_ranked_candidate_id(row)
            if ranked_id:
                out[f"ranked:{ranked_id}"] = resolved_rank
            if isinstance(cid, str) and cid and candidate_counts.get(cid) == 1:
                out[f"candidate:{cid}"] = resolved_rank
    return out


def _rank_sort_key(candidate: ReportCandidate) -> tuple[float, float]:
    """Order key preferring a recorded rank, then a higher score."""
    rank = _as_int(candidate.rank)
    score = _as_float(candidate.score)
    return (
        float(rank) if rank is not None else float("inf"),
        -score if score is not None else float("inf"),
    )


def _final_fallback_sort_key(candidate: ReportCandidate) -> tuple[int, int, int, float, float]:
    """Choose the richest sequence-matched record to represent a submitted pick."""
    source = candidate.source if isinstance(candidate.source, dict) else {}
    source_tool = str(source.get("tool") or "").strip().lower()
    rank_key = _rank_sort_key(candidate)
    return (
        0 if _candidate_ranked_candidate_id(candidate) else 1,
        0 if candidate.structure_path else 1,
        0 if source_tool in _STRUCTURE_GENERATOR_TOOLS else 1,
        rank_key[0],
        rank_key[1],
    )


def _candidate_scoring_rank(
    candidate: ReportCandidate,
    scoring_ranks: dict[str, int],
) -> int | None:
    ranked_id = _candidate_ranked_candidate_id(candidate)
    if ranked_id:
        direct = scoring_ranks.get(f"ranked:{ranked_id}")
        if direct is not None:
            return direct
    direct = scoring_ranks.get(f"candidate:{candidate.candidate_id}")
    if direct is not None:
        return direct
    scores = candidate.scores if isinstance(candidate.scores, dict) else {}
    objective_rank = scores.get("objective_rank")
    if isinstance(objective_rank, dict):
        return _as_int(objective_rank.get("rank"))
    return None


def _candidate_rank_identity(candidate: ReportCandidate) -> str:
    ranked_id = _candidate_ranked_candidate_id(candidate)
    if ranked_id:
        return f"ranked:{ranked_id}"
    return f"candidate:{candidate.candidate_id}" if candidate.candidate_id else ""


def _first_str(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _row_score_map(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores")
    out = dict(scores) if isinstance(scores, dict) else {}
    objective = dict(out.get("objective_rank") or {})
    for key in (
        "ranked_candidate_id",
        "objective_name",
        "objective_score",
        "score",
        "score_formula",
        "formula_text",
        "score_direction",
        "gate_summary",
        "score_components",
        "passed_gates",
        "failed_gates",
        "warnings",
        "tie_breaker_values",
        "rank",
        "material_key",
    ):
        value = row.get(key)
        if value is not None:
            objective.setdefault(key, value)
    if objective:
        out["objective_rank"] = objective
    return out


def _merge_score_namespaces(
    base: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            merged = dict(value)
            merged.update(out[key])
            out[key] = merged
        elif key not in out:
            out[key] = value
    return out


def _ranked_row_candidate(
    row: dict[str, Any],
    *,
    node: WorkflowNodeSummary,
    index: int,
) -> ReportCandidate:
    candidate_id = str(row.get("candidate_id") or row.get("name") or index + 1)
    ranked_candidate_id = _row_ranked_candidate_id(row)
    material_key = _row_material_key(row)
    structure_path = _first_str(row, *_STRUCTURE_PATH_KEYS)
    scores = _row_score_map(row)
    objective = dict(scores.get("objective_rank") or {})
    if ranked_candidate_id:
        objective.setdefault("ranked_candidate_id", ranked_candidate_id)
    if material_key:
        objective.setdefault("material_key", material_key)
    objective.setdefault("rank", _as_int(row.get("rank")) or index + 1)
    objective_score = _as_float(
        row.get("objective_score") if row.get("objective_score") is not None else row.get("score")
    )
    if objective_score is not None:
        objective.setdefault("objective_score", objective_score)
        objective.setdefault("score", objective_score)
    if objective:
        scores["objective_rank"] = objective
    return ReportCandidate(
        candidate_id=candidate_id,
        rank=_as_int(row.get("rank")) or index + 1,
        score=objective_score,
        sequence=_first_str(row, "sequence"),
        structure_path=structure_path,
        scores=scores,
        qc_flags=_string_list(row.get("qc_flags")),
        liabilities=row.get("liabilities") if isinstance(row.get("liabilities"), list) else [],
        source={
            "node_id": node.node_id,
            "job_id": node.job_id,
            "tool": node.tool,
            "kind": "ranking_manifest",
            "local_id": ranked_candidate_id or candidate_id,
            "ranked_candidate_id": ranked_candidate_id,
            "identity_key": row.get("identity_key"),
            "material_key": material_key,
            "candidate_ref": row.get("candidate_ref")
            if isinstance(row.get("candidate_ref"), dict)
            else {},
        },
    )


def _unique_support_candidate(
    ranked: ReportCandidate,
    candidates: list[ReportCandidate],
) -> ReportCandidate | None:
    ranked_id = _candidate_ranked_candidate_id(ranked)
    if ranked_id:
        matches = [c for c in candidates if _candidate_ranked_candidate_id(c) == ranked_id]
        if len(matches) == 1:
            return matches[0]
        structure_stem = _candidate_structure_stem_identity(ranked)
        if structure_stem:
            stem_matches = [
                c for c in matches if _candidate_structure_stem_identity(c) == structure_stem
            ]
            if len(stem_matches) == 1:
                return stem_matches[0]
    material_key = _candidate_material_key(ranked)
    if material_key:
        matches = [c for c in candidates if _candidate_material_key(c) == material_key]
        if len(matches) == 1:
            return matches[0]
    structure = _candidate_structure_identity(ranked)
    if structure:
        matches = [c for c in candidates if _candidate_structure_identity(c) == structure]
        if len(matches) == 1:
            return matches[0]
    structure_stem = _candidate_structure_stem_identity(ranked)
    if structure_stem:
        matches = [
            c
            for c in candidates
            if _candidate_structure_stem_identity(c) == structure_stem
        ]
        if len(matches) == 1:
            return matches[0]
    candidate_id = str(ranked.candidate_id or "")
    sequence = _candidate_sequence_identity(ranked)
    if candidate_id and sequence:
        matches = [
            c
            for c in candidates
            if str(c.candidate_id or "") == candidate_id
            and _candidate_sequence_identity(c) == sequence
        ]
        if len(matches) == 1:
            return matches[0]
    if candidate_id:
        matches = [c for c in candidates if str(c.candidate_id or "") == candidate_id]
        if len(matches) == 1:
            return matches[0]
    return None


def _candidate_structure_stem_identity(candidate: ReportCandidate) -> str | None:
    structure = _candidate_structure_identity(candidate)
    if not structure:
        return None
    return Path(structure).stem or structure


def _enrich_ranked_candidate(
    ranked: ReportCandidate,
    support: ReportCandidate | None,
) -> ReportCandidate:
    if support is None:
        return ranked
    source = dict(ranked.source if isinstance(ranked.source, dict) else {})
    support_source = support.source if isinstance(support.source, dict) else {}
    source.setdefault("support_candidate_id", support.candidate_id)
    if support_source.get("tool"):
        source.setdefault("support_tool", support_source.get("tool"))
    artifacts = list(ranked.artifacts)
    seen_artifacts = {artifact.path for artifact in artifacts}
    for artifact in support.artifacts:
        if artifact.path in seen_artifacts:
            continue
        seen_artifacts.add(artifact.path)
        artifacts.append(artifact)
    return replace(
        ranked,
        sequence=ranked.sequence or support.sequence,
        structure_path=ranked.structure_path or support.structure_path,
        scores=_merge_score_namespaces(ranked.scores, support.scores),
        qc_flags=ranked.qc_flags or support.qc_flags,
        liabilities=ranked.liabilities or support.liabilities,
        source=source,
        artifacts=artifacts or support.artifacts,
    )


def _ranking_manifest_candidates(model: ReportModel) -> list[ReportCandidate]:
    """Build canonical ranked rows directly from objective_rank node output.

    The collected candidate list may contain raw generator/evaluator aliases
    that share sequence or candidate_id with ranked rows. The review board uses
    these objective_rank rows as the manifest whenever they exist, then attaches
    matching raw rows only as support evidence.
    """

    ranked: list[ReportCandidate] = []
    support_pool = [
        candidate
        for candidate in model.candidates
        if str((candidate.source or {}).get("tool") or "").strip().lower() != "objective_rank"
    ]
    objective_rank_support_pool = [
        candidate
        for candidate in model.candidates
        if str((candidate.source or {}).get("tool") or "").strip().lower() == "objective_rank"
    ]
    for node in model.graph.nodes:
        if (node.tool or "").strip().lower() != "objective_rank":
            continue
        output = node.output if isinstance(node.output, dict) else {}
        rows = output.get("ranked_candidates")
        if not isinstance(rows, list):
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            ranked_row = dict(row)
            ranked_row.setdefault("formula_text", output.get("formula_text"))
            ranked_row.setdefault("gate_summary", output.get("gate_summary"))
            ranked_row.setdefault("objective_name", output.get("objective_name"))
            ranked_row.setdefault("score_direction", output.get("score_direction"))
            canonical = _ranked_row_candidate(ranked_row, node=node, index=index)
            support = _unique_support_candidate(canonical, support_pool)
            if support is None:
                support = _unique_support_candidate(canonical, objective_rank_support_pool)
            if support is None:
                support = _unique_support_candidate(canonical, model.final_candidates)
            ranked.append(
                _enrich_ranked_candidate(
                    canonical,
                    support,
                )
            )
    if not ranked:
        return list(model.candidates)

    ranked_identities = {_candidate_manifest_identity(candidate) for candidate in ranked}
    out = list(ranked)
    for candidate in model.candidates:
        identity = _candidate_manifest_identity(candidate)
        if identity and identity in ranked_identities:
            continue
        if _unique_support_candidate(candidate, ranked) is not None:
            continue
        out.append(candidate)
    return out


def _resolve_final_records(model: ReportModel) -> dict[int, str]:
    """Map ``model.candidates`` indices that are genuinely final → provenance.

    Fragmented designs can share candidate_id or sequence across objective-rank
    rows. The submitted identity therefore wins in this order:
    ranked_candidate_id, material_key, structure path, then unique candidate_id.
    Sequence fallback is allowed only for old submissions that recorded no
    stronger material key.
    """
    resolved: dict[int, str] = {}
    matched_finals: set[int] = set()

    def unique_match(predicate: Any) -> int | None:
        matches = [
            idx
            for idx, cand in enumerate(model.candidates)
            if idx not in resolved and predicate(cand)
        ]
        return matches[0] if len(matches) == 1 else None

    # Pass 1: exact submitted material identities.
    for final_idx, fc in enumerate(model.final_candidates):
        ranked_id = _submitted_ranked_candidate_id(fc)
        if ranked_id:
            match = unique_match(
                lambda cand, ranked_id=ranked_id: (
                    _candidate_ranked_candidate_id(cand) == ranked_id
                )
            )
            if match is not None:
                resolved[match] = Provenance.EXACT
                matched_finals.add(final_idx)
                continue
        material_key = _submitted_material_key(fc)
        if material_key:
            match = unique_match(
                lambda cand, material_key=material_key: (
                    _candidate_material_key(cand) == material_key
                )
            )
            if match is not None:
                resolved[match] = Provenance.EXACT
                matched_finals.add(final_idx)
                continue
        fc_structure = _candidate_structure_identity(fc)
        if fc_structure:
            match = unique_match(
                lambda cand, fc_structure=fc_structure: (
                    _candidate_structure_identity(cand) == fc_structure
                )
            )
            if match is not None:
                resolved[match] = Provenance.EXACT
                matched_finals.add(final_idx)
                continue
        fcid = str(fc.candidate_id or "")
        if fcid:
            match = unique_match(
                lambda cand, fcid=fcid: str(cand.candidate_id or "") == fcid
            )
            if match is not None:
                resolved[match] = Provenance.EXACT
                matched_finals.add(final_idx)

    # Pass 2: sequence fallback, only for final picks unresolved by id. When
    # several records share the submitted sequence, prefer the record carrying
    # the submitted structure/objective-rank evidence over QC-only aliases.
    for final_idx, fc in enumerate(model.final_candidates):
        if final_idx in matched_finals:
            continue
        seq = _degap(fc.sequence or "")
        if not seq:
            continue
        candidates = [
            idx
            for idx, cand in enumerate(model.candidates)
            if idx not in resolved and _degap(cand.sequence or "") == seq
        ]
        if not candidates:
            continue
        fc_structure = (fc.structure_path or "").strip()
        if fc_structure:
            fc_path = Path(fc_structure)
            structure_matches = [
                idx
                for idx in candidates
                if (model.candidates[idx].structure_path or "").strip()
                and (
                    model.candidates[idx].structure_path == fc_structure
                    or Path(model.candidates[idx].structure_path or "").name == fc_path.name
                )
            ]
            if structure_matches:
                candidates = structure_matches
            else:
                # A submitted final with an explicit structure is already the
                # authoritative material. Do not promote an older raw row just
                # because it shares the binder sequence; doing so can attach
                # the final rationale to preliminary triage metrics.
                continue
        best = min(candidates, key=lambda i: _final_fallback_sort_key(model.candidates[i]))
        resolved[best] = Provenance.INFERRED
    return resolved


def _classify_status(
    candidate: ReportCandidate,
    is_final: bool,
    *,
    workflow_type: str,
    scoring_rank: int | None = None,
) -> str:
    if is_final:
        return CandidateStatus.FINAL
    if _qc_fails(candidate):
        return CandidateStatus.FAILED_OR_REJECTED
    if workflow_type == WorkflowType.RETRIEVAL:
        return CandidateStatus.REFERENCE_GROUNDING
    if scoring_rank is not None or isinstance(candidate.scores.get("objective_rank"), dict):
        return CandidateStatus.RANKED_ALTERNATIVE
    if _has_evaluation_evidence(candidate):
        return CandidateStatus.EVALUATED_NOT_SELECTED
    return CandidateStatus.GENERATED_NOT_EVALUATED


def _missing_evidence(candidate: ReportCandidate) -> list[str]:
    missing: list[str] = []
    if not (candidate.structure_path or "").strip():
        missing.append("structure")
    if "protein_qc" not in (candidate.scores or {}):
        missing.append("qc")
    if not _has_evaluation_evidence(candidate):
        missing.append("evaluation")
    if _as_int(candidate.rank) is None:
        missing.append("rank")
    return missing


def _selection_basis(candidate: ReportCandidate, status: str) -> dict[str, Any]:
    if status != CandidateStatus.FINAL:
        return {}
    src = candidate.source if isinstance(candidate.source, dict) else {}
    return {
        "submitted": True,
        "rank": _as_int(candidate.rank),
        "score": _as_float(candidate.score),
        "auto_selected": bool(src.get("auto_selected", False)),
        "justification": src.get("justification") or src.get("reasoning"),
    }


def _non_selection_basis(candidate: ReportCandidate, status: str) -> dict[str, Any]:
    if status == CandidateStatus.FINAL:
        return {}
    return {
        "submitted": False,
        "rank": _as_int(candidate.rank),
        "score": _as_float(candidate.score),
        "qc_flags": _string_list(candidate.qc_flags),
        "missing_evidence": _missing_evidence(candidate),
    }


# --------------------------------------------------------------------------- #
# Evidence path (shared pipeline + candidate overlay)
# --------------------------------------------------------------------------- #


def _node_evidence_summary(candidate: ReportCandidate, tool: str) -> str:
    """One short candidate-scoped line for a node, from candidate.scores[tool]."""
    block = _score_namespace(candidate, tool)
    if not block:
        return ""
    bits: list[str] = []
    name = (tool or "").lower()
    if name == "esm3":
        p = _plddt_100(block.get("mean_plddt"))
        if p is not None:
            bits.append(f"pLDDT {p}")
        t = _fmt(block.get("ptm"))
        if t:
            bits.append(f"pTM {t}")
    elif name == "protein_qc":
        flags = _string_list(block.get("flags"))
        bits.append("QC pass" if block.get("passes_filters") else "QC fail")
        if flags:
            bits.append(f"{len(flags)} flag" + ("s" if len(flags) != 1 else ""))
    elif name == "sequence_analysis":
        g = _fmt(block.get("gravy"))
        if g:
            bits.append(f"GRAVY {g}")
    elif name == "proteinmpnn":
        s = _fmt(block.get("score"))
        if s:
            bits.append(f"MPNN {s}")
    elif name in {"bindcraft", "boltz2_affinity", "protenix_v2", "proteina_complexa"}:
        p = _plddt_100(block.get("plddt"))
        if p is not None:
            bits.append(f"pLDDT {p}")
        i = _fmt(block.get("iptm"))
        if i:
            bits.append(f"ipTM {i}")
        a = _fmt(_first_recorded(block.get("affinity_pred_value"), block.get("predicted_affinity")))
        if a:
            bits.append(f"Affinity {a}")
    elif name == "boltzgen":
        c = _fmt(block.get("confidence"))
        if c:
            bits.append(f"Confidence {c}")
        a = _fmt(block.get("affinity_pred_value"))
        if a:
            bits.append(f"Affinity {a}")
    elif name == "pxdesign":
        s = _fmt(block.get("score"))
        if s:
            bits.append(f"Generation score {s}")
    elif name == "ipsae":
        i = _fmt(
            _first_recorded(
                block.get("ipsae_score"),
                block.get("ipsae_max"),
                block.get("ipsae"),
            )
        )
        if i:
            bits.append(f"ipSAE {i}")
    elif name == "pxmeter":
        d = _dockq_disp(block.get("dockq"))
        if d:
            bits.append(f"DockQ {d}")
        lddt = _fmt(block.get("lddt"))
        if lddt:
            bits.append(f"lDDT {lddt}")
    return " · ".join(bits)


def _build_evidence_path(
    candidate: ReportCandidate,
    ordered_nodes: list[WorkflowNodeSummary],
    status: str,
) -> EvidencePath:
    src = candidate.source if isinstance(candidate.source, dict) else {}
    src_node_id = src.get("source_node_id") or src.get("node_id")
    src_tool = (src.get("tool") or "").strip().lower()

    steps: list[EvidenceStep] = []
    generators: list[EvidenceStep] = []
    evaluators: list[EvidenceStep] = []
    ranker: EvidenceStep | None = None
    submission: EvidenceStep | None = None

    for node in ordered_nodes:
        role = classify_tool_role(node.tool)
        if role == ToolRole.PLUMBING:
            continue  # plumbing is not evidence; keep it out of the candidate path
        tool = (node.tool or "").lower()
        summary = _node_evidence_summary(candidate, tool)
        # provenance for this node's overlay onto the candidate
        if node.node_id and node.node_id == src_node_id:
            prov = Provenance.EXACT
        elif summary:
            prov = Provenance.DERIVED  # this candidate's metrics are present
        else:
            prov = Provenance.NOT_RECORDED
        step = EvidenceStep(
            node_id=node.node_id,
            tool=node.tool,
            role=role,
            state=node.state,
            job_id=node.job_id,
            evidence_summary=summary,
            provenance=prov,
        )
        steps.append(step)
        if role == ToolRole.GENERATOR:
            generators.append(step)
        elif role == ToolRole.EVALUATOR and summary:
            evaluators.append(step)
        elif role == ToolRole.RANKER and ranker is None:
            ranker = step
        elif role == ToolRole.SUBMISSION and submission is None:
            submission = step

    # Producer attribution.
    producer: EvidenceStep | None = None
    producer_prov = Provenance.NOT_RECORDED
    exact_gen = [s for s in generators if s.node_id == src_node_id]
    if exact_gen:
        producer, producer_prov = exact_gen[0], Provenance.EXACT
    elif len(generators) == 1:
        # Single generator in the pipeline — credit it, flagged as inferred.
        producer, producer_prov = generators[0], Provenance.INFERRED
    elif src_tool:
        tool_gen = [s for s in generators if (s.tool or "").lower() == src_tool]
        if tool_gen:
            producer, producer_prov = tool_gen[0], Provenance.INFERRED

    if status == CandidateStatus.FINAL and submission is None:
        submission = EvidenceStep(
            node_id="final_submission",
            tool="submit_candidate",
            role=ToolRole.SUBMISSION,
            evidence_summary="selected",
            provenance=Provenance.EXACT,
        )

    return EvidencePath(
        shared_pipeline=steps,
        producer=producer,
        producer_provenance=producer_prov,
        evaluators=evaluators,
        ranker=ranker,
        submission=submission,
    )


# --------------------------------------------------------------------------- #
# MSA / structure association
# --------------------------------------------------------------------------- #


def _msa_for_candidate(candidate: ReportCandidate, msa_blocks: list[MSABlock]) -> list[MSABlock]:
    seq = _degap(candidate.sequence or "")
    if not seq or not msa_blocks:
        return []
    matched: list[MSABlock] = []
    for block in msa_blocks:
        for _label, aligned in block.sequences[:1]:  # query row only
            if _degap(aligned) == seq:
                matched.append(block)
                break
    return matched


def _structure_variants(candidate: ReportCandidate) -> list:
    seen: set[str] = set()
    out = []
    for art in candidate.artifacts:
        kind = (art.kind or "").lower()
        if kind in {"pdb", "cif", "mmcif", "ent"} and art.path not in seen:
            seen.add(art.path)
            out.append(art)
    return out


# --------------------------------------------------------------------------- #
# Bundle + board assembly
# --------------------------------------------------------------------------- #


def _build_bundle(
    candidate: ReportCandidate,
    *,
    workflow_type: str,
    ordered_nodes: list[WorkflowNodeSummary],
    msa_blocks: list[MSABlock],
    final_provenance: str | None,
    scoring_rank: int | None = None,
    top_ranked_id: str | None = None,
) -> CandidateEvidenceBundle:
    is_final = final_provenance is not None
    status = _classify_status(
        candidate,
        is_final,
        workflow_type=workflow_type,
        scoring_rank=scoring_rank,
    )
    cand_msa = _msa_for_candidate(candidate, msa_blocks)
    # Authoritative rank for the board badge + selector card: objective_rank's own
    # rank when known, else the recorded collection-order rank.
    display_rank = scoring_rank if scoring_rank is not None else _as_int(candidate.rank)
    # Flag the case where the agent submitted a design objective_rank did NOT score
    # #1, so the board badge (#N) doesn't read as a contradiction.
    submitted_despite_rank = (
        display_rank
        if (
            is_final
            and scoring_rank is not None
            and scoring_rank != 1
            and top_ranked_id is not None
            and top_ranked_id != _candidate_rank_identity(candidate)
        )
        else None
    )
    return CandidateEvidenceBundle(
        candidate=candidate,
        workflow_type=workflow_type,
        status=status,
        status_basis=final_provenance if is_final else Provenance.DERIVED,
        display_rank=display_rank,
        submitted_despite_rank=submitted_despite_rank,
        selection_basis=_selection_basis(candidate, status),
        non_selection_basis=_non_selection_basis(candidate, status),
        evidence_path=_build_evidence_path(candidate, ordered_nodes, status),
        score_groups=_build_score_groups(candidate, cand_msa),
        structure_variants=_structure_variants(candidate),
        msa_blocks=cand_msa,
        tool_evidence={
            tool: block
            for tool, block in (candidate.scores or {}).items()
            if isinstance(block, dict)
        },
        artifacts=list(candidate.artifacts),
    )


# Board groups, in display order. Failed / rejected are collapsed by default.
_GROUP_ORDER: list[tuple[str, str, bool]] = [
    (CandidateStatus.FINAL, "Final / Submitted", False),
    (CandidateStatus.RANKED_ALTERNATIVE, "Ranked alternatives", False),
    (CandidateStatus.EVALUATED_NOT_SELECTED, "Evaluated, not selected", False),
    (CandidateStatus.GENERATED_NOT_EVALUATED, "Generated, not evaluated", True),
    (CandidateStatus.REFERENCE_GROUNDING, "Reference grounding", True),
    (CandidateStatus.FAILED_OR_REJECTED, "Failed or rejected", True),
]


def _rank_key(bundle: CandidateEvidenceBundle) -> tuple[float, float]:
    # Prefer the authoritative objective_rank display rank; fall back to the
    # recorded rank.
    rank = (
        bundle.display_rank if bundle.display_rank is not None else _as_int(bundle.candidate.rank)
    )
    score = _as_float(bundle.candidate.score)
    return (
        float(rank) if rank is not None else float("inf"),
        -score if score is not None else float("inf"),
    )


def _board_group_title(status: str, default_title: str, workflow_type: str) -> str:
    """Group title, with a retrieval-aware override for unscored hits.

    In a retrieval workflow the "incomplete evidence" bucket isn't a quality
    failure: these are database sequences pulled by a query that were never
    individually folded or scored, so they legitimately lack evaluator
    evidence. Calling them "Incomplete evidence" reads as a defect; relabel so
    the reader understands the bucket is "retrieved but not evaluated".
    """
    if workflow_type == WorkflowType.RETRIEVAL and status == CandidateStatus.REFERENCE_GROUNDING:
        return "Retrieved grounding evidence"
    return default_title


def _build_board_groups(
    bundles: list[CandidateEvidenceBundle],
    workflow_type: str = WorkflowType.GENERIC,
) -> list[BoardGroup]:
    by_status: dict[str, list[CandidateEvidenceBundle]] = {}
    for b in bundles:
        by_status.setdefault(b.status, []).append(b)
    groups: list[BoardGroup] = []
    for status, title, collapsed in _GROUP_ORDER:
        items = by_status.get(status)
        if not items:
            continue
        items.sort(key=_rank_key)
        groups.append(
            BoardGroup(
                status=status,
                title=_board_group_title(status, title, workflow_type),
                bundles=items,
                collapsed=collapsed,
            )
        )
    return groups


def build_review_model(model: ReportModel) -> ReportReviewModel:
    """Derive the candidate-centric :class:`ReportReviewModel` from a report."""
    workflow_type = detect_workflow_type(model)
    ordered_nodes = _topo_order(list(model.graph.nodes))
    manifest_candidates = _ranking_manifest_candidates(model)
    has_ranking_manifest = any(
        (node.tool or "").strip().lower() == "objective_rank"
        and isinstance(node.output, dict)
        and isinstance(node.output.get("ranked_candidates"), list)
        for node in model.graph.nodes
    )
    manifest_model = (
        replace(model, candidates=manifest_candidates) if has_ranking_manifest else model
    )
    final_map = _resolve_final_records(manifest_model)
    scoring_ranks = _scoring_rank_map(model)
    top_ranked_id = next((cid for cid, rank in scoring_ranks.items() if rank == 1), None)

    bundles = [
        _build_bundle(
            candidate,
            workflow_type=workflow_type,
            ordered_nodes=ordered_nodes,
            msa_blocks=model.msa_blocks,
            final_provenance=final_map.get(idx),
            scoring_rank=_candidate_scoring_rank(candidate, scoring_ranks),
            top_ranked_id=top_ranked_id,
        )
        for idx, candidate in enumerate(manifest_model.candidates)
    ]

    # A final pick may be recorded as its own candidate record that never appears
    # in ``model.candidates`` (distinct id *and* sequence — a genuinely disjoint
    # submission record). The submitted design must never be missing from the
    # board, so synthesise a FINAL bundle directly from any such record.
    represented_identities = {
        _candidate_manifest_identity(manifest_model.candidates[i])
        for i in final_map
    } - {""}
    represented_ids = {
        str(manifest_model.candidates[i].candidate_id or "") for i in final_map
    } - {""}
    represented_seqs = {
        _degap(manifest_model.candidates[i].sequence or "") for i in final_map
    } - {""}
    for fc in model.final_candidates:
        fcid = str(fc.candidate_id or "")
        fseq = _degap(fc.sequence or "")
        f_identity = _candidate_manifest_identity(fc)
        strong_identity = _strong_candidate_manifest_identity(fc)
        if strong_identity:
            if f_identity in represented_identities:
                continue
        elif (
            (fcid and fcid in represented_ids)
            or (fseq and fseq in represented_seqs and not (fc.structure_path or "").strip())
        ):
            continue
        bundles.append(
            _build_bundle(
                fc,
                workflow_type=workflow_type,
                ordered_nodes=ordered_nodes,
                msa_blocks=model.msa_blocks,
                final_provenance=Provenance.EXACT,
                scoring_rank=_candidate_scoring_rank(fc, scoring_ranks),
                top_ranked_id=top_ranked_id,
            )
        )
        if f_identity:
            represented_identities.add(f_identity)
        if fcid:
            represented_ids.add(fcid)
        if fseq:
            represented_seqs.add(fseq)

    board_groups = _build_board_groups(bundles, workflow_type)
    return ReportReviewModel(
        base=manifest_model,
        workflow_type=workflow_type,
        board_groups=board_groups,
        bundles=bundles,
        grouping_policy=(
            "ranking_manifest" if manifest_candidates != list(model.candidates) else "raw"
        ),
    )


__all__ = [
    "build_review_model",
    "classify_generation_stage_key",
    "classify_tool_role",
    "classify_workflow_stage_key",
    "detect_workflow_type",
]
