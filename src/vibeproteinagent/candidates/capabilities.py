from __future__ import annotations

from dataclasses import dataclass
from typing import Any

CANONICAL_PRODUCERS: set[str] = {
    "search_sequence",
    "go_sequences",
    "sequence_analysis",
    "merge_candidates",
    "rfdiffusion",
    "proteinmpnn",
    "bindcraft",
    "structure_analysis",
    "ipsae",
    "prodigy",
    "protein_qc",
    "tmalign",
    "pxmeter",
    "boltz2_affinity",
    "esm2",
    "esm3",
    "protenix_v2",
    "family_profile",
}

CANONICAL_CONSUMERS: dict[str, str] = {
    "rank_designs": "candidates",
    "rfdiffusion": "candidates",
    "proteinmpnn": "candidates",
    "bindcraft": "candidates",
    "sequence_analysis": "candidates",
    "structure_analysis": "candidates",
    "ipsae": "candidates",
    "prodigy": "candidates",
    "protein_qc": "candidates",
    "tmalign": "candidates",
    "pxmeter": "candidates",
    "boltz2_affinity": "candidates",
    "esm2": "candidates",
    "esm3": "candidates",
    "protenix_v2": "candidates",
    "family_profile": "candidates",
}

CANONICAL_SCALAR_CONSUMERS: dict[str, dict[str, str]] = {}

CANONICAL_PROFILE_PRODUCERS: dict[str, str] = {
    "family_profile": "profile",
}

CANONICAL_PROFILE_CONSUMERS: dict[str, str] = {
    "family_profile": "profile",
    "esm2": "profile_guidance",
    "esm3": "profile_guidance",
    "rfdiffusion": "profile_guidance",
    "proteinmpnn": "profile_guidance",
}

CANONICAL_TERM_PRODUCERS: dict[str, dict[str, str]] = {
    "go_term_search": {"go_id": "selected_go_id"},
    "ontology_lookup": {"go_id": "resolved_terms[0].id"},
    "interpro_entry_search": {"interpro_id": "selected_interpro_id"},
    "interpro_entry_lookup": {"interpro_id": "resolved_entries[0].accession"},
    "function_term_resolver": {
        "go_id": "selected_go_id",
        "go_terms": "selected_go_terms",
        "interpro_id": "selected_interpro_id",
        "interpro_terms": "selected_interpro_terms",
    },
}

CANONICAL_TERM_CONSUMERS: dict[str, str] = {
    "go_sequences": "go_id",
    "interpro_sequences": "interpro_id",
}

# Tools whose input shape alone cannot uniquely determine the operating
# mode. LLM must set `inputs.mode` to one of the listed values.
EXPLICIT_MODE_TOOLS: dict[str, list[str]] = {
    "rfdiffusion": [
        "unconditional",
        "unconditional_generation",
        "backbone_generation",
        "motif_scaffolding",
        "symmetric_motif_scaffolding",
        "partial_diffusion",
        "binder_design",
        "binder_backbone_design",
        "fold_conditioning",
        "symmetric_oligomer",
        "symmetric_oligomer_design",
    ],
    "esm2": ["embedding", "sequence_generation"],
    # Parallel migration PRs append additional explicit-mode registrations below.
}

CANONICAL_PRODUCERS.update(
    {
        "interpro_sequences",
        "foldseek",
        "rfdiffusion3",
        "rfpeptides",
        "pxdesign",
        "boltzgen",
        "ligandmpnn",
        "sequence_edit",
        "pyrosetta",
        "vina",
        "mmseqs2_gpu",
        "template_select",
        "proteina_complexa",
    }
)

CANONICAL_CONSUMERS.update(
    {
        "foldseek": "candidates",
        "rfdiffusion3": "candidates",
        "rfpeptides": "candidates",
        "pxdesign": "candidates",
        "boltzgen": "candidates",
        "ligandmpnn": "candidates",
        "sequence_edit": "candidates",
        "pyrosetta": "candidates",
        "vina": "candidates",
        "mmseqs2_gpu": "candidates",
        "template_select": "candidates",
        "proteina_complexa": "candidates",
    }
)

EXPLICIT_MODE_TOOLS.update(
    {
        "foldseek": ["easy-search", "easy-cluster"],
        "sequence_edit": ["mutate_residues", "check_wt_consistency"],
        "pyrosetta": ["score", "interface", "relax", "pack"],
        "proteina_complexa": ["protein_binder", "ligand_binder", "motif"],
    }
)

# Most explicit-mode tools expose their operating mode through `inputs.mode`.
# Some adapters already have a domain-specific field name in their public
# schema. Keep the selected-mode contract planner-facing while compiling into
# the adapter's real input field.
EXPLICIT_MODE_INPUT_FIELDS: dict[str, str] = {
    "sequence_edit": "operation",
}


def explicit_mode_input_field(tool_name: str) -> str:
    return EXPLICIT_MODE_INPUT_FIELDS.get(tool_name, "mode")

CandidateCapability = str

CANDIDATE_CAPABILITY_FIELDS: tuple[CandidateCapability, ...] = (
    "has_backbone_structure",
    "has_full_structure",
    "has_sequence",
    "has_candidate_scores",
)


@dataclass(frozen=True)
class CandidateCapabilityContract:
    consumes: frozenset[CandidateCapability] = frozenset()
    produces: frozenset[CandidateCapability] = frozenset()
    preserves_upstream: bool = True
    min_candidate_items: int = 1


_SEQUENCE = frozenset({"has_sequence"})
_BACKBONE = frozenset({"has_backbone_structure"})
_FULL_STRUCTURE = frozenset({"has_backbone_structure", "has_full_structure"})
_SCORES = frozenset({"has_candidate_scores"})
_SEQUENCE_SCORES = _SEQUENCE | _SCORES
_BACKBONE_SCORES = _BACKBONE | _SCORES
_BACKBONE_SEQUENCE_SCORES = _BACKBONE | _SEQUENCE | _SCORES
_FULL_SEQUENCE_SCORES = _FULL_STRUCTURE | _SEQUENCE | _SCORES


def _contract(
    *,
    consumes: frozenset[CandidateCapability] = frozenset(),
    produces: frozenset[CandidateCapability] = frozenset(),
    preserves_upstream: bool = True,
    min_candidate_items: int = 1,
) -> CandidateCapabilityContract:
    return CandidateCapabilityContract(
        consumes=consumes,
        produces=produces,
        preserves_upstream=preserves_upstream,
        min_candidate_items=min_candidate_items,
    )


TOOL_CANDIDATE_CONTRACTS: dict[str, CandidateCapabilityContract] = {
    # Sequence or family retrieval producers.
    "search_sequence": _contract(produces=_SEQUENCE_SCORES),
    "go_sequences": _contract(produces=_SEQUENCE),
    "interpro_sequences": _contract(produces=_SEQUENCE),
    "mmseqs2_gpu": _contract(consumes=_SEQUENCE, produces=_SEQUENCE_SCORES),
    "sequence_analysis": _contract(consumes=_SEQUENCE, produces=_SEQUENCE_SCORES),
    "family_profile": _contract(consumes=_SEQUENCE, preserves_upstream=False),
    "sequence_edit": _contract(consumes=_SEQUENCE, produces=_SEQUENCE),
    "esm2": _contract(consumes=_SEQUENCE, produces=_SEQUENCE_SCORES),
    "esm3": _contract(consumes=_SEQUENCE, produces=_FULL_STRUCTURE | _SEQUENCE | _SCORES),
    # Backbone/structure generators.
    "rfdiffusion": _contract(produces=_BACKBONE, preserves_upstream=False),
    "rfdiffusion3": _contract(consumes=_FULL_STRUCTURE, produces=_BACKBONE, preserves_upstream=False),
    "rfpeptides": _contract(produces=_BACKBONE, preserves_upstream=False),
    "boltzgen": _contract(produces=_FULL_SEQUENCE_SCORES),
    # Sequence design, backbone-design, or full-structure producers.
    "proteinmpnn": _contract(consumes=_BACKBONE, produces=_SEQUENCE),
    "ligandmpnn": _contract(consumes=_BACKBONE, produces=_SEQUENCE),
    "bindcraft": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_SEQUENCE_SCORES),
    "pxdesign": _contract(consumes=_FULL_STRUCTURE, produces=_BACKBONE_SEQUENCE_SCORES),
    "proteina_complexa": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_SEQUENCE_SCORES),
    "protenix_v2": _contract(consumes=_SEQUENCE, produces=_FULL_SEQUENCE_SCORES),
    "boltz2_affinity": _contract(consumes=_SEQUENCE, produces=_FULL_SEQUENCE_SCORES),
    # Structure and score consumers.
    "structure_analysis": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES),
    "foldseek": _contract(consumes=_BACKBONE, produces=_SCORES, preserves_upstream=False),
    "template_select": _contract(consumes=_BACKBONE, produces=_BACKBONE_SCORES),
    "tmalign": _contract(consumes=_BACKBONE, produces=_BACKBONE_SCORES, min_candidate_items=2),
    "ipsae": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES),
    "prodigy": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES),
    "pxmeter": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES, min_candidate_items=2),
    "pyrosetta": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES),
    "vina": _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES),
    "protein_qc": _contract(produces=_SCORES),
    "rank_designs": _contract(produces=_SCORES),
    "merge_candidates": _contract(),
}


def candidate_contract(tool_name: str) -> CandidateCapabilityContract:
    return TOOL_CANDIDATE_CONTRACTS.get(tool_name, CandidateCapabilityContract())


def planner_facing_candidate_contract(tool_name: str) -> CandidateCapabilityContract:
    if tool_name == "pxdesign":
        return _contract(consumes=_FULL_STRUCTURE, produces=_FULL_SEQUENCE_SCORES)
    return candidate_contract(tool_name)


TOOL_CANDIDATE_CAPABILITY_PROFILES: dict[str, list[dict[str, object]]] = {
    "rfdiffusion": [
        {
            "name": "unconditional_backbone_generation",
            "default": True,
            "inputs": {"mode": "unconditional"},
            "consumes": [],
            "produces": sorted(_BACKBONE),
            "description": "Generates backbone structures from noise without an upstream target structure.",
        },
        {
            "name": "target_conditioned_backbone_generation",
            "default": False,
            "inputs": {"mode": "partial_diffusion"},
            "consumes": sorted(_BACKBONE),
            "produces": sorted(_BACKBONE),
            "description": "Uses an upstream or user-provided target/motif structure and emits backbone candidates.",
        },
    ],
    "rfpeptides": [
        {
            "name": "targetless_peptide_backbone_generation",
            "default": True,
            "inputs": {},
            "consumes": [],
            "produces": sorted(_BACKBONE),
            "description": "Generates cyclic peptide backbones without a target when no target_pdb/candidates are supplied.",
        },
        {
            "name": "target_conditioned_peptide_backbone_generation",
            "default": False,
            "inputs": {"target_pdb": "<path or canonical candidates>"},
            "consumes": sorted(_BACKBONE),
            "produces": sorted(_BACKBONE),
            "description": "Uses a target structure and emits peptide backbone candidates.",
        },
    ],
    "rfdiffusion3": [
        {
            "name": "target_conditioned_binder_backbone_generation",
            "default": True,
            "inputs": {"target_pdb": "<path or canonical candidates>"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_BACKBONE),
            "description": "Current adapter requires a target structure and emits binder backbone candidates.",
        },
    ],
    "boltzgen": [
        {
            "name": "targetless_design_inverse_fold_folding_analyze",
            "default": True,
            "inputs": {},
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": (
                "Runs design, inverse-folding, folding, and analysis. Filtering is not part "
                "of the VPA adapter contract."
            ),
        },
        {
            "name": "target_conditioned_design_inverse_fold_folding_analyze",
            "default": False,
            "inputs": {"target_pdb": "<path or canonical candidates>"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": (
                "Consumes a full target structure and runs BoltzGen design, inverse-folding, "
                "folding, and analysis."
            ),
        },
    ],
    "bindcraft": [
        {
            "name": "target_conditioned_full_binder_design",
            "default": True,
            "inputs": {"target_pdb": "<path or canonical candidates>"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": "Requires a target structure and emits sequenced, scored binder structures.",
        },
    ],
    "pxdesign": [
        {
            "name": "pipeline",
            "default": True,
            "inputs": {"target_pdb": "<path or canonical candidates>"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": (
                "Designs protein binders against a protein target and returns "
                "sequenced, scored, full-structure binder candidates for downstream "
                "selection. It is not a small-molecule ligand binder-design route; "
                "use a ligand-binder capable tool for ligand targets."
            ),
        },
    ],
    "proteina_complexa": [
        {
            "name": "protein_binder",
            "default": True,
            "inputs": {"mode": "protein_binder", "target_pdb": "<path or canonical candidates>"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": "Requires a protein target structure and emits sequenced, scored binder candidates.",
        },
        {
            "name": "ligand_binder",
            "default": False,
            "inputs": {"mode": "ligand_binder", "target_pdb": "<path>", "ligand_smiles": "<SMILES>"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": "Requires ligand target context and emits sequenced, scored binder candidates.",
        },
        {
            "name": "motif",
            "default": False,
            "inputs": {"mode": "motif", "motif_pdb": "<path or canonical candidates>"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": "Requires a motif structure and emits sequenced, scored scaffold candidates.",
        },
    ],
    "esm3": [
        {
            "name": "sequence_embedding",
            "default": False,
            "inputs": {"tracks": ["embedding"]},
            "consumes": sorted(_SEQUENCE),
            "produces": sorted(_SEQUENCE_SCORES),
            "description": "Consumes sequence and emits sequence-level embeddings/scores without a predicted structure.",
        },
        {
            "name": "sequence_to_structure",
            "default": True,
            "inputs": {"tracks": ["sequence", "structure"]},
            "consumes": sorted(_SEQUENCE),
            "produces": sorted(_FULL_SEQUENCE_SCORES),
            "description": "Consumes sequence and emits sequence plus predicted structure and confidence scores.",
        },
    ],
    "pxmeter": [
        {
            "name": "reference_predicted_structure_metrics",
            "default": True,
            "inputs": {"candidates": "at least two full-structure candidates"},
            "consumes": sorted(_FULL_STRUCTURE),
            "produces": sorted(_FULL_STRUCTURE | _SCORES),
            "min_candidate_items": 2,
            "description": "Scores predicted structure against reference structure; requires two structure candidates or explicit CIF paths.",
        },
    ],
    "tmalign": [
        {
            "name": "pairwise_structure_alignment",
            "default": True,
            "inputs": {"candidates": "at least two structure candidates"},
            "consumes": sorted(_BACKBONE),
            "produces": sorted(_BACKBONE_SCORES),
            "min_candidate_items": 2,
            "description": "Aligns two structures; backbone coordinates are sufficient, full structures are also accepted.",
        },
    ],
}


def candidate_capability_profiles(tool_name: str) -> list[dict[str, object]]:
    return TOOL_CANDIDATE_CAPABILITY_PROFILES.get(tool_name, [])


def candidate_contract_for_inputs(
    tool_name: str,
    inputs: dict[str, Any] | None,
) -> CandidateCapabilityContract:
    inputs = inputs if isinstance(inputs, dict) else {}
    if tool_name == "family_profile":
        if inputs.get("profile") is not None:
            return _contract(consumes=_SEQUENCE, produces=_SEQUENCE_SCORES)
        return _contract(consumes=_SEQUENCE, preserves_upstream=False)
    if tool_name == "rfdiffusion":
        mode = inputs.get("mode") or "unconditional"
        if mode in {
            "unconditional",
            "unconditional_generation",
            "backbone_generation",
            "symmetric_oligomer",
            "symmetric_oligomer_design",
        }:
            return _contract(produces=_BACKBONE, preserves_upstream=False)
        return _contract(consumes=_BACKBONE, produces=_BACKBONE, preserves_upstream=False)
    if tool_name == "rfpeptides":
        if inputs.get("target_pdb") or inputs.get("candidates"):
            return _contract(consumes=_BACKBONE, produces=_BACKBONE, preserves_upstream=False)
        return _contract(produces=_BACKBONE, preserves_upstream=False)
    if tool_name == "esm3":
        tracks = inputs.get("tracks")
        track_set = {str(item).lower() for item in tracks} if isinstance(tracks, list) else {"sequence", "structure"}
        if "structure" in track_set:
            return _contract(consumes=_SEQUENCE, produces=_FULL_SEQUENCE_SCORES)
        return _contract(consumes=_SEQUENCE, produces=_SEQUENCE_SCORES)
    if tool_name == "pxdesign":
        return _contract(consumes=_FULL_STRUCTURE, produces=_BACKBONE_SEQUENCE_SCORES)
    if tool_name == "proteina_complexa":
        return _contract(consumes=_FULL_STRUCTURE, produces=_FULL_SEQUENCE_SCORES)
    if tool_name == "boltzgen":
        if inputs.get("target_pdb") or inputs.get("target_yaml") or inputs.get("candidates"):
            return _contract(consumes=_FULL_STRUCTURE, produces=_FULL_SEQUENCE_SCORES)
        return _contract(produces=_FULL_SEQUENCE_SCORES)
    if tool_name == "pyrosetta":
        mode = inputs.get("mode") or "score"
        if mode == "interface":
            return _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES)
        return _contract(consumes=_FULL_STRUCTURE, produces=_FULL_STRUCTURE | _SCORES)
    return candidate_contract(tool_name)


def candidate_contract_for_node(
    tool_name: str,
    inputs: dict[str, Any] | None,
    upstreams: list[Any] | tuple[Any, ...] | None,
) -> CandidateCapabilityContract:
    if tool_name == "family_profile" and _has_profile_source_upstream(upstreams):
        prepared = dict(inputs or {})
        prepared.setdefault("profile", "${profile_source}")
        return candidate_contract_for_inputs(tool_name, prepared)
    return candidate_contract_for_inputs(tool_name, inputs)


def profile_guided_default_inputs(
    tool_name: str,
    upstreams: list[Any] | tuple[Any, ...] | None,
) -> dict[str, Any]:
    if tool_name == "esm2" and _has_profile_source_upstream(upstreams):
        return {"mode": "sequence_generation"}
    return {}


def _has_profile_source_upstream(upstreams: list[Any] | tuple[Any, ...] | None) -> bool:
    for upstream in upstreams or []:
        if isinstance(upstream, dict) and upstream.get("relationship") == "profile_source":
            return True
    return False


def candidate_capabilities_from_record(candidate: dict[str, object]) -> set[CandidateCapability]:
    capabilities: set[CandidateCapability] = set()
    explicit = candidate.get("capabilities")
    explicit_values = explicit if isinstance(explicit, dict) else {}
    explicitly_false = {
        field
        for field in CANDIDATE_CAPABILITY_FIELDS
        if explicit_values.get(field) is False
    }
    for field in CANDIDATE_CAPABILITY_FIELDS:
        if explicit_values.get(field) is True:
            capabilities.add(field)
    sequence = candidate.get("sequence")
    if (
        "has_sequence" not in explicit_values
        and isinstance(sequence, str)
        and sequence.strip()
    ):
        capabilities.add("has_sequence")
    if (
        candidate.get("structure_path")
        or candidate.get("backbone_path")
        or candidate.get("full_structure_path")
        or candidate.get("pdb_path")
        or candidate.get("cif_path")
    ):
        if "has_backbone_structure" not in explicitly_false:
            capabilities.add("has_backbone_structure")
        if (
            "has_full_structure" not in explicit_values
            and (
                candidate.get("full_structure") is not False
                or candidate.get("full_structure_path")
            )
        ):
            capabilities.add("has_full_structure")
    scores = candidate.get("scores")
    if "has_candidate_scores" not in explicit_values and isinstance(scores, dict) and scores:
        capabilities.add("has_candidate_scores")
    for key in ("score", "rank", "plddt", "ptm", "iptm", "ranking_score"):
        if "has_candidate_scores" not in explicit_values and key in candidate:
            capabilities.add("has_candidate_scores")
            break
    return capabilities


def candidate_capability_flags(
    *,
    has_sequence: bool = False,
    has_backbone_structure: bool = False,
    has_full_structure: bool = False,
    has_candidate_scores: bool = False,
) -> dict[str, bool]:
    return {
        "has_sequence": bool(has_sequence),
        "has_backbone_structure": bool(has_backbone_structure),
        "has_full_structure": bool(has_full_structure),
        "has_candidate_scores": bool(has_candidate_scores),
    }

def annotate_candidate_capabilities(candidate: dict[str, object]) -> dict[str, object]:
    capabilities = candidate_capabilities_from_record(candidate)
    out = dict(candidate)
    out["capabilities"] = {
        field: field in capabilities
        for field in CANDIDATE_CAPABILITY_FIELDS
    }
    return out


def build_canonical_wiring_catalog() -> dict[str, object]:
    producers = [
        producer
        for producer in sorted(CANONICAL_PRODUCERS)
        if producer != "merge_candidates"
    ]
    return {
        "candidate_producers": producers,
        "candidate_consumers": dict(CANONICAL_CONSUMERS),
        "term_producers": sorted(CANONICAL_TERM_PRODUCERS),
        "term_consumers": dict(CANONICAL_TERM_CONSUMERS),
        "profile_producers": dict(CANONICAL_PROFILE_PRODUCERS),
        "profile_consumers": dict(CANONICAL_PROFILE_CONSUMERS),
        "relationship": "candidate_source",
        "term_relationship": "term_source",
        "profile_relationship": "profile_source",
        "candidate_capabilities": list(CANDIDATE_CAPABILITY_FIELDS),
        "candidate_contracts": {
            name: {
                "consumes": sorted(planner_facing_candidate_contract(name).consumes),
                "produces": sorted(planner_facing_candidate_contract(name).produces),
                "preserves_upstream": planner_facing_candidate_contract(
                    name
                ).preserves_upstream,
                "min_candidate_items": planner_facing_candidate_contract(
                    name
                ).min_candidate_items,
            }
            for name in sorted(TOOL_CANDIDATE_CONTRACTS)
        },
        "candidate_capability_profiles": TOOL_CANDIDATE_CAPABILITY_PROFILES,
    }
