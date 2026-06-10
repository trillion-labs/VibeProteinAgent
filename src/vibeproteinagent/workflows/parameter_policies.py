from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ParameterOwnership = Literal[
    "intent",
    "budget",
    "mode",
    "source_artifact",
    "candidate_stream",
    "profile_evidence",
    "structural_selector",
    "length_constraint",
    "scoring_policy",
    "environment",
    "advanced_override",
]

ParameterSource = Literal[
    "user_explicit",
    "known_input",
    "planner_literal",
    "compiled_wiring",
    "upstream_candidate",
    "runtime_derived",
    "adapter_default",
]

STRONG_USER_SOURCES: frozenset[str] = frozenset({"user_explicit", "known_input"})
RUNTIME_SOURCES: frozenset[str] = frozenset(
    {"compiled_wiring", "upstream_candidate", "runtime_derived"}
)


class ParameterPolicy(BaseModel):
    field: str
    ownership: ParameterOwnership
    root_planner_allowed: bool = True
    downstream_planner_allowed: bool = True
    derive_when_upstream: bool = False
    user_value_must_validate_against_runtime: bool = True
    soft_preference: bool = False


def _policy(
    field: str,
    ownership: ParameterOwnership,
    *,
    root_planner_allowed: bool = True,
    downstream_planner_allowed: bool = True,
    derive_when_upstream: bool = False,
    user_value_must_validate_against_runtime: bool = True,
    soft_preference: bool = False,
) -> ParameterPolicy:
    return ParameterPolicy(
        field=field,
        ownership=ownership,
        root_planner_allowed=root_planner_allowed,
        downstream_planner_allowed=downstream_planner_allowed,
        derive_when_upstream=derive_when_upstream,
        user_value_must_validate_against_runtime=user_value_must_validate_against_runtime,
        soft_preference=soft_preference,
    )


def _policies(*items: ParameterPolicy) -> dict[str, ParameterPolicy]:
    return {item.field: item for item in items}


TOOL_PARAMETER_POLICIES: dict[str, dict[str, ParameterPolicy]] = {
    "proteinmpnn": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy(
            "profile_guidance",
            "profile_evidence",
            root_planner_allowed=False,
            downstream_planner_allowed=False,
        ),
        _policy("pdb_path", "source_artifact", derive_when_upstream=True),
        _policy("chains", "structural_selector", derive_when_upstream=True),
        _policy("num_designs", "budget"),
        _policy("temperature", "budget"),
        _policy("omit_aas", "intent"),
        _policy("seed", "budget"),
    ),
    "ligandmpnn": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("pdb_path", "source_artifact", derive_when_upstream=True),
        _policy("chains", "structural_selector", derive_when_upstream=True),
        _policy("num_designs", "budget"),
        _policy("temperature", "budget"),
        _policy("omit_aas", "intent"),
        _policy("seed", "budget"),
        _policy(
            "model_type",
            "intent",
            root_planner_allowed=False,
            downstream_planner_allowed=False,
        ),
        _policy("redesigned_residues", "structural_selector"),
        _policy("redesigned_positions", "structural_selector"),
        _policy("sequence", "source_artifact", derive_when_upstream=True),
        _policy("fasta_path", "source_artifact"),
        _policy("position_basis", "intent"),
    ),
    "pyrosetta": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("pdb_path", "source_artifact", derive_when_upstream=True),
        _policy("mode", "mode"),
        _policy("score_fn", "intent"),
        _policy("chains_a", "structural_selector", derive_when_upstream=True),
        _policy("chains_b", "structural_selector", derive_when_upstream=True),
        _policy("max_relax_cycles", "budget"),
        _policy("seed", "budget"),
    ),
    "rfdiffusion": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy(
            "profile_guidance",
            "profile_evidence",
            root_planner_allowed=False,
            downstream_planner_allowed=False,
        ),
        _policy("target_pdb", "source_artifact", derive_when_upstream=True),
        _policy("target_chain", "structural_selector", derive_when_upstream=True),
        _policy("target_residues", "structural_selector"),
        _policy("target_hotspots", "structural_selector"),
        _policy("target_hotspot_residue_ids", "structural_selector"),
        _policy("target_sequence", "source_artifact", derive_when_upstream=True),
        _policy("target_fasta_path", "source_artifact"),
        _policy("position_basis", "intent"),
        _policy("contigs", "advanced_override"),
        _policy("length_min", "length_constraint"),
        _policy("length_max", "length_constraint"),
        _policy("total_length", "length_constraint"),
        _policy("binder_length_min", "length_constraint"),
        _policy("binder_length_max", "length_constraint"),
        _policy("mode", "mode"),
        _policy("symmetry", "mode"),
        _policy("num_designs", "budget"),
        _policy("num_diffusion_steps", "budget"),
        _policy("partial_T", "mode"),
        _policy("noise_scale_ca", "intent"),
        _policy("noise_scale_frame", "intent"),
        _policy("inpaint_seq", "structural_selector"),
        _policy("inpaint_str", "structural_selector"),
        _policy("provide_seq", "structural_selector"),
        _policy("contig_length", "length_constraint"),
        _policy("checkpoint_path", "environment", root_planner_allowed=False),
        _policy("model_directory_path", "environment", root_planner_allowed=False),
        _policy("scaffold_dir", "source_artifact"),
        _policy("target_ss", "source_artifact"),
        _policy("target_adj", "source_artifact"),
        _policy("guiding_potentials", "advanced_override"),
        _policy("guide_scale", "advanced_override"),
        _policy("guide_decay", "advanced_override"),
        _policy("olig_intra_all", "advanced_override"),
        _policy("olig_inter_all", "advanced_override"),
        _policy("hydra_overrides", "advanced_override", root_planner_allowed=False),
        _policy("seed", "budget"),
    ),
    "rfdiffusion3": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("target_pdb", "source_artifact", derive_when_upstream=True),
        _policy("target_chain", "structural_selector", derive_when_upstream=True),
        _policy("target_hotspots", "structural_selector"),
        _policy("target_hotspot_residue_ids", "structural_selector"),
        _policy("target_sequence", "source_artifact", derive_when_upstream=True),
        _policy("target_fasta_path", "source_artifact"),
        _policy("position_basis", "intent"),
        _policy("binder_length", "length_constraint"),
        _policy("num_designs", "budget"),
        _policy("num_diffusion_steps", "budget"),
        _policy("seed", "budget"),
    ),
    "rfpeptides": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("target_pdb", "source_artifact", derive_when_upstream=True),
        _policy("target_chain", "structural_selector", derive_when_upstream=True),
        _policy("target_residues", "structural_selector"),
        _policy("target_hotspots", "structural_selector"),
        _policy("target_hotspot_residue_ids", "structural_selector"),
        _policy("target_sequence", "source_artifact", derive_when_upstream=True),
        _policy("target_fasta_path", "source_artifact"),
        _policy("position_basis", "intent"),
        _policy("peptide_length", "length_constraint"),
        _policy("num_designs", "budget"),
        _policy("num_diffusion_steps", "budget"),
        _policy("noise_scale", "intent"),
        _policy("seed", "budget"),
    ),
    "bindcraft": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("target_pdb", "source_artifact", derive_when_upstream=True),
        _policy("target_chain", "structural_selector", derive_when_upstream=True),
        _policy("target_hotspots", "structural_selector"),
        _policy("target_hotspot_residue_ids", "structural_selector"),
        _policy("target_sequence", "source_artifact", derive_when_upstream=True),
        _policy("target_fasta_path", "source_artifact"),
        _policy("position_basis", "intent"),
        _policy("binder_length", "length_constraint"),
        _policy("num_designs", "budget"),
        _policy("num_recycles", "budget"),
        _policy("seed", "budget"),
        _policy("max_trajectories", "budget"),
        _policy("max_attempted_trajectories", "budget"),
        _policy("max_runtime_s", "budget"),
    ),
    "pxdesign": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("target_pdb", "source_artifact", derive_when_upstream=True),
        _policy("target_chain", "structural_selector", derive_when_upstream=True),
        _policy("binder_length", "length_constraint"),
        _policy("num_samples", "budget"),
        _policy("num_steps", "budget"),
        _policy("seed", "budget"),
        _policy("temperature", "budget"),
        _policy("hotspots", "structural_selector"),
        _policy("crop", "structural_selector"),
        _policy("top_k_designs", "budget"),
        _policy("mpnn_temperature", "budget"),
    ),
    "proteina_complexa": _policies(
        _policy("mode", "mode"),
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("target_pdb", "source_artifact", derive_when_upstream=True),
        _policy("target_chain", "structural_selector", derive_when_upstream=True),
        _policy("target_pocket_residues", "structural_selector"),
        _policy("target_pocket_residue_ids", "structural_selector"),
        _policy("target_sequence", "source_artifact", derive_when_upstream=True),
        _policy("target_fasta_path", "source_artifact"),
        _policy("ligand_smiles", "intent"),
        _policy("motif_pdb", "source_artifact", derive_when_upstream=True),
        _policy("motif_chain", "structural_selector"),
        _policy("motif_residues", "structural_selector"),
        _policy("motif_residue_ids", "structural_selector"),
        _policy("motif_sequence", "source_artifact", derive_when_upstream=True),
        _policy("motif_fasta_path", "source_artifact"),
        _policy("position_basis", "intent"),
        _policy("total_length", "length_constraint"),
        _policy("binder_length", "length_constraint"),
        _policy("num_designs", "budget"),
        _policy("seed", "budget"),
        _policy("run_name", "intent"),
        _policy("search_algorithm", "intent"),
        _policy("reward_model", "intent"),
    ),
    "boltzgen": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("target_pdb", "source_artifact", derive_when_upstream=True),
        _policy("target_yaml", "source_artifact"),
        _policy("target_chain", "structural_selector", derive_when_upstream=True),
        _policy("binder_length", "length_constraint"),
        _policy("num_designs", "budget"),
        _policy("num_diffusion_steps", "budget"),
        _policy("protocol", "intent"),
        _policy("seed", "budget"),
    ),
    "esm2": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy(
            "profile_guidance",
            "profile_evidence",
            root_planner_allowed=False,
            downstream_planner_allowed=False,
        ),
        _policy("sequence", "source_artifact", derive_when_upstream=True),
        _policy("mode", "mode"),
        _policy("model_name", "environment"),
        _policy("return_attention", "intent"),
        _policy("mask_positions", "structural_selector"),
        _policy("mask_indices_zero_based", "structural_selector"),
    ),
    "esm3": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy(
            "profile_guidance",
            "profile_evidence",
            root_planner_allowed=False,
            downstream_planner_allowed=False,
        ),
        _policy("sequence", "source_artifact", derive_when_upstream=True),
        _policy("model_name", "environment"),
        _policy("tracks", "mode"),
        _policy("num_steps", "budget"),
        _policy("temperature", "budget"),
    ),
    "protenix_v2": _policies(
        _policy("input_json", "source_artifact", derive_when_upstream=True),
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("seeds", "budget"),
        _policy("use_msa", "intent"),
        _policy("msa_backend", "intent"),
        _policy("use_template", "intent"),
        _policy("use_rna_msa", "intent"),
        _policy("enable_cache", "environment"),
        _policy("enable_tf32", "environment"),
        _policy("n_sample", "budget"),
        _policy("n_step", "budget"),
        _policy("n_cycle", "budget"),
    ),
    "boltz2_affinity": _policies(
        _policy("input_yaml", "source_artifact", derive_when_upstream=True),
        _policy("sequences", "source_artifact", derive_when_upstream=True),
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("binder_chain", "structural_selector", derive_when_upstream=True),
        _policy("seeds", "budget"),
        _policy("recycling_steps", "budget"),
        _policy("sampling_steps", "budget"),
        _policy("diffusion_samples", "budget"),
        _policy("sampling_steps_affinity", "budget"),
        _policy("diffusion_samples_affinity", "budget"),
        _policy("use_msa", "intent"),
        _policy("msa_backend", "intent"),
        _policy("affinity_mw_correction", "intent"),
    ),
    "objective_rank": _policies(
        _policy("job_ids", "candidate_stream", root_planner_allowed=False),
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("objective_name", "intent"),
        _policy("gates", "scoring_policy"),
        _policy("score_formula", "scoring_policy"),
        _policy("score_direction", "scoring_policy"),
        _policy("constants", "scoring_policy"),
        _policy("missing", "scoring_policy"),
        _policy("tie_breakers", "scoring_policy"),
        _policy("top_k", "budget"),
        _policy("output_prefix", "intent"),
    ),
    "family_profile": _policies(
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy(
            "profile",
            "profile_evidence",
            root_planner_allowed=False,
            downstream_planner_allowed=False,
        ),
        _policy("fasta_path", "source_artifact"),
        _policy("alignment_path", "source_artifact"),
        _policy("family_id", "intent"),
        _policy("family_label", "intent"),
        _policy("min_conservation", "scoring_policy"),
        _policy("motif_window", "scoring_policy"),
        _policy("output_prefix", "intent"),
    ),
    "structure_analysis": _policies(
        _policy("structure_path", "source_artifact", derive_when_upstream=True),
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("sequence", "source_artifact", derive_when_upstream=True),
        _policy("fasta_path", "source_artifact"),
        _policy("position_basis", "intent"),
        _policy("chain_id", "structural_selector", derive_when_upstream=True),
        _policy("interface_chain_ids", "structural_selector"),
        _policy("ligand_resnames", "structural_selector"),
        _policy("positions", "structural_selector"),
        _policy("max_annotations", "budget"),
        _policy("contact_distance_angstrom", "scoring_policy"),
        _policy("surface_rsa_threshold", "scoring_policy"),
        _policy("buried_rsa_threshold", "scoring_policy"),
        _policy("run_dssp", "intent"),
        _policy("output_prefix", "intent"),
    ),
    "sequence_analysis": _policies(
        _policy("sequence", "source_artifact", derive_when_upstream=True),
        _policy("fasta_path", "source_artifact"),
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("positions", "structural_selector"),
        _policy("ph", "scoring_policy"),
        _policy("max_annotations", "budget"),
        _policy("include_regions", "intent"),
        _policy("include_residue_table", "intent"),
        _policy("output_prefix", "intent"),
    ),
    "protein_qc": _policies(
        _policy("sequence", "source_artifact", derive_when_upstream=True),
        _policy("pdb_path", "source_artifact", derive_when_upstream=True),
        _policy("candidates", "candidate_stream", root_planner_allowed=False),
        _policy("metrics", "scoring_policy"),
    ),
}


def policy_for(tool: str, field: str) -> ParameterPolicy:
    explicit = TOOL_PARAMETER_POLICIES.get(tool, {}).get(field)
    if explicit is not None:
        return explicit
    return fallback_policy(field)


def fallback_policy(field: str) -> ParameterPolicy:
    lowered = field.lower()
    if lowered in {"candidates", "candidate_sets", "predictions", "analyses", "qc_results"}:
        return _policy(field, "candidate_stream", root_planner_allowed=False)
    if lowered in {"profile", "profile_guidance"}:
        return _policy(
            field,
            "profile_evidence",
            root_planner_allowed=False,
            downstream_planner_allowed=False,
        )
    if (
        lowered.endswith("_path")
        or lowered.endswith("_file")
        or lowered.endswith("_pdb")
        or lowered.endswith("_cif")
        or lowered in {"sequence", "sequences", "input_json", "input_yaml"}
    ):
        return _policy(field, "source_artifact", derive_when_upstream=True)
    if lowered in {"mode", "protocol", "model_type", "tracks", "symmetry"}:
        return _policy(field, "mode")
    if (
        lowered.startswith("num_")
        or lowered in {"top_k", "top_k_designs", "temperature", "seed", "seeds"}
        or lowered.endswith("_steps")
        or lowered.endswith("_samples")
    ):
        return _policy(field, "budget")
    if "length" in lowered:
        return _policy(field, "length_constraint")
    if "chain" in lowered or "residue" in lowered or "hotspot" in lowered:
        return _policy(field, "structural_selector")
    return _policy(field, "intent")


def field_ownership_summary(tool: str, fields: list[str]) -> dict[str, str]:
    return {field: policy_for(tool, field).ownership for field in fields}


def planner_ownership_hint(tool: str, field: str) -> str:
    policy = policy_for(tool, field)
    if policy.ownership == "candidate_stream":
        return "compiler_controller_wired"
    if policy.derive_when_upstream:
        return "runtime_derived_when_downstream_or_known_input"
    if policy.root_planner_allowed and policy.downstream_planner_allowed:
        return "planner_allowed"
    if policy.root_planner_allowed:
        return "planner_allowed_root_only"
    return policy.ownership


def planner_ownership_hints(tool: str, fields: list[str]) -> dict[str, str]:
    return {field: planner_ownership_hint(tool, field) for field in fields}


__all__ = [
    "RUNTIME_SOURCES",
    "STRONG_USER_SOURCES",
    "TOOL_PARAMETER_POLICIES",
    "ParameterOwnership",
    "ParameterPolicy",
    "ParameterSource",
    "fallback_policy",
    "field_ownership_summary",
    "planner_ownership_hint",
    "planner_ownership_hints",
    "policy_for",
]
