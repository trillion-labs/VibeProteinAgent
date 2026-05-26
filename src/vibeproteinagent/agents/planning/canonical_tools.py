from __future__ import annotations

CANONICAL_PRODUCERS: set[str] = {
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
    "dssp",
    "tmalign",
    "pxmeter",
    "boltz2_affinity",
    "esm2",
    "esm3",
    "protenix_v2",
    # Virtual in-memory producer populated by the controller between rounds.
    "prior_round",
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
    "dssp": "candidates",
    "tmalign": "candidates",
    "pxmeter": "candidates",
    "boltz2_affinity": "candidates",
    "esm2": "candidates",
    "esm3": "candidates",
    "protenix_v2": "candidates",
}

CANONICAL_SCALAR_CONSUMERS: dict[str, dict[str, str]] = {}

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
        "pxdesign_pipeline_multi_gpu",
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
        "pxdesign_pipeline_multi_gpu": "candidates",
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
        "pxdesign": ["infer", "pipeline"],
        "sequence_edit": ["mutate_residues", "check_wt_consistency"],
        "pyrosetta": ["score", "interface", "relax", "pack"],
        "proteina_complexa": ["protein_binder", "ligand_binder", "motif"],
    }
)


def build_canonical_wiring_for_planner(
    *,
    has_prior_round_context: bool,
) -> dict[str, object]:
    producers = [
        producer
        for producer in sorted(CANONICAL_PRODUCERS)
        if producer != "merge_candidates"
    ]
    if not has_prior_round_context:
        producers = [
            producer for producer in producers if producer != "prior_round"
        ]
    return {
        "candidate_producers": producers,
        "candidate_consumers": dict(CANONICAL_CONSUMERS),
        "relationship": "candidate_source",
    }
