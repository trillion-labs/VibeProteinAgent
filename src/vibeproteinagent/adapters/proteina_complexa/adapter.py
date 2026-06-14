from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters import _proteina_complexa_common as common
from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.adapters.proteina_complexa import availability_skip_reason
from vibeproteinagent.adapters.residue_mapping import (
    PositionBasis,
    parse_residue_id,
    parse_structure_residue_refs,
    resolve_positions,
)
from vibeproteinagent.candidates.capabilities import candidate_capability_flags
from vibeproteinagent.candidates.chain_roles import (
    chain_sequence_from_structure,
    complex_chain_metadata,
)
from vibeproteinagent.candidates.schema import stable_candidate_id
from vibeproteinagent.core.errors import VPAError

ProteinaComplexaMode = Literal["protein_binder", "ligand_binder", "motif"]
ProteinaComplexaSearchAlgorithm = Literal[
    "default",
    "single-pass",
    "best-of-n",
    "beam-search",
    "fk-steering",
    "mcts",
]
ProteinaComplexaRewardModel = Literal["default", "none"]
ProteinaComplexaStep = Literal["generate", "evaluate"]

_MODE_CONFIG = {
    "protein_binder": {
        "pipeline": "configs/search_binder_local_pipeline.yaml",
        "ckpt": "complexa.ckpt",
        "ae_ckpt": "complexa_ae.ckpt",
    },
    "ligand_binder": {
        "pipeline": "configs/search_ligand_binder_local_pipeline.yaml",
        "ckpt": "complexa_ligand.ckpt",
        "ae_ckpt": "complexa_ligand_ae.ckpt",
    },
    "motif": {
        "pipeline": "configs/search_motif_local_pipeline.yaml",
        "ckpt": "complexa_ame.ckpt",
        "ae_ckpt": "complexa_ame_ae.ckpt",
    },
}

_MODE_DEFAULT_NUM_DESIGNS: dict[ProteinaComplexaMode, int] = {
    "protein_binder": 4,
    "ligand_binder": 2,
    "motif": 10,
}


class ProteinaComplexaInput(BaseModel):
    mode: ProteinaComplexaMode = Field(
        ...,
        description=(
            "Which Proteina-Complexa task to run: protein_binder, ligand_binder, "
            "or motif."
        ),
    )
    target_pdb: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Protein target PDB for protein_binder, or a structure containing "
            "the ligand pocket for ligand_binder."
        ),
    )
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    target_chain: str = Field(default="A", min_length=1, max_length=1)
    target_pocket_residues: list[int] = Field(default_factory=list)
    target_pocket_residue_ids: list[str] = Field(default_factory=list)
    target_sequence: str | None = None
    target_fasta_path: str | None = Field(default=None, min_length=1)
    ligand_smiles: str | None = Field(default=None, min_length=1)
    motif_pdb: str | None = Field(default=None, min_length=1)
    motif_chain: str = Field(default="A", min_length=1, max_length=1)
    motif_residues: list[int] = Field(default_factory=list)
    motif_residue_ids: list[str] = Field(default_factory=list)
    motif_sequence: str | None = None
    motif_fasta_path: str | None = Field(default=None, min_length=1)
    position_basis: PositionBasis = "auto"
    total_length: int | None = Field(default=None, ge=30, le=400)
    binder_length: int = Field(default=80, ge=30, le=200)
    num_designs: int | None = Field(default=None, ge=1, le=128)
    seed: int | None = None
    run_name: str | None = None
    search_algorithm: ProteinaComplexaSearchAlgorithm = Field(
        default="default",
        description=(
            "Generation search strategy. default preserves Proteina-Complexa's "
            "upstream mode config; single-pass is useful for small smoke runs."
        ),
    )
    reward_model: ProteinaComplexaRewardModel = Field(
        default="default",
        description=(
            "Reward model policy. default preserves the upstream config; none "
            "disables reward scoring/guidance for fast compatibility smoke runs."
        ),
    )
    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_structure(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        candidates = data.get("candidates")
        if not candidates:
            return data
        canonical = [_is_canonical_candidate(item) for item in candidates]
        if any(canonical) and not all(canonical):
            raise ValueError(
                "Mixed canonical and legacy candidate items are not supported. "
                "Pass either all canonical Candidate[] or all legacy items."
            )
        if all(canonical):
            structure_path = candidates[0].get("structure_path")
            if not structure_path:
                raise ValueError(
                    "proteina_complexa canonical Candidate[] input requires structure_path"
                )
            out = dict(data)
            if out.get("mode") == "motif":
                out.setdefault("motif_pdb", structure_path)
                if candidates[0].get("sequence"):
                    out.setdefault("motif_sequence", candidates[0].get("sequence"))
            else:
                out.setdefault("target_pdb", structure_path)
                if candidates[0].get("sequence"):
                    out.setdefault("target_sequence", candidates[0].get("sequence"))
            return out
        return data

    @model_validator(mode="after")
    def _validate_mode_specific_inputs(self) -> ProteinaComplexaInput:
        if self.num_designs is None:
            self.num_designs = _MODE_DEFAULT_NUM_DESIGNS[self.mode]
        if self.mode == "protein_binder":
            if not self.target_pdb:
                raise ValueError("protein_binder mode requires target_pdb")
        elif self.mode == "ligand_binder":
            missing = []
            if not self.target_pdb:
                missing.append("target_pdb")
            if not self.target_pocket_residues and not self.target_pocket_residue_ids:
                missing.append("target_pocket_residues")
            if not self.ligand_smiles:
                missing.append("ligand_smiles")
            if missing:
                raise ValueError(
                    "ligand_binder mode requires " + ", ".join(missing)
                )
        elif self.mode == "motif":
            missing = []
            if not self.motif_pdb:
                missing.append("motif_pdb")
            if not self.motif_residues and not self.motif_residue_ids:
                missing.append("motif_residues")
            if self.total_length is None:
                missing.append("total_length")
            if missing:
                raise ValueError("motif mode requires " + ", ".join(missing))
        return self


class ProteinaComplexaDesign(BaseModel):
    pdb_path: str
    sequence: str | None = None
    length: int
    score: float | None = None
    iptm: float | None = None
    plddt: float | None = None
    binder_refold_rmsd: float | None = None
    ligand_contacts: int | None = None
    ligand_rmsd: float | None = None
    motif_rmsd: float | None = None
    scaffold_plddt: float | None = None
    designability: float | None = None


class ProteinaComplexaOutput(BaseModel):
    mode: ProteinaComplexaMode
    designs: list[ProteinaComplexaDesign]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    runtime_ms: int
    num_designs_returned: int
    seeds_used: list[int]
    position_mappings: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    recovered_after_evaluation_failure: bool = False


@register_adapter("descriptor.yaml")
class ProteinaComplexaAdapter(ToolAdapter):
    """Unified product-facing adapter for Proteina-Complexa task modes."""

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        del config, weights_dir
        self.device = device
        self._sidecars: dict[
            tuple[int | None, tuple[tuple[str, str], ...]], common.PersistentPythonSidecar
        ] = {}

    @torch.no_grad()
    def run(
        self, params: ProteinaComplexaInput, ctx: AdapterRunContext
    ) -> ProteinaComplexaOutput:
        started = time.time()
        ctx.emit_progress(0.02)

        if not torch.cuda.is_available():
            raise RuntimeError("Proteina-Complexa adapter requires CUDA")

        skip_reason = availability_skip_reason()
        if skip_reason is not None:
            raise RuntimeError(skip_reason)

        mode_config = _MODE_CONFIG[params.mode]
        ckpt = str(mode_config["ckpt"])
        ae_ckpt = str(mode_config["ae_ckpt"])
        if not common.ckpt_pair_present(ckpt, ae_ckpt):
            raise FileNotFoundError(
                f"Proteina-Complexa checkpoints not found ({ckpt} + {ae_ckpt}) "
                f"under {common.proteina_complexa_ckpts()}."
            )

        prepared_input, task_name = _prepare_mode_input(params, ctx.job_dir)
        position_mappings, mapping_warnings = _position_mapping_audit(params)
        run_name = params.run_name or f"vpa_{ctx.job_id}"
        overrides = _build_overrides(
            params=params,
            run_name=run_name,
            prepared_input=prepared_input,
        )
        pipeline = str(mode_config["pipeline"])
        config_path = common.pipeline_config(pipeline)
        extra_env = {"USE_V2_COMPLEXA_ARCH": "True"} if params.mode == "motif" else None
        steps = _resolve_steps(params)

        run_dir = common.inference_run_dir(
            config_stem=Path(pipeline).stem,
            task_name=task_name,
            run_name=run_name,
        )
        run_started_at = time.time()
        run_failed: Exception | None = None
        ctx.emit_progress(0.10)
        try:
            common.run_design(
                sidecar=common.sidecar_for(
                    cache=self._sidecars,
                    seed=params.seed,
                    extra_env=extra_env,
                ),
                config_path=config_path,
                overrides=overrides,
                cwd=common.proteina_complexa_tree(),
                steps=steps,
            )
        except Exception as exc:
            run_failed = exc
        ctx.emit_progress(0.92)

        recovered_after_evaluation_failure = False
        try:
            designs = _collect_designs(run_dir=run_dir, params=params)
        except Exception:
            if run_failed is not None:
                raise run_failed
            raise
        if run_failed is not None:
            if "evaluate" not in (steps or []):
                raise run_failed
            designs = _fresh_designs(designs, min_mtime_s=run_started_at - 2.0)
            if not designs:
                raise run_failed
            recovered_after_evaluation_failure = True
            mapping_warnings.append(_partial_recovery_warning(run_failed, run_dir, len(designs)))
        runtime_ms = int((time.time() - started) * 1000)
        ctx.emit_progress(1.0)
        return ProteinaComplexaOutput(
            mode=params.mode,
            designs=designs,
            candidates=_proteina_complexa_candidates(params, designs),
            runtime_ms=runtime_ms,
            num_designs_returned=len(designs),
            seeds_used=[params.seed] if params.seed is not None else [],
            position_mappings=position_mappings,
            warnings=mapping_warnings,
            recovered_after_evaluation_failure=recovered_after_evaluation_failure,
        )

    def estimate_vram(self, params: ProteinaComplexaInput) -> int:
        del params
        return 24 * 1024

    def healthcheck(self) -> bool:
        return availability_skip_reason() is None

    def unload(self) -> None:
        for sidecar in self._sidecars.values():
            sidecar.close()
        self._sidecars.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _partial_recovery_warning(exc: Exception, run_dir: Path, design_count: int) -> str:
    hint = None
    if isinstance(exc, VPAError):
        hint = exc.details.get("root_cause_hint")
    if hint is None:
        details = getattr(exc, "details", None)
        hint = details.get("root_cause_hint") if isinstance(details, dict) else None
    suffix = f" Root cause hint: {hint}." if hint else ""
    return (
        "Proteina-Complexa generated "
        f"{design_count} structure{'s' if design_count != 1 else ''} under {run_dir}, "
        "but the evaluate step failed. Candidate structures are recovered; "
        "Proteina-Complexa reward/confidence metrics are missing and must not be "
        f"treated as recorded evidence.{suffix}"
    )


def _fresh_designs(
    designs: list[ProteinaComplexaDesign],
    *,
    min_mtime_s: float,
) -> list[ProteinaComplexaDesign]:
    fresh: list[ProteinaComplexaDesign] = []
    for design in designs:
        try:
            if Path(design.pdb_path).stat().st_mtime >= min_mtime_s:
                fresh.append(design)
        except OSError:
            continue
    return fresh


def _resolve_steps(params: ProteinaComplexaInput) -> list[ProteinaComplexaStep] | None:
    del params
    return ["generate", "evaluate"]


def _prepare_mode_input(params: ProteinaComplexaInput, job_dir: Path) -> tuple[Path, str]:
    if params.mode == "motif":
        prepared = _prepare_motif_only_pdb(params, job_dir)
        return prepared, common.hydra_task_key(prepared, prefix="motif")

    assert params.target_pdb is not None
    target = Path(params.target_pdb).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"target_pdb not found: {target}")
    if params.mode == "ligand_binder":
        assert params.ligand_smiles is not None
        _validate_ligand_only_target(target, params.ligand_smiles)
        prepared = _prepare_ligand_only_target(target, job_dir)
        return prepared, common.hydra_task_key(prepared, prefix="ligand_target")
    return target, common.hydra_task_key(target, prefix="target")


def _build_overrides(
    *,
    params: ProteinaComplexaInput,
    run_name: str,
    prepared_input: Path,
) -> list[str]:
    if params.mode == "protein_binder":
        return _protein_binder_overrides(
            params=params,
            run_name=run_name,
            target=prepared_input,
        )
    if params.mode == "ligand_binder":
        return _ligand_binder_overrides(
            params=params,
            run_name=run_name,
            target=prepared_input,
        )
    return _motif_overrides(params=params, run_name=run_name, motif=prepared_input)


def _protein_binder_overrides(
    *,
    params: ProteinaComplexaInput,
    run_name: str,
    target: Path,
) -> list[str]:
    task_name = common.hydra_task_key(target, prefix="target")
    target_input = common.protein_chain_input_spec(target, params.target_chain)
    return [
        *_base_overrides(params.mode, run_name),
        f"++generation.task_name={task_name}",
        f"++generation.target_dict_cfg.{task_name}.source=vpa_custom",
        f"++generation.target_dict_cfg.{task_name}.target_filename={target.stem}",
        f"++generation.target_dict_cfg.{task_name}.target_path={target}",
        f"++generation.target_dict_cfg.{task_name}.target_input={target_input}",
        f"++generation.target_dict_cfg.{task_name}.hotspot_residues=[]",
        f"++generation.target_dict_cfg.{task_name}.binder_length=[{params.binder_length},{params.binder_length}]",
        f"++generation.target_dict_cfg.{task_name}.pdb_id=vpa_custom",
        f"++generation.dataloader.dataset.conditional_features.0.pdb_path={target}",
        f"++generation.dataloader.dataset.conditional_features.0.input_spec={target_input}",
        "++generation.dataloader.dataset.conditional_features.0.target_hotspots=[]",
        f"++generation.dataloader.dataset.nres.low={params.binder_length}",
        f"++generation.dataloader.dataset.nres.high={params.binder_length}",
        f"++generation.dataloader.dataset.nres.nsamples={params.num_designs}",
        *_generation_policy_overrides(params),
        *_seed_override(params),
    ]


def _ligand_binder_overrides(
    *,
    params: ProteinaComplexaInput,
    run_name: str,
    target: Path,
) -> list[str]:
    task_name = common.hydra_task_key(target, prefix="ligand_target")
    pocket_csv = ",".join(_target_pocket_labels(params, target))
    base = f"++generation.target_dict_cfg.{task_name}"
    return [
        *_base_overrides(params.mode, run_name),
        f"++generation.task_name={task_name}",
        f"{base}.source=vpa_custom",
        f"{base}.target_filename={target.stem}",
        f"{base}.target_path={target}",
        f"{base}.hotspot_residues=[{pocket_csv}]",
        f"{base}.binder_length=[{params.binder_length}]",
        f"{base}.pdb_id=vpa_custom",
        f"{base}.ligand=null",
        f"{base}.ligand_only=True",
        f"{base}.SMILES={params.ligand_smiles!r}",
        f"{base}.use_bonds_from_file=False",
        f"++generation.dataloader.dataset.nres.nsamples={params.num_designs}",
        *_generation_policy_overrides(params),
        *_seed_override(params),
    ]


def _motif_overrides(
    *,
    params: ProteinaComplexaInput,
    run_name: str,
    motif: Path,
) -> list[str]:
    assert params.total_length is not None
    task_name = common.hydra_task_key(motif, prefix="motif")
    motif_residues = _resolved_motif_resseqs(params)
    contig = common.motif_contig_string(
        chain=params.motif_chain,
        residues=motif_residues,
        total_length=params.total_length,
    )
    segment_order = common.motif_segment_order(motif_residues)
    base = f"++generation.motif_dict_cfg.{task_name}"
    return [
        *_base_overrides(params.mode, run_name),
        "++env_vars.USE_V2_COMPLEXA_ARCH=True",
        f"++generation.task_name={task_name}",
        f"{base}.contig_string={contig}",
        f"{base}.motif_pdb_path={motif}",
        f"{base}.motif_only=True",
        f"{base}.motif_min_length={params.total_length}",
        f"{base}.motif_max_length={params.total_length}",
        f"{base}.segment_order={segment_order}",
        f"{base}.atom_selection_mode=all_atom",
        f"++generation.dataloader.dataset.conditional_features.0.nsamples={params.num_designs}",
        *_generation_policy_overrides(params),
        *_seed_override(params),
    ]


def _base_overrides(mode: ProteinaComplexaMode, run_name: str) -> list[str]:
    mode_config = _MODE_CONFIG[mode]
    return [
        f"++run_name={run_name}",
        f"++ckpt_path={common.proteina_complexa_ckpts()}",
        f"++ckpt_name={mode_config['ckpt']}",
        f"++autoencoder_ckpt_path={common.proteina_complexa_ckpts() / str(mode_config['ae_ckpt'])}",
    ]


def _seed_override(params: ProteinaComplexaInput) -> list[str]:
    return [f"++seed={params.seed}"] if params.seed is not None else []


def _generation_policy_overrides(params: ProteinaComplexaInput) -> list[str]:
    overrides: list[str] = []
    if params.search_algorithm != "default":
        overrides.append(f"++generation.search.algorithm={params.search_algorithm}")
    if params.reward_model == "none":
        overrides.append("++generation.reward_model=null")
    return overrides


def _validate_ligand_only_target(target_pdb: Path, ligand_smiles: str) -> None:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        raise ValueError(f"ligand_smiles is not parseable by RDKit: {ligand_smiles!r}")
    expected_heavy_atoms = mol.GetNumHeavyAtoms()
    pdb_atom_count = 0
    pdb_heavy_atoms = 0
    with target_pdb.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            pdb_atom_count += 1
            element = line[76:78].strip() or line[12:16].strip()[:1]
            if element.upper() != "H":
                pdb_heavy_atoms += 1
    if pdb_heavy_atoms != expected_heavy_atoms:
        raise ValueError(
            "Proteina-Complexa ligand_binder mode expects target_pdb to be a "
            "ligand-only PDB matching ligand_smiles. Got "
            f"{pdb_heavy_atoms} heavy atoms in {target_pdb}, but RDKit parsed "
            f"{expected_heavy_atoms} heavy atoms from ligand_smiles."
        )
    if pdb_atom_count != expected_heavy_atoms:
        raise ValueError(
            "Proteina-Complexa ligand_binder mode expects a heavy-atom-only "
            "ligand PDB. Got "
            f"{pdb_atom_count} atoms in {target_pdb}, but RDKit parsed "
            f"{expected_heavy_atoms} heavy atoms from ligand_smiles."
        )


def _prepare_ligand_only_target(target_pdb: Path, job_dir: Path) -> Path:
    lines = target_pdb.read_text().splitlines(keepends=True)
    changed = False
    stripped_hydrogen = False
    fixed: list[str] = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            element = line[76:78].strip() or line[12:16].strip()[:1]
            if element.upper() == "H":
                changed = True
                stripped_hydrogen = True
                continue
            if len(line) > 21 and line[21] == " ":
                fixed.append(f"{line[:21]}A{line[22:]}")
                changed = True
            else:
                fixed.append(line)
        elif stripped_hydrogen and line.startswith("CONECT"):
            changed = True
            continue
        else:
            fixed.append(line)
    if not changed:
        return target_pdb
    prepared = job_dir / f"{target_pdb.stem}_heavy_chainA{target_pdb.suffix}"
    prepared.write_text("".join(fixed))
    return prepared


def _prepare_motif_only_pdb(params: ProteinaComplexaInput, job_dir: Path) -> Path:
    assert params.motif_pdb is not None
    source = Path(params.motif_pdb).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"motif_pdb not found: {source}")
    residues = set(_resolved_motif_resseqs(params))
    selected: list[str] = []
    for line in source.read_text().splitlines(keepends=True):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip()
        if chain != params.motif_chain:
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        if resseq in residues:
            selected.append(line)
    if not selected:
        raise ValueError(
            f"No atoms found for motif chain {params.motif_chain!r} residues "
            f"{sorted(residues)} in {source}."
        )
    prepared = job_dir / f"{source.stem}_motif_only.pdb"
    prepared.write_text("".join(selected) + "TER\nEND\n")
    return prepared


def _target_pocket_labels(params: ProteinaComplexaInput, target: Path) -> list[str]:
    refs = parse_structure_residue_refs(target, params.target_chain)
    if not refs:
        return [
            *[str(pos) for pos in params.target_pocket_residues],
            *params.target_pocket_residue_ids,
        ]
    resolution = resolve_positions(
        residues=refs,
        positions=params.target_pocket_residues,
        explicit_residue_ids=params.target_pocket_residue_ids,
        chain_id=params.target_chain,
        sequence=params.target_sequence,
        fasta_path=params.target_fasta_path,
        position_basis=params.position_basis,
        label="target_pocket_residues",
    )
    return resolution.residue_ids


def _resolved_motif_resseqs(params: ProteinaComplexaInput) -> list[int]:
    assert params.motif_pdb is not None
    source = Path(params.motif_pdb).expanduser().resolve()
    resolution = resolve_positions(
        residues=parse_structure_residue_refs(source, params.motif_chain),
        positions=params.motif_residues,
        explicit_residue_ids=params.motif_residue_ids,
        chain_id=params.motif_chain,
        sequence=params.motif_sequence,
        fasta_path=params.motif_fasta_path,
        position_basis=params.position_basis,
        label="motif_residues",
    )
    return [
        parse_residue_id(residue_id, default_chain=params.motif_chain).resseq
        for residue_id in resolution.residue_ids
    ]


def _position_mapping_audit(params: ProteinaComplexaInput) -> tuple[dict[str, object], list[str]]:
    mappings: dict[str, object] = {}
    warnings: list[str] = []
    if params.mode == "motif" and params.motif_pdb:
        source = Path(params.motif_pdb).expanduser().resolve()
        resolution = resolve_positions(
            residues=parse_structure_residue_refs(source, params.motif_chain),
            positions=params.motif_residues,
            explicit_residue_ids=params.motif_residue_ids,
            chain_id=params.motif_chain,
            sequence=params.motif_sequence,
            fasta_path=params.motif_fasta_path,
            position_basis=params.position_basis,
            label="motif_residues",
        )
        mappings["motif_residues"] = resolution.to_audit()
        warnings.extend(resolution.warnings)
    if params.mode == "ligand_binder" and params.target_pdb:
        target = Path(params.target_pdb).expanduser().resolve()
        refs = parse_structure_residue_refs(target, params.target_chain)
        if refs:
            resolution = resolve_positions(
                residues=refs,
                positions=params.target_pocket_residues,
                explicit_residue_ids=params.target_pocket_residue_ids,
                chain_id=params.target_chain,
                sequence=params.target_sequence,
                fasta_path=params.target_fasta_path,
                position_basis=params.position_basis,
                label="target_pocket_residues",
            )
            mappings["target_pocket_residues"] = resolution.to_audit()
            warnings.extend(resolution.warnings)
        elif params.target_sequence or params.target_fasta_path:
            warnings.append(
                "target_pocket_residues could not be sequence-mapped because "
                "target_pdb has no protein residues"
            )
    return mappings, warnings


def _collect_designs(
    *,
    run_dir: Path,
    params: ProteinaComplexaInput,
) -> list[ProteinaComplexaDesign]:
    pdbs = common.collect_pdbs(run_dir)
    if not pdbs:
        raise RuntimeError(
            f"Proteina-Complexa {params.mode} mode produced no PDBs under {run_dir}. "
            "The generate/evaluate pipeline did not materialize design artifacts."
        )

    metrics_by_stem = _load_results_csv(run_dir / "results.csv")
    designs: list[ProteinaComplexaDesign] = []
    default_length = params.total_length if params.mode == "motif" else params.binder_length
    assert default_length is not None
    for pdb in pdbs:
        row = metrics_by_stem.get(pdb.stem, {})
        designs.append(
            ProteinaComplexaDesign(
                pdb_path=str(pdb),
                sequence=_maybe_str(row.get("sequence")),
                length=int(row.get("length", default_length) or default_length),
                score=_as_float(row.get("reward") or row.get("score")),
                iptm=_as_float(row.get("iptm")),
                plddt=_as_float(row.get("plddt")),
                binder_refold_rmsd=_as_float(row.get("binder_refold_rmsd")),
                ligand_contacts=_as_int(row.get("ligand_contacts")),
                ligand_rmsd=_as_float(row.get("ligand_rmsd")),
                motif_rmsd=_as_float(row.get("motif_rmsd")),
                scaffold_plddt=_as_float(row.get("scaffold_plddt") or row.get("plddt")),
                designability=_as_float(row.get("designability")),
            )
        )
    return designs


def _load_results_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stem = (
                row.get("design")
                or row.get("Design")
                or row.get("sample_id")
                or row.get("pdb")
                or ""
            )
            if stem:
                rows[Path(stem).stem] = row
    return rows


def _as_float(value: object) -> float | None:
    if value is None or value == "" or value == "NA":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    if value is None or value == "" or value == "NA":
        return None
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _maybe_str(value: object) -> str | None:
    if value is None or value == "" or value == "NA":
        return None
    return str(value)


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _proteina_complexa_candidates(
    params: ProteinaComplexaInput,
    designs: list[ProteinaComplexaDesign],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, design in enumerate(designs):
        local_id = f"design_{index:04d}"
        proteina_artifacts = _proteina_complexa_candidate_artifacts(params, design)
        candidate: dict[str, Any] = {
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "proteina_complexa",
                local_id,
                sequence=design.sequence,
                structure_path=design.pdb_path,
                index=index,
            ),
            "source": {
                "tool": "proteina_complexa",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": None,
                "index": index,
            },
            "structure_path": design.pdb_path,
            "scores": {
                "proteina_complexa": {
                    "mode": params.mode,
                    "length": design.length,
                    "score": design.score,
                    "iptm": design.iptm,
                    "plddt": design.plddt,
                    "binder_refold_rmsd": design.binder_refold_rmsd,
                    "ligand_contacts": design.ligand_contacts,
                    "ligand_rmsd": design.ligand_rmsd,
                    "motif_rmsd": design.motif_rmsd,
                    "scaffold_plddt": design.scaffold_plddt,
                    "designability": design.designability,
                }
            },
            "artifacts": {"proteina_complexa": proteina_artifacts}
            if proteina_artifacts
            else {},
            "capabilities": candidate_capability_flags(
                has_sequence=bool(design.sequence),
                has_backbone_structure=bool(design.pdb_path),
                has_full_structure=bool(design.pdb_path),
                has_candidate_scores=True,
            ),
        }
        if design.sequence:
            candidate["sequence"] = design.sequence
        if params.candidates and _is_canonical_candidate(params.candidates[0]):
            upstream_scores = params.candidates[0].get("scores")
            if isinstance(upstream_scores, dict):
                candidate["scores"] = {**upstream_scores, **candidate["scores"]}
        candidates.append(candidate)
    return candidates


def _proteina_complexa_candidate_artifacts(
    params: ProteinaComplexaInput,
    design: ProteinaComplexaDesign,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {"mode": params.mode}
    if params.target_pdb:
        artifacts["target_pdb"] = params.target_pdb
        artifacts["target_chain"] = params.target_chain
    if params.mode == "protein_binder" and params.target_pdb:
        chain_rows = complex_chain_metadata(
            target_sequence=chain_sequence_from_structure(
                params.target_pdb,
                params.target_chain,
            ),
            binder_sequence=design.sequence,
            structure_path=design.pdb_path,
            target_chain_id=params.target_chain,
            binder_input_order=1,
        )
        if chain_rows:
            artifacts["complex_chains"] = chain_rows
    elif params.mode == "motif":
        artifacts["motif_pdb"] = params.motif_pdb
        artifacts["motif_chain"] = params.motif_chain
        chain_rows = complex_chain_metadata(
            target_sequence=chain_sequence_from_structure(
                params.motif_pdb,
                params.motif_chain,
            ),
            binder_sequence=design.sequence,
            structure_path=design.pdb_path,
            target_chain_id=params.motif_chain,
            target_role="motif",
            binder_role="design",
        )
        if chain_rows:
            artifacts["complex_chains"] = chain_rows
    elif params.mode == "ligand_binder":
        artifacts["ligand_smiles"] = params.ligand_smiles
    return artifacts


__all__ = [
    "ProteinaComplexaAdapter",
    "ProteinaComplexaDesign",
    "ProteinaComplexaInput",
    "ProteinaComplexaOutput",
]
