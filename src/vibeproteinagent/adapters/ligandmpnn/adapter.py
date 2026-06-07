from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.candidates.paths import candidate_pdb_path
from vibeproteinagent.candidates.schema import stable_candidate_id
from vibeproteinagent.adapters.residue_mapping import (
    PositionBasis,
    parse_residue_id,
    parse_structure_residue_refs,
    resolve_positions,
)

# vendor/ligandmpnn/ holds the upstream package under `ligandmpnn_vendor/`.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _PROJECT_ROOT / "vendor" / "ligandmpnn"
_VENDOR_WEIGHTS_DIR = _VENDOR_ROOT / "weights"

# Single bundled checkpoint — see vendor/ligandmpnn/UPSTREAM.md.
_DEFAULT_CKPTS: dict[str, str] = {
    "ligand_mpnn": "ligandmpnn_v_32_020_25.pt",
    "protein_mpnn": "proteinmpnn_v_48_020.pt",
    "soluble_mpnn": "solublempnn_v_48_020.pt",
    "ca_mpnn": "proteinmpnn_v_48_020_ca.pt",
}

if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from ligandmpnn_vendor import (  # noqa: E402 — vendored import after sys.path tweak
    ProteinMPNN as _LigandProteinMPNN,
)
from ligandmpnn_vendor import (  # noqa: E402
    featurize as _featurize,
)
from ligandmpnn_vendor import (  # noqa: E402
    parse_PDB as _parse_PDB,
)
from ligandmpnn_vendor import (  # noqa: E402
    restype_int_to_str as _RESTYPE_INT_TO_STR,
)


class LigandMPNNInput(BaseModel):
    pdb_path: str | None = Field(None, description="Absolute path to the input PDB.")
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    chains: list[str] | None = Field(
        None, description="Chain identifiers to load; None loads every chain."
    )
    num_designs: int = Field(8, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=2.0)
    omit_aas: str = Field("X", description="Amino acids disallowed at every position.")
    seed: int | None = Field(None, description="Optional RNG seed for reproducibility.")
    model_type: str = Field(
        "ligand_mpnn",
        description="One of protein_mpnn / ligand_mpnn / soluble_mpnn / ca_mpnn.",
    )
    redesigned_residues: str | None = Field(
        None,
        description=(
            "Space-separated residue ids (e.g. 'A10 A11 B25') marking which "
            "positions are designable. None marks every position in `chains` "
            "designable."
        ),
    )
    redesigned_positions: list[int] = Field(default_factory=list)
    sequence: str | None = None
    fasta_path: str | None = Field(default=None, min_length=1)
    position_basis: PositionBasis = "auto"

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_pdb_path(cls, data: Any) -> Any:
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
            structure_path = candidate_pdb_path(candidates[0])
            if not structure_path:
                raise ValueError(
                    "ligandmpnn canonical Candidate[] input requires a PDB artifact "
                    "(pdb_path or predicted_pdb); CIF-only candidates are not accepted"
                )
            out = dict(data)
            out.setdefault("pdb_path", structure_path)
            if candidates[0].get("sequence"):
                out.setdefault("sequence", candidates[0].get("sequence"))
            return out
        return data

    @model_validator(mode="after")
    def _require_pdb_path(self) -> LigandMPNNInput:
        if not self.pdb_path:
            raise ValueError("ligandmpnn requires pdb_path")
        return self


class LigandMPNNDesign(BaseModel):
    sequence: str
    score: float
    seq_recovery: float | None = None


class LigandMPNNOutput(BaseModel):
    designs: list[LigandMPNNDesign]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    designs_fasta: str
    num_residues: int
    runtime_ms: int
    position_mappings: dict[str, object] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def _resolve_checkpoint(weights_dir: Path, model_type: str) -> Path:
    filename = _DEFAULT_CKPTS.get(model_type)
    if filename is None:
        raise ValueError(
            f"unsupported LigandMPNN model_type {model_type!r}; "
            f"expected one of {sorted(_DEFAULT_CKPTS)}"
        )
    candidate = Path(weights_dir) / filename
    if candidate.exists():
        return candidate
    vendor_candidate = _VENDOR_WEIGHTS_DIR / filename
    if vendor_candidate.exists():
        return vendor_candidate
    raise FileNotFoundError(
        f"LigandMPNN checkpoint {filename!r} not found in {weights_dir!s} "
        f"or vendored fallback {vendor_candidate!s}. "
        f"See vendor/ligandmpnn/UPSTREAM.md to reproduce."
    )


@register_adapter("descriptor.yaml")
class LigandMPNNAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        # Cache the most recently loaded checkpoint under `model_type` keying.
        # `__init__` defaults to `ligand_mpnn` and the worker keeps the adapter
        # warm across runs of the same `model_type`.
        self.loaded_model_type: str = "ligand_mpnn"
        self.ckpt_path = _resolve_checkpoint(self.weights_dir, self.loaded_model_type)
        self._build_model(self.ckpt_path, self.loaded_model_type)

    def _build_model(self, ckpt_path: Path, model_type: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        atom_context_num = (
            int(ckpt["atom_context_num"]) if model_type == "ligand_mpnn" else 1
        )
        k_neighbors = int(ckpt["num_edges"])
        self.model = _LigandProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type=model_type,
            ligand_mpnn_use_side_chain_context=0,
        )
        self.model.to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.loaded_model_type = model_type
        self.ckpt_path = ckpt_path

    def _ensure_model(self, model_type: str) -> None:
        if model_type == self.loaded_model_type and hasattr(self, "model"):
            return
        ckpt_path = _resolve_checkpoint(self.weights_dir, model_type)
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._build_model(ckpt_path, model_type)

    @torch.no_grad()
    def run(self, params: LigandMPNNInput, ctx: AdapterRunContext) -> LigandMPNNOutput:
        t0 = time.time()
        ctx.emit_progress(0.05)

        self._ensure_model(params.model_type)

        if params.seed is not None:
            torch.manual_seed(int(params.seed))
            np.random.seed(int(params.seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(params.seed))

        assert params.pdb_path is not None
        pdb_path = str(Path(params.pdb_path).expanduser())
        parsed_list = _parse_PDB(
            pdb_path,
            device=self.device,
            chains=list(params.chains) if params.chains else [],
            parse_all_atoms=False,
        )
        if not parsed_list:
            raise ValueError(f"LigandMPNN parse_PDB returned no records for {pdb_path}")
        protein = parsed_list[0]

        # Build chain_mask before featurize: 1.0 = designable, 0.0 = fixed.
        chain_letters = protein["chain_letters"]
        if params.chains:
            allowed = set(params.chains)
            chain_mask = torch.tensor(
                [c in allowed for c in chain_letters],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            chain_mask = torch.ones(len(chain_letters), dtype=torch.float32, device=self.device)

        # Optional: further narrow designable positions by `redesigned_residues`.
        redesigned_resolution = _resolve_redesigned_positions(params, Path(pdb_path))
        if redesigned_resolution is not None:
            wanted = {
                _ligandmpnn_residue_key(residue_id, params.chains[0] if params.chains else "A")
                for residue_id in redesigned_resolution.residue_ids
            }
            # residue id format used by LigandMPNN: f"{chain}{resnum}"
            r_idx = protein["R_idx"].detach().cpu().tolist()
            keep_mask = torch.zeros_like(chain_mask)
            for i, (chain, ri) in enumerate(zip(chain_letters, r_idx, strict=True)):
                if f"{chain}{int(ri)}" in wanted:
                    keep_mask[i] = 1.0
            chain_mask = chain_mask * keep_mask

        if float(chain_mask.sum().item()) == 0.0:
            raise ValueError(
                "No designable positions selected; check `chains` and "
                "`redesigned_residues` against the input PDB."
            )

        protein["chain_mask"] = chain_mask

        fd = _featurize(
            protein,
            cutoff_for_score=8.0,
            use_atom_context=True,
            number_of_ligand_atoms=25,
            model_type=params.model_type,
        )

        B_in, L = fd["S"].shape
        ctx.emit_progress(0.20)

        # Inject the sampling controls expected by ProteinMPNN.sample(). All
        # entries are required keys per `model_utils.ProteinMPNN.sample`.
        batch_size = min(int(params.num_designs), 8)
        remaining = int(params.num_designs)
        n_done = 0

        designs: list[LigandMPNNDesign] = []
        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = out_dir / "designs.fa"

        with fasta_path.open("w") as fh:
            # Convert true sequence ints to a string for the header record.
            true_seq = "".join(
                _RESTYPE_INT_TO_STR.get(int(i), "X")
                for i in fd["S"][0].detach().cpu().tolist()
            )
            fh.write(f">input|num_residues={L}\n{true_seq}\n")

        progress_start = 0.25
        progress_end = 0.95

        while remaining > 0:
            this_batch = min(batch_size, remaining)
            fd_batch = dict(fd)
            fd_batch["batch_size"] = int(this_batch)
            fd_batch["temperature"] = float(params.temperature) if params.temperature > 0 else 1e-6
            fd_batch["bias"] = torch.zeros(B_in, L, 21, device=self.device)
            # Hard-omit the requested amino acids by adding -inf to their bias.
            for aa in params.omit_aas:
                from ligandmpnn_vendor import restype_str_to_int as _RESTYPE_STR_TO_INT
                idx = _RESTYPE_STR_TO_INT.get(aa)
                if idx is not None:
                    fd_batch["bias"][..., idx] = -1e9
            # randn shape must be (batch_size, L) — LigandMPNN's sample() derives
            # the per-decode-order from broadcasting it against chain_mask.
            fd_batch["randn"] = torch.randn(this_batch, L, device=self.device)
            fd_batch["symmetry_residues"] = [[]]
            fd_batch["symmetry_weights"] = [[]]

            out = self.model.sample(fd_batch)
            S_sample: torch.Tensor = out["S"]  # [B_dec, L]
            log_probs: torch.Tensor = out["log_probs"]  # [B_dec, L, 21]

            # Score = NLL of sampled sequence over designable positions only.
            mask = fd["mask"][0]  # [L]
            designable = (mask * chain_mask).float()
            S_one_hot = torch.nn.functional.one_hot(S_sample, 21).float()
            loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
            denom = designable.sum().clamp(min=1.0)
            scores = (loss_per_residue * designable).sum(-1) / denom

            # Sequence recovery vs native (over designable positions only).
            true_S = fd["S"][0]  # [L]
            match = (S_sample == true_S.unsqueeze(0)).float()
            recovery = (match * designable).sum(-1) / denom

            with fasta_path.open("a") as fh:
                for b_ix in range(this_batch):
                    seq = "".join(
                        _RESTYPE_INT_TO_STR.get(int(i), "X")
                        for i in S_sample[b_ix].detach().cpu().tolist()
                    )
                    score = float(scores[b_ix].item())
                    rec = float(recovery[b_ix].item())
                    designs.append(
                        LigandMPNNDesign(sequence=seq, score=score, seq_recovery=rec)
                    )
                    sample_idx = n_done + b_ix + 1
                    fh.write(
                        f">design={sample_idx}, T={params.temperature}, "
                        f"score={score:.4f}, seq_recovery={rec:.4f}\n{seq}\n"
                    )

            n_done += this_batch
            remaining -= this_batch
            frac = progress_start + (progress_end - progress_start) * (
                n_done / max(int(params.num_designs), 1)
            )
            ctx.emit_progress(min(max(frac, progress_start), progress_end))

        ctx.emit_progress(1.0)
        runtime_ms = int((time.time() - t0) * 1000)
        return LigandMPNNOutput(
            designs=designs,
            candidates=_ligandmpnn_candidates(params, designs, pdb_path),
            designs_fasta=str(fasta_path),
            num_residues=int(L),
            runtime_ms=runtime_ms,
            position_mappings={
                "redesigned_positions": redesigned_resolution.to_audit()
            } if redesigned_resolution is not None else {},
            warnings=redesigned_resolution.warnings if redesigned_resolution is not None else [],
        )

    def estimate_vram(self, params: LigandMPNNInput) -> int:
        # Same scale as ProteinMPNN; ligand context adds <50 MiB.
        return 1024

    def healthcheck(self) -> bool:
        return self.ckpt_path.exists()

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "LigandMPNNAdapter",
    "LigandMPNNDesign",
    "LigandMPNNInput",
    "LigandMPNNOutput",
]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _ligandmpnn_candidates(
    params: LigandMPNNInput,
    designs: list[LigandMPNNDesign],
    structure_path: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, design in enumerate(designs):
        local_id = f"seq_{index:04d}"
        candidate = {
            "schema_version": "candidate_v1",
            "candidate_id": stable_candidate_id(
                "ligandmpnn",
                local_id,
                sequence=design.sequence,
                structure_path=structure_path,
                index=index,
            ),
            "source": {
                "tool": "ligandmpnn",
                "node_id": None,
                "job_id": None,
                "local_id": local_id,
                "native_id": None,
                "index": index,
            },
            "sequence": design.sequence,
            "structure_path": structure_path,
            "scores": {
                "ligandmpnn": {
                    "score": design.score,
                    "seq_recovery": design.seq_recovery,
                }
            },
            "artifacts": {},
        }
        if params.candidates and _is_canonical_candidate(params.candidates[0]):
            upstream_scores = params.candidates[0].get("scores")
            if isinstance(upstream_scores, dict):
                candidate["scores"] = {**upstream_scores, **candidate["scores"]}
        candidates.append(candidate)
    return candidates


def _resolve_redesigned_positions(params: LigandMPNNInput, pdb_path: Path):
    explicit = params.redesigned_residues.split() if params.redesigned_residues else []
    if not explicit and not params.redesigned_positions:
        return None
    chain = params.chains[0] if params.chains else "A"
    return resolve_positions(
        residues=parse_structure_residue_refs(pdb_path, chain if params.chains else None),
        positions=params.redesigned_positions,
        explicit_residue_ids=explicit,
        chain_id=chain,
        sequence=params.sequence,
        fasta_path=params.fasta_path,
        position_basis=params.position_basis,
        label="redesigned_positions",
    )


def _ligandmpnn_residue_key(residue_id: str, default_chain: str) -> str:
    ref = parse_residue_id(residue_id, default_chain=default_chain)
    return f"{ref.chain_id}{ref.resseq}"
