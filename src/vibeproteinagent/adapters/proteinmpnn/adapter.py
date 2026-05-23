from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, Field

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext

# Project root is parents[4] from this file:
#   src/vibeproteinagent/adapters/proteinmpnn/adapter.py
#   └─0─┘└─────────1──────────┘└────2────┘└──3──────┘└─4─┘
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _PROJECT_ROOT / "vendor" / "proteinmpnn"
_VENDOR_WEIGHTS_DEFAULT = _VENDOR_ROOT / "weights" / "v_48_020.pt"

# The vendored package lives at vendor/proteinmpnn/proteinmpnn_vendor/.
# We add `vendor/proteinmpnn` to sys.path so `proteinmpnn_vendor` is importable.
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from proteinmpnn_vendor import (  # noqa: E402 — vendored import after sys.path tweak
    ProteinMPNN,
    _S_to_seq,
    _scores,
    parse_PDB,
    tied_featurize,
)

# ProteinMPNN's amino-acid alphabet (21 letters; final 'X' is the unknown token).
_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class ProteinMPNNInput(BaseModel):
    pdb_path: str = Field(..., description="Absolute path to the input PDB.")
    chains: list[str] | None = Field(
        None, description="Chain identifiers to design; None designs every chain."
    )
    num_designs: int = Field(8, ge=1, le=1024)
    temperature: float = Field(0.1, gt=0.0, le=2.0)
    omit_aas: str = Field("X", description="Amino acids disallowed at every position.")
    seed: int | None = Field(None, description="Optional RNG seed for reproducibility.")


class ProteinMPNNDesign(BaseModel):
    candidate_id: str | None = None
    sequence: str
    score: float
    seq_recovery: float | None = None


class ProteinMPNNOutput(BaseModel):
    designs: list[ProteinMPNNDesign]
    designs_fasta: str
    log_prob_npz: str
    num_residues: int
    runtime_ms: int


def _resolve_checkpoint(weights_dir: Path) -> Path:
    """Pick the v_48_020.pt checkpoint, falling back to the vendored copy."""
    candidate = Path(weights_dir) / "v_48_020.pt"
    if candidate.exists():
        return candidate
    if _VENDOR_WEIGHTS_DEFAULT.exists():
        return _VENDOR_WEIGHTS_DEFAULT
    raise FileNotFoundError(
        f"ProteinMPNN checkpoint v_48_020.pt not found in {weights_dir!s} "
        f"or vendored fallback {_VENDOR_WEIGHTS_DEFAULT!s}"
    )


def _candidate_id(pdb_path: str, sample_idx: int, sequence: str) -> str:
    stem = Path(pdb_path).stem.replace(" ", "_") or "input"
    token = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:10]
    return f"proteinmpnn:{stem}:design_{sample_idx}:{token}"


@register_adapter("descriptor.yaml")
class ProteinMPNNAdapter(ToolAdapter):
    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.ckpt_path = _resolve_checkpoint(self.weights_dir)
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        self.model = ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=ckpt["num_edges"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def run(self, params: ProteinMPNNInput, ctx: AdapterRunContext) -> ProteinMPNNOutput:
        t0 = time.time()
        ctx.emit_progress(0.02)

        if params.seed is not None:
            torch.manual_seed(int(params.seed))
            np.random.seed(int(params.seed))
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(int(params.seed))

        pdb_path = str(Path(params.pdb_path).expanduser())
        pdb_dict_list = parse_PDB(pdb_path, input_chain_list=params.chains, ca_only=False)
        if not pdb_dict_list:
            raise ValueError(f"parse_PDB returned no records for {pdb_path}")
        protein = pdb_dict_list[0]
        ctx.emit_progress(0.10)

        chain_id_dict: dict[str, tuple[list[str], list[str]]] | None = None
        if params.chains is not None:
            # All listed chains are "designed"; nothing visible-only.
            chain_id_dict = {protein["name"]: (list(params.chains), [])}

        omit_aas_set = set(params.omit_aas)
        omit_AAs_np = np.array(
            [aa in omit_aas_set for aa in _ALPHABET], dtype=np.float32
        )
        bias_AAs_np = np.zeros(len(_ALPHABET), dtype=np.float32)

        # Sample in mini-batches to keep peak VRAM bounded. 8 designs/batch matches
        # ProteinMPNN's reference `--batch_size` default and is comfortable on
        # any modern GPU for sub-1k-residue inputs.
        batch_size = min(params.num_designs, 8)
        remaining = params.num_designs

        designs: list[ProteinMPNNDesign] = []
        log_probs_chunks: list[np.ndarray] = []
        sampled_sequences_chunks: list[np.ndarray] = []
        num_residues: int | None = None

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = out_dir / "designs.fa"
        npz_path = out_dir / "logprobs.npz"

        # Write FASTA header (input sequence) up-front; we append per-design records.
        with fasta_path.open("w") as fh:
            input_seq = protein["seq"]
            fh.write(
                f">{protein['name']}|input, num_residues={len(input_seq)}\n{input_seq}\n"
            )

        progress_start = 0.20
        progress_end = 0.95

        n_done = 0
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            batch_clones = [
                {k: v for k, v in protein.items()} for _ in range(this_batch)
            ]
            (
                X, S, mask, _lengths, chain_M, chain_encoding_all,
                _chain_list_list, _visible_list_list, _masked_list_list,
                _masked_chain_length_list_list, chain_M_pos, omit_AA_mask,
                residue_idx, _dihedral_mask, _tied_pos, pssm_coef, pssm_bias,
                pssm_log_odds_all, bias_by_res_all, _tied_beta,
            ) = tied_featurize(
                batch_clones,
                self.device,
                chain_id_dict,
                None,
                None,
                None,
                None,
                None,
                ca_only=False,
            )
            pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float()

            if num_residues is None:
                # Count of residues actually designed in this run (chain_M masks
                # designed positions; chain_M_pos masks those not fixed).
                designed_mask = (mask * chain_M * chain_M_pos)[0]
                num_residues = int(designed_mask.sum().item())

            randn = torch.randn(chain_M.shape, device=X.device)
            sample_dict = self.model.sample(
                X, randn, S, chain_M, chain_encoding_all, residue_idx,
                mask=mask,
                temperature=params.temperature,
                omit_AAs_np=omit_AAs_np,
                bias_AAs_np=bias_AAs_np,
                chain_M_pos=chain_M_pos,
                omit_AA_mask=omit_AA_mask,
                pssm_coef=pssm_coef,
                pssm_bias=pssm_bias,
                pssm_multi=0.0,
                pssm_log_odds_flag=False,
                pssm_log_odds_mask=pssm_log_odds_mask,
                pssm_bias_flag=False,
                bias_by_res=bias_by_res_all,
            )
            S_sample = sample_dict["S"]
            log_probs = self.model(
                X, S_sample, mask, chain_M * chain_M_pos, residue_idx,
                chain_encoding_all, randn,
                use_input_decoding_order=True,
                decoding_order=sample_dict["decoding_order"],
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            scores = _scores(S_sample, log_probs, mask_for_loss).detach().cpu().numpy()

            log_probs_chunks.append(log_probs.detach().cpu().numpy())
            sampled_sequences_chunks.append(S_sample.detach().cpu().numpy())

            with fasta_path.open("a") as fh:
                for b_ix in range(this_batch):
                    one_hot = torch.nn.functional.one_hot(S[b_ix], 21) * \
                        torch.nn.functional.one_hot(S_sample[b_ix], 21)
                    recovery = (
                        (one_hot.sum(-1) * mask_for_loss[b_ix]).sum()
                        / torch.clamp(mask_for_loss[b_ix].sum(), min=1.0)
                    ).item()
                    seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix] * chain_M_pos[b_ix])
                    score = float(scores[b_ix])
                    sample_idx = n_done + b_ix + 1
                    candidate_id = _candidate_id(pdb_path, sample_idx, seq)
                    designs.append(
                        ProteinMPNNDesign(
                            candidate_id=candidate_id,
                            sequence=seq,
                            score=score,
                            seq_recovery=float(recovery),
                        )
                    )
                    fh.write(
                        f">{protein['name']}|design={sample_idx}, "
                        f"candidate_id={candidate_id}, "
                        f"T={params.temperature}, score={score:.4f}, "
                        f"seq_recovery={recovery:.4f}\n{seq}\n"
                    )

            n_done += this_batch
            remaining -= this_batch
            frac = progress_start + (progress_end - progress_start) * (
                n_done / max(params.num_designs, 1)
            )
            ctx.emit_progress(min(max(frac, progress_start), progress_end))

        np.savez(
            npz_path,
            log_probs=np.concatenate(log_probs_chunks, axis=0),
            S_sample=np.concatenate(sampled_sequences_chunks, axis=0),
            num_designs=np.int64(params.num_designs),
            num_residues=np.int64(num_residues or 0),
            temperature=np.float32(params.temperature),
        )

        # Strict length check: every designed sequence must equal num_residues.
        for d in designs:
            if num_residues is not None and len(d.sequence) != num_residues:
                raise RuntimeError(
                    f"ProteinMPNN sequence length mismatch: got {len(d.sequence)} "
                    f"expected {num_residues}"
                )

        ctx.emit_progress(1.0)
        runtime_ms = int((time.time() - t0) * 1000)
        return ProteinMPNNOutput(
            designs=designs,
            designs_fasta=str(fasta_path),
            log_prob_npz=str(npz_path),
            num_residues=int(num_residues or 0),
            runtime_ms=runtime_ms,
        )

    def estimate_vram(self, params: ProteinMPNNInput) -> int:
        # ProteinMPNN is tiny (~6 MiB checkpoint, ~128-dim hidden). 1 GiB is a safe
        # ceiling for sub-1k-residue inputs across any reasonable num_designs.
        return 1024

    def healthcheck(self) -> bool:
        return self.ckpt_path.exists()

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "ProteinMPNNAdapter",
    "ProteinMPNNDesign",
    "ProteinMPNNInput",
    "ProteinMPNNOutput",
]
