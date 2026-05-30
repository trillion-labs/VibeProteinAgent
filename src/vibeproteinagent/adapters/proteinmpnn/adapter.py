from __future__ import annotations

import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import stable_candidate_id

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
    pdb_path: str | None = Field(None, description="Absolute path to the input PDB.")
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    chains: list[str] | None = Field(
        None, description="Chain identifiers to design; None designs every chain."
    )
    num_designs: int = Field(8, ge=1, le=1024)
    temperature: float = Field(0.1, gt=0.0, le=2.0)
    omit_aas: str = Field("X", description="Amino acids disallowed at every position.")
    seed: int | None = Field(None, description="Optional RNG seed for reproducibility.")

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
            structure_path = _mpnn_structure_path_from_candidate(candidates[0])
            if not structure_path:
                raise ValueError("proteinmpnn canonical Candidate[] input requires structure_path")
            out = dict(data)
            out.setdefault("pdb_path", structure_path)
            return out
        return data

    @model_validator(mode="after")
    def _require_pdb_path(self) -> ProteinMPNNInput:
        if not self.pdb_path:
            raise ValueError("proteinmpnn requires pdb_path")
        if self.chains is None:
            chains = _infer_protein_chain_ids(Path(self.pdb_path).expanduser())
            if chains:
                self.chains = chains
        return self


class ProteinMPNNDesign(BaseModel):
    candidate_id: str | None = None
    sequence: str
    score: float
    seq_recovery: float | None = None


class ProteinMPNNOutput(BaseModel):
    designs: list[ProteinMPNNDesign]
    candidates: list[dict[str, Any]] = Field(default_factory=list)
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

        assert params.pdb_path is not None
        source_structure_path = Path(params.pdb_path).expanduser()
        pdb_path, input_chains, conversion_metadata = _materialize_mpnn_pdb_input(
            source_structure_path,
            requested_chains=params.chains,
            job_dir=ctx.job_dir,
        )
        if conversion_metadata:
            out_meta = ctx.job_dir / "outputs" / "proteinmpnn_input_chain_map.json"
            out_meta.parent.mkdir(parents=True, exist_ok=True)
            out_meta.write_text(json.dumps(conversion_metadata, indent=2, sort_keys=True) + "\n")

        pdb_dict_list = parse_PDB(str(pdb_path), input_chain_list=input_chains, ca_only=False)
        if not pdb_dict_list:
            raise ValueError(f"parse_PDB returned no records for {pdb_path}")
        protein = pdb_dict_list[0]
        ctx.emit_progress(0.10)

        chain_id_dict: dict[str, tuple[list[str], list[str]]] | None = None
        if input_chains is not None:
            # All listed chains are "designed"; nothing visible-only.
            chain_id_dict = {protein["name"]: (list(input_chains), [])}

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
            candidates=[
                _canonical_candidate_from_design(
                    design.model_dump(),
                    index,
                    params=params,
                    structure_path=str(source_structure_path),
                    mpnn_input_pdb_path=(
                        str(pdb_path) if Path(pdb_path) != source_structure_path else None
                    ),
                )
                for index, design in enumerate(designs)
            ],
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


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"


def _mpnn_structure_path_from_candidate(candidate: dict[str, Any]) -> str | None:
    direct = candidate.get("structure_path")
    if isinstance(direct, str) and direct.lower().endswith(".pdb"):
        return direct
    artifacts = candidate.get("artifacts") if isinstance(candidate.get("artifacts"), dict) else {}
    source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
    source_tool = str(source.get("tool") or "")
    preferred_namespaces = [source_tool] if source_tool else []
    preferred_namespaces.extend(
        name
        for name in (
            "proteinmpnn",
            "bindcraft",
            "pxdesign",
            "boltzgen",
            "proteina_complexa",
        )
        if name not in preferred_namespaces
    )
    for namespace in preferred_namespaces:
        payload = artifacts.get(namespace)
        if isinstance(payload, dict):
            value = payload.get("pdb_path")
            if isinstance(value, str) and value:
                return value
    if isinstance(direct, str) and direct:
        return direct
    for key in ("pdb_path", "full_structure_path", "backbone_path"):
        value = candidate.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _infer_protein_chain_ids(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        return _infer_cif_protein_chain_ids(path)
    return _infer_pdb_protein_chain_ids(path)


def _infer_pdb_protein_chain_ids(path: Path) -> list[str]:
    chains: set[str] = set()
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain = line[21:22].strip()
        if chain:
            chains.add(chain)
    return sorted(chains)


def _infer_cif_protein_chain_ids(path: Path) -> list[str]:
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return []
    cols: list[str] = []
    chains: set[str] = set()
    in_atom_loop = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            cols = []
            in_atom_loop = False
            continue
        if stripped.startswith("_atom_site."):
            in_atom_loop = True
            cols.append(stripped.removeprefix("_atom_site."))
            continue
        if not in_atom_loop:
            continue
        parts = stripped.split()
        if not parts or parts[0].startswith("#"):
            in_atom_loop = False
            continue
        if len(parts) < len(cols):
            continue
        row = {col: parts[idx] for idx, col in enumerate(cols)}
        if row.get("group_PDB") != "ATOM":
            continue
        chain = row.get("auth_asym_id") or row.get("label_asym_id")
        if chain and chain not in {".", "?"}:
            chains.add(str(chain))
    return sorted(chains)


def _materialize_mpnn_pdb_input(
    structure_path: Path,
    *,
    requested_chains: list[str] | None,
    job_dir: Path,
) -> tuple[Path, list[str] | None, dict[str, Any] | None]:
    if structure_path.suffix.lower() not in {".cif", ".mmcif"}:
        chains = requested_chains
        if chains is None:
            inferred = _infer_protein_chain_ids(structure_path)
            chains = inferred or None
        return structure_path, chains, None

    rows = _read_cif_atom_rows(structure_path)
    raw_to_pdb = _raw_to_pdb_chain_map(rows)
    pdb_path = job_dir / "inputs" / f"{structure_path.stem}_proteinmpnn_input.pdb"
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    _write_cif_rows_as_pdb(rows, pdb_path, raw_to_pdb)
    mapped_chains = _map_requested_chains(requested_chains, raw_to_pdb)
    metadata = {
        "source_structure_path": str(structure_path),
        "mpnn_input_pdb_path": str(pdb_path),
        "raw_to_pdb_chain": raw_to_pdb,
        "requested_chains": requested_chains,
        "mpnn_chains": mapped_chains,
    }
    return pdb_path, mapped_chains, metadata


def _read_cif_atom_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines = path.read_text(errors="replace").splitlines()
    index = 0
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        headers: list[str] = []
        while index < len(lines) and lines[index].strip().startswith("_"):
            header = lines[index].strip()
            if header.startswith("_atom_site."):
                headers.append(header.removeprefix("_atom_site."))
            index += 1
        if not headers:
            while index < len(lines) and lines[index].strip() != "#":
                index += 1
            continue
        while index < len(lines):
            raw = lines[index].strip()
            if not raw or raw == "#" or raw == "loop_" or raw.startswith("_"):
                break
            parts = shlex.split(raw)
            if len(parts) >= len(headers):
                row = {header: parts[pos] for pos, header in enumerate(headers)}
                if str(row.get("group_PDB") or "").upper() == "ATOM":
                    rows.append(row)
            index += 1
    if not rows:
        raise ValueError(f"No _atom_site ATOM rows found in {path}")
    return rows


def _raw_to_pdb_chain_map(rows: list[dict[str, str]]) -> dict[str, str]:
    used: set[str] = set()
    out: dict[str, str] = {}
    for row in rows:
        raw_chain = _cif_chain_id(row)
        if raw_chain in out:
            continue
        pdb_chain = _pdb_chain_name(raw_chain, used)
        out[raw_chain] = pdb_chain
        used.add(pdb_chain)
    return out


def _cif_chain_id(row: dict[str, str]) -> str:
    return _clean_cif_value(row.get("auth_asym_id")) or _clean_cif_value(
        row.get("label_asym_id")
    ) or "_"


def _clean_cif_value(value: str | None) -> str:
    if value in {None, "", ".", "?"}:
        return ""
    return str(value)


def _pdb_chain_name(raw_chain: str, used: set[str]) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    if len(raw_chain) == 1 and raw_chain in alphabet and raw_chain not in used:
        return raw_chain
    for chain in alphabet:
        if chain not in used:
            return chain
    raise ValueError("Too many mmCIF chains to map into one-character PDB chain IDs")


def _map_requested_chains(
    requested_chains: list[str] | None,
    raw_to_pdb: dict[str, str],
) -> list[str]:
    if requested_chains is None:
        return list(raw_to_pdb.values())
    pdb_values = set(raw_to_pdb.values())
    mapped: list[str] = []
    missing: list[str] = []
    for chain in requested_chains:
        if chain in raw_to_pdb:
            mapped.append(raw_to_pdb[chain])
        elif chain in pdb_values:
            mapped.append(chain)
        else:
            missing.append(chain)
    if missing:
        raise ValueError(
            "Requested ProteinMPNN chain(s) are absent from mmCIF input: "
            f"{missing}; available={sorted(raw_to_pdb)}"
        )
    return mapped


def _write_cif_rows_as_pdb(
    rows: list[dict[str, str]],
    pdb_path: Path,
    raw_to_pdb: dict[str, str],
) -> None:
    records: list[dict[str, Any]] = []
    coords: list[tuple[float, float, float]] = []
    residue_numbers: dict[tuple[str, str], int] = {}
    next_residue: dict[str, int] = {}
    for row in rows:
        raw_chain = _cif_chain_id(row)
        raw_resseq = (
            _clean_cif_value(row.get("auth_seq_id"))
            or _clean_cif_value(row.get("label_seq_id"))
            or str(next_residue.get(raw_chain, 1))
        )
        try:
            x = float(row["Cartn_x"])
            y = float(row["Cartn_y"])
            z = float(row["Cartn_z"])
        except (KeyError, ValueError):
            continue
        coords.append((x, y, z))
        records.append(
            {
                "row": row,
                "chain": raw_to_pdb[raw_chain],
                "resseq": _pdb_resseq(
                    raw_chain, raw_resseq, residue_numbers, next_residue
                ),
                "coords": (x, y, z),
            }
        )
    if not records:
        raise ValueError("No mmCIF ATOM rows with numeric coordinates could be written")
    mean_x = sum(x for x, _, _ in coords) / len(coords)
    mean_y = sum(y for _, y, _ in coords) / len(coords)
    mean_z = sum(z for _, _, z in coords) / len(coords)
    max_abs = max(
        max(abs(x - mean_x), abs(y - mean_y), abs(z - mean_z))
        for x, y, z in coords
    )
    scale = 200.0 / max_abs if max_abs > 200.0 else 1.0

    with pdb_path.open("w") as handle:
        handle.write("MODEL        1\n")
        serial = 1
        for record in records:
            row = record["row"]
            x, y, z = record["coords"]
            x = (x - mean_x) * scale
            y = (y - mean_y) * scale
            z = (z - mean_z) * scale
            atom_name = (
                _clean_cif_value(row.get("auth_atom_id"))
                or _clean_cif_value(row.get("label_atom_id"))
                or "CA"
            )
            resname = (
                _clean_cif_value(row.get("auth_comp_id"))
                or _clean_cif_value(row.get("label_comp_id"))
                or "GLY"
            ).upper()
            if resname == "MSE":
                resname = "MET"
            if resname == "XPB":
                resname = "GLY"
            elem = (
                _clean_cif_value(row.get("type_symbol"))
                or atom_name.strip()[:1]
                or "C"
            ).upper()[:2]
            atom_field = f" {atom_name:<3}" if len(atom_name) < 4 else atom_name[:4]
            handle.write(
                f"ATOM  {serial:5d} {atom_field} {resname:>3s} {record['chain']:1s}"
                f"{record['resseq']:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {elem:>2s}\n"
            )
            serial += 1
        handle.write("TER\nENDMDL\nEND\n")


def _pdb_resseq(
    raw_chain: str,
    raw_resseq: str,
    residue_numbers: dict[tuple[str, str], int],
    next_residue: dict[str, int],
) -> int:
    try:
        value = int(raw_resseq)
    except ValueError:
        key = (raw_chain, raw_resseq)
        if key not in residue_numbers:
            value = next_residue.get(raw_chain, 1)
            residue_numbers[key] = value
            next_residue[raw_chain] = value + 1
        return residue_numbers[key]
    next_residue[raw_chain] = max(next_residue.get(raw_chain, 1), value + 1)
    return value


def _canonical_candidate_from_design(
    design: dict[str, Any],
    index: int,
    *,
    params: ProteinMPNNInput,
    structure_path: str,
    mpnn_input_pdb_path: str | None = None,
) -> dict[str, Any]:
    local_id = f"seq_{index:04d}"
    sequence = str(design.get("sequence") or "")
    candidate = {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            "proteinmpnn",
            local_id,
            native_id=design.get("candidate_id"),
            sequence=sequence,
            structure_path=structure_path,
            index=index,
        ),
        "source": {
            "tool": "proteinmpnn",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": design.get("candidate_id"),
            "index": index,
        },
        "sequence": sequence,
        "structure_path": structure_path,
        "scores": {
            "proteinmpnn": {
                "score": design.get("score"),
                "seq_recovery": design.get("seq_recovery"),
            }
        },
        "artifacts": {},
    }
    if mpnn_input_pdb_path:
        candidate["artifacts"]["proteinmpnn"] = {
            "mpnn_input_pdb_path": mpnn_input_pdb_path,
        }
    if params.candidates and _is_canonical_candidate(params.candidates[0]):
        upstream_scores = params.candidates[0].get("scores")
        if isinstance(upstream_scores, dict):
            candidate["scores"] = {**upstream_scores, **candidate["scores"]}
    return candidate
