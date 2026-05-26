from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import stable_candidate_id
from vibeproteinagent.core.compat import ensure_esm3_shared_hf_cache
from vibeproteinagent.core.config import model_cache_root
from vibeproteinagent.core.errors import DependencyError

_DEFAULT_MODEL = "esm3_sm_open_v1"
_HF_CREDENTIAL_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "VPA_HF_CREDENTIAL")

# Gated model HF repo paths use a Hugging Face credential plus accepted T&Cs.
# The Cambrian Open License only covers the small variant.
_GATED_MODEL_HINTS = (
    "esm3-medium",
    "esm3-large",
    "esm3-open-2024",
    "esm3_medium",
    "esm3_large",
)


def _hf_credential() -> str | None:
    for name in _HF_CREDENTIAL_ENV:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _ensure_shared_hf_cache_env() -> None:
    """Prefer VPA's shared model cache over node-local Hugging Face defaults."""
    ensure_esm3_shared_hf_cache(str(model_cache_root() / "huggingface"))
    credential = _hf_credential()
    if credential:
        os.environ.setdefault("HF_TOKEN", credential)
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", credential)


def _normalize_model_name(name: str) -> str:
    """Translate HF-style names (``EvolutionaryScale/esm3-sm-open-v1``) to the
    SDK's ``esm3_sm_open_v1``. The SDK's ``LOCAL_MODEL_REGISTRY`` keys use
    underscores; HF repo paths use hyphens. We accept either and normalise.
    """
    name = name.strip()
    if "/" in name:
        # "EvolutionaryScale/esm3-sm-open-v1" → "esm3-sm-open-v1"
        name = name.split("/", 1)[1]
    # Hyphens-to-underscores for the SDK registry only when this looks like the open small variant.
    if name == "esm3-sm-open-v1":
        return "esm3_sm_open_v1"
    return name


def _looks_gated(model_name: str) -> bool:
    lower = model_name.lower()
    return any(hint in lower for hint in _GATED_MODEL_HINTS)


class ESM3Input(BaseModel):
    sequence: str = Field(..., min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str = Field(_DEFAULT_MODEL)
    tracks: list[str] = Field(default_factory=lambda: ["sequence", "structure"])
    num_steps: int = Field(default=20, ge=1)
    temperature: float = Field(default=0.5, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _canonical_candidates_to_sequence(cls, data: Any) -> Any:
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
        if not all(canonical):
            return data
        first = candidates[0]
        sequence = first.get("sequence")
        if not sequence:
            raise ValueError("esm3 canonical Candidate[] input requires sequence")
        out = dict(data)
        out.setdefault("sequence", sequence)
        return out


class ESM3Output(BaseModel):
    per_residue_embeddings_npz: str | None
    predicted_pdb: str | None
    confidence_scores: dict[str, Any] | None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    num_residues: int
    model_name: str
    runtime_ms: int


@register_adapter("descriptor.yaml")
class ESM3Adapter(ToolAdapter):
    """ESM3 wrapper that returns per-residue embeddings + optional structure.

    Defaults to ``esm3_sm_open_v1`` (the 1.4 B-parameter open variant on the
    Cambrian Open License; the SDK auto-downloads
    weights into the local HF cache on first access).

    Gated variants (``esm3-medium-2024-08``, etc.) are supported by setting
    ``HF_TOKEN`` (or ``VPA_HF_CREDENTIAL``) in the worker env and accepting
    EvolutionaryScale's terms at https://huggingface.co/EvolutionaryScale/<model>
    before loading.

    Embedding stripping (see note ``8f956f56-…``): the SDK tokenizer prepends
    a CLS and appends an EOS, so the per-residue tensor is shaped
    ``[L+2, hidden]``. We slice ``[1:L+1, :]`` to align with the input
    sequence length.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.config = dict(config) if config else {}
        self.model = None
        self.loaded_model_name: str | None = None
        self._load(self.config.get("model_name") or _DEFAULT_MODEL)

    def _load(self, model_name: str) -> None:
        _ensure_shared_hf_cache_env()
        from esm.models.esm3 import ESM3

        normalized = _normalize_model_name(model_name)
        if normalized == self.loaded_model_name and self.model is not None:
            return
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            self.model = ESM3.from_pretrained(normalized, device=self.device).eval()
        except Exception as exc:
            hint = (
                "https://huggingface.co/EvolutionaryScale/esm3-medium-2024-08"
                if _looks_gated(normalized)
                else "https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
            )
            raise DependencyError(
                f"ESM3 model {normalized!r} not accessible — set HF_TOKEN "
                f"(or VPA_HF_CREDENTIAL) and accept the EvolutionaryScale terms at {hint}",
                details={
                    "model_name": normalized,
                    "requested_name": model_name,
                    "hf_credential_set": bool(_hf_credential()),
                    "cause": f"{type(exc).__name__}: {exc}",
                },
            ) from exc
        self.loaded_model_name = normalized

    @torch.no_grad()
    def run(self, params: ESM3Input, ctx: AdapterRunContext) -> ESM3Output:
        from esm.sdk.api import ESMProtein, GenerationConfig, LogitsConfig

        t0 = time.time()
        ctx.emit_progress(0.05)
        self._load(params.model_name)
        ctx.emit_progress(0.25)

        tracks = {t.lower() for t in params.tracks}
        seq_len = len(params.sequence)
        protein = ESMProtein(sequence=params.sequence)
        encoded = self.model.encode(protein)
        ctx.emit_progress(0.40)

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        per_residue_npz: str | None = None
        if {"sequence", "embedding"} & tracks:
            logits_out = self.model.logits(
                encoded, LogitsConfig(sequence=True, return_embeddings=True)
            )
            embeddings = logits_out.embeddings  # [1, L+2, H]
            if embeddings is None:
                raise RuntimeError(
                    "ESM3.logits returned no embeddings even though return_embeddings=True"
                )
            # CLS at 0, EOS at L+1 — slice [1:L+1] to recover [L, H].
            per_residue = embeddings[0, 1 : seq_len + 1, :]
            arr = per_residue.float().detach().cpu().numpy().astype(np.float32)
            npz_path = out_dir / "esm3_per_residue.npz"
            np.savez(npz_path, per_residue=arr)
            per_residue_npz = str(npz_path)
        ctx.emit_progress(0.60)

        predicted_pdb: str | None = None
        confidence: dict[str, Any] | None = None
        if "structure" in tracks:
            gen_cfg = GenerationConfig(
                track="structure",
                num_steps=int(params.num_steps),
                temperature=float(params.temperature),
            )
            gen = self.model.generate(protein, gen_cfg)
            pdb_path = out_dir / "esm3_predicted.pdb"
            gen.to_pdb(str(pdb_path))
            predicted_pdb = str(pdb_path)
            confidence = {
                "mean_plddt": (
                    float(gen.plddt.mean().item()) if gen.plddt is not None else None
                ),
                "ptm": float(gen.ptm.item()) if gen.ptm is not None else None,
            }
        ctx.emit_progress(0.95)

        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        output = ESM3Output(
            per_residue_embeddings_npz=per_residue_npz,
            predicted_pdb=predicted_pdb,
            confidence_scores=confidence,
            num_residues=seq_len,
            model_name=self.loaded_model_name or params.model_name,
            runtime_ms=runtime_ms,
        )
        output.candidates = _esm3_candidates(params, output)
        return output

    def estimate_vram(self, params: ESM3Input) -> int:
        # esm3_sm_open_v1: 1.4 B params x bf16 = ~3 GiB resident, plus
        # ~5 GiB peak during structure generation on moderate (<512 aa) inputs.
        return 8 * 1024

    def healthcheck(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.loaded_model_name = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "ESM3Adapter",
    "ESM3Input",
    "ESM3Output",
    "_looks_gated",
    "_normalize_model_name",
]


def _esm3_candidates(params: ESM3Input, output: ESM3Output) -> list[dict[str, Any]]:
    local_id = "esm3_0000"
    scores: dict[str, Any] = {
        "model_name": output.model_name,
        "num_residues": output.num_residues,
        "tracks": [track.lower() for track in params.tracks],
        "embedding_present": output.per_residue_embeddings_npz is not None,
        "structure_predicted": output.predicted_pdb is not None,
    }
    if output.confidence_scores is not None:
        scores.update(output.confidence_scores)

    candidate: dict[str, Any] = {
        "schema_version": "candidate_v1",
        "candidate_id": stable_candidate_id(
            "esm3",
            local_id,
            sequence=params.sequence,
            structure_path=output.predicted_pdb,
            index=0,
        ),
        "source": {
            "tool": "esm3",
            "node_id": None,
            "job_id": None,
            "local_id": local_id,
            "native_id": None,
            "index": 0,
        },
        "sequence": params.sequence,
        "scores": {"esm3": scores},
        "artifacts": {},
    }
    if output.predicted_pdb:
        candidate["structure_path"] = output.predicted_pdb
    if output.per_residue_embeddings_npz:
        candidate["artifacts"]["esm3"] = {
            "per_residue_embeddings_npz": output.per_residue_embeddings_npz
        }
    return [candidate]


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"
