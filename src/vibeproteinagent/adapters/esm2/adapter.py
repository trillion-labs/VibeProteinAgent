from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

from vibeproteinagent.adapters.base import ToolAdapter, register_adapter
from vibeproteinagent.adapters.context import AdapterRunContext
from vibeproteinagent.agents.planning.candidate import stable_candidate_id

_DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_MASK_CHARS = frozenset({"X", "?", "_", "-", "."})


class ESM2Input(BaseModel):
    sequence: str = Field(..., min_length=1)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    mode: Literal["embedding", "sequence_generation"] = Field("embedding")
    model_name: str = Field(_DEFAULT_MODEL)
    return_attention: bool = Field(False)
    mask_positions: list[int] = Field(default_factory=list)
    mask_indices_zero_based: list[int] = Field(default_factory=list)

    @field_validator("sequence")
    @classmethod
    def _normalise_sequence(cls, value: str) -> str:
        return value.strip().upper()

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
            raise ValueError("esm2 canonical Candidate[] input requires sequence")
        out = dict(data)
        out.setdefault("sequence", sequence)
        return out

    @model_validator(mode="after")
    def _validate_design_template(self) -> ESM2Input:
        allowed = set(_AMINO_ACIDS) | {"X"}
        if self.mode == "sequence_generation":
            allowed |= set(_MASK_CHARS)
        bad = sorted({c for c in self.sequence if c not in allowed})
        if bad:
            raise ValueError(f"sequence contains unsupported residue/template characters: {bad}")
        L = len(self.sequence)
        seen: set[int] = set()
        for pos in self.mask_positions:
            if pos < 1 or pos > L:
                raise ValueError(f"mask position {pos} is outside sequence length {L}")
            if pos in seen:
                raise ValueError(f"duplicate mask position {pos}")
            seen.add(pos)
        seen_zero: set[int] = set()
        for idx in self.mask_indices_zero_based:
            if idx < 0 or idx >= L:
                raise ValueError(f"zero-based mask index {idx} is outside sequence length {L}")
            if idx in seen_zero:
                raise ValueError(f"duplicate zero-based mask index {idx}")
            seen_zero.add(idx)
        return self


class ESM2Output(BaseModel):
    per_residue_embeddings_npz: str | None = None
    pooled_embedding: list[float] | None = None
    generated_sequences: list[str] = Field(default_factory=list)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    filled_positions: list[int] = Field(default_factory=list)
    generation_strategy: str | None = None
    fallback_used: bool = False
    num_residues: int
    model_name: str
    mode: Literal["embedding", "sequence_generation"] = "embedding"
    runtime_ms: int


@register_adapter("descriptor.yaml")
class ESM2Adapter(ToolAdapter):
    """Hugging Face ESM2 wrapper for embeddings and masked sequence design.

    Strips the CLS (idx 0) and EOS (final non-pad position via attention mask)
    so the returned ``[L, hidden]`` tensor aligns exactly with the input
    sequence length. See note `8f956f56-…` for the gotcha.
    """

    def __init__(self, config: dict, weights_dir: Path, device: torch.device) -> None:
        self.device = device
        self.weights_dir = Path(weights_dir)
        self.config = dict(config) if config else {}
        # Default-load the small ESM2 so daemon warm hits HF cache once,
        # then run() can swap models on demand (rare in practice).
        self.loaded_model_name: str | None = None
        self.tokenizer = None
        self.model = None
        if not self.config.get("defer_model_load", False):
            self._load(self.config.get("model_name") or _DEFAULT_MODEL)

    def _load(self, model_name: str) -> None:
        # Defer transformers import so adapter discovery doesn't pay the cost.
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        if model_name == self.loaded_model_name and self.model is not None:
            return
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.model = model
        self.loaded_model_name = model_name
        self.hidden_size = int(model.config.hidden_size)

    @torch.no_grad()
    def run(self, params: ESM2Input, ctx: AdapterRunContext) -> ESM2Output:
        t0 = time.time()
        ctx.emit_progress(0.05)
        if params.mode == "sequence_generation":
            return self._run_sequence_generation(params, ctx, t0)

        self._load(params.model_name)
        ctx.emit_progress(0.30)

        enc = self.tokenizer(params.sequence, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        ctx.emit_progress(0.45)

        outputs = self.model(
            **enc,
            output_attentions=params.return_attention,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1]  # [1, L+2, H]
        attn = enc["attention_mask"][0]
        # EOS sits right after the last residue; PADs (if any) come after EOS.
        last_idx = int(attn.sum().item()) - 1
        per_residue = hidden[0, 1:last_idx, :]  # [L, H], CLS & EOS removed

        L = int(per_residue.shape[0])
        if L != len(params.sequence):
            raise RuntimeError(
                f"ESM2 per-residue length {L} does not match input "
                f"sequence length {len(params.sequence)}; CLS/EOS stripping "
                "failed."
            )

        ctx.emit_progress(0.85)

        out_dir = ctx.job_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / "esm2_per_residue.npz"
        import numpy as np

        arr = per_residue.float().detach().cpu().numpy().astype(np.float32)
        save_kwargs: dict[str, Any] = {"per_residue": arr}
        if params.return_attention:
            # attentions: tuple of [B, heads, T, T], length=num_layers
            stacked = torch.stack(outputs.attentions, dim=1)  # [B, layers, heads, T, T]
            stacked = stacked[0, :, :, 1:last_idx, 1:last_idx]
            save_kwargs["attention"] = stacked.detach().cpu().numpy().astype(np.float32)
        np.savez(npz_path, **save_kwargs)

        pooled = per_residue.mean(dim=0)
        runtime_ms = int((time.time() - t0) * 1000)
        ctx.emit_progress(1.0)
        output = ESM2Output(
            per_residue_embeddings_npz=str(npz_path),
            pooled_embedding=pooled.float().detach().cpu().tolist(),
            num_residues=L,
            model_name=params.model_name,
            mode="embedding",
            runtime_ms=runtime_ms,
        )
        output.candidates = _esm2_candidates(params, output)
        return output

    def _run_sequence_generation(
        self, params: ESM2Input, ctx: AdapterRunContext, t0: float
    ) -> ESM2Output:
        user_positions = _masked_positions(
            params.sequence,
            params.mask_positions,
            params.mask_indices_zero_based,
        )
        indices = [pos - 1 for pos in user_positions]
        if not indices:
            ctx.emit_progress(1.0)
            output = ESM2Output(
                generated_sequences=[params.sequence],
                filled_positions=[],
                generation_strategy="identity_no_masks",
                num_residues=len(params.sequence),
                model_name=params.model_name,
                mode="sequence_generation",
                runtime_ms=int((time.time() - t0) * 1000),
            )
            output.candidates = _esm2_candidates(params, output)
            return output

        self._load(params.model_name)
        ctx.emit_progress(0.30)
        tokens = _tokens_for_generation(params.sequence, indices, self.tokenizer.mask_token)
        special_tokens = [self.tokenizer.cls_token, *tokens, self.tokenizer.eos_token]
        ids = self.tokenizer.convert_tokens_to_ids(special_tokens)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        ctx.emit_progress(0.55)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        aa_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in _AMINO_ACIDS],
            dtype=torch.long,
            device=self.device,
        )
        logits = outputs.logits[0]
        generated = list(params.sequence)
        for pos in indices:
            token_pos = pos + 1  # account for CLS
            scores = logits[token_pos, aa_ids]
            generated[pos] = _AMINO_ACIDS[int(torch.argmax(scores).item())]

        ctx.emit_progress(1.0)
        filled = "".join(generated)
        output = ESM2Output(
            generated_sequences=[filled],
            filled_positions=user_positions,
            generation_strategy="esm2_mlm_argmax",
            num_residues=len(filled),
            model_name=params.model_name,
            mode="sequence_generation",
            runtime_ms=int((time.time() - t0) * 1000),
        )
        output.candidates = _esm2_candidates(params, output)
        return output

    def estimate_vram(self, params: ESM2Input) -> int:
        # Small ESM2 (t12_35M) ~ 200 MiB peak. Larger variants override.
        return 1024

    def healthcheck(self) -> bool:
        return self.model is not None

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = ["ESM2Adapter", "ESM2Input", "ESM2Output"]


def _masked_positions(
    sequence: str,
    explicit_positions: list[int],
    zero_based_indices: list[int] | None = None,
) -> list[int]:
    positions = {idx + 1 for idx, aa in enumerate(sequence) if aa in _MASK_CHARS}
    positions.update(explicit_positions)
    positions.update(idx + 1 for idx in zero_based_indices or [])
    return sorted(positions)


def _tokens_for_generation(sequence: str, positions: list[int], mask_token: str) -> list[str]:
    masked = set(positions)
    return [mask_token if idx in masked else aa for idx, aa in enumerate(sequence)]


def _esm2_candidates(params: ESM2Input, output: ESM2Output) -> list[dict[str, Any]]:
    if output.mode == "sequence_generation":
        sequences = output.generated_sequences or [params.sequence]
    else:
        sequences = [params.sequence]

    candidates: list[dict[str, Any]] = []
    for index, sequence in enumerate(sequences):
        local_id = f"esm2_{index:04d}"
        scores: dict[str, Any] = {
            "mode": output.mode,
            "model_name": output.model_name,
            "num_residues": output.num_residues,
            "fallback_used": output.fallback_used,
        }
        artifacts: dict[str, Any] = {}
        if output.mode == "embedding":
            scores["embedding_dim"] = (
                len(output.pooled_embedding) if output.pooled_embedding is not None else None
            )
            scores["return_attention"] = params.return_attention
            if output.per_residue_embeddings_npz:
                artifacts["esm2"] = {
                    "per_residue_embeddings_npz": output.per_residue_embeddings_npz
                }
        else:
            scores["filled_positions"] = list(output.filled_positions)
            scores["generation_strategy"] = output.generation_strategy

        candidates.append(
            {
                "schema_version": "candidate_v1",
                "candidate_id": stable_candidate_id(
                    "esm2",
                    local_id,
                    sequence=sequence,
                    index=index,
                ),
                "source": {
                    "tool": "esm2",
                    "node_id": None,
                    "job_id": None,
                    "local_id": local_id,
                    "native_id": None,
                    "index": index,
                },
                "sequence": sequence,
                "scores": {"esm2": scores},
                "artifacts": artifacts,
            }
        )
    return candidates


def _is_canonical_candidate(item: Any) -> bool:
    return isinstance(item, dict) and item.get("schema_version") == "candidate_v1"
