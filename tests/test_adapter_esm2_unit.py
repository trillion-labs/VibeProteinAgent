"""Fast unit tests for the ESM2 adapter (no GPU, no HF download)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.esm2.adapter import (
    _AMINO_ACIDS,
    _DEFAULT_MODEL,
    ESM2Adapter,
    ESM2Input,
    ESM2Output,
    _masked_positions,
)


def test_descriptor_fields():
    d = ESM2Adapter.descriptor
    assert d.name == "esm2"
    assert d.version == "1.1.0"
    assert d.category.value == "protein_design"
    assert "sequence_generation" in d.supported_tasks
    assert "partial_sequence_design" in d.supported_tasks
    assert "embedding" in d.supported_tasks
    assert d.gpu_required is True
    assert d.pin_warm is True


def test_input_defaults():
    p = ESM2Input(sequence="MKKL")
    assert p.mode == "embedding"
    assert p.model_name == _DEFAULT_MODEL
    assert p.return_attention is False
    assert p.mask_positions == []


def test_input_rejects_empty():
    with pytest.raises(ValidationError):
        ESM2Input(sequence="")


def test_embedding_input_rejects_template_mask_characters():
    with pytest.raises(ValidationError):
        ESM2Input(sequence="MK?L")


def test_sequence_generation_accepts_partial_template():
    p = ESM2Input(mode="sequence_generation", sequence="MKX?L", mask_positions=[1])
    assert p.sequence == "MKX?L"
    assert _masked_positions(p.sequence, p.mask_positions) == [1, 3, 4]


def test_output_round_trip():
    out = ESM2Output(
        per_residue_embeddings_npz="/tmp/e.npz",
        pooled_embedding=[0.1, 0.2, 0.3],
        num_residues=3,
        model_name="facebook/esm2_t12_35M_UR50D",
        mode="embedding",
        runtime_ms=12,
    )
    rebuilt = ESM2Output.model_validate(out.model_dump())
    assert rebuilt == out


def test_sequence_generation_uses_masked_lm_logits(tmp_path):
    adapter = ESM2Adapter(
        config={"defer_model_load": True},
        weights_dir=tmp_path,
        device=torch.device("cpu"),
    )

    class FakeTokenizer:
        cls_token = "<cls>"
        eos_token = "<eos>"
        mask_token = "<mask>"

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, list):
                return [self.convert_tokens_to_ids(t) for t in tokens]
            if tokens in _AMINO_ACIDS:
                return _AMINO_ACIDS.index(tokens)
            return len(_AMINO_ACIDS)

    class FakeModel:
        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            logits = torch.zeros(
                (1, int(input_ids.shape[1]), len(_AMINO_ACIDS) + 1),
                dtype=torch.float32,
            )
            logits[:, :, _AMINO_ACIDS.index("A")] = 5.0
            return SimpleNamespace(logits=logits)

    def fake_load(model_name: str) -> None:
        adapter.tokenizer = FakeTokenizer()
        adapter.model = FakeModel()
        adapter.loaded_model_name = model_name

    adapter._load = fake_load  # type: ignore[method-assign]

    params = ESM2Input(mode="sequence_generation", sequence="ACX?D")
    out = adapter.run(params, null_context(tmp_path))
    assert out.mode == "sequence_generation"
    assert out.generated_sequences == ["ACAAD"]
    assert out.filled_positions == [3, 4]
    assert out.generation_strategy == "esm2_mlm_argmax"
    assert out.fallback_used is False
    assert out.per_residue_embeddings_npz is None
    assert out.pooled_embedding is None
