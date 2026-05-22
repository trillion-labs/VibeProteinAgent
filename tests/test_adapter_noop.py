from __future__ import annotations

import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.noop.adapter import NoopAdapter, NoopInput, NoopOutput


def test_noop_healthcheck(tmp_path):
    a = NoopAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    assert a.healthcheck() is True


def test_noop_run_echoes_payload(tmp_path):
    a = NoopAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    out = a.run(NoopInput(payload="hello"), null_context(tmp_path))
    assert isinstance(out, NoopOutput)
    assert out.payload == "hello"
    assert out.echoed_at > 0


def test_noop_estimate_vram_zero(tmp_path):
    a = NoopAdapter(config={}, weights_dir=tmp_path, device=torch.device("cpu"))
    assert a.estimate_vram(NoopInput(payload="x")) == 0


def test_noop_descriptor_attached():
    assert NoopAdapter.descriptor.name == "noop"
    assert NoopAdapter.descriptor.gpu_required is False
