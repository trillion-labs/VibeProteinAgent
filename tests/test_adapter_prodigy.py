"""Tests for the PRODIGY adapter (CPU-only, vendored haddocking/prodigy).

Fast: runs in ~250ms on the bundled 3BZD complex fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.prodigy.adapter import (
    ProdigyAdapter,
    ProdigyInput,
    ProdigyOutput,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE = _PROJECT_ROOT / "tests" / "fixtures" / "prodigy_3BZD.pdb"


def test_descriptor_fields():
    d = ProdigyAdapter.descriptor
    assert d.name == "prodigy"
    assert d.category.value == "scoring"
    assert d.gpu_required is False
    assert d.pin_warm is True


def test_input_defaults():
    p = ProdigyInput(pdb_path="/tmp/a.pdb", chain_a="A", chain_b="B")
    assert p.temperature_k == pytest.approx(298.15)


def test_input_rejects_nonpositive_temp():
    with pytest.raises(ValidationError):
        ProdigyInput(pdb_path="/tmp/a.pdb", chain_a="A", chain_b="B", temperature_k=0.0)


def test_output_round_trip():
    out = ProdigyOutput(
        delta_g_kcal_mol=-9.37,
        kd_nm=133.27,
        interface_contacts=51,
        runtime_ms=200,
    )
    rebuilt = ProdigyOutput.model_validate(out.model_dump())
    assert rebuilt == out


@pytest.mark.skipif(not _FIXTURE.exists(), reason="3BZD.pdb fixture missing")
def test_run_end_to_end_on_3bzd(tmp_path):
    a = ProdigyAdapter(config={}, weights_dir=Path("/nonexistent"), device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    out = a.run(
        ProdigyInput(pdb_path=str(_FIXTURE), chain_a="A", chain_b="B"),
        null_context(job_dir),
    )
    # PRODIGY's reference value on 3BZD ≈ -9.37 kcal/mol (Phase 1.7).
    assert -15.0 <= out.delta_g_kcal_mol <= -3.0
    assert out.kd_nm > 0.0
    assert out.interface_contacts > 10
    assert out.runtime_ms >= 0


@pytest.mark.skipif(not _FIXTURE.exists(), reason="3BZD.pdb fixture missing")
def test_run_rejects_unknown_chain(tmp_path):
    a = ProdigyAdapter(config={}, weights_dir=Path("/nonexistent"), device=torch.device("cpu"))
    job_dir = tmp_path / "jobs" / "j"
    job_dir.mkdir(parents=True)
    with pytest.raises(ValueError):
        a.run(
            ProdigyInput(pdb_path=str(_FIXTURE), chain_a="A", chain_b="Z"),
            null_context(job_dir),
        )
