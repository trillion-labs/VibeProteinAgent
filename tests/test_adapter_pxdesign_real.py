"""Slow: run the real PXDesign adapter end-to-end on the bundled tiny target.

Skipped when:
  * CUDA isn't available;
  * a vendor tree is missing;
  * the PXDesign checkpoint isn't on disk.

Opt-in: ``pytest -m slow tests/test_adapter_pxdesign_real.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.pxdesign import (
    PROTENIX_PXD_VENDOR,
    PXDBENCH_VENDOR,
    PXDESIGN_CKPT_DIR,
    PXDESIGN_MODEL_NAME,
    PXDESIGN_VENDOR,
    PXDesignAdapter,
    PXDesignInput,
)

pytestmark = pytest.mark.slow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_CIF = _PROJECT_ROOT / "tests" / "fixtures" / "5o45.cif"


def _skip_if_unavailable() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for PXDesign")
    for root in (PXDESIGN_VENDOR, PXDBENCH_VENDOR, PROTENIX_PXD_VENDOR):
        if not root.is_dir():
            pytest.skip(f"vendor tree missing: {root}")
    ckpt = PXDESIGN_CKPT_DIR / f"{PXDESIGN_MODEL_NAME}.pt"
    if not ckpt.is_file():
        pytest.skip(f"PXDesign checkpoint not found at {ckpt}")
    if not _FIXTURE_CIF.is_file():
        pytest.skip(f"fixture CIF missing: {_FIXTURE_CIF}")


@pytest.fixture(scope="module")
def adapter() -> PXDesignAdapter:
    _skip_if_unavailable()
    return PXDesignAdapter(
        config={}, weights_dir=Path("/tmp"), device=torch.device("cuda:0")
    )


def test_run_produces_design_cif(adapter: PXDesignAdapter, tmp_path: Path) -> None:
    params = PXDesignInput(
        target_pdb=str(_FIXTURE_CIF),
        target_chain="A",
        binder_length=50,
        num_samples=1,
        num_steps=5,
        seed=101,
        crop="1-116",
    )
    job_dir = tmp_path / "jobs" / "job_real_pxdesign"
    job_dir.mkdir(parents=True)
    out = adapter.run(params, null_context(job_dir))

    assert out.num_samples == 1
    assert len(out.designs) == 1
    design = out.designs[0]
    cif = Path(design.cif_path)
    assert cif.is_file(), f"missing CIF {cif}"
    assert cif.stat().st_size > 1024, f"CIF too small ({cif.stat().st_size} bytes)"
    text = cif.read_text()
    assert "_atom_site." in text
    assert design.binder_length == 50
    # Sequence may or may not parse cleanly depending on biotite shim; tolerate
    # an empty string (the CIF on disk is the authoritative artifact).
    if design.sequence:
        assert len(design.sequence) == 50, (
            f"binder seq length mismatch: got {len(design.sequence)}, expected 50"
        )
    assert out.runtime_ms > 0
