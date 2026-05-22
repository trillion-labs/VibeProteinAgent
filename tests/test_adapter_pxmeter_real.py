"""Tests for the PXMeter adapter (vendored ByteDance structure scorer).

PXMeter is CPU-only but takes ~5 s on the bundled 7rss CIF pair, so the
end-to-end run is marked ``slow``. Fast tests cover descriptor metadata,
pydantic validation, and the ``_extract_metrics`` parser without invoking
the vendored package.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from vibeproteinagent.adapters import null_context
from vibeproteinagent.adapters.pxmeter.adapter import (
    _VENDOR_ROOT,
    ChainPair,
    PXMeterAdapter,
    PXMeterInput,
    PXMeterOutput,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REF_CIF = _VENDOR_ROOT / "examples" / "7rss.cif"
_PRED_CIF = _VENDOR_ROOT / "examples" / "7rss_protenix_pred.cif"


def test_descriptor_fields():
    d = PXMeterAdapter.descriptor
    assert d.name == "pxmeter"
    assert d.category.value == "scoring"
    assert d.gpu_required is False
    assert d.pin_warm is True
    assert d.warm_load_capable is True


def test_vendored_cli_exists():
    assert (_VENDOR_ROOT / "pxmeter" / "cli.py").is_file()


def test_input_defaults():
    p = PXMeterInput(predicted_cif="/tmp/a.cif", reference_cif="/tmp/b.cif")
    assert p.metrics == ["lddt", "dockq"]
    assert p.chain_pairs is None


def test_input_rejects_missing_paths():
    with pytest.raises(ValidationError):
        PXMeterInput(predicted_cif="", reference_cif="/tmp/b.cif")


def test_chain_pair_validates():
    cp = ChainPair(a="A", b="B")
    assert cp.a == "A" and cp.b == "B"
    with pytest.raises(ValidationError):
        ChainPair(a="", b="B")


def test_output_round_trip():
    out = PXMeterOutput(
        metrics={"lddt": 0.95, "dockq": [{"chain_pair": "A,B", "score": 0.84}]},
        report_json_path="/tmp/report.json",
        runtime_ms=42,
    )
    rebuilt = PXMeterOutput.model_validate(out.model_dump())
    assert rebuilt.metrics["lddt"] == 0.95
    assert rebuilt.runtime_ms == 42


def test_extract_metrics_lddt_only():
    report = {
        "complex": {"lddt": 0.947},
        "chain": {"A": {"lddt": 0.96}},
        "interface": {"A,B": {"lddt": 0.92, "dockq": 0.91}},
    }
    params = PXMeterInput(
        predicted_cif="/tmp/a.cif", reference_cif="/tmp/b.cif", metrics=["lddt"]
    )
    metrics = PXMeterAdapter._extract_metrics(report, params)
    assert metrics["lddt"] == pytest.approx(0.947)
    assert metrics["per_chain_lddt"] == {"A": 0.96}
    assert metrics["per_interface_lddt"] == {"A,B": 0.92}
    assert "dockq" not in metrics


def test_extract_metrics_dockq_chain_pair_filter():
    report = {
        "complex": {"lddt": 0.9},
        "interface": {
            "A,B": {"lddt": 0.92, "dockq": 0.85},
            "A,C": {"lddt": 0.91, "dockq": 0.78},
        },
    }
    params = PXMeterInput(
        predicted_cif="/tmp/a.cif",
        reference_cif="/tmp/b.cif",
        metrics=["dockq"],
        chain_pairs=[ChainPair(a="A", b="B")],
    )
    metrics = PXMeterAdapter._extract_metrics(report, params)
    assert metrics["dockq"] == [{"chain_pair": "A,B", "score": 0.85}]


def test_extract_metrics_chain_pair_filter_order_insensitive():
    """``chain_pairs=[A,B]`` should match an interface labeled ``B,A`` too."""
    report = {"interface": {"B,A": {"dockq": 0.42}}}
    params = PXMeterInput(
        predicted_cif="/tmp/a.cif",
        reference_cif="/tmp/b.cif",
        metrics=["dockq"],
        chain_pairs=[ChainPair(a="A", b="B")],
    )
    metrics = PXMeterAdapter._extract_metrics(report, params)
    assert metrics["dockq"] == [{"chain_pair": "B,A", "score": 0.42}]


def test_estimate_vram_zero():
    a_cls = PXMeterAdapter
    # Construct without actually loading: call estimate_vram on a dummy instance.
    # We use __new__ to bypass __init__'s import requirement.
    inst = a_cls.__new__(a_cls)
    p = PXMeterInput(predicted_cif="/tmp/a.cif", reference_cif="/tmp/b.cif")
    assert inst.estimate_vram(p) == 0


@pytest.mark.slow
@pytest.mark.skipif(
    not (_REF_CIF.exists() and _PRED_CIF.exists()),
    reason="PXMeter example CIFs not present",
)
def test_run_end_to_end(tmp_path):
    a = PXMeterAdapter(
        config={}, weights_dir=Path("/nonexistent"), device=torch.device("cpu")
    )
    job_dir = tmp_path / "jobs" / "pxm"
    job_dir.mkdir(parents=True)
    out = a.run(
        PXMeterInput(
            reference_cif=str(_REF_CIF),
            predicted_cif=str(_PRED_CIF),
            metrics=["lddt", "dockq"],
        ),
        null_context(job_dir),
    )
    # Phase 1.7 reference: lddt ~= 0.947, DockQ ~= 0.84-0.92 on the 7rss pair.
    assert out.metrics["lddt"] == pytest.approx(0.947, abs=0.01)
    dockq = out.metrics["dockq"]
    assert dockq is not None and len(dockq) == 3
    for row in dockq:
        assert 0.8 <= row["score"] <= 1.0
    assert {"A", "B", "C"} <= set(out.metrics["per_chain_lddt"].keys())
    report = json.loads(Path(out.report_json_path).read_text())
    assert report["entry_id"] == "7rss"
    assert out.runtime_ms > 0
