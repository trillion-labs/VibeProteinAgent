"""Fast unit tests for the PXDesign adapter shell.

These tests do NOT load the PXDesign model — they only validate descriptor,
pydantic I/O shapes, env-var resolution, and package-level exports.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vibeproteinagent.adapters import load_descriptor
from vibeproteinagent.adapters.pxdesign import (
    PROTENIX_PXD_VENDOR,
    PXDBENCH_VENDOR,
    PXDESIGN_CKPT_DIR,
    PXDESIGN_VENDOR,
    PXDesignAdapter,
    PXDesignDesign,
    PXDesignInput,
    PXDesignOutput,
)
from vibeproteinagent.adapters.pxdesign import adapter as adapter_mod

_ADAPTER_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "vibeproteinagent"
    / "adapters"
    / "pxdesign"
)
_DESCRIPTOR = _ADAPTER_DIR / "descriptor.yaml"
_INPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "pxdesign.input.json"
_OUTPUT_SCHEMA = _ADAPTER_DIR / "schemas" / "pxdesign.output.json"


def test_descriptor_loads_and_matches_spec() -> None:
    d = load_descriptor(_DESCRIPTOR)
    assert d.name == "pxdesign"
    assert d.version == "1.0.0"
    assert d.category.value == "protein_design"
    assert d.runtime.value == "python"
    assert d.backend.value == "pytorch"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.vram_min_gib == 24
    assert d.startup_cost.value == "high"
    assert d.warm_load_capable is True
    assert d.pin_warm is False
    assert d.supported_tasks == ["binder_design"]
    assert [p.value for p in d.supported_precisions] == ["bf16"]
    assert d.incompatible_with == ["protenix_v2"]
    assert d.support_status.value == "verified"
    assert d.availability_probe == (
        "vibeproteinagent.adapters.pxdesign:availability_skip_reason"
    )
    assert d.support_status_reason is None
    assert d.entry_point == (
        "vibeproteinagent.adapters.pxdesign.adapter:PXDesignAdapter"
    )


def test_descriptor_input_schema_is_valid_json() -> None:
    raw = json.loads(_INPUT_SCHEMA.read_text())
    assert raw["$id"].endswith("pxdesign.input.json")
    assert {"required": ["target_pdb"]} in raw["anyOf"]
    assert {"required": ["candidates"]} in raw["anyOf"]
    assert "target_chain" not in raw.get("required", [])
    assert "binder_length" not in raw.get("required", [])
    props = raw["properties"]
    assert props["num_samples"]["default"] == 5
    assert props["num_steps"]["default"] == 400
    assert raw["additionalProperties"] is False


def test_input_schema_accepts_canonical_candidate_source() -> None:
    from jsonschema import Draft202012Validator

    raw = json.loads(_INPUT_SCHEMA.read_text())
    Draft202012Validator.check_schema(raw)
    errors = list(
        Draft202012Validator(raw).iter_errors(
            {
                "candidates": [
                    {
                        "schema_version": "candidate_v1",
                        "candidate_id": "c0",
                        "source": {"tool": "producer", "local_id": "0"},
                        "structure_path": "/tmp/upstream.cif",
                    }
                ]
            }
        )
    )

    assert errors == []
    params = PXDesignInput(
        candidates=[
            {
                "schema_version": "candidate_v1",
                "candidate_id": "c0",
                "source": {"tool": "producer", "local_id": "0"},
                "structure_path": "/tmp/upstream.cif",
            }
        ]
    )
    assert params.target_pdb == "/tmp/upstream.cif"


def test_descriptor_output_schema_is_valid_json() -> None:
    raw = json.loads(_OUTPUT_SCHEMA.read_text())
    assert raw["$id"].endswith("pxdesign.output.json")
    for required in ("designs", "runtime_ms", "num_samples"):
        assert required in raw["required"]
    design_required = raw["properties"]["designs"]["items"]["required"]
    for k in ("pdb_path", "cif_path", "score", "sequence", "binder_length"):
        assert k in design_required


def test_descriptor_yaml_round_trips() -> None:
    yml = yaml.safe_load(_DESCRIPTOR.read_text())
    assert ":" in yml["entry_point"]
    assert yml["incompatible_with"] == ["protenix_v2"]


def test_input_pydantic_defaults() -> None:
    p = PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50)
    assert p.num_samples == 5
    assert p.num_steps == 400
    assert p.seed is None
    assert p.temperature == pytest.approx(1.0)
    assert p.hotspots == []
    assert p.crop is None
    assert not hasattr(p, "use_msa")


def test_input_pydantic_validates_bounds() -> None:
    with pytest.raises(ValidationError):
        PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=10)
    with pytest.raises(ValidationError):
        PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50,
                      num_samples=0)
    with pytest.raises(ValidationError):
        PXDesignInput(target_pdb="/tmp/x.pdb", target_chain="A", binder_length=50,
                      num_steps=1)


def test_output_pydantic_round_trip() -> None:
    design = PXDesignDesign(
        pdb_path=None,
        cif_path="/tmp/a.cif",
        score=0.74,
        sequence="GAGAGAGAGA",
        binder_length=10,
    )
    out = PXDesignOutput(
        designs=[design], runtime_ms=4242, num_samples=1, task_name="x", seed=101,
    )
    again = PXDesignOutput.model_validate(out.model_dump())
    assert again.designs[0].sequence == "GAGAGAGAGA"
    assert again.runtime_ms == 4242
    assert again.task_name == "x"


def test_env_var_overrides_default_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VPA_PXDESIGN_VENDOR", "/tmp/__vpa_pxd_vendor")
    monkeypatch.setenv("VPA_PXDBENCH_VENDOR", "/tmp/__vpa_pxdbench_vendor")
    monkeypatch.setenv("VPA_PROTENIX_PXD_VENDOR", "/tmp/__vpa_protenix_pxd_vendor")
    monkeypatch.setenv("VPA_PXDESIGN_CKPT_DIR", "/tmp/__vpa_pxd_ckpt")
    import importlib

    reloaded = importlib.reload(adapter_mod)
    try:
        assert str(reloaded.PXDESIGN_VENDOR) == "/tmp/__vpa_pxd_vendor"
        assert str(reloaded.PXDBENCH_VENDOR) == "/tmp/__vpa_pxdbench_vendor"
        assert str(reloaded.PROTENIX_PXD_VENDOR) == "/tmp/__vpa_protenix_pxd_vendor"
        assert str(reloaded.PXDESIGN_CKPT_DIR) == "/tmp/__vpa_pxd_ckpt"
    finally:
        monkeypatch.delenv("VPA_PXDESIGN_VENDOR")
        monkeypatch.delenv("VPA_PXDBENCH_VENDOR")
        monkeypatch.delenv("VPA_PROTENIX_PXD_VENDOR")
        monkeypatch.delenv("VPA_PXDESIGN_CKPT_DIR")
        importlib.reload(adapter_mod)


def test_protenix_data_root_uses_vpa_ckpt_common(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    ckpt_root = tmp_path / "protenix"
    common = ckpt_root / "common"
    common.mkdir(parents=True)
    monkeypatch.delenv("VPA_PROTENIX_DATA_ROOT_DIR", raising=False)
    monkeypatch.delenv("PROTENIX_DATA_ROOT_DIR", raising=False)
    monkeypatch.setenv("VPA_PROTENIX_CKPT", str(ckpt_root))

    assert adapter_mod._default_protenix_data_root() == str(common)


def test_resolve_checkpoint_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(adapter_mod, "PXDESIGN_CKPT_DIR", tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        adapter_mod._resolve_checkpoint()


def test_pxdesign_preflights_target_chain_for_existing_mmcif(tmp_path: Path) -> None:
    pdb = tmp_path / "target.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1      11.104  13.207   9.100  1.00 20.00           N\n"
        "ATOM      2  CA  ALA A   1      12.560  13.300   9.100  1.00 20.00           C\n"
        "ATOM      3  C   ALA A   1      13.000  14.700   9.100  1.00 20.00           C\n"
        "TER\n"
        "END\n"
    )
    cif = tmp_path / "target.cif"
    adapter_mod._convert_pdb_to_mmcif(pdb, cif, target_chain="A")

    with pytest.raises(Exception, match="chain preflight"):
        adapter_mod._prepare_pxdesign_structure_file(cif, target_chain="B")


def test_registry_finds_pxdesign() -> None:
    from vibeproteinagent.adapters import ToolAdapter, ToolRegistry

    r = ToolRegistry()
    r.discover()
    names = {d.name for d in r.list()}
    assert "pxdesign" in names
    desc, cls = r.get("pxdesign")
    assert desc.name == "pxdesign"
    assert desc.incompatible_with == ["protenix_v2"]
    assert cls.__name__ == "PXDesignAdapter"
    assert issubclass(cls, ToolAdapter)


def test_protenix_v2_reciprocal_incompatibility() -> None:
    """Protenix-v2 declares the same incompatibility back at pxdesign."""
    from vibeproteinagent.adapters import ToolRegistry

    r = ToolRegistry()
    r.discover()
    desc, _ = r.get("protenix_v2")
    assert "pxdesign" in desc.incompatible_with


def test_module_exports() -> None:
    import vibeproteinagent.adapters.pxdesign as pkg

    for name in (
        "PXDesignAdapter",
        "PXDesignInput",
        "PXDesignOutput",
        "PXDesignDesign",
        "PXDESIGN_VENDOR",
        "PXDBENCH_VENDOR",
        "PROTENIX_PXD_VENDOR",
        "PXDESIGN_CKPT_DIR",
        "PXDESIGN_MODEL_NAME",
    ):
        assert hasattr(pkg, name), f"missing export: {name}"


def test_paths_are_pathlib() -> None:
    assert isinstance(PXDESIGN_VENDOR, Path)
    assert isinstance(PXDBENCH_VENDOR, Path)
    assert isinstance(PROTENIX_PXD_VENDOR, Path)
    assert isinstance(PXDESIGN_CKPT_DIR, Path)


def test_vendor_trees_exist_on_disk() -> None:
    """Phase 5.1 vendored these; the unit test checks they remain in place."""
    assert PXDESIGN_VENDOR.is_dir(), f"missing {PXDESIGN_VENDOR}"
    assert PXDBENCH_VENDOR.is_dir(), f"missing {PXDBENCH_VENDOR}"
    assert PROTENIX_PXD_VENDOR.is_dir(), f"missing {PROTENIX_PXD_VENDOR}"


def test_descriptor_default_pin_warm_is_false() -> None:
    """PXDesign opts out of pin_warm: its 24 GiB ceiling would block protenix_v2."""
    d = load_descriptor(_DESCRIPTOR)
    assert d.pin_warm is False


def test_healthcheck_includes_pipeline_availability_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = PXDesignAdapter.__new__(PXDesignAdapter)
    adapter.ckpt_path = tmp_path / "pxdesign_v0.1.0.pt"
    adapter.ckpt_path.write_text("fake")
    vendor = tmp_path / "vendor"
    vendor.mkdir()
    monkeypatch.setattr(adapter_mod, "PXDESIGN_VENDOR", vendor)

    monkeypatch.setattr(
        adapter_mod, "availability_skip_reason", lambda: "pipeline env missing",
    )
    assert adapter.healthcheck() is False

    monkeypatch.setattr(adapter_mod, "availability_skip_reason", lambda: None)
    assert adapter.healthcheck() is True
