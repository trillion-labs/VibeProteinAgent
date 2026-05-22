from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from vibeproteinagent.adapters import (
    ToolDescriptor,
    ToolRegistry,
    load_descriptor,
    serialize_descriptor,
)
from vibeproteinagent.adapters.descriptor import _SCHEMA_PATH
from vibeproteinagent.core.errors import ErrorCode, InvalidParamsError, VPAError

_ADAPTERS_ROOT = Path(__file__).resolve().parents[1] / "src" / "vibeproteinagent" / "adapters"
_NOOP_DESCRIPTOR = _ADAPTERS_ROOT / "noop" / "descriptor.yaml"
_MPNN_DESCRIPTOR = _ADAPTERS_ROOT / "proteinmpnn" / "descriptor.yaml"


def test_schema_is_self_applicable():
    schema = json.loads(_SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    assert schema["$id"] == "https://vibeproteinagent.dev/schemas/tool-descriptor.json"


def test_load_noop_descriptor():
    d = load_descriptor(_NOOP_DESCRIPTOR)
    assert isinstance(d, ToolDescriptor)
    assert d.name == "noop"
    assert d.version == "1.0.0"
    assert d.category.value == "misc"
    assert d.gpu_required is False
    assert d.estimated_runtime_s >= 0
    assert "noop" in d.supported_tasks
    assert d.supported_precisions[0].value == "fp32"
    assert d.entry_point == "vibeproteinagent.adapters.noop.adapter:NoopAdapter"


def test_load_proteinmpnn_descriptor():
    d = load_descriptor(_MPNN_DESCRIPTOR)
    assert isinstance(d, ToolDescriptor)
    assert d.name == "proteinmpnn"
    assert d.version == "1.0.1"
    assert d.category.value == "inverse_folding"
    assert d.gpu_required is True
    assert d.gpus_required == 1
    assert d.weight_uris is not None and len(d.weight_uris) == 1
    assert d.weight_uris[0].sha256 == (
        "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
    )
    assert d.fusion is not None and d.fusion.output_kind.value == "disk"


def test_serialize_round_trips(tmp_path):
    d = load_descriptor(_NOOP_DESCRIPTOR)
    out = serialize_descriptor(d)
    assert "vram_min_gib" in out
    assert out["vram_min_gib"] is None
    p = tmp_path / "descriptor.json"
    p.write_text(json.dumps(out))
    again = load_descriptor(p)
    assert again.model_dump(mode="json") == d.model_dump(mode="json")


def test_support_status_fields_default_to_verified():
    d = load_descriptor(_NOOP_DESCRIPTOR)
    assert d.support_status.value == "verified"
    assert d.support_status_reason is None
    assert d.availability_probe is None


def test_support_status_fields_validate(tmp_path):
    raw = yaml.safe_load(_NOOP_DESCRIPTOR.read_text())
    raw["support_status"] = "partial"
    raw["support_status_reason"] = "mode=x requires an operator-provisioned env"
    raw["availability_probe"] = "vibeproteinagent.adapters.noop:availability_skip_reason"
    p = tmp_path / "descriptor.yaml"
    p.write_text(yaml.safe_dump(raw))
    d = load_descriptor(p)
    assert d.support_status.value == "partial"
    assert d.availability_probe == (
        "vibeproteinagent.adapters.noop:availability_skip_reason"
    )


def test_builtin_catalog_is_descriptor_complete():
    """Tool catalog must list every built-in descriptor, independent of runtime deps."""
    expected = {
        p.parent.name
        for p in _ADAPTERS_ROOT.glob("*/descriptor.yaml")
        if not p.parent.name.startswith("_")
    }
    registry = ToolRegistry()
    registry.discover(refresh=True)
    actual = {d.name for d in registry.list()}
    assert expected == actual


def test_builtin_descriptors_do_not_ship_unresolved_support_statuses():
    non_verified = {}
    for path in _ADAPTERS_ROOT.glob("*/descriptor.yaml"):
        if path.parent.name.startswith("_"):
            continue
        descriptor = load_descriptor(path)
        if descriptor.support_status.value != "verified":
            non_verified[descriptor.name] = descriptor.support_status.value
    assert non_verified == {}


def test_builtin_descriptors_define_estimated_runtime():
    for path in _ADAPTERS_ROOT.glob("*/descriptor.yaml"):
        if path.parent.name.startswith("_"):
            continue
        descriptor = load_descriptor(path)
        assert descriptor.estimated_runtime_s >= 0, descriptor.name


def test_missing_required_field_raises_invalid_params(tmp_path):
    raw = yaml.safe_load(_NOOP_DESCRIPTOR.read_text())
    raw.pop("name")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(VPAError) as ei:
        load_descriptor(bad)
    assert ei.value.code == ErrorCode.INVALID_PARAMS
    assert isinstance(ei.value, InvalidParamsError)
    assert "name" in str(ei.value).lower() or "name" in str(ei.value.details).lower()


def test_invalid_enum_raises_invalid_params(tmp_path):
    raw = yaml.safe_load(_NOOP_DESCRIPTOR.read_text())
    raw["category"] = "not_a_real_category"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(InvalidParamsError) as ei:
        load_descriptor(bad)
    assert ei.value.code == ErrorCode.INVALID_PARAMS
    assert ei.value.details["field"] == "category"


def test_malformed_yaml_raises_invalid_params(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: noop\n  bad indent: x: y:\n")
    with pytest.raises(InvalidParamsError):
        load_descriptor(bad)


def test_load_descriptor_accepts_json(tmp_path):
    raw = yaml.safe_load(_NOOP_DESCRIPTOR.read_text())
    p = tmp_path / "descriptor.json"
    p.write_text(json.dumps(raw))
    d = load_descriptor(p)
    assert d.name == "noop"
