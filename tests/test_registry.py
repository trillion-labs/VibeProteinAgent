from __future__ import annotations

import pytest

from vibeproteinagent.adapters import ToolAdapter, ToolDescriptor, ToolRegistry
from vibeproteinagent.adapters.noop.adapter import NoopAdapter
from vibeproteinagent.adapters.proteinmpnn.adapter import ProteinMPNNAdapter
from vibeproteinagent.core.errors import ErrorCode, UnknownToolError


def test_discover_finds_builtins():
    r = ToolRegistry()
    r.discover()
    names = {d.name for d in r.list()}
    assert {"noop", "proteinmpnn"}.issubset(names)


def test_get_returns_descriptor_and_class():
    r = ToolRegistry()
    desc, cls = r.get("noop")
    assert isinstance(desc, ToolDescriptor)
    assert desc.name == "noop"
    assert issubclass(cls, ToolAdapter)
    assert cls is NoopAdapter


def test_get_unknown_tool_raises_e_unknown_tool():
    r = ToolRegistry()
    with pytest.raises(UnknownToolError) as ei:
        r.get("nope")
    assert ei.value.code == ErrorCode.UNKNOWN_TOOL


def test_get_with_version_specifier():
    r = ToolRegistry()
    desc, cls = r.get("noop", ">=1.0.0")
    assert desc.version == "1.0.0"
    assert cls is NoopAdapter

    with pytest.raises(UnknownToolError):
        r.get("noop", ">=2.0.0")


def test_list_filters_by_category():
    r = ToolRegistry()
    misc = r.list(category="misc")
    names = {d.name for d in misc}
    assert "noop" in names
    assert "proteinmpnn" not in names

    inverse = r.list(category="inverse_folding")
    names = {d.name for d in inverse}
    assert "proteinmpnn" in names


def test_list_is_sorted():
    r = ToolRegistry()
    items = r.list()
    keys = [(d.category.value, d.name, d.version) for d in items]
    assert keys == sorted(keys)


def test_refresh_re_runs_discovery():
    r = ToolRegistry()
    r.discover()
    before = {d.name for d in r.list()}
    r.refresh()
    after = {d.name for d in r.list()}
    assert before == after
    assert "noop" in after


def test_proteinmpnn_resolves_to_shell_class():
    r = ToolRegistry()
    desc, cls = r.get("proteinmpnn")
    assert cls is ProteinMPNNAdapter
    assert desc.gpu_required is True
